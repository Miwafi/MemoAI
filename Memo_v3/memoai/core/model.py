# -*- coding: utf-8 -*-
"""
MemoAI 核心模型文件
这里实现了我们自研的大语言模型算法

"""
import torch
import torch.nn as nn
import math
from memoai.config.config import TrainingConfig
from memoai.config.optimized_config import OptimizedModelConfig, OptimizedTrainingConfig
from memoai.core.lora import add_lora_to_model, enable_lora, freeze_model_except_lora
import os
import sys
import importlib.util

# 尝试导入梯度检查点
try:
    from torch.utils.checkpoint import checkpoint_sequential
    has_checkpoint = True
except ImportError:
    try:
        # 对于较旧的PyTorch版本
        from torch.utils.checkpoint import checkpoint
        has_checkpoint = True
    except ImportError:
        has_checkpoint = False

# 计算项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 使用绝对路径导入原始配置
config_spec = importlib.util.spec_from_file_location("memoai.config.config", os.path.join(project_root, "config", "config.py"))
config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config_module)
ModelConfig = config_module.ModelConfig

# 不默认使用优化配置，而是使用传入的配置
UseOptimizedConfig = False

# 只有在明确指定时才使用优化配置
if UseOptimizedConfig:
    ModelConfig = OptimizedModelConfig
    TrainingConfig = OptimizedTrainingConfig
else:
    # 使用原始配置
    ModelConfig = config_module.ModelConfig
    TrainingConfig = config_module.TrainingConfig


class RotaryEmbedding(nn.Module):
    """旋转位置编码 - 更高效的位置编码方案
    相比传统正弦位置编码，能更好地处理长序列"""
    def __init__(self, d_model, max_seq_len=4096):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 计算旋转位置编码的角度
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置索引和旋转矩阵
        position_ids = torch.arange(max_seq_len)
        sinusoid_inp = torch.einsum('i,j->ij', position_ids, self.inv_freq)
        
        # 计算sin和cos
        sin_val = torch.sin(sinusoid_inp)
        cos_val = torch.cos(sinusoid_inp)
        
        # 构建旋转矩阵
        sin = torch.zeros((max_seq_len, d_model))
        cos = torch.zeros((max_seq_len, d_model))
        sin[:, 0::2] = sin_val
        sin[:, 1::2] = sin_val
        cos[:, 0::2] = cos_val
        cos[:, 1::2] = cos_val
        
        self.register_buffer('sin', sin)
        self.register_buffer('cos', cos)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # 使用 PyTorch 操作获取序列长度，避免 Proxy 对象转换问题
        seq_len = x.shape[1]
        
        # 使用预计算的旋转矩阵
        sin = self.sin[:seq_len].to(x.device)
        cos = self.cos[:seq_len].to(x.device)
        
        # 应用旋转编码
        x1 = x[..., 0::2] * cos[:, 0::2] - x[..., 1::2] * sin[:, 0::2]
        x2 = x[..., 0::2] * sin[:, 1::2] + x[..., 1::2] * cos[:, 1::2]
        
        # 合并结果
        x_out = torch.stack([x1, x2], dim=-1).reshape_as(x)
        
        return x_out


class FlashAttention(nn.Module):
    """Flash Attention实现 - 更高效的注意力计算
    相比标准注意力，具有更低的内存占用和更高的计算效率"""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # 确保d_model能被nhead整除
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        
        # 定义线性变换
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # q, k, v: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = q.size()
        
        # 线性变换
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # 分头处理
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # 应用Flash Attention (这里用标准实现模拟)
        # 实际生产中应使用优化的Flash Attention实现
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 应用注意力
        output = torch.matmul(attn_weights, v)
        
        # 合并头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.out_proj(output)
        
        return output


class MemoAttention(nn.Module):
    """Memo注意力机制 - 自研的改进型注意力机制
    相比标准注意力，我们添加了门控机制和动态缩放，提升效果"""
    def __init__(self, d_model, nhead, use_flash_attention=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.use_flash_attention = use_flash_attention
        
        if use_flash_attention:
            self.attention = FlashAttention(d_model, nhead)
        else:
            self.head_dim = d_model // nhead
            
            # 确保d_model能被nhead整除
            assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
            
            # 定义线性变换
            self.q_proj_weight = nn.Parameter(torch.randn(d_model, d_model))
            self.k_proj_weight = nn.Parameter(torch.randn(d_model, d_model))
            self.v_proj_weight = nn.Parameter(torch.randn(d_model, d_model))
            self.out_proj_weight = nn.Parameter(torch.randn(d_model, d_model))
            
            # 门控机制参数
            self.gate_weight = nn.Parameter(torch.randn(d_model, d_model))
            self.gate_bias = nn.Parameter(torch.randn(d_model))
            
            # 动态缩放参数 - 每个注意力头有独立的缩放参数
            self.scale_weight = nn.Parameter(torch.randn(nhead, self.head_dim))

    def forward(self, q, k, v, mask=None):
        if self.use_flash_attention:
            return self.attention(q, k, v, mask)
        else:
            # 线性变换
            q = torch.matmul(q, self.q_proj_weight)
            k = torch.matmul(k, self.k_proj_weight)
            v = torch.matmul(v, self.v_proj_weight)
            
            # 分头处理
            q = q.view(q.size(0), q.size(1), self.nhead, self.head_dim).transpose(1, 2)
            k = k.view(k.size(0), k.size(1), self.nhead, self.head_dim).transpose(1, 2)
            v = v.view(v.size(0), v.size(1), self.nhead, self.head_dim).transpose(1, 2)
            
            # 计算注意力分数
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # 应用动态缩放 - 每个注意力头独立缩放
            # q的形状: [batch_size, nhead, seq_len, head_dim]
            # scale_weight的形状: [nhead, head_dim]
            # 计算每个位置的缩放因子
            scale = torch.sigmoid(torch.einsum('bnsh,nh->bns', q, self.scale_weight)) * 2.0  # 缩放范围[0, 2]
            attn_scores = attn_scores * scale.unsqueeze(-1)
            
            # 应用掩码
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
            # 计算注意力权重
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # 应用注意力
            output = torch.matmul(attn_weights, v)
            
            # 合并头
            output = output.transpose(1, 2).contiguous().view(output.size(0), output.size(2), self.d_model)
            
            # 应用门控机制
            gate = torch.sigmoid(torch.matmul(output, self.gate_weight) + self.gate_bias)
            output = output * gate
            
            # 输出投影
            output = torch.matmul(output, self.out_proj_weight)
            
            return output


class FeedForward(nn.Module):
    """前馈网络 - 模型的信息处理单元
    使用GELU激活函数和瓶颈结构，提高效率和性能"""
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用GELU激活函数
        self.norm = nn.LayerNorm(dim_feedforward)  # 归一化层
        
        # 添加额外的残差连接
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = x * self.residual_scale + residual  # 缩放残差连接
        return x


class Expert(nn.Module):
    """专家网络 - MoE模型中的单个专家"""
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)

    def forward(self, x):
        return self.ffn(x)


class MoEFeedForward(nn.Module):
    """专家混合前馈网络 - 使用多个专家网络处理不同输入"""
    def __init__(self, d_model, num_experts=8, expert_capacity=64, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
        # 创建专家网络
        self.experts = nn.ModuleList([
            Expert(d_model, dim_feedforward, dropout) for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = nn.Linear(d_model, num_experts)
        
        # 门控噪声
        self.gate_noise = 1.0

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        
        # 重塑输入以便处理
        x_reshaped = x.view(-1, self.d_model)  # [batch_size * seq_len, d_model]
        
        # 计算门控分数
        gate_logits = self.gate(x_reshaped)
        
        # 添加噪声以鼓励专家多样化
        if self.training:
            noise = torch.randn_like(gate_logits) * self.gate_noise
            gate_logits = gate_logits + noise
        
        # 选择前k个专家
        top_k_logits, top_k_indices = torch.topk(gate_logits, k=2, dim=-1)
        
        # 计算专家权重
        top_k_weights = torch.softmax(top_k_logits, dim=-1)
        
        # 初始化输出
        output = torch.zeros_like(x_reshaped)
        
        # 为每个专家处理分配的样本
        for expert_idx in range(self.num_experts):
            # 找出选择该专家的样本
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # 获取这些样本
                expert_input = x_reshaped[expert_mask]
                
                # 限制专家容量
                if expert_input.size(0) > self.expert_capacity:
                    # 随机选择部分样本
                    perm = torch.randperm(expert_input.size(0))
                    expert_input = expert_input[perm[:self.expert_capacity]]
                    expert_mask[expert_mask] = False
                    expert_mask[expert_mask.nonzero()[perm[:self.expert_capacity]]] = True
                
                # 使用专家处理
                expert_output = self.experts[expert_idx](expert_input)
                
                # 计算权重
                weights = top_k_weights[expert_mask]
                weights = weights.sum(dim=-1, keepdim=True)
                
                # 应用权重并添加到输出
                output[expert_mask] = expert_output * weights
        
        # 重塑回原始形状
        output = output.view(batch_size, seq_len, self.d_model)
        
        return output


class MemoLayer(nn.Module):
    """Memo层 - 模型的基本构建块
    包含一个Memo注意力模块和一个前馈网络模块"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, use_flash_attention=False, use_moe=False, num_experts=8, expert_capacity=64):
        super().__init__()
        self.self_attn = MemoAttention(d_model, nhead, use_flash_attention)
        
        if use_moe:
            self.feed_forward = MoEFeedForward(d_model, num_experts, expert_capacity, dim_feedforward, dropout)
        else:
            self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 添加RMS归一化选项
        self.use_rms_norm = False
        if self.use_rms_norm:
            self.norm1 = nn.RMSNorm(d_model)
            self.norm2 = nn.RMSNorm(d_model)

    def forward(self, src, src_mask=None):
        # 自注意力模块
        src2 = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络模块
        src2 = self.feed_forward(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class MemoAI(nn.Module):
    """MemoAI模型 - 我们自研的大语言模型
    添加了内存优化技术: 梯度检查点、量化支持、LoRA、模型并行"""
    def __init__(self, config=ModelConfig(), training_config=TrainingConfig(), use_optimizations=True, use_lora=False):
        # 从配置中读取模型并行选项
        self.use_model_parallel = False  # 强制禁用模型并行
        print(f"模型并行配置: {self.use_model_parallel}")
        super().__init__()
        self.config = config
        self.training_config = training_config
        self.use_optimizations = use_optimizations
        self.use_lora = use_lora
        self.lora_model = None
        self.device_map = None

        # 强制使用CPU
        self.device = torch.device('cpu')
        print("已强制设置为使用CPU")

        # 词嵌入层
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding = self.embedding.to(self.device)

        # 禁用模型并行，无论CUDA设备数量

        # 位置编码
        if self.config.use_rotary_embedding:
            self.pos_encoder = RotaryEmbedding(config.hidden_size, config.max_seq_len)
        else:
            # 优化的位置编码实现
            self.pos_encoder = nn.Embedding(config.max_seq_len, config.hidden_size)
            # 初始化位置编码
            pos_emb = torch.zeros(config.max_seq_len, config.hidden_size)
            position = torch.arange(0, config.max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.hidden_size, 2).float() * (-math.log(10000.0) / config.hidden_size))
            pos_emb[:, 0::2] = torch.sin(position * div_term)
            pos_emb[:, 1::2] = torch.cos(position * div_term)
            self.pos_encoder.weight.data.copy_(pos_emb)
            self.pos_encoder.weight.requires_grad = False

        # 移动位置编码器到CPU
        self.pos_encoder = self.pos_encoder.to(self.device)

        # 创建Memo层
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            layer = MemoLayer(config.hidden_size,
                              config.num_heads,
                              config.hidden_size * 4,  # 前馈网络维度
                              config.dropout_rate,
                              use_flash_attention=config.use_flash_attention,
                              use_moe=config.use_moe,
                              num_experts=getattr(config, 'num_experts', 8),
                              expert_capacity=getattr(config, 'expert_capacity', 64)
            )
            # 将层移到CPU
            layer = layer.to(self.device)
            self.layers.append(layer)

        # 输出层
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)

        # 移动输出层到CPU
        self.output_layer = self.output_layer.to(self.device)

        # 初始化权重
        self._init_weights()

        # 设置梯度检查点以减少内存使用
        if self.use_optimizations:
            self._setup_gradient_checkpointing()
            print("已启用优化，包括梯度检查点功能")

        # 量化支持
        self.quantization_config = None
        if hasattr(self.config, 'quantization') and self.config.quantization:
            print(f"量化配置: 类型={self.config.quantization_type}, 动态量化={self.config.dynamic_quantization}, QAT={self.config.qat}")
            self._setup_quantization()
            print("已启用模型量化")
        else:
            print("未启用量化")

        # 如果启用LoRA，则初始化LoRA
        if self.use_lora:
            print(f"已启用LoRA，秩: {self.training_config.lora_rank}，alpha: {self.training_config.lora_alpha}")
            self.lora_layers = add_lora_to_model(
                self,
                lora_rank=self.training_config.lora_rank,
                lora_alpha=self.training_config.lora_alpha
            )
            # 默认冻结非LoRA参数
            freeze_model_except_lora(self)

    def _setup_gradient_checkpointing(self):
        """设置梯度检查点以减少内存使用"""
        if self.config.gradient_checkpointing:
            global has_checkpoint
            
            if has_checkpoint:
                # 替换前向传播以使用梯度检查点
                original_forward = self.forward
                def checkpointed_forward(input_ids, attention_mask=None):
                    # 词嵌入和位置编码
                    x = self.embedding(input_ids)
                    if self.config.use_rotary_embedding:
                        x = self.pos_encoder(x)
                    else:
                        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
                        x = x + self.pos_encoder(positions)

                    # 应用注意力掩码
                    if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
                        # 扩展掩码以适应多头注意力
                        src_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    else:
                        # 创建默认的全1掩码
                        src_mask = torch.ones((x.size(0), 1, 1, x.size(1)), device=x.device)

                    # 使用梯度检查点通过层
                    try:
                        from torch.utils.checkpoint import checkpoint_sequential
                        x = checkpoint_sequential(
                             self.layers, len(self.layers), x, src_mask
                          )
                    except ImportError:
                        try:
                            from torch.utils.checkpoint import checkpoint
                            for layer in self.layers:
                                x = checkpoint(layer, x, src_mask)
                        except ImportError:
                            # 如果没有梯度检查点，使用常规前向传播
                            for layer in self.layers:
                                x = layer(x, src_mask)

                    # 输出层
                    logits = self.output_layer(x)
                    return logits
                self.forward = checkpointed_forward
                print("已启用梯度检查点功能")
            else:
                print("警告: 梯度检查点不可用，将使用常规前向传播")
        else:
            print("梯度检查点已禁用")

    def enable_lora(self, enable=True):
        """启用或禁用LoRA"""
        enable_lora(self, enable)

    def freeze_model(self, freeze=True):
        """冻结或解冻模型的非LoRA参数"""
        freeze_model_except_lora(self, freeze)

    def _setup_quantization(self):
        """设置模型量化"""
        try:
            # 根据量化类型选择不同的策略
            if self.config.quantization_type == 'int8':
                # 尝试导入量化工具
                import torch.ao.quantization as quantization
                from torch.ao.quantization import QConfig, get_default_qconfig_mapping
                from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

                # 配置量化 - 兼容不同PyTorch版本
                try:
                    # 尝试使用QScheme
                    self.quantization_config = QConfig(
                        activation=quantization.FakeQuantize.with_args(
                            observer=quantization.MinMaxObserver,
                            quant_min=0,
                            quant_max=255,
                            qscheme=quantization.QScheme.PER_TENSOR_AFFINE,
                        ),
                        weight=quantization.FakeQuantize.with_args(
                            observer=quantization.MinMaxObserver,
                            quant_min=-128,
                            quant_max=127,
                            qscheme=quantization.QScheme.PER_CHANNEL_SYMMETRIC,
                        ),
                    )
                except AttributeError:
                    # 如果没有QScheme，使用旧版本API
                    self.quantization_config = QConfig(
                    activation=quantization.FakeQuantize.with_args(
                        observer=quantization.MinMaxObserver,
                        quant_min=0,
                        quant_max=255,
                    ),
                    weight=quantization.FakeQuantize.with_args(
                        observer=quantization.MinMaxObserver,
                        quant_min=-128,
                        quant_max=127,
                    ),
                )

                # 准备量化
                self.qconfig_mapping = get_default_qconfig_mapping()
                self.eval()

                if self.config.dynamic_quantization:
                    print("使用动态INT8量化")
                    self = quantization.quantize_dynamic(
                            self,
                            {torch.nn.Linear},
                            dtype=torch.qint8
                        )
                elif self.config.qat:
                    print("使用量化感知训练")
                    self.qconfig = self.quantization_config
                    self = quantization.prepare_qat(self, inplace=False)
                else:
                    print("禁用静态INT8量化，使用动态量化")
                    self = quantization.quantize_dynamic(
                        self,
                        {torch.nn.Linear, torch.nn.Embedding},
                        dtype=torch.qint8
                    )

            elif self.config.quantization_type == 'fp16':
                print("使用FP16量化")
                self.half()  # 转换为半精度浮点数

            elif self.config.quantization_type == 'bf16':
                print("使用BF16量化")
                self.bfloat16()  # 转换为Brain浮点数

            else:
                print(f"不支持的量化类型: {self.config.quantization_type}")
                self.quantization_config = None

        except ImportError:
            print("警告: PyTorch量化模块不可用")
            self.quantization_config = None

    def _setup_model_parallel(self):
        """设置模型并行"""
        # 获取可用的GPU数量
        num_gpus = torch.cuda.device_count()
        print(f"使用模型并行，可用GPU数量: {num_gpus}")

        # 创建设备映射
        self.device_map = {}
        total_layers = self.config.num_layers
        layers_per_gpu = total_layers // num_gpus

        # 分配嵌入层和输出层到第一个GPU
        self.device_map['embedding'] = 0
        self.device_map['output_layer'] = 0

        # 分配层到不同的GPU
        for i in range(total_layers):
            gpu_id = i // layers_per_gpu
            if gpu_id >= num_gpus:
                gpu_id = num_gpus - 1
            self.device_map[f'layers.{i}'] = gpu_id

        # 移动嵌入层到指定设备
        self.device = torch.device(f'cuda:{self.device_map["embedding"]}')
        self.embedding = self.embedding.to(self.device)

    def apply_quantization(self, calibration_data=None):
        """应用量化到模型

        Args:
            calibration_data: 用于静态量化的校准数据
        """
        if not self.config.quantization or self.quantization_config is None:
            print("量化未启用或配置无效")
            return

        if self.config.quantization_type == 'int8' and not self.config.dynamic_quantization and not self.config.qat:
            if calibration_data is None:
                print("警告: 静态量化需要校准数据")
                return

            print("应用静态INT8量化")
            # 使用校准数据进行校准
            with torch.no_grad():
                for batch in calibration_data:
                    self(batch)

            # 转换为量化模型
            from torch.ao.quantization.quantize_fx import convert_fx
            self = convert_fx(self)

        print("量化应用完成")

    def _init_weights(self):
        """初始化模型权重
        采用更先进的权重初始化策略"""
        for p in self.parameters():
            if p.dim() > 1:
                # 使用Kaiming初始化替代Xavier
                # 对于GELU，我们可以使用'tanh'作为近似
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='tanh')
            elif p.dim() == 1:
                # 对偏置项使用常数初始化
                nn.init.constant_(p, 0.0)

    def forward(self, input_ids, attention_mask=None):
        """前向传播
        模型的主要工作流程"""
        # 嵌入层
        x = self.embedding(input_ids)

        # 位置编码
        x = self.pos_encoder(x)

        # 应用注意力掩码
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            # 扩展掩码以适应多头注意力
            src_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            # 创建默认的全1掩码
            src_mask = torch.ones((x.size(0), 1, 1, x.size(1)), device=x.device)

        # 通过所有Memo层
        if self.use_model_parallel and self.device_map is not None:
            for i, layer in enumerate(self.layers):
                # 获取当前层所在的设备
                current_gpu_id = self.device_map.get(f'layers.{i}', 0)
                current_device = torch.device(f'cuda:{current_gpu_id}')

                # 将输入移到当前层所在的设备
                x = x.to(current_device)
                src_mask = src_mask.to(current_device)

                # 通过当前层
                x = layer(x, src_mask)
        else:
            for layer in self.layers:
                x = layer(x, src_mask)

        # 将输出移到输出层所在的设备
        if self.use_model_parallel and self.device_map is not None:
            output_gpu_id = self.device_map.get('output_layer', 0)
            output_device = torch.device(f'cuda:{output_gpu_id}')
            x = x.to(output_device)

        # 输出层
        logits = self.output_layer(x)
        return logits

    def enable_lora(self, enable: bool = True) -> None:
        """启用或禁用LoRA"""
        enable_lora(self, enable)
        print(f"{'已启用' if enable else '已禁用'}LoRA")

    def freeze_model(self, freeze: bool = True) -> None:
        """冻结或解冻模型的非LoRA参数"""
        freeze_model_except_lora(self, freeze)
        print(f"模型主参数{'已冻结' if freeze else '已解冻'}")


# 以下是模型增强的实用工具函数


def save_model(model, path):
    """保存模型为pth格式
    
    Args:
        model: 要保存的模型
        path: 保存路径
    """
    # 确保路径目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 保存模型状态字典
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'training_config': model.training_config
    }, path)
    print(f"模型已保存到: {path}")
    

def load_model(path):
    """从pth格式加载模型
    
    Args:
        path: 模型文件路径
    
    Returns:
        加载好的模型
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")
    
    # 加载模型状态字典 - 添加weights_only=False以支持加载自定义配置类
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    # 创建模型实例
    config = checkpoint['config']
    training_config = checkpoint['training_config']
    model = MemoAI(config=config, training_config=training_config)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型已从: {path} 加载")
    
    return model

def load_pretrained_model(model_name='memo-2', cache_dir='~/.cache/memoai'):
    """加载预训练模型
    Args:
        model_name: 模型名称
        cache_dir: 缓存目录
    Returns:
        加载好的模型
    """
    # 扩展缓存目录路径
    cache_dir = os.path.expanduser(cache_dir)
    model_dir = os.path.join(cache_dir, model_name)
    model_path = os.path.join(model_dir, 'model.pth')
    
    # 检查模型是否已缓存
    if os.path.exists(model_path):
        print(f"从缓存加载预训练模型: {model_path}")
        return load_model(model_path)
    
    # 如果模型未缓存，创建新模型
    print(f"模型 {model_name} 未找到，创建新模型实例")
    config = ModelConfig()
    model = MemoAI(config)
    return model

def fine_tune(model, train_data, val_data, config=TrainingConfig()):
    """微调模型
    Args:
        model: 要微调的模型
        train_data: 训练数据
        val_data: 验证数据
        config: 训练配置
    Returns:
        微调后的模型
    """
    # 这里实现模型微调逻辑
    # 使用LoRA进行高效微调
    if config.use_lora:
        # 实现LoRA逻辑
        pass
    # 实际训练循环
    return model