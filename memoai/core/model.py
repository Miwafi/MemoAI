import torch
import torch.nn as nn
import math
from memoai.config.config import TrainingConfig
from memoai.config.optimized_config import OptimizedModelConfig, OptimizedTrainingConfig
from memoai.core.lora import add_lora_to_model, enable_lora, freeze_model_except_lora
import os
import sys
import importlib.util
try:
    from torch.utils.checkpoint import checkpoint_sequential
    has_checkpoint = True
except ImportError:
    try:
        from torch.utils.checkpoint import checkpoint
        has_checkpoint = True
    except ImportError:
        has_checkpoint = False
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_spec = importlib.util.spec_from_file_location("memoai.config.config", os.path.join(project_root, "config", "config.py"))
config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config_module)
ModelConfig = config_module.ModelConfig
UseOptimizedConfig = False
if UseOptimizedConfig:
    ModelConfig = OptimizedModelConfig
    TrainingConfig = OptimizedTrainingConfig
else:
    ModelConfig = config_module.ModelConfig
    TrainingConfig = config_module.TrainingConfig
class RotaryEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=4096):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        position_ids = torch.arange(max_seq_len)
        sinusoid_inp = torch.einsum('i,j->ij', position_ids, self.inv_freq)
        sin_val = torch.sin(sinusoid_inp)
        cos_val = torch.cos(sinusoid_inp)
        sin = torch.zeros((max_seq_len, d_model))
        cos = torch.zeros((max_seq_len, d_model))
        sin[:, 0::2] = sin_val
        sin[:, 1::2] = sin_val
        cos[:, 0::2] = cos_val
        cos[:, 1::2] = cos_val
        self.register_buffer('sin', sin)
        self.register_buffer('cos', cos)
    def forward(self, x):
        seq_len = x.shape[1]
        sin = self.sin[:seq_len].to(x.device)
        cos = self.cos[:seq_len].to(x.device)
        x1 = x[..., 0::2] * cos[:, 0::2] - x[..., 1::2] * sin[:, 0::2]
        x2 = x[..., 0::2] * sin[:, 1::2] + x[..., 1::2] * cos[:, 1::2]
        x_out = torch.stack([x1, x2], dim=-1).reshape_as(x)
        return x_out
class FlashAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.size()
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        return output
class MemoAttention(nn.Module):
    def __init__(self, d_model, nhead, use_flash_attention=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.use_flash_attention = use_flash_attention
        if use_flash_attention:
            self.attention = FlashAttention(d_model, nhead)
        else:
            self.head_dim = d_model // nhead
            assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
            self.q_proj_weight = nn.Parameter(torch.randn(d_model, d_model))
            self.k_proj_weight = nn.Parameter(torch.randn(d_model, d_model))
            self.v_proj_weight = nn.Parameter(torch.randn(d_model, d_model))
            self.out_proj_weight = nn.Parameter(torch.randn(d_model, d_model))
            self.gate_weight = nn.Parameter(torch.randn(d_model, d_model))
            self.gate_bias = nn.Parameter(torch.randn(d_model))
            self.scale_weight = nn.Parameter(torch.randn(nhead, self.head_dim))
    def forward(self, q, k, v, mask=None):
        if self.use_flash_attention:
            return self.attention(q, k, v, mask)
        else:
            q = torch.matmul(q, self.q_proj_weight)
            k = torch.matmul(k, self.k_proj_weight)
            v = torch.matmul(v, self.v_proj_weight)
            q = q.view(q.size(0), q.size(1), self.nhead, self.head_dim).transpose(1, 2)
            k = k.view(k.size(0), k.size(1), self.nhead, self.head_dim).transpose(1, 2)
            v = v.view(v.size(0), v.size(1), self.nhead, self.head_dim).transpose(1, 2)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scale = torch.sigmoid(torch.einsum('bnsh,nh->bns', q, self.scale_weight)) * 2.0
            attn_scores = attn_scores * scale.unsqueeze(-1)
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            output = torch.matmul(attn_weights, v)
            output = output.transpose(1, 2).contiguous().view(output.size(0), output.size(2), self.d_model)
            gate = torch.sigmoid(torch.matmul(output, self.gate_weight) + self.gate_bias)
            output = output * gate
            output = torch.matmul(output, self.out_proj_weight)
            return output
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(dim_feedforward)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = x * self.residual_scale + residual
        return x
class Expert(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
    def forward(self, x):
        return self.ffn(x)
class MoEFeedForward(nn.Module):
    def __init__(self, d_model, num_experts=8, expert_capacity=64, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.experts = nn.ModuleList([
            Expert(d_model, dim_feedforward, dropout) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)
        self.gate_noise = 1.0
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x_reshaped = x.view(-1, self.d_model)
        gate_logits = self.gate(x_reshaped)
        if self.training:
            noise = torch.randn_like(gate_logits) * self.gate_noise
            gate_logits = gate_logits + noise
        top_k_logits, top_k_indices = torch.topk(gate_logits, k=2, dim=-1)
        top_k_weights = torch.softmax(top_k_logits, dim=-1)
        output = torch.zeros_like(x_reshaped)
        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            if expert_mask.any():
                expert_input = x_reshaped[expert_mask]
                if expert_input.size(0) > self.expert_capacity:
                    perm = torch.randperm(expert_input.size(0))
                    expert_input = expert_input[perm[:self.expert_capacity]]
                    expert_mask[expert_mask] = False
                    expert_mask[expert_mask.nonzero()[perm[:self.expert_capacity]]] = True
                expert_output = self.experts[expert_idx](expert_input)
                weights = top_k_weights[expert_mask]
                weights = weights.sum(dim=-1, keepdim=True)
                output[expert_mask] = expert_output * weights
        output = output.view(batch_size, seq_len, self.d_model)
        return output
class MemoLayer(nn.Module):
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
        self.use_rms_norm = False
        if self.use_rms_norm:
            self.norm1 = nn.RMSNorm(d_model)
            self.norm2 = nn.RMSNorm(d_model)
    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.feed_forward(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
class MemoAI(nn.Module):
    def __init__(self, config=ModelConfig(), training_config=TrainingConfig(), use_optimizations=True, use_lora=False):
        self.use_model_parallel = False
        print(f"模型并行配置: {self.use_model_parallel}")
        super().__init__()
        self.config = config
        self.training_config = training_config
        self.use_optimizations = use_optimizations
        self.use_lora = use_lora
        self.lora_model = None
        self.device_map = None
        self.device = torch.device('cpu')
        print("已强制设置为使用CPU")
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding = self.embedding.to(self.device)
        if self.config.use_rotary_embedding:
            self.pos_encoder = RotaryEmbedding(config.hidden_size, config.max_seq_len)
        else:
            self.pos_encoder = nn.Embedding(config.max_seq_len, config.hidden_size)
            pos_emb = torch.zeros(config.max_seq_len, config.hidden_size)
            position = torch.arange(0, config.max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.hidden_size, 2).float() * (-math.log(10000.0) / config.hidden_size))
            pos_emb[:, 0::2] = torch.sin(position * div_term)
            pos_emb[:, 1::2] = torch.cos(position * div_term)
            self.pos_encoder.weight.data.copy_(pos_emb)
            self.pos_encoder.weight.requires_grad = False
        self.pos_encoder = self.pos_encoder.to(self.device)
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            layer = MemoLayer(config.hidden_size,
                              config.num_heads,
                              config.hidden_size * 4,
                              config.dropout_rate,
                              use_flash_attention=config.use_flash_attention,
                              use_moe=config.use_moe,
                              num_experts=getattr(config, 'num_experts', 8),
                              expert_capacity=getattr(config, 'expert_capacity', 64)
            )
            layer = layer.to(self.device)
            self.layers.append(layer)
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)
        self.output_layer = self.output_layer.to(self.device)
        self._init_weights()
        if self.use_optimizations:
            self._setup_gradient_checkpointing()
            print("已启用优化，包括梯度检查点功能")
        self.quantization_config = None
        if hasattr(self.config, 'quantization') and self.config.quantization:
            print(f"量化配置: 类型={self.config.quantization_type}, 动态量化={self.config.dynamic_quantization}, QAT={self.config.qat}")
            self._setup_quantization()
            print("已启用模型量化")
        else:
            print("未启用量化")
        if self.use_lora:
            print(f"已启用LoRA，秩: {self.training_config.lora_rank}，alpha: {self.training_config.lora_alpha}")
            self.lora_layers = add_lora_to_model(
                self,
                lora_rank=self.training_config.lora_rank,
                lora_alpha=self.training_config.lora_alpha
            )
            freeze_model_except_lora(self)
    def _setup_gradient_checkpointing(self):
        if self.config.gradient_checkpointing:
            global has_checkpoint
            if has_checkpoint:
                original_forward = self.forward
                def checkpointed_forward(input_ids, attention_mask=None):
                    x = self.embedding(input_ids)
                    if self.config.use_rotary_embedding:
                        x = self.pos_encoder(x)
                    else:
                        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
                        x = x + self.pos_encoder(positions)
                    if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
                        src_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    else:
                        src_mask = torch.ones((x.size(0), 1, 1, x.size(1)), device=x.device)
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
                            for layer in self.layers:
                                x = layer(x, src_mask)
                    logits = self.output_layer(x)
                    return logits
                self.forward = checkpointed_forward
                print("已启用梯度检查点功能")
            else:
                print("警告: 梯度检查点不可用，将使用常规前向传播")
        else:
            print("梯度检查点已禁用")
    def enable_lora(self, enable=True):
        enable_lora(self, enable)
    def freeze_model(self, freeze=True):
        freeze_model_except_lora(self, freeze)
    def _setup_quantization(self):
        try:
            if self.config.quantization_type == 'int8':
                import torch.ao.quantization as quantization
                from torch.ao.quantization import QConfig, get_default_qconfig_mapping
                from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
                try:
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
                self.half()
            elif self.config.quantization_type == 'bf16':
                print("使用BF16量化")
                self.bfloat16()
            else:
                print(f"不支持的量化类型: {self.config.quantization_type}")
                self.quantization_config = None
        except ImportError:
            print("警告: PyTorch量化模块不可用")
            self.quantization_config = None
    def _setup_model_parallel(self):
        num_gpus = torch.cuda.device_count()
        print(f"使用模型并行，可用GPU数量: {num_gpus}")
        self.device_map = {}
        total_layers = self.config.num_layers
        layers_per_gpu = total_layers // num_gpus
        self.device_map['embedding'] = 0
        self.device_map['output_layer'] = 0
        for i in range(total_layers):
            gpu_id = i // layers_per_gpu
            if gpu_id >= num_gpus:
                gpu_id = num_gpus - 1
            self.device_map[f'layers.{i}'] = gpu_id
        self.device = torch.device(f'cuda:{self.device_map["embedding"]}')
        self.embedding = self.embedding.to(self.device)
    def apply_quantization(self, calibration_data=None):
        if not self.config.quantization or self.quantization_config is None:
            print("量化未启用或配置无效")
            return
        if self.config.quantization_type == 'int8' and not self.config.dynamic_quantization and not self.config.qat:
            if calibration_data is None:
                print("警告: 静态量化需要校准数据")
                return
            print("应用静态INT8量化")
            with torch.no_grad():
                for batch in calibration_data:
                    self(batch)
            from torch.ao.quantization.quantize_fx import convert_fx
            self = convert_fx(self)
        print("量化应用完成")
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='tanh')
            elif p.dim() == 1:
                nn.init.constant_(p, 0.0)
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            src_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            src_mask = torch.ones((x.size(0), 1, 1, x.size(1)), device=x.device)
        if self.use_model_parallel and self.device_map is not None:
            for i, layer in enumerate(self.layers):
                current_gpu_id = self.device_map.get(f'layers.{i}', 0)
                current_device = torch.device(f'cuda:{current_gpu_id}')
                x = x.to(current_device)
                src_mask = src_mask.to(current_device)
                x = layer(x, src_mask)
        else:
            for layer in self.layers:
                x = layer(x, src_mask)
        if self.use_model_parallel and self.device_map is not None:
            output_gpu_id = self.device_map.get('output_layer', 0)
            output_device = torch.device(f'cuda:{output_gpu_id}')
            x = x.to(output_device)
        logits = self.output_layer(x)
        return logits
    def enable_lora(self, enable: bool = True) -> None:
        enable_lora(self, enable)
        print(f"{'已启用' if enable else '已禁用'}LoRA")
    def freeze_model(self, freeze: bool = True) -> None:
        freeze_model_except_lora(self, freeze)
        print(f"模型主参数{'已冻结' if freeze else '已解冻'}")
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'training_config': model.training_config
    }, path)
    print(f"模型已保存到: {path}")
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    training_config = checkpoint['training_config']
    model = MemoAI(config=config, training_config=training_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型已从: {path} 加载")
    return model
def load_pretrained_model(model_name='memo-2', cache_dir='~/.cache/memoai'):
    cache_dir = os.path.expanduser(cache_dir)
    model_dir = os.path.join(cache_dir, model_name)
    model_path = os.path.join(model_dir, 'model.pth')
    if os.path.exists(model_path):
        print(f"从缓存加载预训练模型: {model_path}")
        return load_model(model_path)
    print(f"模型 {model_name} 未找到，创建新模型实例")
    config = ModelConfig()
    model = MemoAI(config)
    return model
def fine_tune(model, train_data, val_data, config=TrainingConfig()):
    if config.use_lora:
        pass
    return model