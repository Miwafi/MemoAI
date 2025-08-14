# -*- coding: utf-8 -*-
"""
MemoAI 配置文件
这里存放着MemoAI的所有超参数和配置项
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class ModelConfig:
    """模型配置类 - 定义模型的各种超参数
"""
    # 模型架构参数
    hidden_size = 4096  # 隐藏层大小 - 增大到4096以提升模型容量
    num_layers = 32     # 网络层数 - 增加到32层以提升模型深度
    num_heads = 32      # 注意力头数 - 增加到32头以提升模型感知能力
    vocab_size = 100000 # 词汇表大小 - 扩展到10万以支持更多词汇
    max_seq_len = 4096  # 最大序列长度 - 增加到4096以处理更长文本
    dropout_rate = 0.1  # dropout率 - 保持0.1防止过拟合

    # 高级特性
    use_flash_attention = True  # 使用Flash Attention加速计算
    use_rotary_embedding = True # 使用旋转位置编码
    use_moe = True              # 使用专家混合模型
    num_experts = 8             # 专家数量
    expert_capacity = 64        # 每个专家的容量
    moe_gate_noise = 1.0        # MoE门控噪声

    # 量化配置
    quantization = True         # 启用INT8量化
    quantization_type = 'int8'  # 量化类型: 'int8', 'fp16', 'bf16'
    dynamic_quantization = True  # 启用动态量化
    qat = False                 # 启用量化感知训练

    # 模型并行配置
    use_model_parallel = False  # 启用模型并行
    parallel_strategy = 'layer_wise' # 并行策略: 'layer_wise', 'tensor_wise'

    # 内存优化配置
    gradient_checkpointing = True  # 启用梯度检查点以节省内存

class TrainingConfig:
    """训练配置类 - 定义训练过程的参数
"""
    batch_size = 64        # 批次大小 - 增大批次以加速训练
    learning_rate = 2e-5   # 学习率 - 稍降低学习率以适应更大模型
    epochs = 30            # 训练轮数 - 增加到30轮以充分训练
    warmup_steps = 5000    # 预热步数 - 增加预热步数
    weight_decay = 0.01    # 权重衰减 - 保持0.01
    gradient_accumulation = 8  # 梯度累积 - 模拟更大批次
    use_adafactor = True        # 使用AdaFactor优化器
    use_lora = True             # 使用LoRA进行高效微调
    lora_rank = 64              # LoRA秩
    lora_alpha = 16             # LoRA alpha参数
    save_steps = 1000           # 保存模型间隔
    logging_steps = 100         # 日志记录间隔

class DataConfig:
    """数据配置类 - 定义数据处理的参数
"""
    data_dir = os.getenv("DATA_DIR", "../data")       # 数据目录
    train_file = "train.txt"   # 训练数据文件
    valid_file = "valid.txt"   # 验证数据文件
    test_file = "test.txt"     # 测试数据文件
    max_samples = 1000000      # 最大样本数量

class InferenceConfig:
    """推理配置类 - 定义推理过程的参数
"""
    model_path = os.getenv("MODEL_PATH", "models/memo-1.gguf")  # 模型路径 (GGUF格式)

    # 修复Windows路径转义问题
    llama_cpp_path = os.getenv("LLAMA_CPP_PATH", "C:\\llama.cpp\\main.exe")  # llama.cpp可执行文件路径
    temperature = 0.7                     # 温度参数 - 控制输出的多样性
    top_k = 50                            # _top_k采样 - 只从概率最高的k个词中选择
    top_p = 0.95                          # 核采样 - 累积概率超过p的词被考虑