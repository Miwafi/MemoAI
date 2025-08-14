"""
MemoAI 优化配置文件
专为减少内存和显存占用而设计
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class OptimizedModelConfig:
    """优化的模型配置类 - 减少内存和显存占用
    """
    # 模型架构参数 - 减小模型规模
    hidden_size = 1024  # 隐藏层大小 - 从4096减小到1024
    num_layers = 12     # 网络层数 - 从32减小到12
    num_heads = 16      # 注意力头数 - 从32减小到16
    vocab_size = 50000  # 词汇表大小 - 从100000减小到50000
    max_seq_len = 2048  # 最大序列长度 - 从4096减小到2048
    dropout_rate = 0.1  # dropout率 - 保持不变

    # 高级特性 - 选择性启用
    use_flash_attention = True  # 保持启用以提高效率
    use_rotary_embedding = True # 保持启用以提高效率
    use_moe = False             # 禁用MoE以减少内存使用
    # num_experts = 4           # 如果启用MoE，减少专家数量
    # expert_capacity = 32      # 减小专家容量
    # moe_gate_noise = 1.0

class OptimizedTrainingConfig:
    """优化的训练配置类 - 减少内存和显存占用
    """
    batch_size = 32        # 批次大小 - 从64减小到32
    learning_rate = 3e-5   # 学习率 - 稍提高以适应较小模型
    epochs = 20            # 训练轮数 - 从30减小到20
    warmup_steps = 3000    # 预热步数 - 从5000减小到3000
    weight_decay = 0.01    # 权重衰减 - 保持不变
    gradient_accumulation = 4  # 梯度累积 - 从8减小到4
    use_adafactor = True        # 保持启用AdaFactor优化器
    use_lora = True             # 保持启用LoRA
    lora_rank = 32              # LoRA秩 - 从64减小到32
    lora_alpha = 16             # LoRA alpha参数 - 保持不变
    save_steps = 1000           # 保存模型间隔 - 保持不变
    logging_steps = 100         # 日志记录间隔 - 保持不变

class OptimizedDataConfig:
    """数据配置类 - 保持不变
    """
    data_dir = os.getenv("DATA_DIR", "../data")       # 数据目录
    train_file = "train.txt"   # 训练数据文件
    valid_file = "valid.txt"   # 验证数据文件
    test_file = "test.txt"     # 测试数据文件
    max_samples = 1000000      # 最大样本数量

class OptimizedInferenceConfig:
    """推理配置类 - 添加量化支持
    """
    model_path = os.getenv("MODEL_PATH", "models/memo-1.gguf")  # 模型路径
    llama_cpp_path = os.getenv("LLAMA_CPP_PATH", "C:\llama.cpp\main.exe")  # llama.cpp路径
    temperature = 0.7                     # 温度参数
    top_k = 50                            # top_k采样
    top_p = 0.95                          # 核采样
    use_quantization = True               # 启用量化
    quantization_bits = 4                 # 量化位数 (4bit或8bit)