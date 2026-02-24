import os
from dotenv import load_dotenv
load_dotenv()
class ModelConfig:
    hidden_size = 4096
    num_layers = 32
    num_heads = 32
    vocab_size = 100000
    max_seq_len = 4096
    dropout_rate = 0.1
    use_flash_attention = True
    use_rotary_embedding = True
    use_moe = True
    num_experts = 8
    expert_capacity = 64
    moe_gate_noise = 1.0
    quantization = True
    quantization_type = 'int8'
    dynamic_quantization = True
    qat = False
    use_model_parallel = False
    parallel_strategy = 'layer_wise'
    gradient_checkpointing = True
class TrainingConfig:
    batch_size = 64
    learning_rate = 2e-5
    epochs = 30
    warmup_steps = 5000
    weight_decay = 0.01
    gradient_accumulation = 8
    use_adafactor = True
    use_lora = True
    lora_rank = 64
    lora_alpha = 16
    save_steps = 1000
    logging_steps = 100
class DataConfig:
    data_dir = os.getenv("DATA_DIR", "../data")
    train_file = "train.txt"
    valid_file = "valid.txt"
    test_file = "test.txt"
    max_samples = 1000000
class InferenceConfig:
    model_path = os.getenv("MODEL_PATH", "models/memo-1.gguf")
    llama_cpp_path = os.getenv("LLAMA_CPP_PATH", "C:\\llama.cpp\\main.exe")
    temperature = 0.7
    top_k = 50
    top_p = 0.95