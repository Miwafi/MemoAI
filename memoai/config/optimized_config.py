import os
from dotenv import load_dotenv
load_dotenv()
class OptimizedModelConfig:
    hidden_size = 1024
    num_layers = 12
    num_heads = 16
    vocab_size = 50000
    max_seq_len = 2048
    dropout_rate = 0.1
    use_flash_attention = True
    use_rotary_embedding = True
    use_moe = False
class OptimizedTrainingConfig:
    batch_size = 32
    learning_rate = 3e-5
    epochs = 20
    warmup_steps = 3000
    weight_decay = 0.01
    gradient_accumulation = 4
    use_adafactor = True
    use_lora = True
    lora_rank = 32
    lora_alpha = 16
    save_steps = 1000
    logging_steps = 100
class OptimizedDataConfig:
    data_dir = os.getenv("DATA_DIR", "../data")
    train_file = "train.txt"
    valid_file = "valid.txt"
    test_file = "test.txt"
    max_samples = 1000000
class OptimizedInferenceConfig:
    model_path = os.getenv("MODEL_PATH", "models/memo-1.gguf")
    llama_cpp_path = os.getenv("LLAMA_CPP_PATH", "C:\llama.cpp\main.exe")
    temperature = 0.7
    top_k = 50
    top_p = 0.95
    use_quantization = True
    quantization_bits = 4