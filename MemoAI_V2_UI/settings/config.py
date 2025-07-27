import json
import os

class Config:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 加载配置项
        self.model_path = config_data.get('model_path', 'model/dialog_model.pth')
        self.use_gpu = config_data.get('use_gpu', True)
        self.temperature = config_data.get('temperature', 5.0)
        self.log_dir = config_data.get('log_dir', 'log')
        self.memory_dir = config_data.get('memory_dir', 'memory')
        self.vocab_path = config_data.get('vocab_path', 'model/vocab.json')
        self.hidden_size = config_data.get('hidden_size', 256)
        # 添加num_layers配置
        self.num_layers = config_data.get('num_layers', 2)
        # 添加embedding_dim配置
        self.embedding_dim = config_data.get('embedding_dim', 128)
        self.learning_rate = config_data.get('learning_rate', 0.001)
        self.fetch_from_huggingface = config_data.get('fetch_from_huggingface', False)
        
        # 设备配置
        self.device = 'cuda' if (self.use_gpu and self._is_cuda_available()) else 'cpu'

    @staticmethod
    def _is_cuda_available():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False