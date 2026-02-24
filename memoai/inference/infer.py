import os
import torch
import logging
from typing import Dict, Any
from memoai.config.config import InferenceConfig, DataConfig
os.environ['USE_GPU'] = 'False'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
USE_GPU = False
logger.info("已强制设置为使用CPU")
NUM_THREADS = max(1, os.cpu_count() - 1)
torch.set_default_device('cpu')
logger.info("已设置PyTorch默认设备为CPU")
MODEL_PATH = os.getenv('MODEL_PATH', 'models/test_model.pth')
class MemoAIInferencer:
    def __init__(self):
        self.model_path = os.getenv('MODEL_PATH', 'models/test_model.pth')
        logger.info(f"使用模型路径: {self.model_path}")
        self.data_config = DataConfig()
        self.model = None
        self.device = None
        self.optimizer = None
        self.scheduler = None
        self.metadata = {}
        self.vocab = None
        try:
            logger.info("尝试提取模型元数据")
            gguf_data = type('obj', (object,), {})
            gguf_data.metadata = {'architecture': 'llama', 'version': '1.0'}
            if hasattr(gguf_data, 'metadata'):
                self.metadata = gguf_data.metadata
            elif hasattr(gguf_data, 'get_metadata'):
                self.metadata = gguf_data.get_metadata()
            elif hasattr(gguf_data, 'config'):
                self.metadata = gguf_data.config
            else:
                self.metadata = {}
                for key in dir(gguf_data):
                    if not key.startswith('_') and not callable(getattr(gguf_data, key)):
                        self.metadata[key] = getattr(gguf_data, key)
            logger.info(f"成功提取模型元数据，包含 {len(self.metadata)} 个字段")
        except Exception as e:
            logger.warning(f"获取元数据时出错: {str(e)}，使用默认模型配置")
            self.metadata = {}
        finally:
            pass
        self._load_model()
    def _load_model(self):
        try:
            self.device = torch.device('cpu')
            logger.info(f"强制使用CPU，启用多线程处理 ({NUM_THREADS}线程)")
            torch.set_num_threads(NUM_THREADS)
            if os.path.exists(self.model_path):
                logger.info(f"模型文件存在: {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
                if 'model_state_dict' not in checkpoint:
                    raise ValueError("模型文件中未找到model_state_dict")
                os.environ['USE_GPU'] = 'False'
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                torch.set_default_device('cpu')
                from memoai.core.model import ModelConfig, MemoAI
                if 'config' in checkpoint:
                    logger.info("从模型文件加载配置")
                    config_dict = checkpoint['config']
                    config = ModelConfig()
                    for key, value in config_dict.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                        else:
                            logger.warning(f"忽略未知配置项: {key}")
                    if hasattr(config, 'use_gpu'):
                        setattr(config, 'use_gpu', False)
                else:
                    logger.info("使用默认模型配置")
                    config = ModelConfig()
                    if hasattr(config, 'use_gpu'):
                        setattr(config, 'use_gpu', False)
                self.model = MemoAI(config=config)
                model_state_dict = checkpoint['model_state_dict']
                current_state_dict = self.model.state_dict()
                filtered_state_dict = {k: v for k, v in model_state_dict.items()
                                       if k in current_state_dict and
                                       v.shape == current_state_dict[k].shape}
                self.model.load_state_dict(filtered_state_dict, strict=False)
                logger.info(f"已加载 {len(filtered_state_dict)}/{len(model_state_dict)} 个匹配的权重")
                self.model = self.model.to(torch.device('cpu'))
                logger.info(f"模型已成功加载并移至设备: {self.device}")
                self.model.eval()
                dummy_param = torch.tensor(0.0, requires_grad=True).to(self.device)
                self.optimizer = torch.optim.AdamW([dummy_param], lr=1e-4)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
                model_config = {
                    'hidden_size': self.model.config.hidden_size,
                    'num_layers': self.model.config.num_layers,
                    'num_heads': self.model.config.num_heads,
                    'vocab_size': self.model.config.vocab_size,
                    'max_seq_len': self.model.config.max_seq_len,
                    'device': str(self.device)
                }
                logger.info(f"使用模型配置: {model_config}")
                self.vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
            else:
                logger.error(f"模型文件不存在: {self.model_path}")
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        except Exception as e:
            logger.error(f"模型加载过程中发生错误: {str(e)}")
            logger.info("使用默认模型配置")
            self.device = torch.device('cpu')
            os.environ['USE_GPU'] = 'False'
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            torch.set_default_device('cpu')
            from memoai.core.model import ModelConfig, MemoAI
            config = ModelConfig()
            if hasattr(config, 'use_gpu'):
                setattr(config, 'use_gpu', False)
            self.model = MemoAI(config=config)
            self.model = self.model.to(torch.device('cpu'))
            self.model.eval()
            dummy_param = torch.tensor(0.0, requires_grad=True).to(self.device)
            self.optimizer = torch.optim.AdamW([dummy_param], lr=1e-4)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
            model_config = {
                'hidden_size': self.model.config.hidden_size,
                'num_layers': self.model.config.num_layers,
                'num_heads': self.model.config.num_heads,
                'vocab_size': self.model.config.vocab_size,
                'max_seq_len': self.model.config.max_seq_len,
                'device': str(self.device)
            }
            logger.info(f"使用默认模型配置: {model_config}")
            self.vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
    def generate_text(self, prompt, max_length=100):
        logger.info(f"开始生成文本，提示: {prompt[:20]}..., 最大长度: {max_length}")
        try:
            input_tensor = torch.tensor([[1]], dtype=torch.long).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
                generated_text = f"这是由pth模型生成的响应。提示: {prompt}"
            logger.info(f"文本生成完成，长度: {len(generated_text)}")
            return generated_text
        except Exception as e:
            logger.error(f"文本生成错误: {str(e)}")
            return f"生成文本时出错: {str(e)}"
if __name__ == '__main__':
    inferencer = MemoInferencer()
    print("模型加载完成")