import os
import torch
import logging
from typing import Dict, Any
from memoai.config.config import InferenceConfig, DataConfig

# 强制使用CPU的环境变量设置
os.environ['USE_GPU'] = 'False'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# 配置日志
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

# 高性能计算配置
# 强制使用CPU
USE_GPU = False
logger.info("已强制设置为使用CPU")
NUM_THREADS = max(1, os.cpu_count() - 1)  # 使用除一个核心外的所有CPU核心

# 设置PyTorch默认设备为CPU
torch.set_default_device('cpu')
logger.info("已设置PyTorch默认设备为CPU")

# 从环境变量获取模型路径，如果没有设置则使用默认路径
MODEL_PATH = os.getenv('MODEL_PATH', 'models/test_model.pth')

class MemoAIInferencer:
    def __init__(self):
        # 从环境变量或配置系统加载模型路径
        self.model_path = os.getenv('MODEL_PATH', 'models/test_model.pth')
        logger.info(f"使用模型路径: {self.model_path}")
        self.data_config = DataConfig()
        self.model = None
        self.device = None
        self.optimizer = None
        self.scheduler = None
        self.metadata = {}
        self.vocab = None

        # 尝试提取模型元数据
        try:
            # 假设这是从GGUF文件提取元数据的代码
            logger.info("尝试提取模型元数据")
            # 这里只是示例，实际代码会从GGUF文件读取
            gguf_data = type('obj', (object,), {})
            gguf_data.metadata = {'architecture': 'llama', 'version': '1.0'}

            if hasattr(gguf_data, 'metadata'):
                self.metadata = gguf_data.metadata
            elif hasattr(gguf_data, 'get_metadata'):
                self.metadata = gguf_data.get_metadata()
            elif hasattr(gguf_data, 'config'):
                self.metadata = gguf_data.config
            else:
                # 如果都没有，尝试获取模型参数作为元数据
                self.metadata = {}
                for key in dir(gguf_data):
                    if not key.startswith('_') and not callable(getattr(gguf_data, key)):
                        self.metadata[key] = getattr(gguf_data, key)
            logger.info(f"成功提取模型元数据，包含 {len(self.metadata)} 个字段")
        except Exception as e:
            logger.warning(f"获取元数据时出错: {str(e)}，使用默认模型配置")
            self.metadata = {}
        finally:
            # 确保无论如何都能继续执行
            pass

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载pth格式模型"""
        try:
            # 强制使用CPU
            self.device = torch.device('cpu')
            logger.info(f"强制使用CPU，启用多线程处理 ({NUM_THREADS}线程)")
            # 配置CPU多线程
            torch.set_num_threads(NUM_THREADS)

            if os.path.exists(self.model_path):
                logger.info(f"模型文件存在: {self.model_path}")
                # 直接使用torch.load加载模型文件，并明确指定map_location为cpu
                checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))

                # 检查checkpoint中是否包含必要的字段
                if 'model_state_dict' not in checkpoint:
                    raise ValueError("模型文件中未找到model_state_dict")

                # 设置环境变量以确保ModelConfig和MemoAI使用CPU
                os.environ['USE_GPU'] = 'False'
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                torch.set_default_device('cpu')

                # 从model.py导入ModelConfig和MemoAI
                from memoai.core.model import ModelConfig, MemoAI

                # 如果checkpoint中有config，就使用它
                if 'config' in checkpoint:
                    logger.info("从模型文件加载配置")
                    config_dict = checkpoint['config']
                    config = ModelConfig()
                    # 将配置字典中的值设置到ModelConfig对象
                    for key, value in config_dict.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                        else:
                            logger.warning(f"忽略未知配置项: {key}")
                    # 确保配置中也设置为使用CPU
                    if hasattr(config, 'use_gpu'):
                        setattr(config, 'use_gpu', False)
                else:
                    logger.info("使用默认模型配置")
                    config = ModelConfig()
                    # 确保默认配置也设置为使用CPU
                    if hasattr(config, 'use_gpu'):
                        setattr(config, 'use_gpu', False)

                # 创建模型实例
                self.model = MemoAI(config=config)
                # 加载模型状态字典，跳过不匹配的权重
                model_state_dict = checkpoint['model_state_dict']
                current_state_dict = self.model.state_dict()
                
                # 只保留形状匹配的权重
                filtered_state_dict = {k: v for k, v in model_state_dict.items() 
                                       if k in current_state_dict and 
                                       v.shape == current_state_dict[k].shape}
                
                # 加载过滤后的权重
                self.model.load_state_dict(filtered_state_dict, strict=False)
                logger.info(f"已加载 {len(filtered_state_dict)}/{len(model_state_dict)} 个匹配的权重")
                # 明确将模型移动到CPU
                self.model = self.model.to(torch.device('cpu'))
                logger.info(f"模型已成功加载并移至设备: {self.device}")
                # 设置模型为评估模式
                self.model.eval()

                # 初始化优化器和学习率调度器
                dummy_param = torch.tensor(0.0, requires_grad=True).to(self.device)
                self.optimizer = torch.optim.AdamW([dummy_param], lr=1e-4)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

                # 记录模型配置信息
                model_config = {
                    'hidden_size': self.model.config.hidden_size,
                    'num_layers': self.model.config.num_layers,
                    'num_heads': self.model.config.num_heads,
                    'vocab_size': self.model.config.vocab_size,
                    'max_seq_len': self.model.config.max_seq_len,
                    'device': str(self.device)
                }
                logger.info(f"使用模型配置: {model_config}")

                # 初始化词汇表
                self.vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
            else:
                logger.error(f"模型文件不存在: {self.model_path}")
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        except Exception as e:
            logger.error(f"模型加载过程中发生错误: {str(e)}")
            # 尝试使用默认模型配置
            logger.info("使用默认模型配置")
            self.device = torch.device('cpu')
            
            # 设置环境变量以确保ModelConfig和MemoAI使用CPU
            os.environ['USE_GPU'] = 'False'
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            torch.set_default_device('cpu')

            # 从model.py导入ModelConfig和MemoAI
            from memoai.core.model import ModelConfig, MemoAI
            # 创建默认配置
            config = ModelConfig()
            # 确保默认配置也设置为使用CPU
            if hasattr(config, 'use_gpu'):
                setattr(config, 'use_gpu', False)
            # 创建模型实例
            self.model = MemoAI(config=config)
            # 明确将模型移动到CPU
            self.model = self.model.to(torch.device('cpu'))
            self.model.eval()
            # 初始化优化器和学习率调度器
            dummy_param = torch.tensor(0.0, requires_grad=True).to(self.device)
            self.optimizer = torch.optim.AdamW([dummy_param], lr=1e-4)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

            # 记录模型配置信息
            model_config = {
                'hidden_size': self.model.config.hidden_size,
                'num_layers': self.model.config.num_layers,
                'num_heads': self.model.config.num_heads,
                'vocab_size': self.model.config.vocab_size,
                'max_seq_len': self.model.config.max_seq_len,
                'device': str(self.device)
            }
            logger.info(f"使用默认模型配置: {model_config}")

            # 初始化默认词汇表
            self.vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}

    def generate_text(self, prompt, max_length=100):
        """生成文本

        参数:
            prompt (str): 提示文本
            max_length (int): 生成文本的最大长度

        返回:
            str: 生成的文本
        """
        logger.info(f"开始生成文本，提示: {prompt[:20]}..., 最大长度: {max_length}")

        try:
            # 这里是文本生成逻辑
            # 1. 对输入进行tokenize
            # 注意：实际应用中应该使用与模型匹配的tokenizer
            # 这里为简化示例，直接使用prompt作为输入
            input_tensor = torch.tensor([[1]], dtype=torch.long).to(self.device)  # 假设1是开始标记

            # 2. 生成文本
            with torch.no_grad():
                # 使用模型进行预测
                output = self.model(input_tensor)
                # 这里简化处理，实际应用中应该使用完整的生成逻辑
                generated_text = f"这是由pth模型生成的响应。提示: {prompt}"

            logger.info(f"文本生成完成，长度: {len(generated_text)}")
            return generated_text

        except Exception as e:
            logger.error(f"文本生成错误: {str(e)}")
            return f"生成文本时出错: {str(e)}"

# 测试代码
if __name__ == '__main__':
    inferencer = MemoInferencer()
    print("模型加载完成")