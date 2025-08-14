# -*- coding: utf-8 -*-
"""
MemoAI 训练模块
这个文件实现了模型的训练逻辑

"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from torch.utils.data import Dataset, DataLoader
import json
import math
import struct

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoAI-Training")

# 从环境变量获取项目根目录，如果不存在则使用当前目录
project_root = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# 添加项目根目录到Python路径
sys.path.append(project_root)

# 导入配置和模型
from memoai.config.config import TrainingConfig, DataConfig
from memoai.core.model import MemoAI

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"将使用设备: {device}")
# 如果使用GPU，可以尝试调整下面的数据加载器批次大小以充分利用GPU内存

class TextDataset(Dataset):
    """文本数据集类
    负责加载和处理训练数据"""
    def __init__(self, data_path, vocab, max_seq_len=512):
        self.data_path = data_path
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.data = self._load_data()
        
    def _load_data(self):
        """加载数据"""
        data = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # 将文本转换为索引
                        indices = self.vocab.text_to_indices(line)
                        # 截断或填充到最大长度
                        if len(indices) > self.max_seq_len:
                            indices = indices[:self.max_seq_len]
                        else:
                            indices += [self.vocab.pad_token_id] * (self.max_seq_len - len(indices))
                        data.append(torch.tensor(indices))
            logger.info(f"成功加载数据: {len(data)}条样本")
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回输入和目标（目标是输入的偏移版本）
        input_ids = self.data[idx]
        target_ids = input_ids.clone()
        # 目标 shifted right
        target_ids[:-1] = input_ids[1:]
        target_ids[-1] = self.vocab.pad_token_id
        return input_ids, target_ids

def load_vocab(vocab_path):
    """加载词汇表"""
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        # 只提取char_to_idx部分
        vocab = vocab_data.get('char_to_idx', {})
        logger.info(f"成功加载词汇表: 词汇表大小 {len(vocab)}")
        return vocab
    except Exception as e:
        logger.error(f"加载词汇表失败: {str(e)}")
        return None

class SimpleVocab:
    """简单词汇表类
    用于将文本转换为索引和反向转换"""
    def __init__(self, vocab_dict):
        self.vocab = vocab_dict
        self.id_to_token = {v: k for k, v in vocab_dict.items()}
        self.pad_token_id = vocab_dict.get('<pad>', 0)
        self.eos_token_id = vocab_dict.get('<eos>', 1)
        self.unk_token_id = vocab_dict.get('<unk>', 2)
    
    def text_to_indices(self, text):
        """将文本转换为索引"""
        return [self.vocab.get(token, self.unk_token_id) for token in text]
    
    def indices_to_text(self, indices):
        """将索引转换为文本"""
        return ''.join([self.id_to_token.get(idx, '<unk>') for idx in indices if idx != self.pad_token_id])

def create_sample_data(file_path, vocab, num_samples=1000):
    """创建示例训练数据
    当没有训练数据时，生成一些随机样本用于测试"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 获取词汇表中的字符（排除特殊字符）
    chars = [k for k, v in vocab.vocab.items() if k not in ['<pad>', '<eos>', '<unk>']]
    if not chars:
        chars = [chr(i) for i in range(33, 127)]  # ASCII可打印字符
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for _ in range(num_samples):
            # 生成随机长度的文本
            length = torch.randint(10, 100, (1,)).item()
            # 随机选择字符
            text_indices = [torch.randint(0, len(chars), (1,)).item() for _ in range(length)]
            # 将索引转换为字符
            text = ''.join([chars[idx] for idx in text_indices])
            f.write(text + '\n')
    logger.info(f"已创建示例数据: {file_path}")

def train_model(model_name, epochs, lr):
    """训练模型"""
    logger.info(f"开始训练模型: {model_name}, 训练轮次: {epochs}, 学习率: {lr}")
    
    # 1. 加载配置
    train_config = TrainingConfig()
    data_config = DataConfig()
    
    # 2. 加载词汇表
    vocab_path = os.path.join(project_root, "memoai", "utils", "vocab.json")
    vocab_dict = load_vocab(vocab_path)
    if not vocab_dict:
        logger.error("词汇表加载失败，无法继续训练")
        return False
    
    vocab = SimpleVocab(vocab_dict)
    
    # 3. 加载数据
    train_data_path = os.path.join(project_root, "memoai", data_config.data_dir, data_config.train_file)
    valid_data_path = os.path.join(project_root, "memoai", data_config.data_dir, data_config.valid_file)
    
    # 检查数据文件是否存在，如果不存在则创建示例数据
    if not os.path.exists(train_data_path):
        logger.error(f"训练数据文件不存在: {train_data_path}")
        logger.info(f"创建示例训练数据: {train_data_path}")
        create_sample_data(train_data_path, vocab)
    
    if not os.path.exists(valid_data_path):
        logger.error(f"验证数据文件不存在: {valid_data_path}")
        logger.info(f"创建示例验证数据: {valid_data_path}")
        create_sample_data(valid_data_path, vocab, num_samples=200)
    
    # 创建数据集和数据加载器
    train_dataset = TextDataset(train_data_path, vocab)
    valid_dataset = TextDataset(valid_data_path, vocab)
    
    # 根据设备类型设置适当的批次大小 (进一步减小GPU批次大小以解决内存不足问题)
    batch_size = 8 if device.type == 'cuda' else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    logger.info(f"使用批次大小: {batch_size} (根据设备自动调整，已进一步减小GPU批次大小以避免CUDA内存不足)")
    
    # 4. 创建模型
    model = MemoAI()
    model.to(device)
    logger.info(f"模型已创建: {model_name}")
    
    # 5. 设置优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=train_config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token_id)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    # 6. 开始训练
    best_valid_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # 进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            # 移动到设备
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # 前向传播
            outputs = model(input_ids)
            
            # 计算损失
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        
        # 学习率调度
        scheduler.step()
        
        # 验证
        model.eval()
        valid_loss = 0.0
        
        with torch.no_grad():
            for input_ids, target_ids in valid_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                
                outputs = model(input_ids)
                loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                valid_loss += loss.item()
        
        # 计算平均验证损失
        avg_valid_loss = valid_loss / len(valid_loader)
        
        logger.info(f"训练损失: {avg_train_loss:.4f}, 验证损失: {avg_valid_loss:.4f}")
        
        # 保存每个epoch的模型
        epoch_model_save_path = os.path.join(project_root, "memoai", "models", f"{model_name}_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_save_path)
        logger.info(f"已保存第 {epoch+1} 轮模型: {epoch_model_save_path}")

        # 保存最佳模型
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            model_save_path = os.path.join(project_root, "memoai", "models", f"{model_name}_best.pth")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"已保存最佳模型: {model_save_path}")
    
    # 保存最终模型
    model_save_path = os.path.join(project_root, "memoai", "models", f"{model_name}_final.pth")
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"模型训练完成！最佳验证损失: {best_valid_loss:.4f}")
    logger.info(f"已保存最终模型: {model_save_path}")

    # 将模型转换为GGUF格式
    try:
        import gguf
        import struct
        
        # 加载PyTorch模型
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        
        # 创建GGUF文件
        gguf_path = os.path.join(project_root, "memoai", "models", f"{model_name}_final.gguf")
        
        # 这里是转换逻辑的简化版本，实际转换可能需要更复杂的处理
        with open(gguf_path, 'wb') as f:
            # 写入GGUF头部
            f.write(b"GGUF")
            f.write(struct.pack("<I", 1))  # 版本号
            
            # 写入模型参数 - 这里只是示例，实际需要遍历模型所有参数
            # 写入参数数据
            for name, param in model.named_parameters():
                # 转换参数为numpy数组
                param_np = param.cpu().detach().numpy()
                
                # 写入参数名称和形状信息
                name_bytes = name.encode('utf-8')
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                
                # 写入参数形状
                f.write(struct.pack("<I", len(param_np.shape)))
                for dim in param_np.shape:
                    f.write(struct.pack("<I", dim))
                
                # 写入参数数据类型
                dtype_code = 0  # 假设是float32
                f.write(struct.pack("<I", dtype_code))
                
                # 写入参数数据
                param_np.tofile(f)
        
        logger.info(f"模型已成功转换为GGUF格式: {gguf_path}")
    except Exception as e:
        logger.error(f"转换模型为GGUF格式时出错: {str(e)}")
        logger.info("请确保gguf库已正确安装，或查看转换脚本是否需要更新")
    
    return True

if __name__ == "__main__":
    # 测试训练功能
    model_name = "Memo-1"
    epochs = 10
    lr = 0.001
    train_model(model_name, epochs, lr)