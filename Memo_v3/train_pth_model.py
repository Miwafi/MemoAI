# -*- coding: utf-8 -*-
"""
MemoAI 模型训练程序
这个文件用于训练并保存.pth格式的模型
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoAI-Training")

# 获取项目根目录
project_root = os.path.abspath(os.path.dirname(__file__))
# 添加项目根目录到Python路径
sys.path.append(project_root)

# 导入配置和模型
from memoai.config.config import TrainingConfig, DataConfig
from memoai.core.model import MemoAI

# 检查CUDA是否可用，如果不可用则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"将使用设备: {device}")

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

def train_model(model_name, epochs, lr, batch_size=4):
    """训练模型并保存为.pth格式
    参数:
        model_name: 模型名称
        epochs: 训练轮次
        lr: 学习率
        batch_size: 批次大小
    """
    logger.info(f"开始训练模型: {model_name}, 训练轮次: {epochs}, 学习率: {lr}, 批次大小: {batch_size}")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    logger.info(f"数据加载器已创建: 训练样本 {len(train_dataset)}, 验证样本 {len(valid_dataset)}")
    
    # 4. 创建模型
    model = MemoAI()
    model.to(device)
    logger.info(f"模型已创建并移至设备: {device}")
    
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
        epoch_model_save_path = os.path.join(project_root, "models", f"{model_name}_epoch_{epoch+1}.pth")
        os.makedirs(os.path.dirname(epoch_model_save_path), exist_ok=True)
        torch.save(model.state_dict(), epoch_model_save_path)
        logger.info(f"已保存第 {epoch+1} 轮模型: {epoch_model_save_path}")

        # 保存最佳模型
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            model_save_path = os.path.join(project_root, "models", f"{model_name}_best.pth")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"已保存最佳模型: {model_save_path}")
    
    # 保存最终模型
    model_save_path = os.path.join(project_root, "models", f"{model_name}_final.pth")
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"模型训练完成！最佳验证损失: {best_valid_loss:.4f}")
    logger.info(f"已保存最终模型: {model_save_path}")

    return True

if __name__ == "__main__":
    # 训练配置
    model_name = "Memo-1"
    epochs = 10
    lr = 0.001
    batch_size = 4  # 较小的批次大小以避免内存问题
    
    # 开始训练
    train_model(model_name, epochs, lr, batch_size)