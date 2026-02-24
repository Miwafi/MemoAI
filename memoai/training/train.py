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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoAI-Training")
project_root = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(project_root)
from memoai.config.config import TrainingConfig, DataConfig
from memoai.core.model import MemoAI
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"将使用设备: {device}")
class TextDataset(Dataset):
    def __init__(self, data_path, vocab, max_seq_len=512):
        self.data_path = data_path
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.data = self._load_data()
    def _load_data(self):
        data = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        indices = self.vocab.text_to_indices(line)
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
        input_ids = self.data[idx]
        target_ids = input_ids.clone()
        target_ids[:-1] = input_ids[1:]
        target_ids[-1] = self.vocab.pad_token_id
        return input_ids, target_ids
def load_vocab(vocab_path):
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        vocab = vocab_data.get('char_to_idx', {})
        logger.info(f"成功加载词汇表: 词汇表大小 {len(vocab)}")
        return vocab
    except Exception as e:
        logger.error(f"加载词汇表失败: {str(e)}")
        return None
class SimpleVocab:
    def __init__(self, vocab_dict):
        self.vocab = vocab_dict
        self.id_to_token = {v: k for k, v in vocab_dict.items()}
        self.pad_token_id = vocab_dict.get('<pad>', 0)
        self.eos_token_id = vocab_dict.get('<eos>', 1)
        self.unk_token_id = vocab_dict.get('<unk>', 2)
    def text_to_indices(self, text):
        return [self.vocab.get(token, self.unk_token_id) for token in text]
    def indices_to_text(self, indices):
        return ''.join([self.id_to_token.get(idx, '<unk>') for idx in indices if idx != self.pad_token_id])
def create_sample_data(file_path, vocab, num_samples=1000):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    chars = [k for k, v in vocab.vocab.items() if k not in ['<pad>', '<eos>', '<unk>']]
    if not chars:
        chars = [chr(i) for i in range(33, 127)]
    with open(file_path, 'w', encoding='utf-8') as f:
        for _ in range(num_samples):
            length = torch.randint(10, 100, (1,)).item()
            text_indices = [torch.randint(0, len(chars), (1,)).item() for _ in range(length)]
            text = ''.join([chars[idx] for idx in text_indices])
            f.write(text + '\n')
    logger.info(f"已创建示例数据: {file_path}")
def train_model(model_name, epochs, lr):
    logger.info(f"开始训练模型: {model_name}, 训练轮次: {epochs}, 学习率: {lr}")
    train_config = TrainingConfig()
    data_config = DataConfig()
    vocab_path = os.path.join(project_root, "memoai", "utils", "vocab.json")
    vocab_dict = load_vocab(vocab_path)
    if not vocab_dict:
        logger.error("词汇表加载失败，无法继续训练")
        return False
    vocab = SimpleVocab(vocab_dict)
    train_data_path = os.path.join(project_root, "memoai", data_config.data_dir, data_config.train_file)
    valid_data_path = os.path.join(project_root, "memoai", data_config.data_dir, data_config.valid_file)
    if not os.path.exists(train_data_path):
        logger.error(f"训练数据文件不存在: {train_data_path}")
        logger.info(f"创建示例训练数据: {train_data_path}")
        create_sample_data(train_data_path, vocab)
    if not os.path.exists(valid_data_path):
        logger.error(f"验证数据文件不存在: {valid_data_path}")
        logger.info(f"创建示例验证数据: {valid_data_path}")
        create_sample_data(valid_data_path, vocab, num_samples=200)
    train_dataset = TextDataset(train_data_path, vocab)
    valid_dataset = TextDataset(valid_data_path, vocab)
    batch_size = 8 if device.type == 'cuda' else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    logger.info(f"使用批次大小: {batch_size} (根据设备自动调整，已进一步减小GPU批次大小以避免CUDA内存不足)")
    model = MemoAI()
    model.to(device)
    logger.info(f"模型已创建: {model_name}")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=train_config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token_id)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    best_valid_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for input_ids, target_ids in valid_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                outputs = model(input_ids)
                loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                valid_loss += loss.item()
        avg_valid_loss = valid_loss / len(valid_loader)
        logger.info(f"训练损失: {avg_train_loss:.4f}, 验证损失: {avg_valid_loss:.4f}")
        epoch_model_save_path = os.path.join(project_root, "memoai", "models", f"{model_name}_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_save_path)
        logger.info(f"已保存第 {epoch+1} 轮模型: {epoch_model_save_path}")
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            model_save_path = os.path.join(project_root, "memoai", "models", f"{model_name}_best.pth")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"已保存最佳模型: {model_save_path}")
    model_save_path = os.path.join(project_root, "memoai", "models", f"{model_name}_final.pth")
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"模型训练完成！最佳验证损失: {best_valid_loss:.4f}")
    logger.info(f"已保存最终模型: {model_save_path}")
    try:
        import gguf
        import struct
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        gguf_path = os.path.join(project_root, "memoai", "models", f"{model_name}_final.gguf")
        with open(gguf_path, 'wb') as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 1))
            for name, param in model.named_parameters():
                param_np = param.cpu().detach().numpy()
                name_bytes = name.encode('utf-8')
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<I", len(param_np.shape)))
                for dim in param_np.shape:
                    f.write(struct.pack("<I", dim))
                dtype_code = 0
                f.write(struct.pack("<I", dtype_code))
                param_np.tofile(f)
        logger.info(f"模型已成功转换为GGUF格式: {gguf_path}")
    except Exception as e:
        logger.error(f"转换模型为GGUF格式时出错: {str(e)}")
        logger.info("请确保gguf库已正确安装，或查看转换脚本是否需要更新")
    return True
if __name__ == "__main__":
    model_name = "Memo-1"
    epochs = 10
    lr = 0.001
    train_model(model_name, epochs, lr)