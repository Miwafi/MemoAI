import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import re
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('log', 'training.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 确保必要目录存在
for dir_path in ['log', 'model', 'TR']:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logging.info(f'创建目录: {dir_path}')

class Config:
    """训练配置类"""
    def __init__(self):
        self.embedding_dim = 256  # 增加嵌入维度
        self.hidden_size = 512   # 增加隐藏层大小
        self.num_layers = 3       # 增加LSTM层数
        self.batch_size = 16      # 减小批次大小
        self.epochs = 50          # 增加训练轮次
        self.learning_rate = 0.0005  # 减小学习率
        self.max_length = 100     # 增加序列长度
        self.model_path = os.path.join('model', 'dialog_model.pth')
        self.vocab_path = os.path.join('model', 'vocab.json')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fetch_from_huggingface = False
        self.dropout_rate = 0.5  # 添加此行定义dropout率

class CharEncoder:
    """字符编码解码器"""
    def __init__(self, config):
        self.config = config
        self.start_char = '\ue000'
        self.end_char = '\ue001'
        self.pad_char = '\ue002'
        self.char_to_idx = {
            self.start_char: 0,
            self.end_char: 1,
            self.pad_char: 2
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.load_vocab()

    def load_vocab(self):
        """加载词汇表"""
        if os.path.exists(self.config.vocab_path):
            try:
                with open(self.config.vocab_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.char_to_idx.update(data['char_to_idx'])
                    self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
                logging.info(f'加载词汇表成功，大小: {len(self.char_to_idx)}')
            except Exception as e:
                logging.warning(f'加载词汇表失败，使用默认词汇表: {str(e)}')
        else:
            logging.warning('词汇表文件不存在，使用默认词汇表')

    def save_vocab(self):
        """保存词汇表"""
        try:
            with open(self.config.vocab_path, 'w', encoding='utf-8') as f:
                data = {
                    'char_to_idx': self.char_to_idx,
                    'idx_to_char': self.idx_to_char
                }
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logging.error(f'保存词汇表失败: {str(e)}')
            return False

    def update_vocab(self, text):
        """从文本更新词汇表"""
        updated = False
        for char in text:
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
                updated = True
        if updated:
            self.save_vocab()
        return updated

    def encode(self, text):
        """将文本编码为索引序列"""
        if not text:
            return [self.char_to_idx[self.pad_char]] * self.config.max_length

        filtered_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。,.:;!?、 ]', '', text)
        filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
        self.update_vocab(filtered_text)  # 确保所有字符都被添加到词汇表

        filtered_text = filtered_text[:self.config.max_length-2]
        seq = [self.char_to_idx[self.start_char]]
        # 添加安全检查，防止索引超出范围
        for char in filtered_text:
            if char in self.char_to_idx:
                seq.append(self.char_to_idx[char])
            else:
                # 处理未预料到的字符
                seq.append(self.char_to_idx[self.pad_char])
                logging.warning(f'发现未收录字符: {repr(char)}, 使用填充符替代')
        seq += [self.char_to_idx[self.end_char]]

        # 填充到最大长度
        if len(seq) < self.config.max_length:
            seq += [self.char_to_idx[self.pad_char]] * (self.config.max_length - len(seq))

        return seq[:self.config.max_length]

    def decode(self, idx_seq):
        """将索引序列解码为文本"""
        text = []
        for idx in idx_seq:
            char = self.idx_to_char.get(idx, '')
            if char in [self.start_char, self.end_char, self.pad_char]:
                continue
            text.append(char)
        return ''.join(text)

# 在LSTMDialogNet类中添加（约第140-150行）
class LSTMDialogNet(nn.Module):
    def __init__(self, config, vocab_size):
        super(LSTMDialogNet, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout_rate
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, vocab_size)
        self.config = config
        self.dropout = nn.Dropout(config.dropout_rate, inplace=False)

    def init_hidden(self, batch_size):
        # 使用torch.zeros初始化隐藏状态并移动到正确设备
        hidden = (torch.zeros(self.config.num_layers, batch_size, self.config.hidden_size).to(self.config.device),
                  torch.zeros(self.config.num_layers, batch_size, self.config.hidden_size).to(self.config.device))
        return hidden
    
    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.layer_norm(out)
        out = self.fc(out)
        return out, hidden

    def save_model(self, path):
        """保存模型"""
        try:
            torch.save(self.state_dict(), path)
            logging.info(f'模型已保存到 {path}')
            return True
        except Exception as e:
            logging.error(f'保存模型失败: {str(e)}')
            return False

    @classmethod
    def load_model(cls, config, vocab_size, model_path):
        """加载现有模型继续训练"""
        model = cls(config, vocab_size)
        # 加载模型参数并处理设备映射
        checkpoint = torch.load(model_path, map_location=config.device)
        
        # 提取模型状态字典
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 处理词汇表大小变化导致的维度不匹配问题
        if vocab_size != state_dict['embedding.weight'].size(0):
            logging.warning(f'词汇表大小变化: 模型{state_dict["embedding.weight"].size(0)} → 当前{vocab_size}')
            
            # 创建新的嵌入层权重矩阵并复制旧权重
            new_embedding = torch.randn(vocab_size, config.embedding_dim)
            copy_size = min(state_dict['embedding.weight'].size(0), vocab_size)
            # 使用detach()分离旧权重并避免原地操作
            new_embedding[:copy_size] = state_dict['embedding.weight'][:copy_size].detach()
            state_dict['embedding.weight'] = nn.Parameter(new_embedding)
            
            # 处理输出层权重
            new_fc_weight = torch.randn(vocab_size, config.hidden_size)
            new_fc_bias = torch.randn(vocab_size)
            # 复制输出层可复用权重
            new_fc_weight[:copy_size] = state_dict['fc.weight'][:copy_size].detach()
            new_fc_bias[:copy_size] = state_dict['fc.bias'][:copy_size].detach()
            state_dict['fc.weight'] = nn.Parameter(new_fc_weight)
            state_dict['fc.bias'] = nn.Parameter(new_fc_bias)
        
        model.load_state_dict(state_dict)
        model.to(config.device)
        return model

class ModelTrainer:
    """模型训练器"""
    def __init__(self):
        self.config = Config()
        self.encoder = CharEncoder(self.config)
        # 延迟初始化模型相关组件
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.loss_history = []
        
    def train(self):
        """训练模型"""
        logging.info('开始模型训练...')
        logging.info(f'使用设备: {self.config.device}')

        # 1. 加载训练数据（会更新词汇表）
        corpus = self.load_tr_corpus()
        if not corpus:
            logging.error('训练数据为空，无法继续训练')
            return False

        # 2. 准备训练数据（进一步完善词汇表）
        inputs, targets = self.prepare_training_data(corpus)
        if inputs is None or targets is None:
            return False

        # 3. 使用最终词汇表大小初始化/加载模型
        self.vocab_size = len(self.encoder.char_to_idx)
        logging.info(f'最终词汇表大小: {self.vocab_size}')
        
        # 新增：尝试加载现有模型
        if os.path.exists(self.config.model_path):
            logging.info(f'加载现有模型: {self.config.model_path}')
            self.model = LSTMDialogNet.load_model(self.config, self.vocab_size, self.config.model_path)
        else:
            logging.info('未找到现有模型，创建新模型')
            self.model = LSTMDialogNet(self.config, self.vocab_size).to(self.config.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # 4. 创建数据加载器
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        # 5. 开始训练循环
        self.model.train()
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                
                # 1. 初始化LSTM隐藏状态
                batch_size = inputs.size(0)
                hidden = self.model.init_hidden(batch_size)
                hidden = (hidden[0].to(self.config.device), hidden[1].to(self.config.device))
                
                # 2. 前向传播时传入隐藏状态
                self.optimizer.zero_grad()
                outputs, hidden = self.model(inputs, hidden)  # 传入hidden参数
                loss = self.criterion(outputs.transpose(1, 2), targets)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                # 打印批次进度
                if batch_idx % 10 == 0:
                    logging.info(f'Epoch {epoch+1}/{self.config.epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')

            avg_loss = total_loss / len(dataloader)
            self.loss_history.append(avg_loss)
            logging.info(f'Epoch {epoch+1}/{self.config.epochs}, Average Loss: {avg_loss:.4f}')

        # 保存最终模型
        self.model.save_model(self.config.model_path)
        logging.info('训练完成!')
        return True

    def load_tr_corpus(self, folder_path=None):
        """加载TR文件夹中的文本文件"""
        if folder_path is None:
            # 使用相对于当前脚本文件的路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            folder_path = os.path.join(script_dir, 'TR')
        
        # 获取绝对路径确保稳定性
        abs_folder_path = os.path.abspath(folder_path)
        logging.info(f'尝试加载TR目录: {abs_folder_path}')
        
        if not os.path.exists(abs_folder_path):
            logging.error(f'TR文件夹不存在: {abs_folder_path}')
            return []
        
        corpus = []  # 确保这是字符串列表
        files_found = 0
        files_processed = 0
        
        for filename in os.listdir(abs_folder_path):
            file_path = os.path.join(abs_folder_path, filename)
            # 跳过目录，只处理文件
            if not os.path.isfile(file_path):
                continue
            # 只处理txt文件
            if not filename.lower().endswith('.txt'):
                continue
                
            files_found += 1
            
            try:
                # 尝试多种编码读取
                encodings = ['utf-8', 'mbcs', 'gbk', 'latin-1']
                text = None
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            text = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                
                if text is None:
                    logging.error(f'所有编码均无法解码文件: {file_path}')
                    continue
            
                # 读取全部文本并清洗
                text = re.sub(r'\s+', ' ', text).strip()
                if not text:
                    logging.warning(f'文件内容为空: {filename}')
                    continue
                    
                # 按标点符号分割句子（保留标点）
                sentences = re.split(r'(。|！|？|，|；|\.|!|\?|,)', text)
                
                # 重组句子（将标点与文本合并）
                combined = []
                if len(sentences) == 1:
                    # 如果没有标点符号，按固定长度切分
                    text_content = sentences[0]
                    chunk_size = 50  # 每50字符切分一段
                    for i in range(0, len(text_content), chunk_size):
                        chunk = text_content[i:i+chunk_size].strip()
                        if chunk:
                            combined.append(chunk)
                else:
                    # 正常处理有标点符号的情况
                    for i in range(0, len(sentences), 2):
                        if i+1 < len(sentences):
                            combined.append(f'{sentences[i]}{sentences[i+1]}')
                        elif i < len(sentences):
                            combined.append(sentences[i])
                
                # 确保只扩展字符串列表
                valid_sentences = [s for s in combined if s.strip()]
                corpus.extend(valid_sentences)
                files_processed += 1
                
                logging.info(f'加载文件: {filename}, 句子数: {len(valid_sentences)}')
            except Exception as e:
                logging.error(f'读取文件失败 {file_path}: {str(e)}')
        
        logging.info(f'总共找到 {files_found} 个txt文件，成功处理 {files_processed} 个')
        logging.info(f'最终加载到的总句子数: {len(corpus)}')
        return corpus

    def prepare_training_data(self, corpus):
        """准备古文序列训练数据"""
        if not corpus:
            logging.error('没有加载到训练数据')
            return None, None

        inputs = []
        targets = []
        seq_length = self.config.max_length
        
        # 将所有句子连接成一个长文本
        full_text = ' '.join(corpus)
        text_length = len(full_text)
        
        logging.info(f'合并后的文本总长度: {text_length} 字符')
        logging.info(f'序列长度: {seq_length}, 所需最小长度: {seq_length * 2}')
        
        if text_length < seq_length * 2:
            logging.warning(f'文本长度不足({text_length} < {seq_length * 2})，使用整个文本作为单个样本')
            # 如果文本太短，使用整个文本并填充
            input_seq = self.encoder.encode(full_text)
            # 创建对应的目标序列（偏移一个字符）
            if len(full_text) > 1:
                target_text = full_text[1:] + full_text[0]  # 循环移位
                target_seq = self.encoder.encode(target_text)
            else:
                target_seq = input_seq
            
            inputs.append(input_seq)
            targets.append(target_seq)
        else:
            # 正常分割
            sample_count = 0
            for i in range(0, len(full_text) - seq_length * 2, seq_length):
                input_seq = self.encoder.encode(full_text[i:i+seq_length])
                target_seq = self.encoder.encode(full_text[i+seq_length:i+seq_length*2])
                inputs.append(input_seq)
                targets.append(target_seq)
                sample_count += 1
            
            logging.info(f'生成了 {sample_count} 个训练样本')
        
        if not inputs:
            logging.error('未能生成任何训练样本')
            return None, None
            
        inputs = torch.tensor(inputs, dtype=torch.long).to(self.config.device)
        targets = torch.tensor(targets, dtype=torch.long).to(self.config.device)
        
        logging.info(f'最终训练数据形状: inputs={inputs.shape}, targets={targets.shape}')
        return inputs, targets

if __name__ == '__main__':
    # 在train_model.py开头添加（约第10行附近）
    import torch
    torch.autograd.set_detect_anomaly(True)  # 添加此行启用异常检测
    trainer = ModelTrainer()
    trainer.train()