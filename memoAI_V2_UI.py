import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import threading
import queue
import time
import random
import datetime
import os
import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ------------------------------
# 配置和常量
# ------------------------------
class Config:
    def __init__(self):
        self.model_path = 'model/dialog_model.pth'
        self.vocab_path = 'model/vocab.json'
        self.log_dir = 'log'
        self.memory_dir = 'memory'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = 10
        self.batch_size = 32
        self.learning_rate = 0.001
        self.hidden_size = 128
        self.embedding_dim = 64
        self.num_layers = 2
        self.dropout = 0.3

config = Config()

# ------------------------------
# 工具函数
# ------------------------------
def log_event(message, level='info'):
    """记录事件到日志"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f'[{timestamp}] [{level.upper()}] {message}\n'
    
    # 确保日志目录存在
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    
    # 写入日志文件
    log_file = os.path.join(config.log_dir, 'app.log')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)
    
    # 打印到控制台
    print(log_entry.strip())

# ------------------------------
# 系统自检
# ------------------------------
def system_check():
    """系统环境自检"""
    log_event("开始系统自检...")
    
    # 检查必要目录
    for dir_path in [config.log_dir, config.memory_dir, 'model']:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                log_event(f"创建必要目录: {dir_path}")
            except Exception as e:
                log_event(f"创建目录失败: {dir_path}, 错误: {str(e)}", 'error')
                return False
    
    # 检查PyTorch
    try:
        import torch
        log_event(f"PyTorch版本: {torch.__version__}, 设备: {config.device}")
    except ImportError:
        log_event("未找到PyTorch库，请安装PyTorch", 'error')
        return False
    
    log_event("系统自检完成")
    return True

# ------------------------------
# 数据管理
# ------------------------------
class DataManager:
    """数据管理类，负责记忆存储和加载"""
    def __init__(self):
        self.memory_path = os.path.join(config.memory_dir, 'memory.json')
        self.memory = self.load_memory()
        
    def load_memory(self):
        """加载记忆数据"""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r', encoding='utf-8') as f:
                    import json
                    return json.load(f)
            except Exception as e:
                log_event(f"加载记忆失败: {str(e)}", 'warning')
        return []
    
    def save_memory(self):
        """保存记忆数据"""
        try:
            with open(self.memory_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            log_event(f"保存记忆失败: {str(e)}", 'error')
            return False
    
    def add_memory(self, user_input, ai_response):
        """添加新的对话记忆"""
        memory_item = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user': user_input,
            'ai': ai_response,
            'is_approved': False,  # 新增：答案肯定状态
            'is_disapproved': False  # 新增：否定状态字段
        }
        self.memory.append(memory_item)
        return self.save_memory()
    
    def get_recent_memory(self, count=None):
        """获取最近的记忆"""
        if count is None or count >= len(self.memory):
            return self.memory.copy()
        return self.memory[-count:].copy()
    
    def get_relevant_memories(self, input_text, top_k=3):
        """获取与输入文本相关的记忆（基于关键词匹配）"""
        if not self.memory:  # 检查记忆是否为空
            return []
        
        # 提取输入文本关键词（简单分词）
        input_words = set(input_text.lower().split())
        relevant_scores = []
        
        for item in self.memory:
            # 合并用户输入和AI响应作为匹配文本
            memory_text = f"{item['user']} {item['ai']}".lower()
            # 计算关键词匹配度
            memory_words = set(memory_text.split())
            common_words = input_words.intersection(memory_words)
            score = len(common_words) / (len(input_words) + 1e-6)  # 避免除零
            relevant_scores.append((item, score))
        
        # 按分数排序并取top_k
        relevant_scores.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in relevant_scores[:top_k]]
    
    def clear_memory(self):
        """清除所有记忆"""
        self.memory = []
        return self.save_memory()
    
    def update_memory_approval(self, index, approved):
        """更新记忆项的肯定状态"""
        if 0 <= index < len(self.memory):
            self.memory[index]['is_approved'] = approved
            return self.save_memory()
        return False

    def update_memory_disapproval(self, index, disapproved):
        """更新记忆项的否定状态"""
        if 0 <= index < len(self.memory):
            self.memory[index]['is_disapproved'] = disapproved
            # 确保肯定和否定状态互斥
            if disapproved:
                self.memory[index]['is_approved'] = False
            return self.save_memory()
        return False
    
    def get_latest_memory_index(self):
        """获取最新记忆项的索引"""
        return len(self.memory) - 1

# ------------------------------
# 字符编码
# ------------------------------
class CharEncoder:
    """字符编码解码器"""
    def __init__(self):
        # 修改特殊字符定义，避免使用空字符
        self.start_char = '\ue000'  # 使用罕见Unicode字符作为起始符
        self.end_char = '\ue001'    # 使用罕见Unicode字符作为结束符
        self.pad_char = '\ue002'    # 使用罕见Unicode字符作为填充符，替换原空字符
        
        # 初始化默认词汇表
        self.char_to_idx = {
            self.start_char: 0,
            self.end_char: 1,
            self.pad_char: 2
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        # 加载现有词汇表
        self.load_vocab()
    
    def load_vocab(self):
        """加载词汇表"""
        if os.path.exists(config.vocab_path):
            try:
                with open(config.vocab_path, 'r', encoding='utf-8') as f:
                    import json
                    data = json.load(f)
                    # 合并加载的词汇表与默认词汇表
                    self.char_to_idx.update(data['char_to_idx'])
                    self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
                log_event(f"加载词汇表成功，大小: {len(self.char_to_idx)}")
            except Exception as e:
                log_event(f"加载词汇表失败，使用默认词汇表: {str(e)}", 'warning')
        else:
            log_event("词汇表文件不存在，使用默认词汇表", 'warning')
    
    def save_vocab(self):
        """保存词汇表"""
        try:
            with open(config.vocab_path, 'w', encoding='utf-8') as f:
                import json
                data = {
                    'char_to_idx': self.char_to_idx,
                    'idx_to_char': self.idx_to_char
                }
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            log_event(f"保存词汇表失败: {str(e)}", 'error')
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
    
    def encode(self, text, max_length=50):
        """将文本编码为索引序列"""
        if not text:
            return [self.char_to_idx[self.pad_char]] * max_length
        
        # 增强文本清洗
        filtered_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。,.:;!?、 ]', '', text)
        filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
        
        # 更新词汇表
        self.update_vocab(filtered_text)
        
        # 截断或填充到最大长度
        filtered_text = filtered_text[:max_length-2]  # 预留start和end字符位置
        seq = [self.char_to_idx[self.start_char]]
        seq += [self.char_to_idx[char] for char in filtered_text if char in self.char_to_idx]
        seq += [self.char_to_idx[self.end_char]]
        
        # 填充到最大长度
        if len(seq) < max_length:
            seq += [self.char_to_idx[self.pad_char]] * (max_length - len(seq))
        
        return seq[:max_length]
    
    def decode(self, idx_seq):
        """将索引序列解码为文本"""
        text = []
        for idx in idx_seq:
            char = self.idx_to_char.get(idx, '')
            # 跳过特殊字符和空字符
            if char in [self.start_char, self.end_char, self.pad_char] or char == '\u0000':
                continue
            text.append(char)
        return ''.join(text)
    
    def get_vocab_size(self):
        """获取词汇表大小"""
        return len(self.char_to_idx)

# ------------------------------
# 对话模型
# ------------------------------
class LSTMDialogNet(nn.Module):
    """LSTM对话模型"""
    def __init__(self, vocab_size):
        super(LSTMDialogNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True
        )
        self.fc = nn.Linear(config.hidden_size, vocab_size)
        
    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        """初始化隐藏状态"""
        weight = next(self.parameters()).data
        hidden = (
            weight.new(config.num_layers, batch_size, config.hidden_size).zero_().to(config.device),
            weight.new(config.num_layers, batch_size, config.hidden_size).zero_().to(config.device)
        )
        return hidden
    
    def save_model(self, path):
        """保存模型"""
        try:
            torch.save(self.state_dict(), path)
            log_event(f"模型已保存到 {path}")
            return True
        except Exception as e:
            log_event(f"保存模型失败: {str(e)}", 'error')
            return False
    
    def load_model(self, path):
        """加载模型"""
        if os.path.exists(path):
            try:
                # 尝试加载模型并处理设备兼容性
                state_dict = torch.load(path, map_location=config.device)
                current_state = self.state_dict()
                filtered_state = {}
                
                for name, param in state_dict.items():
                    if name not in current_state:
                        log_event(f"忽略不存在的参数: {name}", 'warning')
                        continue
                    
                    # 处理词汇表大小变化导致的形状不匹配
                    if name == 'embedding.weight':
                        # 嵌入层: [旧词汇表大小, 嵌入维度] -> [新词汇表大小, 嵌入维度]
                        old_size, embed_dim = param.shape
                        new_size = current_state[name].shape[0]
                        if old_size != new_size:
                            log_event(f"嵌入层词汇表大小变化: {old_size} -> {new_size}", 'warning')
                            # 复制旧参数并初始化新增部分
                            new_param = current_state[name].clone()
                            new_param[:old_size] = param
                            filtered_state[name] = new_param
                            continue
                    elif name in ['fc.weight', 'fc.bias']:
                        # 全连接层: [旧词汇表大小, 隐藏层大小] -> [新词汇表大小, 隐藏层大小]
                        if name == 'fc.weight':
                            old_size, hidden_size = param.shape
                            new_size = current_state[name].shape[0]
                        else:  # fc.bias
                            old_size = param.shape[0]
                            new_size = current_state[name].shape[0]
                        
                        if old_size != new_size:
                            log_event(f"全连接层词汇表大小变化: {old_size} -> {new_size}", 'warning')
                            # 复制旧参数并初始化新增部分
                            new_param = current_state[name].clone()
                            new_param[:old_size] = param
                            filtered_state[name] = new_param
                            continue
                    
                    # 常规参数形状检查
                    if param.shape != current_state[name].shape:
                        log_event(f"参数形状不匹配: {name} {param.shape} vs {current_state[name].shape}", 'warning')
                        continue
                    
                    filtered_state[name] = param
                
                current_state.update(filtered_state)
                self.load_state_dict(current_state)
                self.to(config.device)
                log_event(f"模型已从 {path} 加载")
                return True
            except Exception as e:
                log_event(f"加载模型失败: {str(e)}", 'error')
        else:
            log_event(f"模型文件不存在: {path}", 'error')
        return False

# ------------------------------
# 自学习AI核心
# ------------------------------
class AICore:
    """自学习AI核心类"""
    def __init__(self):
        self.data_manager = DataManager()
        self.encoder = CharEncoder()
        self.vocab_size = self.encoder.get_vocab_size()
        self.model = LSTMDialogNet(self.vocab_size).to(config.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.is_training = False
        self.training_stop_event = threading.Event()
        self.loss_history = []
        
        # 尝试加载现有模型
        self.model_loaded = self.model.load_model(config.model_path)
        if not self.model_loaded:
            log_event("警告：未加载任何模型，响应质量可能不佳", 'warning')
            log_event("将使用新模型进行训练")
    
    def preprocess_data(self, memory=None):
        """预处理数据用于训练"""
        if memory is None:
            memory = self.data_manager.get_recent_memory()
            
        # 验证数据有效性
        valid_memory = []
        for item in memory:
            if isinstance(item, dict) and 'user' in item and 'ai' in item and item['user'].strip() and item['ai'].strip():
                valid_memory.append(item)
        memory = valid_memory
            
        if len(memory) < 2:
            log_event("有效记忆数据不足，无法进行训练", 'warning')
            return None, None
        
        inputs = []
        targets = []
        max_length = 50
        
        for i in range(len(memory) - 1):
            # 使用用户输入作为输入
            input_text = memory[i]['user']
            # 使用当前AI响应作为目标（修复索引越界）
            target_text = memory[i]['ai']
            
            # 编码
            input_seq = self.encoder.encode(input_text, max_length)
            target_seq = self.encoder.encode(target_text, max_length)
            
            # 验证序列有效性
            if len(input_seq) != max_length or len(target_seq) != max_length:
                log_event(f"序列长度不匹配: {len(input_seq)} vs {max_length}", 'warning')
                continue
            
            inputs.append(input_seq)
            targets.append(target_seq)
        
        if not inputs or not targets:
            log_event("没有有效的训练数据", 'warning')
            return None, None
        
        # 转换为张量
        inputs = torch.tensor(inputs, dtype=torch.long).to(config.device)
        targets = torch.tensor(targets, dtype=torch.long).to(config.device)
        
        return inputs, targets
    
    def train_step(self, inputs, targets):
        """单步训练"""
        batch_size = inputs.size(0)
        hidden = self.model.init_hidden(batch_size)
        
        self.model.zero_grad()
        loss = 0
        
        for i in range(inputs.size(1)):
            output, hidden = self.model(inputs[:, i:i+1], hidden)
            loss += self.criterion(output.squeeze(1), targets[:, i])
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item() / inputs.size(1)
    
    def _common_training_loop(self, epochs, dataloader, progress_callback, task_name):
        """公共训练循环逻辑"""
        try:
            for epoch in range(epochs):
                if self.training_stop_event.is_set():
                    log_event(f"{task_name}已被用户中断")
                    break
                
                epoch_loss = 0
                for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
                    loss = self.train_step(batch_inputs, batch_targets)
                    epoch_loss += loss
                    
                    # 调用进度回调
                    if progress_callback:
                        progress = (epoch * len(dataloader) + batch_idx + 1) / (epochs * len(dataloader)) * 100
                        progress_callback(progress, f"{task_name} Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss:.4f}")
                
                avg_loss = epoch_loss / len(dataloader)
                self.loss_history.append(avg_loss)
                log_event(f"{task_name} Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
                
                # 调用进度回调
                if progress_callback:
                    progress_callback((epoch+1)/epochs*100, f"{task_name} Epoch {epoch+1}/{epochs} 完成, 平均Loss: {avg_loss:.4f}")
            
            # 保存模型
            self.model.save_model(config.model_path)
            
            # 绘制损失曲线
            self.plot_loss_history()
            
            log_event(f"{task_name}完成")
            return True
        except Exception as e:
            log_event(f"{task_name}过程中出错: {str(e)}", 'error')
            return False
        finally:
            self.is_training = False
    
    def train(self, epochs=None, progress_callback=None):
        """训练模型"""
        if self.is_training:
            log_event("模型正在训练中，请先停止当前训练", 'warning')
            return False
        
        self.is_training = True
        self.training_stop_event.clear()
        self.loss_history = []
        epochs = epochs or config.epochs
        
        # 准备数据
        inputs, targets = self.preprocess_data()
        if inputs is None or targets is None:
            self.is_training = False
            return False
        
        # 检查词汇表大小是否有变化，如果有则重新创建模型
        current_vocab_size = self.encoder.get_vocab_size()
        if current_vocab_size != self.vocab_size:
            log_event(f"词汇表大小已更新: {self.vocab_size} -> {current_vocab_size}，重新创建模型")
            self.vocab_size = current_vocab_size
            self.model = LSTMDialogNet(self.vocab_size).to(config.device)
            
            # 新增：加载预训练模型权重
            if os.path.exists(config.model_path):
                try:
                    state_dict = torch.load(config.model_path, map_location=config.device)
                    self.model.load_state_dict(state_dict)
                    log_event("成功加载预训练模型权重")
                except Exception as e:
                    log_event(f"加载模型权重失败: {str(e)}", 'error')
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # 创建数据加载器
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        log_event(f"开始训练，共 {epochs} 轮，批次大小: {config.batch_size}")
        return self._common_training_loop(epochs, dataloader, progress_callback, "训练")
    
    def start_training_thread(self, epochs=None, progress_callback=None):
        """在新线程中开始训练"""
        if self.is_training:
            log_event("模型正在训练中，请先停止当前训练", 'warning')
            return False
        
        train_thread = threading.Thread(target=self.train, args=(epochs, progress_callback))
        train_thread.daemon = True
        train_thread.start()
        return True
    
    def stop_training(self):
        """停止训练"""
        if self.is_training:
            self.training_stop_event.set()
            log_event("正在停止训练...")
            return True
        return False
    
    def plot_loss_history(self):
        """绘制损失历史曲线"""
        if not self.loss_history:
            log_event("没有损失历史记录可供绘制", 'warning')
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.loss_history)+1), self.loss_history, marker='o', linestyle='-', color='b')
        plt.title('训练损失变化')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 保存图像
        loss_plot_path = os.path.join(config.log_dir, 'loss_history.png')
        plt.savefig(loss_plot_path)
        plt.close()
        log_event(f"损失历史曲线已保存到 {loss_plot_path}")
    
    def online_learn(self, progress_callback=None):
        """联网自学功能，从网络获取知识进行学习"""
        if self.is_training:
            log_event("模型正在训练中，请先停止当前训练", 'warning')
            return False
        
        log_event("开始联网自学...")
        self.is_training = True
        self.training_stop_event.clear()
        self.loss_history = []
        
        # 模拟联网获取数据的过程
        def fetch_online_knowledge():
            try:
                # 这里可以添加实际的联网获取数据代码
                log_event("正在从网络获取最新知识...")
                time.sleep(3)  # 模拟网络请求延迟
                
                # 模拟获取到的知识数据
                online_knowledge = [
                    {"user": "你好", "ai": "你好！很高兴为您服务。"},
                    {"user": "您好", "ai": "您好！请问有什么可以帮助您的吗？"},
                    {"user": "早上好", "ai": "早上好！今天天气不错呢。"},
                    {"user": "晚上好", "ai": "晚上好！今天过得怎么样？"},
                    {"user": "什么是人工智能?", "ai": "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。"},
                    {"user": "机器学习的主要类型有哪些?", "ai": "机器学习主要分为监督学习、无监督学习、半监督学习和强化学习四大类。"}
                ]
                
                log_event(f"成功获取 {len(online_knowledge)} 条新知识")
                return online_knowledge
            except Exception as e:
                log_event(f"联网获取知识失败: {str(e)}", 'error')
                return None
        
        # 获取在线知识
        online_knowledge = fetch_online_knowledge()
        if not online_knowledge:
            self.is_training = False
            return False
        
        # 使用在线知识进行训练
        inputs, targets = self.preprocess_data(online_knowledge)
        if inputs is None or targets is None:
            self.is_training = False
            return False
        
        # 检查词汇表大小是否有变化，如果有则重新创建模型
        current_vocab_size = self.encoder.get_vocab_size()
        if current_vocab_size != self.vocab_size:
            log_event(f"词汇表大小已更新: {self.vocab_size} -> {current_vocab_size}，重新创建模型")
            self.vocab_size = current_vocab_size
            self.model = LSTMDialogNet(self.vocab_size).to(config.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # 创建数据加载器
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        return self._common_training_loop(3, dataloader, progress_callback, "联网自学")
    
    def self_learn(self, progress_callback=None):
        """自主学习，使用现有记忆进行训练"""
        log_event("开始自主学习...")
        return self.start_training_thread(epochs=5, progress_callback=progress_callback)
    
    def infer(self, input_text, max_length=50):
        """推理生成响应"""
        if not input_text.strip():
            return "请输入有效的文本"
        
        # 添加规则匹配处理常见问候语
        greetings = {
            "你好": "你好！很高兴见到你。",
            "您好": "您好！有什么我可以帮助您的吗？",
            "早上好": "早上好！祝您今天有个好心情。",
            "晚上好": "晚上好！今天过得怎么样？",
            "嗨": "嗨！你好呀！"
        }
        
        # 检查输入是否匹配问候语
        input_lower = input_text.strip().lower()
        for greeting, response in greetings.items():
            if greeting in input_lower:
                self.data_manager.add_memory(input_text, response)
                return response
                # 获取相关记忆并构建上下文
        relevant_memories = self.data_manager.get_relevant_memories(input_text)
        context = "".join([f"用户: {m['user']}\nAI: {m['ai']}\n" for m in relevant_memories])
        full_input = f"{context}用户: {input_text}\nAI: "
        
        try:
            # 编码输入
            input_seq = self.encoder.encode(full_input, max_length)
            
            # 检查词汇表是否更新，如果是则重新创建模型
            current_vocab_size = self.encoder.get_vocab_size()
            if current_vocab_size != self.vocab_size:
                log_event(f"推理时词汇表大小已更新: {self.vocab_size} -> {current_vocab_size}，重新创建模型")
                self.vocab_size = current_vocab_size
                self.model = LSTMDialogNet(self.vocab_size).to(config.device)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
                
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(config.device)
            
            # 初始化隐藏状态
            hidden = self.model.init_hidden(1)
            
            # 推理
            self.model.eval()
            output_seq = []
            for i in range(max_length):  # 循环开始
                if i >= input_tensor.size(1):
                    break
                output, hidden = self.model(input_tensor[:, i:i+1], hidden)

                # 添加温度参数调整概率分布
                temperature = 0.5  # 降低温度使输出更确定
                output = output.squeeze(1)
                output = F.softmax(output / temperature, dim=1)
                
                # 过滤低概率字符
                min_prob = 0.01
                output[output < min_prob] = 0
                output = output / output.sum()  # 重新归一化
                
                top_idx = torch.multinomial(output, 1).item()
                output_seq.append(top_idx)

                # 如果生成了结束符，停止生成
                if self.encoder.idx_to_char.get(top_idx, '') == self.encoder.end_char:
                    break  # 现在break位于for循环内部
            # 循环结束
            
            # 解码输出
            response = self.encoder.decode(output_seq)
            self.model.train()
            
            # 添加到记忆
            self.data_manager.add_memory(input_text, response)
            
            # 验证响应
            if not response.strip():
                response = "抱歉，我无法生成有效的响应。请先进行训练或尝试其他问题。"
            
            return response
        except Exception as e:
            log_event(f"推理过程中出错: {str(e)}", 'error')
            return f"抱歉，处理请求时出错: {str(e)}"
    
    def manual_correction(self, input_text, correct_response):
        """手动纠错，添加正确的对话样本"""
        log_event(f"添加手动纠错样本: 用户输入='{input_text}', 正确响应='{correct_response}'")
        self.data_manager.add_memory(input_text, correct_response)
        return True

# ------------------------------
# UI组件
# ------------------------------
class RoundedButton(ttk.Button):
    """圆角按钮组件"""
    def __init__(self, parent, text, command=None, **kwargs):
        super().__init__(parent, text=text, command=command, **kwargs)
        # 使用ttkbootstrap支持的内置样式
        self.configure(style='Primary.TButton')

# ------------------------------
# 主应用界面
# ------------------------------
class App:
    """主应用程序类"""
    def __init__(self, root):
        self.root = root
        self.root.title("MemoAI V2 - 自学习对话系统")
        self.root.geometry("800x600")
        self.root.minsize(600, 500)
        
        # 初始化数据管理器
        self.data_manager = DataManager()
        
        # 初始化AI核心
        self.ai = AICore()
        
        # AI想法相关初始化
        self.thought_queue = queue.Queue()
        self.thought_thread = None
        self.ai_state = "idle"
        self.running = True
        self.default_font = ('SimHei', 10)
        
        # 创建UI
        self.create_widgets()
        
        # 显示欢迎消息
        self.add_message("system", "欢迎使用MemoAI V2！我正在初始化，请稍候...")
        self.add_message("ai", "您好！我是MemoAI，一个可以自我学习的对话AI。有什么我可以帮助您的吗？")
    
    def create_widgets(self):
        """创建UI组件"""
        # 设置样式
        self.style = ttk.Style()
        self._setup_fonts()
        self.style.configure('ChatFrame.TFrame', background='#f0f0f0')
        self.style.configure('UserMessage.TLabel', background='#0078d7', foreground='white')
        self.style.configure('AIMessage.TLabel', background='#e6e6e6', foreground='black')
        self.style.configure('SystemMessage.TLabel', background='#ffd700', foreground='black')
        
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)


        # 右侧内容框架 - 垂直排列聊天区域和控制区域
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 聊天框架
        chat_frame = ttk.Frame(right_frame)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 聊天历史滚动区域
        self.chat_canvas = tk.Canvas(chat_frame)
        self.chat_scrollbar = ttk.Scrollbar(chat_frame, orient=tk.VERTICAL, command=self.chat_canvas.yview)
        self.chat_history = ttk.Frame(self.chat_canvas, style='ChatFrame.TFrame')
        
        self.chat_history.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        )
        
        self.chat_canvas.create_window((0, 0), window=self.chat_history, anchor="nw")
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)
        
        self.chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.chat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定鼠标滚轮事件
        self.chat_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # 输入框架
        input_frame = ttk.Frame(right_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.user_input = ttk.Entry(input_frame, font=('SimHei', 10))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", lambda event: self.send_message())
        
        send_btn = RoundedButton(input_frame, text="发送", command=self.send_message)
        send_btn.pack(side=tk.RIGHT)
        
        # 状态和控制框架
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(control_frame, text="就绪", foreground="green")
        self.status_label.pack(side=tk.LEFT)
        
        # 进度条
        self.progress_bar = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # 功能按钮框架
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X)
        
        self.learn_btn = RoundedButton(btn_frame, text="自主学习", command=self.start_self_learning)
        self.learn_btn.pack(side=tk.LEFT, padx=5)
        
        self.online_learn_btn = RoundedButton(btn_frame, text="联网自学", command=self.start_online_learning)
        self.online_learn_btn.pack(side=tk.LEFT, padx=5)
        
        self.correct_btn = RoundedButton(btn_frame, text="手动纠错", command=self.open_correction_window)
        self.correct_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = RoundedButton(btn_frame, text="清除对话", command=self.clear_chat)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.quit_btn = RoundedButton(btn_frame, text="退出", command=self.quit_app)
        self.quit_btn.pack(side=tk.RIGHT, padx=5)
    

    def _setup_fonts(self):
        """设置中文字体支持"""
        # 配置全局中文字体
        self.default_font = ('SimHei', 10)
        self.title_font = ('SimHei', 12, 'bold')
        self.button_font = ('SimHei', 10)
        
        # 配置ttk样式字体
        self.style.configure('TLabel', font=self.default_font)
        self.style.configure('TButton', font=self.button_font)
        self.style.configure('TFrame', font=self.default_font)
        self.style.configure('TText', font=self.default_font)

    def _on_mousewheel(self, event):
        """鼠标滚轮事件处理"""
        self.chat_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def add_message(self, sender, text, typing_animation=False, memory_index=None):
        """添加消息到聊天历史"""
        # 创建消息框架
        msg_frame = ttk.Frame(self.chat_history)
        msg_frame.pack(fill=tk.X, padx=5, pady=5, anchor="w" if sender == "user" else "e")
        
        # 设置样式和对齐方式
        style = f'{sender.capitalize()}Message.TLabel'
        anchor = tk.E if sender == "user" else tk.W
        
        # 调试: 打印消息内容
        print(f"添加{sender}消息: {text}")
        
        # 添加肯定和否定按钮（仅AI消息）
        if sender == "ai" and memory_index is not None:
            # 否定按钮（先添加，靠右显示）
            disapprove_btn = RoundedButton(
                msg_frame, 
                text="否定", 
                command=lambda idx=memory_index: self.disapprove_answer(idx)
            )
            disapprove_btn.pack(side=tk.RIGHT, padx=2)
            
            # 肯定按钮
            approve_btn = RoundedButton(
                msg_frame, 
                text="肯定", 
                command=lambda idx=memory_index: self.approve_answer(idx)
            )
            approve_btn.pack(side=tk.RIGHT, padx=2)

        # 创建消息标签 - 显式设置颜色确保可见
        bg_color = '#0078d7' if sender == 'user' else '#e6e6e6'
        fg_color = 'white' if sender == 'user' else 'black'
        
        msg_label = ttk.Label(
            msg_frame,
            text="" if typing_animation else text,
            style=style,
            wraplength=600,
            justify=tk.LEFT,
            anchor=anchor,
            padding=10,
            borderwidth=1,
            relief=tk.SOLID,
            background=bg_color,
            foreground=fg_color
        )
        msg_label.pack(fill=tk.X, expand=True)

        # 打字动画效果
        if typing_animation and sender == "ai":
            self._type_text_animation(msg_label, text, 0)
        
        # 强制刷新UI
        self.chat_history.update_idletasks()
        
        # 滚动到底部
        self.chat_canvas.after_idle(lambda: self.chat_canvas.yview_moveto(1.0))
    
    def _type_text_animation(self, label, text, index):
        if index < len(text):
            current_text = label.cget("text") + text[index]
            label.config(text=current_text)
            self.root.after(30, self._type_text_animation, label, text, index + 1)

    def send_message(self):
        """发送用户消息并获取AI响应"""
        user_text = self.user_input.get().strip()
        if not user_text:
            return
        
        # 添加用户消息到聊天历史
        self.add_message("user", user_text)
        self.user_input.delete(0, tk.END)
        
        # 更新状态
        self.status_label.config(text="思考中...", foreground="blue")
        self.root.update_idletasks()
        
        # 在新线程中获取AI响应，避免UI卡顿
        def get_ai_response():
            self.update_ai_state("responding")  # 设置为响应状态
            response = self.ai.infer(user_text)
            # 使用打字动画显示AI回复
            # 获取最新记忆索引
            memory_index = self.ai.data_manager.get_latest_memory_index()
            self.root.after(0, lambda: self.add_message("ai", response, typing_animation=True, memory_index=memory_index))
            self.root.after(0, lambda: self.status_label.config(text="就绪", foreground="green"))
            self.root.after(0, lambda: self.update_ai_state("idle"))  # 恢复空闲状态
        
        threading.Thread(target=get_ai_response).start()
    
    def start_self_learning(self):
        """开始自主学习"""
        if self.ai.is_training:
            self.add_message("system", "AI正在训练中，请稍候...")
            return
        
        self.add_message("system", "开始自主学习，这可能需要几分钟时间...")
        self.status_label.config(text="学习中...", foreground="orange")
        self.learn_btn.config(state=tk.DISABLED)
        self.update_ai_state("learning")  # 设置为学习状态
        
        # 进度回调函数
        def update_progress(progress, status_text):
            self.progress_bar['value'] = progress
            self.status_label.config(text=status_text)
            self.root.update_idletasks()
            
            if progress >= 100:
                self.progress_bar['value'] = 0
                self.learn_btn.config(state=tk.NORMAL)
                self.status_label.config(text="就绪", foreground="green")
                self.add_message("system", "自主学习完成！")
                self.update_ai_state("idle")  # 恢复空闲状态
        
        # 开始训练线程
        success = self.ai.self_learn(progress_callback=update_progress)
        if not success:
            self.status_label.config(text="就绪", foreground="green")
            self.learn_btn.config(state=tk.NORMAL)
            self.update_ai_state("idle")  # 恢复空闲状态
    
    def start_online_learning(self):
        """开始联网自学"""
        if self.ai.is_training:
            self.add_message("system", "AI正在训练中，请稍候...")
            return
        
        self.add_message("system", "开始联网自学，这可能需要几分钟时间...")
        self.status_label.config(text="联网学习中...", foreground="purple")
        self.online_learn_btn.config(state=tk.DISABLED)
        self.update_ai_state("learning")  # 设置为学习状态
        
        # 进度回调函数
        def update_progress(progress, status_text):
            self.progress_bar['value'] = progress
            self.status_label.config(text=status_text)
            self.root.update_idletasks()
            
            if progress >= 100:
                self.progress_bar['value'] = 0
                self.online_learn_btn.config(state=tk.NORMAL)
                self.status_label.config(text="就绪", foreground="green")
                self.add_message("system", "联网自学完成！")
                self.update_ai_state("idle")  # 恢复空闲状态
        
        # 启动联网学习线程
        threading.Thread(target=lambda: self.ai.online_learn(progress_callback=update_progress)).start()
    
    def open_correction_window(self):
        """打开手动纠错窗口"""
        # 使用tkinter原生Toplevel
        correction_window = tk.Toplevel(self.root)
        correction_window.title("手动纠错")
        correction_window.geometry("400x300")
        correction_window.resizable(False, False)
        
        # 设置窗口字体
        correction_window.option_add("*Font", self.default_font)
        
        # 添加纠错内容标签
        ttk.Label(correction_window, text="请输入正确内容:", padding=5).pack(anchor=tk.W)
        
        # 添加文本框 - 指定中文字体
        correction_text = tk.Text(correction_window, height=10, width=40, font=self.default_font)
        correction_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # 添加按钮框架
        btn_frame = ttk.Frame(correction_window)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 添加确认按钮
        confirm_btn = ttk.Button(btn_frame, text="确认", command=lambda: self.save_correction(correction_window, correction_text))
        confirm_btn.pack(side=tk.RIGHT, padx=5)
        
        # 添加取消按钮
        cancel_btn = ttk.Button(btn_frame, text="取消", command=correction_window.destroy)
        cancel_btn.pack(side=tk.RIGHT)
    
    def save_correction(self, correction_window, correction_text):
        """保存手动纠错内容"""
        # 获取文本框中的纠错内容
        correction_content = correction_text.get(1.0, tk.END).strip()
        
        if correction_content:
            # 记录到日志
            log_event(f"手动纠错内容: {correction_content}")
            
            # 更新到数据管理器
            if hasattr(self.data_manager, 'add_correction'):
                self.data_manager.add_correction(correction_content)
            
            # 使用指定中文字体的消息框
            msg_window = tk.Toplevel(self.root)
            msg_window.title("提示")
            msg_window.geometry("200x100")
            msg_window.resizable(False, False)
            msg_window.option_add("*Font", self.default_font)
            
            ttk.Label(msg_window, text="纠错内容已保存！", padding=20).pack()
            ttk.Button(msg_window, text="确定", command=lambda: [msg_window.destroy(), correction_window.destroy()]).pack(pady=5)
        else:
            # 使用指定中文字体的警告框
            msg_window = tk.Toplevel(self.root)
            msg_window.title("警告")
            msg_window.geometry("200x100")
            msg_window.resizable(False, False)
            msg_window.option_add("*Font", self.default_font)
            
            ttk.Label(msg_window, text="纠错内容不能为空！", padding=20).pack()
            ttk.Button(msg_window, text="确定", command=msg_window.destroy).pack(pady=5)
            return
    
    def approve_answer(self, memory_index):
        """处理答案肯定操作"""
        success = self.ai.data_manager.update_memory_approval(memory_index, True)
        if success:
            self.add_message("system", "答案已肯定，感谢反馈！")
        else:
            self.add_message("system", "操作失败，请重试。")

    def disapprove_answer(self, memory_index):
        """处理答案否定操作"""
        success = self.ai.data_manager.update_memory_disapproval(memory_index, True)
        if success:
            self.add_message("system", "答案已否定，将优化后续回复。")
            # 可选：触发重新生成回答逻辑
            # self.regenerate_answer(memory_index)
        else:
            self.add_message("system", "操作失败，请重试。")

    def clear_chat(self):
        """清除聊天历史"""
        # 清空聊天历史框架
        for widget in self.chat_history.winfo_children():
            widget.destroy()
        
        # 询问是否同时清除记忆
        if messagebox.askyesno("确认", "是否同时清除AI的记忆数据？"):
            self.ai.data_manager.clear_memory()
            self.add_message("system", "聊天历史和AI记忆已清除")
        else:
            self.add_message("system", "聊天历史已清除")
    
    def quit_app(self):
        """退出应用程序"""
        if self.ai.is_training:
            if not messagebox.askyesno("确认", "AI正在训练中，确定要退出吗？"):
                return
            self.ai.stop_training()
        
        log_event("应用程序已退出")
        self.running = False
        self.root.destroy()
    
    def start_thought_processing(self):
        """启动想法处理线程"""
        self.thought_thread = threading.Thread(target=self._thought_generator, daemon=True)
        self.thought_thread.start()
        self.root.after(100, self._update_thought_display)
    
    def _thought_generator(self):
        """生成AI想法的后台线程"""
        while self.running:
            if self.ai_state == "idle":
                thoughts = [
                    "我应该如何更好地理解用户需求...",
                    "等待用户输入中...",
                    "思考如何优化我的回答质量..."
                ]
                thought = random.choice(thoughts)
                self.thought_queue.put((datetime.datetime.now(), "idle", thought))
                time.sleep(random.uniform(5, 10))
            elif self.ai_state == "learning":
                thoughts = [
                    f"正在分析新知识... (进度: {random.randint(10, 90)}%)",
                    "这个概念和我之前学到的有相似之处...",
                    "需要将这部分知识与现有知识库整合..."
                ]
                thought = random.choice(thoughts)
                self.thought_queue.put((datetime.datetime.now(), "learning", thought))
                time.sleep(random.uniform(2, 4))
            elif self.ai_state == "responding":
                thoughts = [
                    "正在分析用户问题...",
                    "寻找最合适的回答方式...",
                    "组织语言，确保回答准确易懂..."
                ]
                thought = random.choice(thoughts)
                self.thought_queue.put((datetime.datetime.now(), "responding", thought))
                time.sleep(random.uniform(1, 3))
            else:
                time.sleep(1)
    
    def _update_thought_display(self):
        """更新想法显示UI"""
        try:
            while not self.thought_queue.empty():
                timestamp, state, thought = self.thought_queue.get_nowait()
                time_str = timestamp.strftime("%H:%M:%S")
                
                # 设置不同状态的文本颜色
                color = "#666666"
                if state == "learning":
                    color = "#e67e22"
                elif state == "responding":
                    color = "#3498db"
                
                # 更新文本框
                self.thought_text.config(state=tk.NORMAL)
                self.thought_text.insert(tk.END, f"[{time_str}] ", ("time",))
                self.thought_text.insert(tk.END, f"{thought}\n", (state,))
                self.thought_text.tag_config("time", foreground="#999999")
                self.thought_text.tag_config("idle", foreground=color)
                self.thought_text.tag_config("learning", foreground=color)
                self.thought_text.tag_config("responding", foreground=color)
                self.thought_text.config(state=tk.DISABLED)
                self.thought_text.see(tk.END)
        except queue.Empty:
            pass
        finally:
            # 定期检查队列
            self.root.after(100, self._update_thought_display)
    
    def update_ai_state(self, state):
        """更新AI状态"""
        self.ai_state = state
        if state == "learning":
            self.thought_queue.put((datetime.datetime.now(), "system", "开始学习新知识..."))
        elif state == "responding":
            self.thought_queue.put((datetime.datetime.now(), "system", "开始处理用户请求..."))
        elif state == "idle":
            self.thought_queue.put((datetime.datetime.now(), "system", "处理完成，等待下一个任务..."))
    
    def clear_thoughts(self):
        """清空想法显示"""
        self.thought_text.config(state=tk.NORMAL)
        self.thought_text.delete(1.0, tk.END)
        self.thought_text.config(state=tk.DISABLED)

# ------------------------------
# 主程序入口
# ------------------------------
if __name__ == "__main__":
    try:
        # 系统自检
        if not system_check():
            sys.exit(1)
        
        # 创建主窗口
        root = ttkb.Window(themename="cosmo")
        root.title("MemoAI V2")
        root.geometry("800x600")
        
        # 确保中文显示正常
        root.option_add("*Font", "SimHei 10")
        
        # 创建应用实例
        app = App(root)
        
        # 启动主循环
        root.mainloop()
    except Exception as e:
        # 捕获并记录启动异常
        error_msg = f"程序启动失败: {str(e)}"
        print(error_msg)
        
        # 写入错误日志
        with open("startup_error.log", "w", encoding="utf-8") as f:
            import traceback
            traceback.print_exc(file=f)
            f.write(error_msg)
        
        # 显示错误对话框
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        messagebox.showerror("启动失败", f"程序无法启动: {str(e)}\n详细信息请查看startup_error.log")
        sys.exit(1)