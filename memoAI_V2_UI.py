import os
import sys
import json
import time
import random
import logging
from datetime import datetime
from collections import deque
from threading import Thread, Event
import numpy as np
import ttkbootstrap
from ttkbootstrap import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tkinter as tk
from tkinter import ttk

# 确保中文显示正常
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# ------------------------------
# 系统配置与日志设置
# ------------------------------
class Config:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.join(self.project_dir, 'log')
        self.memory_dir = os.path.join(self.project_dir, 'memory')
        self.model_dir = os.path.join(self.project_dir, 'model')
        self.model_path = os.path.join(self.model_dir, 'dialog_model.pth')
        self.max_memory_length = 1000  # 添加缺失的配置项
        self.encoding_dim = 128
        self.hidden_dim = 256
        self.n_layers = 2
        self.dropout = 0.3
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 10
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化配置
config = Config()

# 确保目录存在
for dir_path in [config.log_dir, config.memory_dir, config.model_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 日志配置
logging.basicConfig(
    filename=os.path.join(config.log_dir, 'app.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MemoAI')

# ------------------------------
# 工具函数
# ------------------------------
def log_event(message, level='info'):
    """记录事件到日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f'[{timestamp}] {message}'
    
    # 输出到控制台
    print(log_message)
    
    # 写入日志文件
    with open(os.path.join(config.log_dir, 'events.log'), 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')
    
    # 同时记录到logging模块
    if level.lower() == 'error':
        logger.error(message)
    elif level.lower() == 'warning':
        logger.warning(message)
    else:
        logger.info(message)

# ------------------------------
# 系统自检
# ------------------------------
def system_check():
    """系统自检函数，检查必要的库和环境"""
    log_event("系统自检开始...")
    success = True
    
    # 检查必要的库
    required_libraries = [
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('ttkbootstrap', 'ttkbootstrap'),
        ('matplotlib', 'matplotlib')
    ]
    
    for lib_name, import_name in required_libraries:
        try:
            __import__(import_name)
            log_event(f"✓ {lib_name} 已安装")
        except ImportError:
            log_event(f"✗ {lib_name} 未安装", 'error')
            success = False
    
    # 检查目录
    for dir_path in [config.log_dir, config.memory_dir, config.model_dir]:
        if os.path.exists(dir_path):
            log_event(f"✓ 目录 {dir_path} 存在")
        else:
            log_event(f"✗ 目录 {dir_path} 不存在", 'error')
            success = False
    
    # 检查CUDA
    if torch.cuda.is_available():
        log_event(f"✓ CUDA 可用，设备: {torch.cuda.get_device_name(0)}")
    else:
        log_event("✗ CUDA 不可用，将使用CPU", 'warning')
    
    if success:
        log_event("系统自检完成，所有必要组件已就绪")
    else:
        log_event("系统自检发现问题，请解决后重试", 'error')
    
    return success

# ------------------------------
# 数据管理
# ------------------------------
class DataManager:
    """数据管理类，处理记忆存储和加载"""
    def __init__(self):
        self.memory_file = os.path.join(config.memory_dir, 'memory.json')
        self.max_memory_length = config.max_memory_length  # 确保初始化该属性
        self.memory = self.load_memory()
        
    def load_memory(self):
        """加载记忆数据"""
        try:
            if os.path.exists(self.memory_file) and os.path.getsize(self.memory_file) > 0:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return deque(json.load(f), maxlen=self.max_memory_length)
            else:
                log_event(f"记忆文件不存在或为空，将创建新的记忆")
                return deque(maxlen=self.max_memory_length)
        except Exception as e:
            log_event(f"加载记忆失败: {str(e)}", 'error')
            return deque(maxlen=self.max_memory_length)
    
    def save_memory(self):
        """保存记忆数据"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.memory), f, ensure_ascii=False, indent=2)
            log_event(f"记忆已保存，当前记忆条数: {len(self.memory)}")
        except Exception as e:
            log_event(f"保存记忆失败: {str(e)}", 'error')
    
    def add_memory(self, user_input, ai_response):
        """添加新的对话记忆"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.memory.append({
            'timestamp': timestamp,
            'user': user_input,
            'ai': ai_response
        })
        self.save_memory()
        return True
    
    def get_recent_memory(self, n=10):
        """获取最近的n条记忆"""
        return list(self.memory)[-n:]
    
    def clear_memory(self):
        """清除所有记忆"""
        self.memory.clear()
        self.save_memory()
        log_event("所有记忆已清除")
        return True

# ------------------------------
# 字符编码
# ------------------------------
class CharEncoder:
    """字符编码器，处理文本与数值之间的转换"""
    def __init__(self):
        self.chars = set()
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.unknown_char = '<UNK>'
        self.pad_char = ' '  # 将多字符填充符改为单字符空格
        self.load_vocab()
    
    def load_vocab(self):
        """加载或初始化词汇表"""
        vocab_file = os.path.join(config.model_dir, 'vocab.json')
        
        # 基本字符集
        basic_chars = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789，。！？、；：‘’“”（）【】《》/\|+-*=<>@#$%^&()_~'
        
        if os.path.exists(vocab_file):
            try:
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    self.char_to_idx = json.load(f)
                self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
                self.chars = set(self.char_to_idx.keys())
                log_event(f"词汇表已加载，包含 {len(self.chars)} 个字符")
                return
            except Exception as e:
                log_event(f"加载词汇表失败: {str(e)}, 将创建新的词汇表", 'warning')
        
        # 初始化新词汇表
        self.chars = set(basic_chars)
        self.chars.add(self.unknown_char)
        self.chars.add(self.pad_char)
        
        # 创建映射
        self.char_to_idx = {char: i for i, char in enumerate(sorted(self.chars))}
        self.idx_to_char = {i: char for char, i in self.char_to_idx.items()}
        
        # 保存词汇表
        try:
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(self.char_to_idx, f, ensure_ascii=False, indent=2)
            log_event(f"新词汇表已创建，包含 {len(self.chars)} 个字符")
        except Exception as e:
            log_event(f"保存词汇表失败: {str(e)}", 'error')
    
    def update_vocab(self, text):
        """根据文本更新词汇表"""
        new_chars = set(text) - self.chars
        if new_chars:
            for char in new_chars:
                self.chars.add(char)
                self.char_to_idx[char] = len(self.char_to_idx)
                self.idx_to_char[len(self.idx_to_char)] = char
            
            vocab_file = os.path.join(config.model_dir, 'vocab.json')
            try:
                with open(vocab_file, 'w', encoding='utf-8') as f:
                    json.dump(self.char_to_idx, f, ensure_ascii=False, indent=2)
                log_event(f"词汇表已更新，新增 {len(new_chars)} 个字符，总计 {len(self.chars)} 个字符")
            except Exception as e:
                log_event(f"更新词汇表失败: {str(e)}", 'error')
            return True
        return False
    
    def encode(self, text, max_length=50):
        """将文本编码为数值序列"""
        # 确保文本中的字符都在词汇表中
        self.update_vocab(text)
        
        # 截断或填充文本到固定长度
        if len(text) > max_length:
            text = text[:max_length]
        else:
            # 使用单字符填充至max_length
            text = text.ljust(max_length, self.pad_char)
        
        # 编码
        return [self.char_to_idx.get(char, self.char_to_idx[self.unknown_char]) for char in text]
    
    def decode(self, indices):
        """将数值序列解码为文本"""
        return ''.join([self.idx_to_char.get(idx, self.unknown_char) for idx in indices]).rstrip(self.pad_char)
    
    def get_vocab_size(self):
        """获取词汇表大小"""
        return len(self.chars)

# ------------------------------
# LSTM对话模型
# ------------------------------
class LSTMDialogNet(nn.Module):
    """LSTM对话模型"""
    def __init__(self, input_size, hidden_dim, n_layers, output_size, dropout=0.3):
        super(LSTMDialogNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, config.encoding_dim)
        self.lstm = nn.LSTM(
            config.encoding_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)
        
        # 移动到适当的设备
        self.to(config.device)
        log_event(f"模型已初始化，使用设备: {config.device}")
    
    def forward(self, x, hidden):
        batch_size = x.size(0)
        
        # 嵌入层
        x = self.embedding(x)
        x = self.dropout(x)
        
        # LSTM层
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 全连接层
        output = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的输出
        output = self.softmax(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        """初始化隐藏状态"""
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(config.device),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(config.device)
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
        """加载模型，处理兼容性问题"""
        try:
            if os.path.exists(path):
                # 加载模型权重
                state_dict = torch.load(path, map_location=config.device)
                
                # 处理可能的键名不匹配问题
                current_keys = self.state_dict().keys()
                new_state_dict = {}
                
                for key, value in state_dict.items():
                    # 移除可能的"module."前缀（如果模型是在多GPU上保存的）
                    if key.startswith('module.'):
                        new_key = key[7:]
                    else:
                        new_key = key
                    
                    if new_key in current_keys:
                        # 检查张量形状是否匹配
                        if self.state_dict()[new_key].shape == value.shape:
                            new_state_dict[new_key] = value
                        else:
                            log_event(f"模型层 {new_key} 尺寸不匹配，将使用随机初始化", 'warning')
                    else:
                        log_event(f"模型中不存在层 {new_key}，将忽略", 'warning')
                
                # 更新模型权重
                self.load_state_dict(new_state_dict, strict=False)
                self.to(config.device)
                log_event(f"模型已加载自 {path}")
                return True
            else:
                log_event(f"模型文件 {path} 不存在", 'warning')
                return False
        except Exception as e:
            log_event(f"加载模型失败: {str(e)}", 'error')
            return False

# ------------------------------
# 自学习AI核心
# ------------------------------
class SelfLearningAI:
    """自学习AI核心类"""
    def __init__(self):
        self.data_manager = DataManager()
        self.encoder = CharEncoder()
        self.vocab_size = self.encoder.get_vocab_size()
        self.model = LSTMDialogNet(
            input_size=self.vocab_size,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            output_size=self.vocab_size,
            dropout=config.dropout
        )
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.is_training = False
        self.training_stop_event = Event()
        self.loss_history = []
        
        # 尝试加载现有模型
        if not self.model.load_model(config.model_path):
            log_event("将使用新模型进行训练")
    
    def preprocess_data(self, memory=None):
        """预处理数据用于训练"""
        if memory is None:
            memory = self.data_manager.get_recent_memory()
            
        if len(memory) < 2:
            log_event("记忆数据不足，无法进行训练", 'warning')
            return None, None
        
        inputs = []
        targets = []
        max_length = 50
        
        for i in range(len(memory) - 1):
            # 使用用户输入作为输入
            input_text = memory[i]['user']
            # 使用下一个AI响应作为目标
            target_text = memory[i+1]['ai']
            
            # 编码
            input_seq = self.encoder.encode(input_text, max_length)
            target_seq = self.encoder.encode(target_text, max_length)
            
            inputs.append(input_seq)
            targets.append(target_seq)
        
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
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        log_event(f"开始训练，共 {epochs} 轮，批次大小: {config.batch_size}")
        
        try:
            for epoch in range(epochs):
                if self.training_stop_event.is_set():
                    log_event("训练已被用户中断")
                    break
                
                epoch_loss = 0
                for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
                    loss = self.train_step(batch_inputs, batch_targets)
                    epoch_loss += loss
                    
                    # 调用进度回调
                    if progress_callback:
                        progress = (epoch * len(dataloader) + batch_idx + 1) / (epochs * len(dataloader)) * 100
                        progress_callback(progress, f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss:.4f}")
                    
                avg_loss = epoch_loss / len(dataloader)
                self.loss_history.append(avg_loss)
                log_event(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
                
                # 调用进度回调
                if progress_callback:
                    progress_callback((epoch+1)/epochs*100, f"Epoch {epoch+1}/{epochs} 完成, 平均Loss: {avg_loss:.4f}")
            
            # 保存模型
            self.model.save_model(config.model_path)
            
            # 绘制损失曲线
            self.plot_loss_history()
            
            log_event("训练完成")
            return True
        except Exception as e:
            log_event(f"训练过程中出错: {str(e)}", 'error')
            return False
        finally:
            self.is_training = False
    
    def start_training_thread(self, epochs=None, progress_callback=None):
        """在新线程中开始训练"""
        if self.is_training:
            log_event("模型正在训练中，请先停止当前训练", 'warning')
            return False
        
        train_thread = Thread(target=self.train, args=(epochs, progress_callback))
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
                    {"user": "什么是人工智能?", "ai": "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。"},
                    {"user": "机器学习的主要类型有哪些?", "ai": "机器学习主要分为监督学习、无监督学习、半监督学习和强化学习四大类。"},
                    {"user": "什么是深度学习?", "ai": "深度学习是机器学习的分支，是一种以人工神经网络为架构，对数据进行表征学习的算法。"}
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
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        try:
            for epoch in range(3):  # 较少的epoch，因为在线数据量通常较小
                if self.training_stop_event.is_set():
                    log_event("联网自学已被用户中断")
                    break
                
                epoch_loss = 0
                for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
                    loss = self.train_step(batch_inputs, batch_targets)
                    epoch_loss += loss
                    
                    # 调用进度回调
                    if progress_callback:
                        progress = (epoch * len(dataloader) + batch_idx + 1) / (3 * len(dataloader)) * 100
                        progress_callback(progress, f"联网自学 Epoch {epoch+1}/3, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss:.4f}")
                
                avg_loss = epoch_loss / len(dataloader)
                self.loss_history.append(avg_loss)
                log_event(f"联网自学 Epoch {epoch+1}/3, Average Loss: {avg_loss:.4f}")
                
                # 调用进度回调
                if progress_callback:
                    progress_callback((epoch+1)/3*100, f"联网自学 Epoch {epoch+1}/3 完成, 平均Loss: {avg_loss:.4f}")
            
            # 保存模型
            self.model.save_model(config.model_path)
            log_event("联网自学完成")
            return True
        except Exception as e:
            log_event(f"联网自学过程中出错: {str(e)}", 'error')
            return False
        finally:
            self.is_training = False
    
    def self_learn(self, progress_callback=None):
        """自主学习，使用现有记忆进行训练"""
        log_event("开始自主学习...")
        return self.start_training_thread(epochs=5, progress_callback=progress_callback)
    
    def infer(self, input_text, max_length=50):
        """推理生成响应"""
        if not input_text.strip():
            return "请输入有效的文本"
        
        try:
            # 编码输入
            input_seq = self.encoder.encode(input_text, max_length)
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(config.device)
            
            # 初始化隐藏状态
            hidden = self.model.init_hidden(1)
            
            # 推理
            self.model.eval()
            with torch.no_grad():
                output_seq = []
                for i in range(max_length):
                    output, hidden = self.model(input_tensor[:, i:i+1], hidden)
                    _, top_idx = output.topk(1)
                    output_seq.append(top_idx.item())
                    
                    # 如果生成了结束符或填充符，停止生成
                    if self.encoder.idx_to_char.get(top_idx.item(), '') == self.encoder.pad_char:
                        break
            
            # 解码输出
            response = self.encoder.decode(output_seq)
            self.model.train()
            
            # 添加到记忆
            self.data_manager.add_memory(input_text, response)
            
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
        # 修改样式为ttkbootstrap支持的内置样式
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
        
        # 初始化AI核心
        self.ai = SelfLearningAI()
        
        # 创建UI
        self.create_widgets()
        
        # 显示欢迎消息
        self.add_message("system", "欢迎使用MemoAI V2！我正在初始化，请稍候...")
        self.add_message("ai", "您好！我是MemoAI，一个可以自我学习的对话AI。有什么我可以帮助您的吗？")
    
    def create_widgets(self):
        """创建UI组件"""
        # 设置样式
        self.style = ttk.Style()
        # 移除自定义Rounded.TButton样式配置
        # self.style.configure('Rounded.TButton', borderwidth=0)
        self.style.configure('ChatFrame.TFrame', background='#f0f0f0')
        self.style.configure('UserMessage.TLabel', background='#0078d7', foreground='white')
        self.style.configure('AIMessage.TLabel', background='#e6e6e6', foreground='black')
        self.style.configure('SystemMessage.TLabel', background='#ffd700', foreground='black')
        
        # 主框架
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=BOTH, expand=True)
        
        # 聊天历史框架
        chat_frame = ttk.Frame(main_frame, style='ChatFrame.TFrame')
        chat_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        # 聊天历史滚动区域
        self.chat_canvas = tk.Canvas(chat_frame)
        self.chat_scrollbar = ttk.Scrollbar(chat_frame, orient=VERTICAL, command=self.chat_canvas.yview)
        self.chat_history = ttk.Frame(self.chat_canvas, style='ChatFrame.TFrame')
        
        self.chat_history.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        )
        
        self.chat_canvas.create_window((0, 0), window=self.chat_history, anchor="nw")
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)
        
        self.chat_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.chat_scrollbar.pack(side=RIGHT, fill=Y)
        
        # 绑定鼠标滚轮事件
        self.chat_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # 输入框架
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=X, pady=(0, 10))
        
        self.user_input = ttk.Entry(input_frame, font=('SimHei', 10))
        self.user_input.pack(side=LEFT, fill=X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", lambda event: self.send_message())
        
        send_btn = RoundedButton(input_frame, text="发送", command=self.send_message)
        send_btn.pack(side=RIGHT)
        
        # 状态和控制框架
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=X, pady=(0, 10))
        
        self.status_label = ttk.Label(control_frame, text="就绪", foreground="green")
        self.status_label.pack(side=LEFT)
        
        # 进度条
        self.progress_bar = ttk.Progressbar(control_frame, orient=HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(side=LEFT, padx=10, fill=X, expand=True)
        
        # 功能按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=X)
        
        self.learn_btn = RoundedButton(btn_frame, text="自主学习", command=self.start_self_learning)
        self.learn_btn.pack(side=LEFT, padx=5)
        
        # 添加联网自学按钮
        self.online_learn_btn = RoundedButton(btn_frame, text="联网自学", command=self.start_online_learning)
        self.online_learn_btn.pack(side=LEFT, padx=5)
        
        self.correct_btn = RoundedButton(btn_frame, text="手动纠错", command=self.open_correction_window)
        self.correct_btn.pack(side=LEFT, padx=5)
        
        self.clear_btn = RoundedButton(btn_frame, text="清除对话", command=self.clear_chat)
        self.clear_btn.pack(side=LEFT, padx=5)
        
        self.quit_btn = RoundedButton(btn_frame, text="退出", command=self.quit_app)
        self.quit_btn.pack(side=RIGHT, padx=5)
    
    def _on_mousewheel(self, event):
        """鼠标滚轮事件处理"""
        self.chat_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def add_message(self, sender, text):
        """添加消息到聊天历史"""
        # 创建消息框架
        msg_frame = ttk.Frame(self.chat_history)
        msg_frame.pack(fill=X, padx=5, pady=5, anchor="w" if sender == "user" else "e")
        
        # 设置样式和对齐方式
        style = f'{sender.capitalize()}Message.TLabel'
        anchor = E if sender == "user" else W
        
        # 创建消息标签
        msg_label = ttk.Label(
            msg_frame,
            text=text,
            style=style,
            wraplength=600,
            justify=LEFT,
            anchor=anchor,
            padding=10,
            borderwidth=1,
            relief=SOLID
        )
        msg_label.pack(fill=X, expand=True)
        
        # 滚动到底部
        self.chat_canvas.after_idle(lambda: self.chat_canvas.yview_moveto(1.0))
    
    def send_message(self):
        """发送用户消息并获取AI响应"""
        user_text = self.user_input.get().strip()
        if not user_text:
            return
        
        # 添加用户消息到聊天历史
        self.add_message("user", user_text)
        self.user_input.delete(0, END)
        
        # 更新状态
        self.status_label.config(text="思考中...", foreground="blue")
        self.root.update_idletasks()
        
        # 在新线程中获取AI响应，避免UI卡顿
        def get_ai_response():
            response = self.ai.infer(user_text)
            self.root.after(0, lambda: self.add_message("ai", response))
            self.root.after(0, lambda: self.status_label.config(text="就绪", foreground="green"))
        
        Thread(target=get_ai_response).start()
    
    def start_self_learning(self):
        """开始自主学习"""
        if self.ai.is_training:
            self.add_message("system", "AI正在训练中，请稍候...")
            return
        
        self.add_message("system", "开始自主学习，这可能需要几分钟时间...")
        self.status_label.config(text="学习中...", foreground="orange")
        self.learn_btn.config(state=DISABLED)
        
        # 进度回调函数
        def update_progress(progress, status_text):
            self.progress_bar['value'] = progress
            self.status_label.config(text=status_text)
            self.root.update_idletasks()
            
            if progress >= 100:
                self.progress_bar['value'] = 0
                self.learn_btn.config(state=NORMAL)
                self.status_label.config(text="就绪", foreground="green")
                self.add_message("system", "自主学习完成！")
        
        # 开始训练线程
        success = self.ai.self_learn(progress_callback=update_progress)
        if not success:
            self.status_label.config(text="就绪", foreground="green")
            self.learn_btn.config(state=NORMAL)
    
    # 添加联网自学方法
    def start_online_learning(self):
        """开始联网自学"""
        if self.ai.is_training:
            self.add_message("system", "AI正在训练中，请稍候...")
            return
        
        self.add_message("system", "开始联网自学，这可能需要几分钟时间...")
        self.status_label.config(text="联网学习中...", foreground="purple")
        self.online_learn_btn.config(state=DISABLED)
        
        # 进度回调函数
        def update_progress(progress, status_text):
            self.progress_bar['value'] = progress
            self.status_label.config(text=status_text)
            self.root.update_idletasks()
            
            if progress >= 100:
                self.progress_bar['value'] = 0
                self.online_learn_btn.config(state=NORMAL)
                self.status_label.config(text="就绪", foreground="green")
                self.add_message("system", "联网自学完成！")
        
        # 启动联网学习线程
        Thread(target=lambda: self.ai.online_learn(progress_callback=update_progress)).start()
    
    def open_correction_window(self):
        """打开手动纠错窗口"""
        correction_window = ttk.Toplevel(self.root)
        correction_window.title("手动纠错")
        correction_window.geometry("500x300")
        correction_window.resizable(False, False)
        correction_window.transient(self.root)
        correction_window.grab_set()
        
        # 窗口内容
        ttk.Label(correction_window, text="用户输入:").pack(anchor=W, padx=10, pady=(10, 0))
        input_text = ttk.Text(correction_window, height=3, width=60)
        input_text.pack(padx=10, pady=5)
        
        ttk.Label(correction_window, text="正确响应:").pack(anchor=W, padx=10, pady=(10, 0))
        response_text = ttk.Text(correction_window, height=5, width=60)
        response_text.pack(padx=10, pady=5)
        
        # 按钮框架
        btn_frame = ttk.Frame(correction_window)
        btn_frame.pack(fill=X, pady=10)
        
        def save_correction():
            input_str = input_text.get(1.0, END).strip()
            response_str = response_text.get(1.0, END).strip()
            
            if input_str and response_str:
                self.ai.manual_correction(input_str, response_str)
                self.add_message("system", "手动纠错样本已保存，将用于AI学习")
                correction_window.destroy()
            else:
                ttk.messagebox.showerror("错误", "用户输入和正确响应都不能为空！")
        
        # 恢复确认(保存)按钮
        ttk.Button(btn_frame, text="确认", command=save_correction).pack(side=RIGHT, padx=10)
        ttk.Button(btn_frame, text="取消", command=correction_window.destroy).pack(side=RIGHT)
    
    def clear_chat(self):
        """清除聊天历史"""
        # 清空聊天历史框架
        for widget in self.chat_history.winfo_children():
            widget.destroy()
        
        # 询问是否同时清除记忆
        if ttk.messagebox.askyesno("确认", "是否同时清除AI的记忆数据？"):
            self.ai.data_manager.clear_memory()
            self.add_message("system", "聊天历史和AI记忆已清除")
        else:
            self.add_message("system", "聊天历史已清除")
    
    def quit_app(self):
        """退出应用程序"""
        if self.ai.is_training:
            if not ttk.messagebox.askyesno("确认", "AI正在训练中，确定要退出吗？"):
                return
            self.ai.stop_training()
        
        log_event("应用程序已退出")
        self.root.destroy()

# ------------------------------
# 主程序入口
# ------------------------------
if __name__ == "__main__":
    try:
        # 系统自检
        if not system_check():
            sys.exit(1)
        
        # 创建主窗口
        root = ttkbootstrap.Window(themename="cosmo")
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
class App:
    """主应用程序类"""
    def __init__(self, root):
        self.root = root
        self.ai = None
        self.data_manager = None
        self.is_initialized = False
        
        # 创建必要的目录
        self._create_directories()
        
        # 初始化UI
        self.create_widgets()
        
        # 在单独线程中初始化AI组件，避免阻塞UI
        self.status_label.config(text="初始化中...", foreground="blue")
        threading.Thread(target=self._initialize_ai_components, daemon=True).start()
    
    def _create_directories(self):
        """创建必要的目录"""
        import os
        for dir_path in ["log", "memory", "model"]:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)
                except Exception as e:
                    log_event(f"创建目录失败: {dir_path}, 错误: {str(e)}", level="error")
    
    def _initialize_ai_components(self):
        """在后台线程初始化AI组件"""
        try:
            self.data_manager = DataManager()
            self.ai = SelfLearningAI(self.data_manager)
            self.is_initialized = True
            self.status_label.config(text="就绪", foreground="green")
            self.add_message("system", "欢迎使用MemoAI V2！我已准备好为您服务。")
        except Exception as e:
            error_msg = f"AI组件初始化失败: {str(e)}"
            log_event(error_msg, level="error")
            self.status_label.config(text="初始化失败", foreground="red")
            self.add_message("system", error_msg)
    
    def preprocess_data(self, memory=None):
        """预处理数据用于训练"""
        if memory is None:
            memory = self.data_manager.get_recent_memory()
            
        if len(memory) < 2:
            log_event("记忆数据不足，无法进行训练", 'warning')
            return None, None
        
        inputs = []
        targets = []
        max_length = 50
        
        for i in range(len(memory) - 1):
            # 使用用户输入作为输入
            input_text = memory[i]['user']
            # 使用下一个AI响应作为目标
            target_text = memory[i+1]['ai']
            
            # 编码
            input_seq = self.encoder.encode(input_text, max_length)
            target_seq = self.encoder.encode(target_text, max_length)
            
            inputs.append(input_seq)
            targets.append(target_seq)
        
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
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        log_event(f"开始训练，共 {epochs} 轮，批次大小: {config.batch_size}")
        
        try:
            for epoch in range(epochs):
                if self.training_stop_event.is_set():
                    log_event("训练已被用户中断")
                    break
                
                epoch_loss = 0
                for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
                    loss = self.train_step(batch_inputs, batch_targets)
                    epoch_loss += loss
                    
                    # 调用进度回调
                    if progress_callback:
                        progress = (epoch * len(dataloader) + batch_idx + 1) / (epochs * len(dataloader)) * 100
                        progress_callback(progress, f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss:.4f}")
                    
                avg_loss = epoch_loss / len(dataloader)
                self.loss_history.append(avg_loss)
                log_event(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
                
                # 调用进度回调
                if progress_callback:
                    progress_callback((epoch+1)/epochs*100, f"Epoch {epoch+1}/{epochs} 完成, 平均Loss: {avg_loss:.4f}")
            
            # 保存模型
            self.model.save_model(config.model_path)
            
            # 绘制损失曲线
            self.plot_loss_history()
            
            log_event("训练完成")
            return True
        except Exception as e:
            log_event(f"训练过程中出错: {str(e)}", 'error')
            return False
        finally:
            self.is_training = False
    
    def start_training_thread(self, epochs=None, progress_callback=None):
        """在新线程中开始训练"""
        if self.is_training:
            log_event("模型正在训练中，请先停止当前训练", 'warning')
            return False
        
        train_thread = Thread(target=self.train, args=(epochs, progress_callback))
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
                    {"user": "什么是人工智能?", "ai": "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。"},
                    {"user": "机器学习的主要类型有哪些?", "ai": "机器学习主要分为监督学习、无监督学习、半监督学习和强化学习四大类。"},
                    {"user": "什么是深度学习?", "ai": "深度学习是机器学习的分支，是一种以人工神经网络为架构，对数据进行表征学习的算法。"}
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
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        try:
            for epoch in range(3):  # 较少的epoch，因为在线数据量通常较小
                if self.training_stop_event.is_set():
                    log_event("联网自学已被用户中断")
                    break
                
                epoch_loss = 0
                for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
                    loss = self.train_step(batch_inputs, batch_targets)
                    epoch_loss += loss
                    
                    # 调用进度回调
                    if progress_callback:
                        progress = (epoch * len(dataloader) + batch_idx + 1) / (3 * len(dataloader)) * 100
                        progress_callback(progress, f"联网自学 Epoch {epoch+1}/3, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss:.4f}")
                
                avg_loss = epoch_loss / len(dataloader)
                self.loss_history.append(avg_loss)
                log_event(f"联网自学 Epoch {epoch+1}/3, Average Loss: {avg_loss:.4f}")
                
                # 调用进度回调
                if progress_callback:
                    progress_callback((epoch+1)/3*100, f"联网自学 Epoch {epoch+1}/3 完成, 平均Loss: {avg_loss:.4f}")
            
            # 保存模型
            self.model.save_model(config.model_path)
            log_event("联网自学完成")
            return True
        except Exception as e:
            log_event(f"联网自学过程中出错: {str(e)}", 'error')
            return False
        finally:
            self.is_training = False
    
    def self_learn(self, progress_callback=None):
        """自主学习，使用现有记忆进行训练"""
        log_event("开始自主学习...")
        return self.start_training_thread(epochs=5, progress_callback=progress_callback)
    
    def infer(self, input_text, max_length=50):
        """推理生成响应"""
        if not input_text.strip():
            return "请输入有效的文本"
        
        try:
            # 编码输入
            input_seq = self.encoder.encode(input_text, max_length)
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(config.device)
            
            # 初始化隐藏状态
            hidden = self.model.init_hidden(1)
            
            # 推理
            self.model.eval()
            with torch.no_grad():
                output_seq = []
                for i in range(max_length):
                    output, hidden = self.model(input_tensor[:, i:i+1], hidden)
                    _, top_idx = output.topk(1)
                    output_seq.append(top_idx.item())
                    
                    # 如果生成了结束符或填充符，停止生成
                    if self.encoder.idx_to_char.get(top_idx.item(), '') == self.encoder.pad_char:
                        break
            
            # 解码输出
            response = self.encoder.decode(output_seq)
            self.model.train()
            
            # 添加到记忆
            self.data_manager.add_memory(input_text, response)
            
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
        # 修改样式为ttkbootstrap支持的内置样式
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
        
        # 初始化AI核心
        self.ai = SelfLearningAI()
        
        # 创建UI
        self.create_widgets()
        
        # 显示欢迎消息
        self.add_message("system", "欢迎使用MemoAI V2！我正在初始化，请稍候...")
        self.add_message("ai", "您好！我是MemoAI，一个可以自我学习的对话AI。有什么我可以帮助您的吗？")
    
    def create_widgets(self):
        """创建UI组件"""
        # 设置样式
        self.style = ttk.Style()
        # 移除自定义Rounded.TButton样式配置
        # self.style.configure('Rounded.TButton', borderwidth=0)
        self.style.configure('ChatFrame.TFrame', background='#f0f0f0')
        self.style.configure('UserMessage.TLabel', background='#0078d7', foreground='white')
        self.style.configure('AIMessage.TLabel', background='#e6e6e6', foreground='black')
        self.style.configure('SystemMessage.TLabel', background='#ffd700', foreground='black')
        
        # 主框架
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=BOTH, expand=True)
        
        # 聊天历史框架
        chat_frame = ttk.Frame(main_frame, style='ChatFrame.TFrame')
        chat_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        # 聊天历史滚动区域
        self.chat_canvas = ttk.Canvas(chat_frame)
        self.chat_scrollbar = ttk.Scrollbar(chat_frame, orient=VERTICAL, command=self.chat_canvas.yview)
        self.chat_history = ttk.Frame(self.chat_canvas, style='ChatFrame.TFrame')
        
        self.chat_history.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        )
        
        self.chat_canvas.create_window((0, 0), window=self.chat_history, anchor="nw")
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)
        
        self.chat_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.chat_scrollbar.pack(side=RIGHT, fill=Y)
        
        # 绑定鼠标滚轮事件
        self.chat_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # 输入框架
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=X, pady=(0, 10))
        
        self.user_input = ttk.Entry(input_frame, font=('SimHei', 10))
        self.user_input.pack(side=LEFT, fill=X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", lambda event: self.send_message())
        
        send_btn = RoundedButton(input_frame, text="发送", command=self.send_message)
        send_btn.pack(side=RIGHT)
        
        # 状态和控制框架
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=X, pady=(0, 10))
        
        self.status_label = ttk.Label(control_frame, text="就绪", foreground="green")
        self.status_label.pack(side=LEFT)
        
        # 进度条
        self.progress_bar = ttk.Progressbar(control_frame, orient=HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(side=LEFT, padx=10, fill=X, expand=True)
        
        # 功能按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=X)
        
        self.learn_btn = RoundedButton(btn_frame, text="自主学习", command=self.start_self_learning)
        self.learn_btn.pack(side=LEFT, padx=5)
        
        # 添加联网自学按钮
        self.online_learn_btn = RoundedButton(btn_frame, text="联网自学", command=self.start_online_learning)
        self.online_learn_btn.pack(side=LEFT, padx=5)
        
        self.correct_btn = RoundedButton(btn_frame, text="手动纠错", command=self.open_correction_window)
        self.correct_btn.pack(side=LEFT, padx=5)
        
        self.clear_btn = RoundedButton(btn_frame, text="清除对话", command=self.clear_chat)
        self.clear_btn.pack(side=LEFT, padx=5)
        
        self.quit_btn = RoundedButton(btn_frame, text="退出", command=self.quit_app)
        self.quit_btn.pack(side=RIGHT, padx=5)
    
    def _on_mousewheel(self, event):
        """鼠标滚轮事件处理"""
        self.chat_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def add_message(self, sender, text):
        """添加消息到聊天历史"""
        # 创建消息框架
        msg_frame = ttk.Frame(self.chat_history)
        msg_frame.pack(fill=X, padx=5, pady=5, anchor="w" if sender == "user" else "e")
        
        # 设置样式和对齐方式
        style = f'{sender.capitalize()}Message.TLabel'
        anchor = E if sender == "user" else W
        
        # 创建消息标签
        msg_label = ttk.Label(
            msg_frame,
            text=text,
            style=style,
            wraplength=600,
            justify=LEFT,
            anchor=anchor,
            padding=10,
            borderwidth=1,
            relief=SOLID
        )
        msg_label.pack(fill=X, expand=True)
        
        # 滚动到底部
        self.chat_canvas.after_idle(lambda: self.chat_canvas.yview_moveto(1.0))
    
    def send_message(self):
        """发送用户消息并获取AI响应"""
        user_text = self.user_input.get().strip()
        if not user_text:
            return
        
        # 添加用户消息到聊天历史
        self.add_message("user", user_text)
        self.user_input.delete(0, END)
        
        # 更新状态
        self.status_label.config(text="思考中...", foreground="blue")
        self.root.update_idletasks()
        
        # 在新线程中获取AI响应，避免UI卡顿
        def get_ai_response():
            response = self.ai.infer(user_text)
            self.root.after(0, lambda: self.add_message("ai", response))
            self.root.after(0, lambda: self.status_label.config(text="就绪", foreground="green"))
        
        Thread(target=get_ai_response).start()
    
    def start_self_learning(self):
        """开始自主学习"""
        if self.ai.is_training:
            self.add_message("system", "AI正在训练中，请稍候...")
            return
        
        self.add_message("system", "开始自主学习，这可能需要几分钟时间...")
        self.status_label.config(text="学习中...", foreground="orange")
        self.learn_btn.config(state=DISABLED)
        
        # 进度回调函数
        def update_progress(progress, status_text):
            self.progress_bar['value'] = progress
            self.status_label.config(text=status_text)
            self.root.update_idletasks()
            
            if progress >= 100:
                self.progress_bar['value'] = 0
                self.learn_btn.config(state=NORMAL)
                self.status_label.config(text="就绪", foreground="green")
                self.add_message("system", "自主学习完成！")
        
        # 开始训练线程
        success = self.ai.self_learn(progress_callback=update_progress)
        if not success:
            self.status_label.config(text="就绪", foreground="green")
            self.learn_btn.config(state=NORMAL)
    
    def open_correction_window(self):
        """打开手动纠错窗口"""
        correction_window = ttk.Toplevel(self.root)
        correction_window.title("手动纠错")
        correction_window.geometry("500x300")
        correction_window.resizable(False, False)
        correction_window.transient(self.root)
        correction_window.grab_set()
        
        # 窗口内容
        ttk.Label(correction_window, text="用户输入:").pack(anchor=W, padx=10, pady=(10, 0))
        input_text = ttk.Text(correction_window, height=3, width=60)
        input_text.pack(padx=10, pady=5)
        
        ttk.Label(correction_window, text="正确响应:").pack(anchor=W, padx=10, pady=(10, 0))
        response_text = ttk.Text(correction_window, height=5, width=60)
        response_text.pack(padx=10, pady=5)
        
        # 按钮框架
        btn_frame = ttk.Frame(correction_window)
        btn_frame.pack(fill=X, pady=10)
        
        def save_correction():
            input_str = input_text.get(1.0, END).strip()
            response_str = response_text.get(1.0, END).strip()
            
            if input_str and response_str:
                self.ai.manual_correction(input_str, response_str)
                self.add_message("system", "手动纠错样本已保存，将用于AI学习")
                correction_window.destroy()
            else:
                ttk.messagebox.showerror("错误", "用户输入和正确响应都不能为空！")
        
        # 恢复确认(保存)按钮
        ttk.Button(btn_frame, text="确认", command=save_correction).pack(side=RIGHT, padx=10)
        ttk.Button(btn_frame, text="取消", command=correction_window.destroy).pack(side=RIGHT)
    
    def clear_chat(self):
        """清除聊天历史"""
        # 清空聊天历史框架
        for widget in self.chat_history.winfo_children():
            widget.destroy()
        
        # 询问是否同时清除记忆
        if ttk.messagebox.askyesno("确认", "是否同时清除AI的记忆数据？"):
            self.ai.data_manager.clear_memory()
            self.add_message("system", "聊天历史和AI记忆已清除")
        else:
            self.add_message("system", "聊天历史已清除")
    
    def quit_app(self):
        """退出应用程序"""
        if self.ai.is_training:
            if not ttk.messagebox.askyesno("确认", "AI正在训练中，确定要退出吗？"):
                return
            self.ai.stop_training()
        
        log_event("应用程序已退出")
        self.root.destroy()

# ------------------------------
# 主程序入口
# ------------------------------
if __name__ == "__main__":
    try:
        # 系统自检
        if not system_check():
            sys.exit(1)
        
        # 创建主窗口
        root = ttkbootstrap.Window(themename="cosmo")
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
