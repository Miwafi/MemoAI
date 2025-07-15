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
import requests
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
# 配置和常量
# 翻译文件
laun = {'中文':{
    'SYSCheck':'系统自检','CREATMenu':'创建必要目录','CREATError':'创建目录失败','ERROR':'错误',
    'SEND_BTN': '发送', 'SELF_LEARN_BTN': '自主学习', 'ONLINE_LEARN_BTN': '联网自学',
    'CORRECT_BTN': '手动纠错', 'COPY_AI_BTN': '复制AI输出', 'CLEAR_BTN': '清除对话',
    'SETTINGS_BTN': '设置', 'QUIT_BTN': '退出', 'READY_STATUS': '就绪',
    'WELCOME_MSG': '欢迎使用MemoAI V2！我正在初始化，请稍候...',
    'AI_GREETING': '您好！我是MemoAI，一个可以自我学习的对话AI。有什么我可以帮助您的吗？',
    'NETWORK_ROAMING': '网络漫游', 'APP_SUBTITLE': '自学习对话系统'
    },'ENG':{
        'SYSCheck':'System checking','CREATMenu':'Creating normal menu','CREATError':'Creating directory failed',
        'ERROR':'Error', 'SEND_BTN': 'Send', 'SELF_LEARN_BTN': 'Self Learning', 
        'ONLINE_LEARN_BTN': 'Online Learning', 'CORRECT_BTN': 'Manual Correction',
        'COPY_AI_BTN': 'Copy AI Output', 'CLEAR_BTN': 'Clear Chat', 'SETTINGS_BTN': 'Settings',
        'QUIT_BTN': 'Quit', 'READY_STATUS': 'Ready',
        'WELCOME_MSG': 'Welcome to MemoAI V2! Initializing, please wait...',
        'AI_GREETING': 'Hello! This is MemoAI, a self-learning conversational AI. How can I help you?',
        'NETWORK_ROAMING': 'Network Roaming', 'APP_SUBTITLE': 'Self-learning Dialogue System'
        },'日本語':{    
            'SYSCheck':'システム自己診断中','CREATMenu':'せっずを作成中','CREATError':'へイルを作成できませんでした',
            'ERROR':'エラー', 'SEND_BTN': '送信', 'SELF_LEARN_BTN': '自己学習', 
            'ONLINE_LEARN_BTN': 'オンライン学習', 'CORRECT_BTN': '手動修正',
            'COPY_AI_BTN': 'AI出力をコピー', 'CLEAR_BTN': 'チャットをクリア', 'SETTINGS_BTN': '設定',
            'QUIT_BTN': '完了', 'READY_STATUS': '準備完了',
            'WELCOME_MSG': 'MemoAI V2へようこそ！初期化中です、しばらくお待ちください...',
            'AI_GREETING': 'こんにちは！ MemoAIです、自己学習ができる会話型AIです。何かお手伝いできますか？',
            'NETWORK_ROAMING': '根とを言います', 'APP_SUBTITLE': '自己学習対話システム'
            }}
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
        self.temperature = 5
config = Config()
# 工具函数
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
# 系统自检
def system_check():
    """系统环境自检"""
    log_event(laun['ENG']['SYSCheck'])
    
    # 检查必要目录
    for dir_path in [config.log_dir, config.memory_dir, 'model']:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                log_event(f"{laun['ENG']['CREATMenu']}: {dir_path}")
            except Exception as e:
                log_event(f"{laun['ENG']['CREATError']}: {dir_path}, {laun['ENG']['ERROR']}: {str(e)}", 'error')
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
# 数据管理
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
# 字符编码
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
# 对话模型
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
# 自学习AI核心
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
                    # 增加批次级别的终止检查
                    if self.training_stop_event.is_set():
                        log_event(f"{task_name}已被用户中断")
                        return False
                    
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
            # 确保无论成功失败都重置训练状态
            self.is_training = False
            log_event(f"{task_name}线程已结束")
    
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
    
    def online_learn(self, query, progress_callback=None):
        online_knowledge = []
        try:
            # 网络请求逻辑
            url = f"https://www.baidu.com/s?wd={quote(query)}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.select('.result')

            for result in results[:3]:  # 取前3条结果
                title_tag = result.select_one('.t a')
                summary_tag = result.select_one('.c-abstract')
                if title_tag and summary_tag:
                    title = title_tag.get_text(strip=True)
                    summary = summary_tag.get_text(strip=True)
                    online_knowledge.append({
                        "user": f"什么是{title}?",
                        "ai": summary
                    })

            log_event(f"成功获取 {len(online_knowledge)} 条新知识")
            return online_knowledge

        except Exception as e:
            # 错误处理逻辑
            log_event(f"联网获取知识失败: {str(e)}, URL: {url if 'url' in locals() else '未知'}", 'error')
            print(f"[网络漫游]错误: {str(e)}, URL: {url if 'url' in locals() else '未知'}")
            return None
        return online_knowledge
        
        # 获取在线知识
        online_knowledge = fetch_online_knowledge()
        if not online_knowledge:
            self.is_training = False
            return False

        if progress_callback:
            progress_callback(50)  # 示例：更新进度为50%    
        
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
    def online_learn(self, query):
        online_knowledge = []
        try:
            # 网络请求逻辑
            url = f"https://www.baidu.com/s?wd={quote(query)}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.select('.result')

            for result in results[:3]:  # 取前3条结果
                title_tag = result.select_one('.t a')
                summary_tag = result.select_one('.c-abstract')
                if title_tag and summary_tag:
                    title = title_tag.get_text(strip=True)
                    summary = summary_tag.get_text(strip=True)
                    online_knowledge.append({
                        "user": f"什么是{title}?",
                        "ai": summary
                    })

            log_event(f"成功获取 {len(online_knowledge)} 条新知识")
            return online_knowledge

        except Exception as e:
            # 错误处理逻辑
            log_event(f"联网获取知识失败: {str(e)}, URL: {url if 'url' in locals() else '未知'}", 'error')
            print(f"[网络漫游]错误: {str(e)}, URL: {url if 'url' in locals() else '未知'}")
            return None
        return online_knowledge
        
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
    
    def infer(self, input_text, max_length=50, enable_network_roaming=False):
        """推理生成响应"""
        if not input_text.strip():
            return "请输入有效的文本"
        
        # 添加规则匹配处理常见问候语
        greetings = {
            "你好": "你好！很高兴见到你。",
            "您好": "您好！有什么我可以帮助您的吗？",
            "早上好": "早上好！祝您今天有个好心情。",
            "晚上好": "晚上好！今天过得怎么样？",
            "嗨": "嗨！你好呀！",
            "hi":"hi",
            "hi":"hi",
            "你好":"你好",
            "hello":"hello!",
            "早上好":"早上好呀",
            "晚上好":"晚上好哟",
            "再见":"再见啦",
            "拜拜":"拜拜咯",
            "谢谢":"不客气",
            "麻烦了":"没关系",
            "不好意思":"没事的",
            "对不起":"没关系的",
            "你好呀":"你好呀",
            "在吗":"在的",
            "在干嘛":"没干嘛呢",
            "吃饭了吗":"吃了",
            "睡觉了吗":"还没",
            "早上好呀":"早呀",
            "晚上好哟":"晚上好",
            "最近好吗":"挺好的",
            "还好吗":"还行",
            "过得怎么样":"就那样",
            "还好吧":"挺好的",
            "开心吗":"开心呀",
            "难过吗":"有点",
            "生气了吗":"没有哦",
            "累不累":"有点累",
            "困不困":"还好",
            "饿不饿":"有点饿",
            "渴不渴":"有点渴",
            "冷不冷":"有点冷",
            "热不热":"有点热",
            "天气真好":"是啊",
            "下雨了":"嗯呢",
            "下雪了":"好美呀",
            "刮风了":"注意保暖",
            "今天好热":"是啊好热",
            "今天好冷":"多穿点",
            "周末愉快":"你也是",
            "节日快乐":"谢谢",
            "生日快乐":"谢谢呀",
            "恭喜你":"谢谢啦",
            "太棒了":"是呀",
            "真厉害":"谢谢夸奖",
            "加油哦":"会的",
            "努力呀":"好的",
            "辛苦了":"还好",
            "休息一下":"好的",
            "慢慢来":"嗯呢",
            "别急呀":"知道了",
            "没关系的":"谢谢",
            "没问题的":"好的",
            "可以吗":"可以呀",
            "行吗":"行呀",
            "好吗":"好呀",
            "对吗":"对的",
            "是吗":"是的",
            "真的吗":"真的呀",
            "假的吧":"不是哦",
            "可能吧":"也许",
            "大概吧":"差不多",
            "应该吧":"可能",
            "一定的":"嗯呢",
            "必须的":"是的",
            "当然啦":"嗯呢",
            "肯定的":"对呀",
            "没错呀":"是的",
            "正确的":"对的",
            "错误的":"是的",
            "好主意":"谢谢",
            "好想法":"不错",
            "真有趣":"是呀",
            "真好玩":"对呀",
            "真无聊":"是呀",
            "真没劲":"嗯呢",
            "真漂亮":"谢谢",
            "真好看":"谢谢呀",
            "真难看":"是吗",
            "真丑呀":"别这样",
            "真好吃":"是呀",
            "真难吃":"有点",
            "真难闻":"嗯呢",
            "真香呀":"是的",
            "真臭呀":"确实",
            "真安静":"是啊",
            "真吵呀":"有点",
            "真干净":"谢谢",
            "真脏呀":"会打扫的",
            "真整齐":"还好",
            "真乱呀":"会整理的",
            "真快呀":"是的",
            "真慢呀":"抱歉",
            "真高呀":"还好",
            "真矮呀":"没办法",
            "真胖呀":"会减肥的",
            "真瘦呀":"谢谢",
            "真聪明":"谢谢夸奖",
            "真笨呀":"别这样说",
            "真可爱":"谢谢呀",
            "真讨厌":"别这样",
            "真善良":"谢谢",
            "真狠心":"不是的",
            "真勇敢":"谢谢",
            "真胆小":"有点",
            "真大方":"应该的",
            "真小气":"不是的",
            "真热情":"谢谢",
            "真冷漠":"不是故意的",
            "什么是力？":"力是物体对物体的作用，它可以改变物体的运动状态或使物体发生形变",
            "重力的方向是什么？":"重力的方向总是竖直向下的",
            "惯性是什么？":"惯性是物体保持原有运动状态不变的性质，是物体的固有属性",
            "牛顿第一定律又叫什么？":"牛顿第一定律又叫惯性定律",
            "压强的计算公式是什么？":"压强的计算公式是p=F/S，其中p表示压强，F表示压力，S表示受力面积",
            "什么是浮力？":"浮力是指浸在液体或气体中的物体受到液体或气体竖直向上托的力",
            "阿基米德原理的内容是什么？":"浸在液体中的物体受到向上的浮力，浮力的大小等于物体排开液体所受的重力",
            "声音是如何产生的？":"声音是由物体的振动产生的",
            "声音的传播需要介质吗？":"是的，声音的传播需要介质，真空不能传声",
            "光在真空中的传播速度是多少？":"光在真空中的传播速度约为3×10⁸米/秒",
            "光的反射定律是什么？":"反射光线、入射光线和法线在同一平面内，反射光线和入射光线分居法线两侧，反射角等于入射角",
            "什么是光的折射？":"光从一种介质斜射入另一种介质时，传播方向发生偏折的现象叫做光的折射",
            "凸透镜对光线有什么作用？":"凸透镜对光线有会聚作用",
            "凹透镜对光线有什么作用？":"凹透镜对光线有发散作用",
            "什么是平面镜成像？":"平面镜成像是由于光的反射形成的，所成的像是正立、等大的虚像，像与物到镜面的距离相等",
            "什么是导体？":"容易导电的物体叫做导体，如金属、人体、大地、石墨、酸碱盐的水溶液等",
            "什么是绝缘体？":"不容易导电的物体叫做绝缘体，如橡胶、玻璃、陶瓷、塑料、油等",
            "电流的方向是如何规定的？":"物理学中规定正电荷定向移动的方向为电流的方向",
            "电压的作用是什么？":"电压是使电路中形成电流的原因，电源是提供电压的装置",
            "电阻的大小与哪些因素有关？":"电阻的大小与导体的材料、长度、横截面积和温度有关",
            "欧姆定律的内容是什么？":"导体中的电流跟导体两端的电压成正比，跟导体的电阻成反比，公式为I=U/R",
            "串联电路中电流的特点是什么？":"串联电路中各处的电流都相等，即I=I₁=I₂=…=In",
            "并联电路中电压的特点是什么？":"并联电路中各支路两端的电压相等，都等于电源电压，即U=U₁=U₂=…=Un",
            "什么是电功？":"电流所做的功叫做电功，电流做功的过程就是电能转化为其他形式能的过程",
            "电功率的物理意义是什么？":"电功率是表示电流做功快慢的物理量",
            "什么是额定电压？":"用电器正常工作时的电压叫做额定电压",
            "什么是额定功率？":"用电器在额定电压下工作时的功率叫做额定功率",
            "家庭电路的电压是多少？":"我国家庭电路的电压是220V",
            "安全电压的范围是多少？":"一般情况下，对人体安全的电压不高于36V",
            "什么是磁场？":"磁场是存在于磁体周围的一种特殊物质，它对放入其中的磁体产生磁力的作用",
            "磁感线的特点是什么？":"磁感线是为了描述磁场而引入的假想曲线，磁体外部的磁感线从N极出发，回到S极，磁感线不相交",
            "电流的磁效应是谁发现的？":"电流的磁效应是由奥斯特发现的",
            "什么是电磁铁？":"电磁铁是利用电流的磁效应制成的，由铁芯和线圈组成，其磁性有无可由电流的通断控制",
            "电磁感应现象是谁发现的？":"电磁感应现象是由法拉第发现的",
            "发电机的工作原理是什么？":"发电机是根据电磁感应现象制成的，工作时将机械能转化为电能",
            "电动机的工作原理是什么？":"电动机是根据通电导体在磁场中受到力的作用的原理制成的，工作时将电能转化为机械能",
            "什么是动能？":"物体由于运动而具有的能叫做动能",
            "什么是重力势能？":"物体由于受到重力并处在一定高度而具有的能叫做重力势能",
            "什么是弹性势能？":"物体由于发生弹性形变而具有的能叫做弹性势能",
            "机械能包括哪些？":"机械能包括动能和势能，势能又包括重力势能和弹性势能",
            "能量守恒定律的内容是什么？":"能量既不会凭空消灭，也不会凭空产生，它只会从一种形式转化为其他形式，或者从一个物体转移到其他物体，而在转化和转移的过程中，能量的总量保持不变",
            "什么是杠杆？":"一根在力的作用下能绕着固定点转动的硬棒叫做杠杆",
            "杠杆的五要素是什么？":"杠杆的五要素是支点、动力、阻力、动力臂和阻力臂",
            "什么是省力杠杆？":"动力臂大于阻力臂的杠杆叫做省力杠杆，使用时省力但费距离",
            "什么是费力杠杆？":"动力臂小于阻力臂的杠杆叫做费力杠杆，使用时费力但省距离",
            "什么是等臂杠杆？":"动力臂等于阻力臂的杠杆叫做等臂杠杆，使用时既不省力也不费力",
            "定滑轮的特点是什么？":"定滑轮不省力，但能改变力的方向，实质是一个等臂杠杆",
            "动滑轮的特点是什么？":"动滑轮能省一半的力，但不能改变力的方向，实质是一个动力臂是阻力臂二倍的杠杆",
            "滑轮组的特点是什么？":"滑轮组既能省力又能改变力的方向，省力多少由承担物重的绳子段数决定",
            "什么是机械功？":"力对物体所做的功叫做机械功，做功的两个必要因素是作用在物体上的力和物体在力的方向上通过的距离，公式为W=Fs",
            "什么是机械效率？":"有用功跟总功的比值叫做机械效率，公式为η=W有/W总×100%，机械效率总小于1",
            "什么是功率？":"功率是表示物体做功快慢的物理量，单位时间内所做的功叫做功率，公式为P=W/t",
            "什么是密度？":"单位体积的某种物质的质量叫做这种物质的密度，公式为ρ=m/V",
            "什么是匀速直线运动？":"物体沿着直线且速度不变的运动叫做匀速直线运动",
            "速度的物理意义是什么？":"速度是表示物体运动快慢的物理量，公式为v=s/t",
            "什么是参照物？":"在研究物体的运动时，被选作标准的物体叫做参照物",
            "什么是惯性定律？":"一切物体在没有受到力的作用时，总保持静止状态或匀速直线运动状态，这就是惯性定律，也叫牛顿第一定律",
            "什么是二力平衡？":"物体在两个力的作用下保持静止或匀速直线运动状态，这两个力就彼此平衡，二力平衡的条件是:作用在同一物体上，大小相等，方向相反，作用在同一直线上",
            "什么是力的三要素？":"力的大小、方向和作用点叫做力的三要素，它们都能影响力的作用效果",
            "什么是力的作用效果？":"力可以使物体发生形变，也可以改变物体的运动状态",
            "什么是弹力？":"物体由于发生弹性形变而产生的力叫做弹力",
            "什么是摩擦力？":"两个相互接触的物体，当它们发生相对运动或有相对运动趋势时，在接触面上会产生一种阻碍相对运动的力，这种力叫做摩擦力",
            "影响滑动摩擦力大小的因素是什么？":"影响滑动摩擦力大小的因素有两个:一是压力的大小，二是接触面的粗糙程度，压力越大，接触面越粗糙，滑动摩擦力越大",
            "什么是大气压？":"大气对浸在它里面的物体产生的压强叫做大气压强，简称大气压或气压",
            "标准大气压的数值是多少？":"标准大气压的值约为1.013×10⁵Pa，相当于760mm高水银柱产生的压强",
            "液体压强的特点是什么？":"液体内部朝各个方向都有压强；在同一深度，液体向各个方向的压强相等；液体的压强随深度的增加而增大；不同液体的压强还跟液体的密度有关，在深度相同时，液体的密度越大，压强越大",
            "液体压强的计算公式是什么？":"液体压强的计算公式是p=ρgh，其中ρ表示液体的密度，h表示液体的深度，g取9.8N/kg",
            "什么是升华？":"物质从固态直接变成气态的过程叫做升华，升华需要吸热",
            "什么是凝华？":"物质从气态直接变成固态的过程叫做凝华，凝华需要放热",
            "什么是熔化？":"物质从固态变成液态的过程叫做熔化，熔化需要吸热",
            "什么是凝固？":"物质从液态变成固态的过程叫做凝固，凝固需要放热",
            "什么是汽化？":"物质从液态变成气态的过程叫做汽化，汽化需要吸热，汽化有蒸发和沸腾两种方式",
            "什么是液化？":"物质从气态变成液态的过程叫做液化，液化需要放热",
            "什么是晶体？":"有固定熔化温度的固体叫做晶体，晶体熔化时的温度叫做熔点",
            "什么是非晶体？":"没有固定熔化温度的固体叫做非晶体，非晶体没有熔点",
            "什么是温度？":"温度是表示物体冷热程度的物理量，常用的温度单位是摄氏度，符号是℃",
            "什么是光的直线传播？":"光在同种均匀介质中是沿直线传播的，小孔成像、影子的形成等都是光的直线传播现象",
            "什么是镜面反射？":"平行光线射到光滑表面上时反射光线也是平行的，这种反射叫做镜面反射",
            "什么是漫反射？":"平行光线射到凹凸不平的表面上，反射光线射向各个方向，这种反射叫做漫反射，漫反射也遵循光的反射定律",
            "什么是实像？":"实像是由实际光线会聚而成的，可以用光屏承接",
            "什么是虚像？":"虚像是由实际光线的反向延长线会聚而成的，不能用光屏承接",
            "什么是凸透镜的焦点？":"平行于主光轴的光线经凸透镜折射后，会聚在主光轴上的一点，这个点叫做凸透镜的焦点，用F表示",
            "什么是焦距？":"焦点到凸透镜光心的距离叫做焦距，用f表示",
            "什么是光的色散？":"太阳光通过棱镜后，被分解成各种颜色的光，这种现象叫做光的色散，说明白光是由各种色光混合而成的",
            "什么是导体和绝缘体的区别？":"导体内部有大量的自由电荷，绝缘体内部几乎没有自由电荷，导体和绝缘体在一定条件下可以相互转化",
            "什么是电路？":"由电源、用电器、开关和导线等元件组成的电流路径叫做电路",
            "电路的三种状态是什么？":"电路的三种状态是通路、断路和短路，通路是指处处连通的电路，断路是指某处断开的电路，短路是指电流不经过用电器而直接从电源正极回到负极的电路，短路是非常危险的",
            "串联电路和并联电路的区别是什么？":"串联电路中电流只有一条路径，各用电器相互影响；并联电路中电流有多条路径，各用电器互不影响",
            "什么是电势能？":"电荷在电场中具有的势能叫做电势能，电场力对电荷做功，电势能减少，克服电场力做功，电势能增加",
            "什么是磁场的基本性质？":"磁场的基本性质是对放入其中的磁体产生磁力的作用",
            "什么是磁化？":"使原来没有磁性的物体获得磁性的过程叫做磁化",
            "什么是电磁继电器？":"电磁继电器是利用电磁铁来控制工作电路的一种开关，它可以实现用低电压、弱电流控制高电压、强电流的工作电路",
            "什么是波长？":"相邻两个波峰或波谷之间的距离叫做波长，用λ表示",
            "什么是频率？":"物体每秒振动的次数叫做频率，用f表示，单位是赫兹，简称赫，符号是Hz",
            "什么是波速？":"波在介质中传播的速度叫做波速，公式为v=λf",
            "什么是超声波？":"频率高于20000Hz的声波叫做超声波，超声波具有方向性好、穿透能力强等特点",
            "什么是次声波？":"频率低于20Hz的声波叫做次声波，次声波可以传播很远的距离，具有很强的穿透能力",
            "什么是化学？":"化学是研究物质的组成、结构、性质以及变化规律的科学",
            "物质的三态是什么？":"物质的三态是固态、液态和气态",
            "什么是元素？":"元素是具有相同核电荷数（即质子数）的一类原子的总称",
            "地壳中含量最多的元素是什么？":"地壳中含量最多的元素是氧元素",
            "地壳中含量最多的金属元素是什么？":"地壳中含量最多的金属元素是铝元素",
            "什么是原子？":"原子是化学变化中的最小粒子",
            "什么是分子？":"分子是保持物质化学性质的最小粒子",
            "原子由什么构成？":"原子由原子核和核外电子构成，原子核由质子和中子构成",
            "质子带什么电？":"质子带正电荷",
            "电子带什么电？":"电子带负电荷",
            "中子带什么电？":"中子不带电",
            "原子中质子数、核电荷数、电子数的关系是什么？":"在原子中，质子数=核电荷数=核外电子数",
            "什么是相对原子质量？":"以一种碳原子质量的1/12为标准，其他原子的质量跟它相比较所得到的比，叫做这种原子的相对原子质量",
            "什么是化学式？":"用元素符号和数字的组合表示物质组成的式子叫做化学式",
            "什么是化合价？":"元素的化合价是元素的原子在形成化合物时表现出来的一种性质，在化合物中各元素正负化合价的代数和为零",
            "常见的金属有哪些？":"常见的金属有铁、铝、铜、锌、银、金等",
            "什么是金属活动性顺序？":"金属活动性顺序是指金属在水溶液中失去电子变成离子的能力由强到弱的顺序，常见顺序为:钾、钙、钠、镁、铝、锌、铁、锡、铅、（氢）、铜、汞、银、铂、金",
            "什么是溶液？":"一种或几种物质分散到另一种物质里，形成均一的、稳定的混合物，叫做溶液",
            "溶液由什么组成？":"溶液由溶质和溶剂组成，被溶解的物质叫做溶质，能溶解其他物质的物质叫做溶剂",
            "水是最常用的溶剂吗？":"是的，水是最常用的溶剂，汽油、酒精等也可以作溶剂",
            "什么是饱和溶液？":"在一定温度下，向一定量溶剂里加入某种溶质，当溶质不能继续溶解时，所得到的溶液叫做这种溶质的饱和溶液",
            "什么是不饱和溶液？":"在一定温度下，向一定量溶剂里加入某种溶质，还能继续溶解的溶液叫做这种溶质的不饱和溶液",
            "什么是溶解度？":"固体的溶解度是指在一定温度下，某固态物质在100g溶剂里达到饱和状态时所溶解的质量，单位是克",
            "影响固体溶解度的因素是什么？":"影响固体溶解度的主要因素是温度，大多数固体物质的溶解度随温度的升高而增大，少数物质的溶解度受温度影响很小，如氯化钠，极少数物质的溶解度随温度的升高而减小，如氢氧化钙",
            "什么是酸？":"酸在水溶液中解离出的阳离子全部是氢离子（H⁺），常见的酸有盐酸、硫酸、硝酸等",
            "什么是碱？":"碱在水溶液中解离出的阴离子全部是氢氧根离子（OH⁻），常见的碱有氢氧化钠、氢氧化钙等",
            "什么是盐？":"盐是由金属离子（或铵根离子）和酸根离子构成的化合物，常见的盐有氯化钠、碳酸钠、硫酸铜等",
            "什么是中和反应？":"酸和碱作用生成盐和水的反应叫做中和反应，中和反应在生活和生产中有广泛的应用，如改良土壤的酸碱性、治疗胃酸过多等",
            "什么是pH？":"pH是表示溶液酸碱度的数值，pH的范围通常在0-14之间，pH<7的溶液呈酸性，pH=7的溶液呈中性，pH>7的溶液呈碱性",
            "什么是氧化反应？":"物质与氧发生的化学反应叫做氧化反应",
            "什么是还原反应？":"含氧化合物里的氧被夺去的反应叫做还原反应",
            "什么是催化剂？":"在化学反应里能改变其他物质的化学反应速率，而本身的质量和化学性质在反应前后都没有发生变化的物质叫做催化剂（又叫触媒）",
            "什么是化学方程式？":"用化学式来表示化学反应的式子叫做化学方程式",
            "书写化学方程式要遵守什么原则？":"书写化学方程式要遵守两个原则:一是必须以客观事实为基础，绝不能凭空臆想、臆造事实上不存在的物质和化学反应；二是要遵守质量守恒定律，等号两边各原子的种类与数目必须相等",
            "什么是质量守恒定律？":"参加化学反应的各物质的质量总和，等于反应后生成的各物质的质量总和，这个规律就叫做质量守恒定律",
            "什么是氧气？":"氧气是一种无色、无味的气体，密度比空气略大，不易溶于水，具有助燃性和氧化性，能支持燃烧和供给呼吸",
            "氧气的化学式是什么？":"氧气的化学式是O₂",
            "如何检验氧气？":"将带火星的木条伸入集气瓶中，如果木条复燃，说明该气体是氧气",
            "实验室制取氧气的方法有哪些？":"实验室制取氧气的方法有:加热高锰酸钾制取氧气、加热氯酸钾和二氧化锰的混合物制取氧气、分解过氧化氢溶液制取氧气",
            "什么是氢气？":"氢气是一种无色、无味、密度比空气小、难溶于水的气体，具有可燃性和还原性，是最清洁的能源",
            "氢气的化学式是什么？":"氢气的化学式是H₂",
            "如何检验氢气的纯度？":"收集一试管氢气，用拇指堵住试管口，管口向下移近酒精灯火焰，松开拇指点火，如果听到尖锐的爆鸣声，表明氢气不纯，需要再收集、再检验；如果听到很小的声音，表明氢气已纯净",
            "什么是二氧化碳？":"二氧化碳是一种无色、无味的气体，密度比空气大，能溶于水，不能燃烧，也不支持燃烧，能使澄清石灰水变浑浊",
            "二氧化碳的化学式是什么？":"二氧化碳的化学式是CO₂",
            "如何检验二氧化碳？":"将气体通入澄清石灰水中，如果澄清石灰水变浑浊，说明该气体是二氧化碳",
            "实验室制取二氧化碳的药品是什么？":"实验室制取二氧化碳常用大理石（或石灰石）和稀盐酸",
            "什么是甲烷？":"甲烷是一种无色、无味的气体，密度比空气小，极难溶于水，具有可燃性，是天然气、沼气等的主要成分",
            "甲烷的化学式是什么？":"甲烷的化学式是CH₄",
            "什么是乙醇？":"乙醇俗称酒精，是一种无色、有特殊香味的液体，易挥发，能与水以任意比例互溶，具有可燃性，常用作燃料、溶剂等",
            "乙醇的化学式是什么？":"乙醇的化学式是C₂H₅OH",
            "什么是铁生锈的条件？":"铁生锈的条件是铁与氧气和水同时接触，防止铁生锈的方法有:涂油、刷漆、镀锌、保持铁制品表面洁净干燥等",
            "什么是金属的冶炼？":"金属的冶炼是指从金属矿物中提炼出金属的过程，如炼铁的原理是利用一氧化碳的还原性将铁从其氧化物中还原出来",
            "什么是盐酸？":"盐酸是氯化氢气体的水溶液，是一种无色、有刺激性气味的液体，具有挥发性和腐蚀性，能与活泼金属、金属氧化物、碱等发生反应",
            "盐酸的化学式是什么？":"盐酸中溶质的化学式是HCl",
            "什么是硫酸？":"硫酸是一种无色、粘稠、油状的液体，不易挥发，具有吸水性、脱水性和强腐蚀性，能与活泼金属、金属氧化物、碱、某些盐等发生反应",
            "硫酸的化学式是什么？":"硫酸的化学式是H₂SO₄",
            "什么是氢氧化钠？":"氢氧化钠俗称烧碱、火碱、苛性钠，是一种白色固体，易溶于水，溶解时放出大量的热，具有强烈的腐蚀性，能与非金属氧化物、酸、某些盐等发生反应",
            "氢氧化钠的化学式是什么？":"氢氧化钠的化学式是NaOH",
            "什么是氢氧化钙？":"氢氧化钙俗称熟石灰、消石灰，是一种白色粉末状固体，微溶于水，其水溶液俗称石灰水，具有腐蚀性，能与二氧化碳、酸、某些盐等发生反应",
            "氢氧化钙的化学式是什么？":"氢氧化钙的化学式是Ca(OH)₂",
            "什么是氯化钠？":"氯化钠俗称食盐，是一种白色固体，易溶于水，是重要的调味品，也是人体生理活动必不可少的物质，还可用于配制生理盐水、腌制食品等",
            "氯化钠的化学式是什么？":"氯化钠的化学式是NaCl",
            "什么是碳酸钠？":"碳酸钠俗称纯碱、苏打，是一种白色粉末状固体，易溶于水，水溶液呈碱性，广泛用于玻璃、造纸、纺织、洗涤剂等工业",
            "碳酸钠的化学式是什么？":"碳酸钠的化学式是Na₂CO₃",
            "什么是碳酸氢钠？":"碳酸氢钠俗称小苏打，是一种白色粉末状固体，能溶于水，水溶液呈弱碱性，常用作发酵粉、治疗胃酸过多等",
            "碳酸氢钠的化学式是什么？":"碳酸氢钠的化学式是NaHCO₃",
            "什么是化肥？":"化肥是指以矿物、空气、水为原料，经过化学加工制成的含有农作物生长所需营养元素的物质，主要包括氮肥、磷肥、钾肥和复合肥",
            "什么是氮肥？":"氮肥是含有氮元素的化肥，能促进植物茎叶生长茂盛，叶色浓绿，如尿素、碳酸氢铵、硝酸铵等",
            "什么是磷肥？":"磷肥是含有磷元素的化肥，能促进植物根系发达，增强抗寒抗旱能力，促进作物早熟，籽粒饱满，如磷矿粉、过磷酸钙等",
            "什么是钾肥？":"钾肥是含有钾元素的化肥，能促进植物茎秆健壮，增强抗病虫害和抗倒伏能力，如硫酸钾、氯化钾等",
            "什么是复合肥？":"复合肥是同时含有两种或两种以上营养元素的化肥，如硝酸钾（KNO₃）、磷酸二氢铵（NH₄H₂PO₄）等",
            "什么是有机化合物？":"有机化合物简称有机物，通常是指含有碳元素的化合物（除碳的氧化物、碳酸、碳酸盐等以外），如甲烷、乙醇、葡萄糖、淀粉、蛋白质等",
            "什么是无机化合物？":"无机化合物简称无机物，通常是指不含有碳元素的化合物（包括碳的氧化物、碳酸、碳酸盐等），如水、氯化钠、硫酸、氢氧化钠等",
            "什么是糖类？":"糖类是由C、H、O三种元素组成的化合物，是人体主要的供能物质，常见的糖类有葡萄糖、蔗糖、淀粉等",
            "什么是油脂？":"油脂是重要的营养物质，是维持生命活动的备用能源，分为植物油和动物脂肪两类",
            "什么是蛋白质？":"蛋白质是构成细胞的基本物质，是机体生长及修补受损组织的主要原料，还能为人体提供能量，如瘦肉、鱼、鸡蛋、牛奶等食物中富含蛋白质",
            "什么是维生素？":"维生素在人体内需要量很小，但它们可以起到调节新陈代谢、预防疾病、维持身体健康的作用，如缺乏维生素C会引起坏血病，缺乏维生素A会引起夜盲症等",
            "什么是燃烧？":"燃烧是一种发光、放热的剧烈的氧化反应，燃烧需要同时满足三个条件:可燃物、氧气（或空气）、达到燃烧所需的最低温度（也叫着火点）",
            "灭火的原理是什么？":"灭火的原理是破坏燃烧的条件，即清除可燃物或使可燃物与其他物品隔离、隔绝氧气（或空气）、使温度降到着火点以下",
            "什么是空气污染？":"空气污染是指空气中混入了有害气体和烟尘等污染物，对人体健康和环境造成危害，常见的空气污染物有二氧化硫、一氧化碳、二氧化氮、可吸入颗粒物等",
            "什么是水污染？":"水污染是指水体因某种物质的介入，而导致其化学、物理、生物或者放射性等方面特征的改变，从而影响水的有效利用，危害人体健康或者破坏生态环境的现象，水污染的主要来源有工业废水、农业废水和生活污水等",
            "什么是化学变化？":"有新物质生成的变化叫做化学变化，如燃烧、生锈、食物腐烂等",
            "什么是物理变化？":"没有新物质生成的变化叫做物理变化，如物质的三态变化、形状改变等",
            "化学变化和物理变化的本质区别是什么？":"化学变化和物理变化的本质区别是是否有新物质生成",
            "什么是物质的物理性质？":"物质不需要发生化学变化就表现出来的性质叫做物理性质，如颜色、状态、气味、熔点、沸点、密度、硬度、溶解性等",
            "什么是物质的化学性质？":"物质在化学变化中表现出来的性质叫做化学性质，如可燃性、氧化性、还原性、稳定性、腐蚀性等",
            "什么是催化剂的作用？":"催化剂在化学反应中能改变化学反应速率，它既可以加快反应速率，也可以减慢反应速率，但自身的质量和化学性质在反应前后不变",
            "什么是离子？":"带电的原子或原子团叫做离子，带正电的离子叫做阳离子，带负电的离子叫做阴离子",
            "什么是电离？":"物质溶解于水时，离解成自由移动的离子的过程叫做电离",
            "什么是同素异形体？":"由同种元素组成的不同单质叫做同素异形体，如金刚石、石墨和C₆₀是碳元素的同素异形体",
            "什么是合金？":"合金是由一种金属跟其他金属（或非金属）熔合而成的具有金属特性的物质，合金的性能通常比组成它的纯金属更好，如钢是铁的合金，其硬度比纯铁大，抗腐蚀性比纯铁好",
            "什么是信息技术？":"信息技术是主要用于管理和处理信息所采用的各种技术的总称，包括传感技术、计算机技术和通信技术等",
            "什么是计算机？":"计算机是一种能够按照预先存储的程序，自动、高速地进行大量数值计算和各种信息处理的电子设备",
            "计算机的基本组成包括哪些部分？":"计算机的基本组成包括硬件系统和软件系统，硬件是物理设备，软件是运行在硬件上的程序和数据",
            "计算机硬件系统由哪些部分构成？":"计算机硬件系统由中央处理器（CPU）、存储器、输入设备和输出设备构成",
            "什么是CPU？":"CPU即中央处理器，是计算机的核心部件，负责执行指令、处理数据，主要由运算器和控制器组成",
            "存储器分为哪几类？":"存储器分为内存储器（内存）和外存储器（外存），内存速度快但容量小，外存速度较慢但容量大",
            "常见的输入设备有哪些？":"常见的输入设备有键盘、鼠标、扫描仪、摄像头、麦克风等",
            "常见的输出设备有哪些？":"常见的输出设备有显示器、打印机、音箱、投影仪等",
            "什么是操作系统？":"操作系统是管理计算机硬件与软件资源的系统软件，是计算机系统的核心，如Windows、macOS、Linux等",
            "什么是应用软件？":"应用软件是为满足用户特定需求而开发的软件，如办公软件、游戏软件、图形设计软件等",
            "什么是二进制？":"二进制是一种以2为基数的记数法，计算机内部数据的存储和处理都采用二进制，只有0和1两个数字",
            "什么是字节（Byte）？":"字节是计算机存储信息的基本单位，1字节等于8个二进制位（bit），简写为B",
            "常见的存储单位有哪些？":"常见的存储单位有字节（B）、千字节（KB）、兆字节（MB）、吉字节（GB）、太字节（TB）等，相邻单位间的换算关系是1024倍",
            "什么是计算机网络？":"计算机网络是将地理位置不同的具有独立功能的多台计算机及其外部设备，通过通信线路连接起来，在网络操作系统、网络管理软件及网络通信协议的管理和协调下，实现资源共享和信息传递的计算机系统",
            "计算机网络按覆盖范围可分为哪几类？":"计算机网络按覆盖范围可分为局域网（LAN）、城域网（MAN）和广域网（WAN）",
            "什么是IP地址？":"IP地址是互联网协议地址，是给连接到互联网上的每台主机分配的一个32位（IPv4）或128位（IPv6）的标识符，用于在网络中定位和识别设备",
            "什么是域名？":"域名是互联网上识别和定位计算机的层次结构式字符标识，与IP地址相对应，便于用户记忆，如www.baidu.com",
            "什么是HTTP？":"HTTP即超文本传输协议，是用于从万维网服务器传输超文本到本地浏览器的传送协议，规定了浏览器和服务器之间信息传递的格式和规则",
            "什么是URL？":"URL即统一资源定位符，是互联网上标准资源的地址，用于标识互联网上某一资源的位置，如https://www.example.com/path",
            "什么是搜索引擎？":"搜索引擎是一种用于在互联网上检索信息的工具，通过特定的技术从互联网上抓取、索引和存储信息，用户输入关键词后，能快速返回相关结果，如百度、谷歌",
            "什么是电子邮件（Email）？":"电子邮件是一种通过互联网进行信息交换的通信方式，具有快速、便捷、低成本等特点，格式通常为用户名@域名，如user@example.com",
            "什么是病毒？":"计算机病毒是一种人为编制的具有自我复制能力、能够破坏计算机系统功能或数据的程序，具有传染性、破坏性、潜伏性等特点",
            "如何防范计算机病毒？":"防范计算机病毒的措施包括:安装杀毒软件并定期更新病毒库、不随意打开来历不明的邮件和附件、不访问不安全的网站、及时修补系统漏洞、定期备份重要数据等",
            "什么是防火墙？":"防火墙是一种位于内部网络与外部网络之间的网络安全系统，依照特定的规则，允许或限制传输的数据通过，起到保护内部网络安全的作用",
            "什么是云计算？":"云计算是一种基于互联网的计算方式，通过网络按需提供可动态扩展的虚拟化资源，如服务器、存储、应用软件等，用户无需自建基础设施，按使用量付费",
            "什么是大数据？":"大数据指无法在一定时间范围内用常规软件工具进行捕捉、管理和处理的数据集合，具有海量（Volume）、高速（Velocity）、多样（Variety）、低价值密度（Value）、真实性（Veracity）等特点",
            "什么是人工智能（AI）？":"人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学，包括机器学习、自然语言处理、计算机视觉等领域",
            "什么是物联网（IoT）？":"物联网是通过信息传感设备，按约定的协议，将任何物品与互联网相连接，进行信息交换和通信，以实现对物品的智能化识别、定位、跟踪、监控和管理的一种网络",
            "什么是区块链？":"区块链是一种分布式账本技术，通过去中心化和加密技术，实现数据的不可篡改、可追溯和透明化，在金融、物流、医疗等领域有广泛应用前景",
            "什么是HTML？":"HTML即超文本标记语言，是用于创建网页的标准标记语言，通过一系列标签来描述网页的结构和内容",
            "什么是CSS？":"CSS即层叠样式表，用于描述HTML或XML等文档的呈现方式，控制网页的布局、字体、颜色等外观样式",
            "什么是JavaScript？":"JavaScript是一种具有函数优先的轻量级、解释型或即时编译型的编程语言，主要用于网页交互效果的实现，能使网页更加动态和响应式",
            "什么是数据库？":"数据库是按照数据结构来组织、存储和管理数据的仓库，能高效地实现数据的查询、插入、删除、修改等操作，常见的数据库有MySQL、Oracle、SQL Server等",
            "什么是编程？":"编程是指使用编程语言编写程序的过程，通过指令告诉计算机执行特定的任务，实现所需的功能",
            "常见的编程语言有哪些？":"常见的编程语言有Python、Java、C、C++、C#、JavaScript、PHP、Ruby等，不同语言有不同的应用场景",
            "什么是算法？":"算法是解决特定问题的步骤和方法，具有有穷性、确定性、可行性、输入、输出等特征，是程序的核心",
            "什么是数据结构？":"数据结构是计算机中组织和存储数据的方式，研究数据的逻辑结构、物理结构以及它们之间的相互关系，并对这些结构定义相应的操作，如数组、链表、栈、队列、树、图等",
            "什么是多媒体技术？":"多媒体技术是指将文本、声音、图像、动画、视频等多种媒体信息，通过计算机进行综合处理和有机结合，并能实现人机交互的技术",
            "什么是位图？":"位图也叫像素图，是由许多像素点组成的图像，放大后会失真，常见格式有JPG、PNG、GIF、BMP等",
            "什么是矢量图？":"矢量图是由数学公式定义的线条和形状组成的图像，放大后不会失真，常见格式有AI、SVG、EPS等",
            "什么是分辨率？":"分辨率指单位长度内包含的像素点数量，通常以像素/英寸（PPI）或像素/厘米（PPCM）表示，分辨率越高，图像越清晰",
            "什么是操作系统的功能？":"操作系统的主要功能包括进程管理、内存管理、文件管理、设备管理和用户接口管理等，是计算机系统的核心软件",
            "什么是BIOS？":"BIOS即基本输入输出系统，是固化在计算机主板上的一组程序，负责计算机启动时的硬件检测和初始化，以及引导操作系统加载",
            "什么是U盘？":"U盘是一种使用USB接口的无需物理驱动器的微型高容量移动存储产品，具有体积小、容量大、携带方便、即插即用等特点",
            "什么是固态硬盘（SSD）？":"固态硬盘是用固态电子存储芯片阵列而制成的硬盘，具有读写速度快、防震抗摔、功耗低等优点，相比传统机械硬盘性能更优",
            "什么是IPV4和IPV6？":"IPV4是第四版互联网协议，地址长度为32位，约43亿个地址，已分配殆尽；IPV6是第六版互联网协议，地址长度为128位，能提供海量地址，解决了IP地址短缺问题",
            "什么是FTP？":"FTP即文件传输协议，用于在网络上进行文件传输的标准协议，允许用户通过网络从一台计算机向另一台计算机上传或下载文件",
            "什么是Telnet？":"Telnet是一种远程登录协议，允许用户通过网络连接到远程计算机，并像在本地操作一样控制远程计算机（因安全性问题，现在多被SSH替代）",
            "什么是DNS？":"DNS即域名系统，负责将域名转换为对应的IP地址，使得用户可以通过易记的域名访问互联网资源，而无需记住复杂的IP地址",
            "什么是DHCP？":"DHCP即动态主机配置协议，能够自动为网络中的计算机分配IP地址、子网掩码、网关等网络参数，简化了网络配置过程",
            "什么是局域网（LAN）？":"局域网是指在某一区域内由多台计算机互联成的计算机组，覆盖范围通常在几千米以内，如办公室、学校、家庭内的网络",
            "什么是广域网（WAN）？":"广域网是一种跨越大的、地域性的计算机网络的集合，通常覆盖一个国家或多个国家，因特网是世界上最大的广域网",
            "什么是调制解调器（Modem）？":"调制解调器是一种计算机硬件，能把计算机的数字信号翻译成可沿普通电话线传送的模拟信号，而这些模拟信号又可被线路另一端的另一个调制解调器接收，并译成计算机可懂的数字信号，实现计算机通过电话线等模拟线路接入互联网",
            "什么是路由器？":"路由器是一种连接多个网络或网段的网络设备，能在不同网络之间转发数据分组，根据信道的情况自动选择和设定路由，以最佳路径发送信号",
            "什么是交换机？":"交换机是一种用于电（光）信号转发的网络设备，它可以为接入交换机的任意两个网络节点提供独享的电信号通路，主要用于局域网内的数据交换",
            "什么是网关？":"网关是连接两个不同网络的设备，能实现不同网络协议之间的转换，使不同网络中的计算机可以相互通信，如家庭网络中的路由器通常兼具网关功能",
            "什么是URL的组成部分？":"URL通常由协议类型、服务器地址（域名或IP地址）、端口号（可选）、路径、查询参数（可选）和锚点（可选）组成，如协议://服务器地址:端口/路径?查询参数#锚点",
            "什么是静态网页？":"静态网页是指内容固定不变的网页，由HTML代码编写，用户访问时服务器直接返回固定的HTML文件，无法根据用户操作动态生成内容",
            "什么是动态网页？":"动态网页是指内容可以根据不同情况动态变化的网页，通常结合服务器端脚本语言（如PHP、ASP、JSP）和数据库，能根据用户请求生成个性化内容",
            "什么是Cookie？":"Cookie是网站服务器存储在用户本地计算机上的小型文本文件，用于记录用户的登录状态、偏好设置等信息，以便网站为用户提供个性化服务",
            "什么是Session？":"Session是指服务器为每个用户建立的一个临时存储信息的会话对象，用于在多个请求之间保持用户状态，数据存储在服务器端，相对Cookie更安全",
            "什么是API？":"API即应用程序编程接口，是一些预先定义的函数或协议，为不同软件应用之间的交互提供标准接口，使得开发者可以调用其他服务或组件的功能，而无需了解其内部实现",
            "什么是软件开发流程？":"软件开发流程是指从需求分析、设计、编码、测试到部署、维护的一系列阶段，常见的模型有瀑布模型、敏捷开发、迭代模型等",
            "什么是需求分析？":"需求分析是软件开发的第一个阶段，通过与用户沟通，明确软件需要实现的功能、性能、接口等需求，形成需求规格说明书，为后续开发提供依据",
            "什么是软件测试？":"软件测试是在规定的条件下对软件进行操作，以发现软件错误，衡量软件质量，并对其是否能满足设计要求进行评估的过程，包括单元测试、集成测试、系统测试等",
            "什么是断点续传？":"断点续传是指在文件传输过程中，由于网络中断等原因导致传输中断后，能够从上次中断的位置继续传输，而无需重新开始，节省时间和带宽",
            "什么是P2P技术？":"P2P即对等网络技术，指网络中每个节点既是客户端又是服务器，节点之间可以直接进行数据交换，无需依赖中心服务器，如早期的BT下载、即时通讯工具等",
            "什么是虚拟现实（VR）？":"虚拟现实是一种可以创建和体验虚拟世界的计算机技术，利用计算机生成一种模拟环境，用户通过专门的设备（如VR眼镜）沉浸到该环境中，获得视觉、听觉等感官体验",
            "什么是增强现实（AR）？":"增强现实是一种将虚拟信息与真实世界巧妙融合的技术，通过终端设备（如手机、AR眼镜）将计算机生成的文字、图像、3D模型等虚拟信息叠加到真实场景中，增强用户对现实世界的感知",
            "什么是数据备份？":"数据备份是指将计算机系统中的数据复制到其他存储介质（如硬盘、U盘、云存储）中，以防止数据因意外（如病毒攻击、硬件故障、误删除）而丢失，以便在数据损坏或丢失时进行恢复",
            "什么是数据加密？":"数据加密是指通过加密算法将明文转换为密文的过程，只有拥有解密密钥的人才能将密文还原为明文，从而保证数据在传输和存储过程中的安全性和保密性",
            "什么是二进制的优点？":"二进制的优点包括:电路实现简单（只需表示0和1两种状态）、运算规则简单（加法和乘法规则少）、抗干扰能力强（两种状态区分明确），适合计算机硬件实现",
            "什么是汇编语言？":"汇编语言是一种低级编程语言，使用助记符代替机器指令的操作码，与机器语言一一对应，依赖于具体的计算机硬件，执行效率高，但编写和维护难度大",
            "什么是高级编程语言？":"高级编程语言是一种接近于人类自然语言和数学表达式的编程语言，不依赖于具体的硬件，可读性和可维护性强，需要通过编译或解释转化为机器语言才能执行，如Python、Java等",
            "什么是编译型语言？":"编译型语言是指需要通过编译器将源代码一次性编译成机器语言 executable 文件，之后运行时无需重新编译，直接执行，执行效率高，如C、C++、Java（编译为字节码）",
            "什么是解释型语言？":"解释型语言是指不需要预先编译，而是在运行时由解释器逐行解释源代码并执行，执行效率相对较低，但开发和调试更灵活，如Python、JavaScript、PHP",
            "什么是面向对象编程（OOP）？":"面向对象编程是一种编程范式，将数据和操作数据的方法封装在对象中，通过类和对象、继承、多态、封装等特性，提高代码的复用性、可维护性和扩展性，如Java、C++、Python都支持面向对象编程",
            "什么是函数？":"函数是一段具有特定功能的可重用代码块，通过接收输入参数（可选），执行一系列操作后返回结果（可选），能简化代码结构，提高代码复用性",
            "什么是变量？":"变量是计算机内存中用于存储数据的命名空间，其值可以在程序运行过程中改变，不同编程语言对变量的定义和使用有不同的规则，包括数据类型、作用域等",
            "什么是数据类型？":"数据类型是指变量或表达式可以存储的数据的种类，不同的数据类型决定了数据的存储方式、取值范围和可进行的操作，常见的有整数、浮点数、字符串、布尔值、数组、对象等",
            "什么是计算机的工作原理？":"计算机的工作原理基于冯·诺依曼体系结构，核心是存储程序和程序控制，即事先将程序和数据存入存储器，计算机在运行时自动从存储器中取出指令并执行，按程序的规定逐步完成操作",
            "什么是Cache？":"Cache即高速缓冲存储器，是位于CPU和主存储器之间的一种容量较小但速度很快的存储器，用于临时存储CPU频繁访问的数据和指令，减少CPU访问主存的次数，提高计算机运行速度",
            "什么是内存泄漏？":"内存泄漏是指程序在运行过程中，动态分配的内存空间在使用完毕后没有被正确释放，导致系统内存被逐渐耗尽，最终可能使程序运行变慢甚至崩溃，常见于C、C++等需要手动管理内存的语言",
            "什么是分布式系统？":"分布式系统是由多个相互连接的计算机组成的系统，这些计算机通过网络通信，协同工作，对外呈现为一个统一的整体，具有高可用性、可扩展性、容错性等优点，如分布式数据库、分布式文件系统",
            "什么是开源软件？":"开源软件是指源代码可以被公众获取的软件，任何人都可以查看、修改和分发，遵循开源协议，如Linux操作系统、Python编程语言、Apache服务器等，开源软件促进了知识共享和技术创新",
            "今天天气怎么样":"挺晴朗的，就是有点风",
            "你喜欢吃甜的还是咸的":"我更喜欢甜的，比如蛋糕之类的",
            "周末打算去哪里玩呀":"还没确定呢，可能在家看看剧",
            "最近有没有好看的电影推荐":"我觉得那部新出的悬疑片挺不错的",
            "你平时喜欢听什么类型的音乐":"比较喜欢轻音乐，能让人放松",
            "这周末有个画展，一起去吗":"好呀，我对绘画还挺感兴趣的",
            "你早上一般几点起床":"七点左右吧，习惯了这个时间",
            "晚上睡不着的时候你会做什么":"有时候会看看书，或者听听播客",
            "你最喜欢的季节是什么":"秋天，不冷不热，风景也好看",
            "你会做饭吗":"会一点简单的，比如番茄炒蛋",
            "你觉得养猫好还是养狗好":"各有各的好，猫比较独立，狗更热情",
            "最近工作忙不忙呀":"还好，不算特别忙，能准时下班",
            "你平时运动吗":"偶尔会去跑跑步，或者在家做做瑜伽",
            "你喜欢看什么类型的电视剧":"家庭伦理剧和喜剧都挺喜欢的",
            "你有没有什么特别的爱好":"喜欢收集各种明信片",
            "明天要降温了，记得多穿点":"好的，谢谢你提醒",
            "你去过海边吗":"去过一次，海边的日落特别美",
            "你觉得咖啡好喝还是茶好喝":"我更喜欢茶，尤其是绿茶",
            "你最近在追什么综艺吗":"有个慢生活类的综艺，挺治愈的",
            "你喜欢吃辣吗":"还行，能接受微辣，太辣就受不了了",
            "你周末一般几点睡觉":"比平时晚点，大概十一点左右吧",
            "你有没有养宠物呀":"现在没有，以前养过一只仓鼠",
            "你觉得城市里好还是农村好":"各有优势，城市方便，农村清静",
            "你平时用什么牌子的护肤品":"就用一些基础保湿的，没什么特定牌子",
            "最近有什么新出的游戏吗":"听说有款角色扮演类的游戏评价不错",
            "你喜欢下雨天吗":"喜欢，下雨天在家睡觉特别舒服",
            "你上学的时候最喜欢哪门课":"历史课，觉得能了解过去的事情很有趣",
            "你会骑自行车吗":"会呀，小时候经常骑",
            "你觉得夏天最不能少的东西是什么":"空调和冰西瓜，缺一不可",
            "你有没有什么害怕的东西":"有点怕虫子，尤其是蟑螂",
            "这附近新开了家餐厅，味道不错":"是吗，有空去尝尝看",
            "你平时喜欢逛超市吗":"挺喜欢的，慢慢逛感觉很解压",
            "你多久没回老家了":"快半年了，打算过年回去看看",
            "你喜欢看纪录片吗":"喜欢，尤其是自然和历史题材的",
            "你觉得早睡早起重要吗":"挺重要的，对身体好",
            "你有没有什么想去的地方":"一直想去西藏，看看那边的风景",
            "你平时网购多吗":"还行，主要买些生活用品",
            "你喜欢吃水果吗，最喜欢哪种":"喜欢呀，最喜欢草莓和芒果",
            "你会弹乐器吗":"以前学过一点钢琴，现在差不多忘了",
            "你觉得冬天最适合做什么":"窝在被窝里看电影，或者吃火锅",
            "最近有没有什么烦心事":"还好，没什么特别烦的，心态比较平和",
            "你喜欢穿休闲装还是正装":"肯定是休闲装，穿着舒服",
            "你每天都会喝水吗":"会的，尽量保证每天喝够八杯水",
            "你有没有什么特别的小习惯":"睡前会泡个脚，有助于睡眠",
            "你喜欢看球赛吗":"偶尔看，主要看世界杯的时候",
            "你觉得旅行最重要的是什么":"我觉得是和谁一起去，还有放松的心情",
            "你平时喜欢写东西吗":"会写点日记，记录一下当天的事情",
            "这周末有朋友来我家，做什么菜好呢":"可以做个火锅，方便又热闹",
            "你喜欢晴天还是阴天":"晴天吧，感觉心情都会变好",
            "你有没有丢过很重要的东西":"丢过一次钥匙，当时急死了",
            "你觉得读书有用吗":"肯定有用呀，能增长见识，开阔眼界",
            "你平时怎么缓解压力的":"听听歌，或者出去散散步",
            "你喜欢吃早餐吗，一般吃什么":"喜欢，通常吃包子或者豆浆油条",
            "你觉得养植物难吗":"有点难，我养死过好几盆了",
            "最近有没有什么新的流行语":"好像有个“绝绝子”，经常听到别人说",
            "你会折纸吗":"会折个千纸鹤、小船之类的简单样式",
            "你喜欢看日出还是日落":"日落吧，感觉更温柔一些",
            "你平时出门喜欢走路还是坐车":"不远的话就走路，还能锻炼身体",
            "你有没有什么偶像":"没有特别的偶像，比较欣赏努力的人",
            "你觉得钱重要吗":"挺重要的，但也不是最重要的，开心更重要",
            "你喜欢冬天的雪吗":"喜欢，下雪的时候整个世界都变白了，特别美",
            "你平时喜欢看漫画吗":"偶尔看，更喜欢看长篇的连载漫画",
            "你有没有通宵过":"有过，大学的时候和同学一起通宵复习",
            "你觉得朋友之间最重要的是什么":"真诚和互相理解吧",
            "你喜欢吃面食还是米饭":"都喜欢，换着吃不会腻",
            "你平时会记账吗":"会的，这样能知道钱花在哪里了",
            "你有没有什么技能是别人不知道的":"我会一点点魔术，很简单的那种",
            "你觉得每天运动多久合适":"半小时到一小时吧，太久了容易累",
            "你喜欢去图书馆吗":"喜欢，那里的氛围特别适合看书学习",
            "你有没有做过很勇敢的事":"小时候救过一只掉进水里的小猫",
            "你觉得什么颜色最百搭":"黑色吧，和什么颜色都能搭配",
            "你平时喜欢喝奶茶吗":"偶尔喝，一般点半糖的",
            "你有没有什么后悔的事":"有过，但现在觉得都是经历，也不后悔了",
            "你喜欢看烟花吗":"喜欢，烟花绽放的瞬间特别惊艳",
            "你觉得一个人生活好还是和家人一起好":"各有好处，偶尔想一个人静一静，偶尔也想热闹点",
            "你平时会关注新闻吗":"会的，主要看一些社会和国际新闻",
            "你喜欢吃火锅还是烤肉":"都喜欢，不同时候想吃的不一样",
            "你有没有什么小秘密":"肯定有呀，每个人都有自己的小秘密",
            "你觉得什么星座最靠谱":"我觉得靠谱不靠谱和星座关系不大，主要看个人",
            "你平时喜欢逛公园吗":"喜欢，尤其是早上，能看到很多锻炼的人，很有活力",
            "你有没有学过外语":"学过一点英语，日常交流还行",
            "你觉得夏天最好的解暑方式是什么":"吹着空调吃冰西瓜，太爽了",
            "你喜欢看演唱会吗":"喜欢，现场的氛围特别好，能让人很投入",
            "你有没有什么梦想":"想以后开一家自己的小店，卖喜欢的东西",
            "你觉得什么季节最适合旅行":"春天和秋天吧，天气不冷不热，景色也好",
            "你平时会做手工吗":"会做一些简单的手工，比如折纸、串珠子",
            "你喜欢吃零食吗，最喜欢哪种":"喜欢，最喜欢薯片和巧克力",
            "你觉得什么性格的人最受欢迎":"开朗、真诚、会换位思考的人吧",
            "你有没有和朋友吵过架":"有过，但很快就和好了，不影响感情",
            "你喜欢看日出吗":"喜欢，早起看日出，感觉新的一天充满希望",
            "你觉得每天睡多久合适":"七八个小时吧，睡太多会觉得累",
            "你平时喜欢听播客吗，听什么类型的":"喜欢，主要听一些故事类和知识类的播客",
            "你有没有去过游乐园":"去过，最喜欢坐过山车，很刺激",
            "你觉得什么味道最好闻":"刚洗过的衣服的味道，还有雨后泥土的味道",
            "你喜欢吃早餐还是晚餐":"都喜欢，早餐开启一天，晚餐结束一天",
            "你有没有什么害怕的动物":"害怕蛇，觉得有点吓人",
            "你觉得什么电影最感人":"亲情题材的电影，很容易让人有共鸣",
            "你平时会整理房间吗":"会的，房间整齐了，心情也会变好",
            "你喜欢看足球还是篮球":"篮球吧，觉得节奏更快一些",
            "你有没有收到过很特别的礼物":"收到过朋友亲手织的围巾，很感动",
            "你觉得什么职业最辛苦":"其实每个职业都有自己的辛苦，都不容易",
            "你喜欢吃甜筒还是冰淇淋":"甜筒吧，拿着方便，而且脆皮很好吃",
            "你平时会写日记吗":"会的，写下来能记住很多细节，以后看很有意义",
            "你有没有独自旅行过":"有过，一个人旅行很自由，能认识很多新朋友",
            "你觉得什么天气最适合睡觉":"下雨天，听着雨声特别容易睡着",
            "你喜欢吃海鲜吗":"喜欢，尤其是虾和螃蟹，味道特别鲜",
            "你有没有什么计划在今年完成":"想学会游泳，这个计划好久了",
            "你觉得什么音乐最能让人放松":"轻音乐和自然声音，比如雨声、海浪声",
            "你平时喜欢去健身房吗":"不常去，更喜欢在户外运动",
            "你有没有什么奇怪的癖好":"喜欢闻新书的味道，觉得特别香",
            "你觉得朋友多好还是少好":"不在于多，在于真心，有几个知心朋友就够了",
            "你喜欢吃辣的火锅还是清汤的":"看情况，和朋友一起可能吃辣的，自己吃就清汤的",
            "你有没有熬夜赶过工":"有过，赶报告的时候，第二天特别累",
            "你觉得什么颜色最能让人开心":"黄色吧，看起来很明亮，很有活力",
   
        }
        
        # 检查输入是否匹配问候语
        input_lower = input_text.strip().lower()
        for greeting, response in greetings.items():
            if greeting in input_lower:
                self.data_manager.add_memory(input_text, response)
                return response
        # 新增：检查是否为计算题并计算结果
        math_pattern = re.compile(r'^[\d\s\+\-\*\/\(\)\.]+$')
        if math_pattern.match(input_text.strip()):
            try:
                result = eval(input_text.strip())
                return f"计算结果: {result}"
            except:
                return "无法计算该表达式，请检查输入格式。"
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
            
            # 如果启用网络漫游且本地响应不理想，则进行网络搜索
            if enable_network_roaming and (not response.strip() or len(response) < 20):
                try:
                    log_event(f"进行网络搜索: {input_text}")
                    print(f"[网络漫游] 触发搜索: {input_text}")  # 终端日志
                    url = f"https://www.baidu.com/s?wd={requests.utils.quote(input_text)}"
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    results = soup.select('.result.c-container')[:1]
                    if results:
                        summary = results[0].select_one('.c-abstract').get_text(strip=True) if results[0].select_one('.c-abstract') else ""
                        if summary:
                            response = f"根据网络信息: {summary}"
                except Exception as e:
                    log_event(f"网络搜索失败: {str(e)}", 'error')
            
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
# UI组件
class RoundedButton(ttk.Button):
    """圆角按钮组件"""
    def __init__(self, parent, text, command=None, **kwargs):
        super().__init__(parent, text=text, command=command, **kwargs)
        # 使用ttkbootstrap支持的内置样式
        self.configure(style='Primary.TButton')
# 主应用界面
class App:
    """主应用程序类"""
    def __init__(self, root):
        self.root = root
        self.root.title("MemoAI V2 - 自学习对话系统")
        self.root.geometry("1001x563")  # 修改初始分辨率为1001:563
        self.root.minsize(1001, 563)     # 设置最低分辨率为1001:563
        
        self.current_language = '中文'  # 默认语言为中文
        
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
        
        # 启动系统自检
        self.run_system_check()
        
        # 显示欢迎消息
        self.add_message("system", self.get_text('WELCOME_MSG'))
        self.add_message("ai", self.get_text('AI_GREETING'))
    
    def get_text(self, key):
        """根据当前语言获取文本"""
        # 获取当前语言的字典，如果不存在则使用中文
        lang_dict = laun.get(self.current_language, laun['中文'])
        # 返回对应键的文本，如果不存在则返回键本身
        return lang_dict.get(key, key)
    
    def create_widgets(self):
        """创建UI组件"""
        # 设置样式
        self.style = ttk.Style()
        self._setup_fonts()
        self.style.configure('ChatFrame.TFrame', background='#f0f0f0')
        self.style.configure('UserMessage.TLabel', background='#0078d7', foreground='white')
        self.style.configure('AIMessage.TLabel', background='#e6e6e6', foreground='black')
        self.style.configure('SystemMessage.TLabel', background='#A7A7A7', foreground='black')  # 浅灰色背景，黑色文字
        self.style.configure('SystemSuccess.TLabel', background='#00cc66', foreground='white')   # 保留成功状态的绿色
        
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

        # 就绪状态标签
        self.status_label = ttk.Label(input_frame, text=self.get_text('READY_STATUS'), foreground="green")
        self.status_label.pack(side=tk.LEFT, padx=(0, 10))

        # 用户输入框（缩短并居中）
        self.user_input = ttk.Entry(input_frame, font=('SimHei', 10), width=40)  # 设置固定宽度
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", lambda event: self.send_message())

        # 发送按钮
        self.send_btn = RoundedButton(input_frame, text=self.get_text('SEND_BTN'), command=self.send_message)
        self.send_btn.pack(side=tk.LEFT)
        
        # 状态和控制框架
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 语言选择框架
        language_frame = ttk.Frame(control_frame)
        language_frame.pack(side=tk.RIGHT, padx=10)

        ttk.Label(language_frame, text="语言:").pack(side=tk.LEFT)
        # 创建语言按钮而非下拉框
        lang_frame = ttk.Frame(language_frame)
        lang_frame.pack(side=tk.LEFT)

        # 中文按钮
        self.cn_btn = ttk.Button(lang_frame, text="中文", command=lambda: self.set_language('中文'))
        self.cn_btn.pack(side=tk.LEFT, padx=2)
        self.cn_btn.config(state=tk.DISABLED, style='Disabled.TButton')  # 当前语言置灰

        # 英文按钮
        self.en_btn = ttk.Button(lang_frame, text="ENG", command=lambda: self.set_language('ENG'))
        self.en_btn.pack(side=tk.LEFT, padx=2)

        # 日文按钮
        self.jp_btn = ttk.Button(lang_frame, text="日本語", command=lambda: self.set_language('日本語'))
        self.jp_btn.pack(side=tk.LEFT, padx=2)
        
        # 进度条
        self.progress_bar = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # 功能按钮框架
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X)
        
        # 添加网络漫游开关
        self.network_roaming_var = tk.BooleanVar(value=False)
        self.roaming_switch = ttk.Checkbutton(btn_frame, text=self.get_text('NETWORK_ROAMING'), variable=self.network_roaming_var)
        self.roaming_switch.pack(side=tk.LEFT, padx=5)
        
        self.learn_btn = RoundedButton(btn_frame, text=self.get_text('SELF_LEARN_BTN'), command=self.start_self_learning)
        self.learn_btn.pack(side=tk.LEFT, padx=5)
        
        self.online_learn_btn = RoundedButton(btn_frame, text=self.get_text('ONLINE_LEARN_BTN'), command=self.start_online_learning)
        self.online_learn_btn.pack(side=tk.LEFT, padx=5)
        
        self.correct_btn = RoundedButton(btn_frame, text="手动纠错", command=self.open_correction_window)
        self.correct_btn.pack(side=tk.LEFT, padx=5)
        
        # 添加复制AI输出按钮
        self.copy_ai_btn = RoundedButton(btn_frame, text="复制AI输出", command=self.copy_last_ai_response)
        self.copy_ai_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = RoundedButton(btn_frame, text="清除对话", command=self.clear_chat)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.settings_btn = RoundedButton(btn_frame, text="设置", command=self.open_settings_window)
        self.settings_btn.pack(side=tk.LEFT, padx=5)
        self.quit_btn = RoundedButton(btn_frame, text="退出", command=self.quit_app)
        self.quit_btn.pack(side=tk.RIGHT, padx=5)
    

    def set_language(self, lang):
        """设置语言并更新按钮状态"""
        self.current_language = lang
        # 重置所有按钮状态
        self.cn_btn.config(state=tk.NORMAL, style='TButton')
        self.en_btn.config(state=tk.NORMAL, style='TButton')
        self.jp_btn.config(state=tk.NORMAL, style='TButton')

        # 设置当前语言按钮为灰色
        if lang == '中文':
            self.cn_btn.config(state=tk.DISABLED, style='Disabled.TButton')
        elif lang == 'ENG':
            self.en_btn.config(state=tk.DISABLED, style='Disabled.TButton')
        else:
            self.jp_btn.config(state=tk.DISABLED, style='Disabled.TButton')

        # 更新UI文本
        self.update_ui_texts()
    
    def update_ui_texts(self):
        """更新所有UI元素的文本"""
        # 更新按钮文本
        self.send_btn.config(text=self.get_text('SEND_BTN'))
        self.learn_btn.config(text=self.get_text('SELF_LEARN_BTN'))
        self.online_learn_btn.config(text=self.get_text('ONLINE_LEARN_BTN'))
        self.correct_btn.config(text=self.get_text('CORRECT_BTN'))
        self.copy_ai_btn.config(text=self.get_text('COPY_AI_BTN'))
        self.clear_btn.config(text=self.get_text('CLEAR_BTN'))
        self.settings_btn.config(text=self.get_text('SETTINGS_BTN'))
        self.quit_btn.config(text=self.get_text('QUIT_BTN'))
        
        # 更新状态标签和窗口标题
        self.status_label.config(text=self.get_text('READY_STATUS'))
        self.root.title(f"MemoAI V2 - {self.get_text('APP_SUBTITLE')}")
        
        # 更新网络漫游开关文本
        self.roaming_switch.config(text=self.get_text('NETWORK_ROAMING'))
    
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

    def run_system_check(self):
        """系统自检流程 - 创建独立模态窗口"""
        # 创建自检窗口
        self.check_window = tk.Toplevel(self.root)
        self.check_window.title("系统自检")
        self.check_window.resizable(False, False)
        
        # 设置为模态窗口（阻止主窗口操作）
        self.check_window.transient(self.root)
        self.check_window.grab_set()
        
        # 强制设置窗口固定尺寸（宽度x高度）
        width, height = 400, 200
        x = (self.check_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.check_window.winfo_screenheight() // 2) - (height // 2)
        self.check_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
        # 添加自检标题
        ttk.Label(self.check_window, text="系统自检中...", font=('SimHei', 12, 'bold')).pack(pady=10)
        
        # 添加进度条
        self.check_progress = ttk.Progressbar(self.check_window, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.check_progress.pack(pady=20, padx=20)
        
        # 添加状态标签
        self.check_status = ttk.Label(self.check_window, text="准备开始自检", font=('SimHei', 10))
        self.check_status.pack(pady=10)
        
        # 绑定窗口关闭事件
        self.check_window.protocol("WM_DELETE_WINDOW", self.on_check_window_close)
        
        self.check_steps = [
            ("检查数据管理器", self.check_data_manager),
            ("验证AI模型", self.check_ai_model),
            ("测试网络连接", self.check_network),
            ("验证依赖库", self.check_dependencies)
        ]
        self.check_success = True  # 默认自检成功
        
        def execute_check(index=0):
            if index < len(self.check_steps):
                step_name, step_func = self.check_steps[index]
                self.check_status.config(text=f"正在{step_name}...")
                
                try:
                    result, message = step_func()
                    self.check_success = self.check_success and result
                    
                    # 更新进度条
                    self.check_progress["value"] = (index + 1) * (100 / len(self.check_steps))
                    
                    # 继续下一项检查
                    self.check_window.after(1000, execute_check, index + 1)
                except Exception as e:
                    self.check_success = False
                    self.check_status.config(text=f"{step_name}失败: {str(e)}")
            else:
                # 自检完成处理
                if self.check_success:
                    self.check_status.config(text="系统自检成功！", foreground="green")
                    self.check_window.after(2000, self.check_window.destroy)  # 成功后自动关闭
                    self.status_label.config(text="自检完成", foreground="green")
                    self.add_message("system", "=== 系统自检完成 ===", custom_fg="#9933ff")
                else:
                    # 播放错误提示音
                    import winsound
                    winsound.Beep(1000, 500)  # 1000Hz频率，500ms时长
                    self.check_status.config(text="系统自检失败，请检查问题后重启", foreground="red")
                    # 添加关闭按钮
                    ttk.Button(self.check_window, text="关闭", command=self.on_check_window_close).pack(pady=10)
        
        # 启动检查流程
        self.check_window.after(500, execute_check)
    
    def on_check_window_close(self):
        """自检窗口关闭处理"""
        if not self.check_success:
            self.root.destroy()  # 自检失败时关闭主窗口
        self.check_window.destroy()
    
    def check_data_manager(self):
        """检查数据管理器状态"""
        # 修复属性名错误：memory_file → memory_path
        if hasattr(self.data_manager, 'memory_path'):
            memory_path = self.data_manager.memory_path
            # 添加详细日志输出
            log_event(f"检查记忆文件: {memory_path}")
            if os.path.exists(memory_path):
                return True, f"数据管理器就绪 (记忆文件: {os.path.basename(memory_path)})"
            return False, f"记忆文件不存在于: {memory_path}"
        return False, "数据管理器未初始化"
    
    def check_ai_model(self):
        """检查AI模型状态"""
        model_path = "model/dialog_model.pth"
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            return True, "模型文件正常"
        return False, "模型文件缺失或损坏"
    
    def check_network(self):
        """检查网络连接状态"""
        try:
            import requests
            response = requests.get("https://www.baidu.com", timeout=5)
            return response.status_code == 200, "网络连接正常"
        except:
            return False, "网络连接失败"
    
    def check_dependencies(self):
        """检查关键依赖库"""
        required_libs = ["torch", "requests", "bs4", "ttkbootstrap"]
        missing = [lib for lib in required_libs if not self._is_lib_installed(lib)]
        if not missing:
            return True, "所有依赖库已安装"
        return False, f"缺少依赖: {', '.join(missing)}"
    
    def _is_lib_installed(self, lib_name):
        """检查库是否安装"""
        try:
            __import__(lib_name)
            return True
        except ImportError:
            return False

    def _on_mousewheel(self, event):
        """鼠标滚轮事件处理"""
        self.chat_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def add_message(self, sender, text, typing_animation=False, memory_index=None, custom_bg=None, custom_fg=None):
        """添加消息到聊天历史"""
        # 创建消息框架
        msg_frame = ttk.Frame(self.chat_history)
        msg_frame.pack(fill=tk.X, padx=5, pady=5, anchor="center")  # 修改为居中对齐
        
        # 设置样式和对齐方式
        style = f'{sender.capitalize()}Message.TLabel'
        anchor = tk.CENTER  # 设置为居中
        
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

        # 创建消息标签 - 支持自定义颜色
        bg_color = custom_bg if custom_bg else ('#0078d7' if sender == 'user' else '#e6e6e6' if sender == 'ai' else '#ffd700')
        fg_color = custom_fg if custom_fg else ('white' if sender == 'user' else 'black')
        
        msg_label = ttk.Label(
            msg_frame,
            text="" if typing_animation else text,
            style=style,
            wraplength=600,
            justify=tk.LEFT,
            anchor=anchor,  # 文本居中
            padding=10,
            borderwidth=1,
            relief=tk.SOLID,
            background=bg_color,
            foreground=fg_color
        )
        msg_label.pack(fill=tk.X, expand=True, anchor=tk.CENTER)  # 居中显示

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
            # 获取网络漫游状态
            # 获取网络漫游状态
            enable_roaming = self.network_roaming_var.get() if hasattr(self, 'network_roaming_var') else False
            
            # 生成AI响应
            ai_response = self.ai.infer(user_text, enable_network_roaming=enable_roaming)
            
            # 使用打字动画显示AI回复
            # 获取最新记忆索引
            memory_index = self.ai.data_manager.get_latest_memory_index()
            self.root.after(0, lambda: self.display_ai_response(ai_response, memory_index))
            self.root.after(0, lambda: self.status_label.config(text="就绪", foreground="green"))
            self.root.after(0, lambda: self.update_ai_state("idle"))  # 恢复空闲状态
        
        threading.Thread(target=get_ai_response).start()

    def display_ai_response(self, response, memory_index):
        """显示AI响应并保存到最后一条响应变量"""
        self.last_ai_response = response
        self.add_message("ai", response, typing_animation=True, memory_index=memory_index)

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
        query_text = self.user_input.get().strip()
        if not query_text:
            self.add_message("system", "请输入查询内容")
            self.online_learn_btn.config(state=tk.NORMAL)
            self.status_label.config(text="就绪", foreground="green")
            self.update_ai_state("idle")
            return
        threading.Thread(target=lambda: self.ai.online_learn(
            query_text, 
            progress_callback=update_progress
        )).start()

    
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
    
    def open_settings_window(self):
        """打开设置窗口"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("设置")
        settings_window.geometry("400x400")  # 增大窗口高度
        settings_window.resizable(False, False)
        settings_window.option_add("*Font", self.default_font)

        # 创建设置框架
        settings_frame = ttk.Frame(settings_window, padding=20)
        settings_frame.pack(fill=tk.BOTH, expand=True)

        # 添加进度条
        ttk.Label(settings_frame, text="操作进度:").grid(row=0, column=0, sticky=tk.W, pady=10)
        self.progress_bar = ttk.Progressbar(settings_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress_bar.grid(row=0, column=1, sticky=tk.EW, pady=10)

        # 温度设置
        ttk.Label(settings_frame, text="推理温度设置:").grid(row=1, column=0, sticky=tk.W, pady=10)
        
        temp_frame = ttk.Frame(settings_frame)
        temp_frame.grid(row=1, column=1, sticky=tk.EW)
        
        # 修复AICore无config属性的问题，直接使用实例变量或默认值
        default_temp = getattr(self.ai, 'temperature', 5)
        self.temp_var = tk.IntVar(value=default_temp)
        
        # 设置滑动条固定长度并确保填充
        temp_scale = ttk.Scale(temp_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                              variable=self.temp_var, command=self.update_temp_label, length=200)
        temp_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.temp_label = ttk.Label(temp_frame, text=str(default_temp))
        self.temp_label.pack(side=tk.LEFT)
        
        # 功能按钮区域
        functions_frame = ttk.LabelFrame(settings_frame, text="功能选项")
        functions_frame.grid(row=2, column=0, columnspan=2, sticky=tk.NSEW, pady=15)

        # 网络漫游开关
        ttk.Checkbutton(functions_frame, text=self.get_text('NETWORK_ROAMING'), variable=self.network_roaming_var).pack(anchor=tk.W, pady=5)

        # 手动纠错按钮
        ttk.Button(functions_frame, text=self.get_text('CORRECT_BTN'), command=self.open_correction_window).pack(anchor=tk.W, pady=5)

        # 自我学习按钮
        ttk.Button(functions_frame, text=self.get_text('SELF_LEARN_BTN'), command=self.start_self_learning).pack(anchor=tk.W, pady=5)

        # 联网自学按钮
        ttk.Button(functions_frame, text=self.get_text('ONLINE_LEARN_BTN'), command=self.start_online_learning).pack(anchor=tk.W, pady=5)

        # 保存按钮
        save_btn = RoundedButton(settings_window, text="保存设置", command=lambda: self.save_settings(settings_window))
        save_btn.pack(pady=20)
        
        # 配置列权重，使第二列可扩展
        settings_frame.columnconfigure(1, weight=1)
        # 确保第一列也有适当宽度
        settings_frame.columnconfigure(0, minsize=120)

    def update_temp_label(self, value):
        """更新温度标签显示"""
        self.temp_label.config(text=str(int(float(value))))

    def save_settings(self, window):
        """保存设置"""
        # 获取温度值
        temperature = self.temp_var.get()
        
        # 直接设置AICore的temperature属性
        self.ai.temperature = temperature
        
        # 显示保存成功消息
        self.add_message("system", f"设置已保存，温度值: {temperature}")
        window.destroy()
    
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
    
    def copy_last_ai_response(self):
        """复制最后一条AI输出到剪贴板"""
        if hasattr(self, 'last_ai_response') and self.last_ai_response:
            self.root.clipboard_clear()  # 清除剪贴板
            self.root.clipboard_append(self.last_ai_response)  # 添加文本到剪贴板
            self.status_label.config(text="AI输出已复制到剪贴板")
        else:
            self.status_label.config(text="暂无AI输出可复制")
    
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
# 主程序入口
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