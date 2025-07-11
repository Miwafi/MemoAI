
 

import os
import zhconv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import pycorrector
from textblob import TextBlob

# 1. 数据采集与处理模块
class DataManager:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = []
    def load_data(self):
        memory_path = os.path.join('.', 'memory.txt')
        if not os.path.exists(memory_path):
            return []
        with open(memory_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            try:
                pair = json.loads(line.strip())
                if isinstance(pair, list) and len(pair) == 2:
                    data.append(pair)
            except Exception:
                continue
        return data
    def add_data(self, user_input, ai_reply):
        self.data.append([user_input, ai_reply])

# 2. 模型定义

class CharEncoder:
    def __init__(self):
        # 英文+数字+标点+全部中文字符（通过zhconv库处理）
        self.chars = list("abcdefghijklmnopqrstuvwxyz0123456789,.!?;:()[]{}<>-_=+@#$%^&*~|/\\'\" ")
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.max_len = 32
    def encode(self, text):
        # 使用zhconv库将繁体转为简体，过滤非英文、数字、标点和中文
        text = zhconv.convert(text, 'zh-cn')
        text = text[:self.max_len]
        arr = np.zeros(self.max_len, dtype=int)
        for i, c in enumerate(text):
            if c in self.char2idx:
                arr[i] = self.char2idx[c]
            elif '\u4e00' <= c <= '\u9fff':
                # 中文字符统一映射到一个特殊索引（如最后一个）
                arr[i] = self.vocab_size - 1
            else:
                arr[i] = 0
        return arr
    def decode(self, arr):
        # 反向解码时，特殊索引显示为“中”
        return ''.join([self.idx2char.get(int(i), '中') if int(i) < self.vocab_size - 1 else '中' for i in arr]).strip()

class SimpleDialogNet(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.fc = nn.Linear(max_len * hidden_size, max_len * vocab_size)
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
    def forward(self, x):
        # x: (batch, max_len)
        emb = self.embed(x)  # (batch, max_len, hidden)
        emb = emb.reshape(emb.size(0), -1)  # (batch, max_len*hidden)
        out = self.fc(emb)  # (batch, max_len*vocab)
        out = out.view(-1, self.max_len, self.vocab_size)
        return out

class SelfLearningAI:
    def grammar_correct(self, text):
        # 自动检测语言
        def is_chinese(s):
            for ch in s:
                if '\u4e00' <= ch <= '\u9fff':
                    return True
            return False
        def is_english(s):
            for ch in s:
                if 'a' <= ch.lower() <= 'z':
                    return True
            return False
        # 中文纠错
        if is_chinese(text):
            corrected, detail = pycorrector.correct(text)
            return corrected
        # 英文纠错（使用TextBlob，无需联网和下载）
        elif is_english(text):
            blob = TextBlob(text)
            return str(blob.correct())
        else:
            return text
    def correct_text(self, text):
        # 简单纠错映射，可自行扩展
        corrections = {
            '裡': '里', '裏': '里', '妳': '你', '麼': '么', '麼': '么', '喫': '吃', '瞭': '了', '訊': '信', '訊息': '信息',
            '髮': '发', '髮型': '发型', '髮絲': '发丝', '髮膚': '发肤', '髮色': '发色', '髮量': '发量',
            '裏面': '里面', '裡面': '里面', '裡頭': '里头', '裏頭': '里头',
            '幫': '帮', '幫助': '帮助', '訊息': '信息', '訊號': '信号', '訊問': '询问',
            '覺': '觉', '覺得': '觉得', '覺醒': '觉醒', '覺察': '觉察',
            '麼': '么', '麼事': '么事', '麼么': '么么', '麼么哒': '么么哒',
            '嘗': '尝', '嘗試': '尝试', '嘗味': '尝味', '嘗鮮': '尝鲜',
            '訊': '信', '訊息': '信息', '訊號': '信号', '訊問': '询问',
            '瞭解': '了解', '瞭望': '了望', '瞭然': '了然',
        }
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
        return text
    def __init__(self, input_size, hidden_size, output_size, data_path):
        self.data_manager = DataManager(data_path)
        self.encoder = CharEncoder()
        self.model = SimpleDialogNet(self.encoder.vocab_size, self.encoder.max_len, hidden_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    def train(self, epochs=10):
        print("[训练] 开始训练对话模型...")
        data = self.data_manager.load_data()
        if not data:
            print("[训练] 没有可用对话数据，请先添加问答对。")
            return
        for epoch in range(epochs):
            total_loss = 0
            with tqdm(data, desc=f"第{epoch+1}/{epochs}轮训练进度") as pbar:
                for user_input, ai_reply in pbar:
                    x = torch.tensor([self.encoder.encode(user_input)], dtype=torch.long)
                    y = torch.tensor([self.encoder.encode(ai_reply)], dtype=torch.long)
                    out = self.model(x)  # (1, max_len, vocab)
                    loss = self.criterion(out.view(-1, self.encoder.vocab_size), y.view(-1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    pbar.set_postfix({"损失": f"{loss.item():.4f}"})
            print(f"第{epoch+1}/{epochs}轮训练完成，平均损失: {total_loss/len(data):.4f}")
    def self_learn(self, user_input=None, ai_reply=None):
        print("[自我学习] 采集新对话并增量训练...")
        if user_input is not None and ai_reply is not None:
            self.save_memory(user_input, ai_reply)
        self.train(epochs=20)  # 每次自我学习多训练轮数

    def save_memory(self, user_input, ai_reply):
        
        memory_path = os.path.join('.', 'memory.txt')
        with open(memory_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps([user_input, ai_reply], ensure_ascii=False) + '\n')
    def infer(self, user_input, base_penalty=1.2):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor([self.encoder.encode(user_input)], dtype=torch.long)
            out = self.model(x)  # (1, max_len, vocab)
            out = out.squeeze(0)  # (max_len, vocab)
            generated = []
            last_idx = None
            for i in range(self.encoder.max_len):
                logits = out[i].cpu().numpy()
                # 固定重复惩罚：如果已用过该字符，则惩罚系数为2
                for idx in range(self.encoder.vocab_size):
                    if idx in generated:
                        logits[idx] /= 2
                # 避免连续输出同一字符
                if last_idx is not None:
                    logits[last_idx] *= 0.3
                next_idx = int(np.argmax(logits))
                generated.append(next_idx)
                last_idx = next_idx
            reply = self.encoder.decode(generated)
            # 仅保留中英文、数字和常用标点
            allowed = set(self.encoder.chars)
            reply = ''.join([c for c in reply if c in allowed]).strip()
            reply = ' '.join(reply.split())
            # 删除连续重复内容
            if reply:
                dedup_reply = reply[0]
                for c in reply[1:]:
                    if c != dedup_reply[-1]:
                        dedup_reply += c
            else:
                dedup_reply = reply
            # 答复纠错（错别字+语法）
            dedup_reply = self.correct_text(dedup_reply)
            dedup_reply = self.grammar_correct(dedup_reply)
            return dedup_reply

def log_event(event):
    print(f"[LOG] {event}")

if __name__ == "__main__":
    ai = SelfLearningAI(input_size=32, hidden_size=64, output_size=32, data_path="./data")
    ai.train(epochs=3)
    print("\n按L键输入问答对并自我学习，按A键让AI自动回复，按K键自主学习，按Q键退出。")
    last_user_input = None
    last_ai_reply = None
    while True:
        key = input("请输入指令(L=人工标注/A=AI自动/K=自主学习/M=手动纠错/Q): ").strip().upper()
        if key == 'L':
            user_input = input("你: ")
            golden_reply = input("期望AI回复: ")
            ai.self_learn(user_input, golden_reply)
            print("已完成一次自我学习。")
            last_user_input = user_input
            last_ai_reply = golden_reply
        elif key == 'A':
            user_input = input("你: ")
            ai_reply = ai.infer(user_input)
            print("AI:", ai_reply)
            last_user_input = user_input
            last_ai_reply = ai_reply
        elif key == 'K':
            print("[自主学习] 正在读取全部历史对话并进行模型训练，请稍候...")
            ai.train(epochs=20)
            print("[自主学习] 已完成全部历史对话训练，模型已更新！")
        elif key == 'M':
            if last_user_input is None or last_ai_reply is None:
                print("[手动纠错] 暂无可纠正的AI输出，请先进行一次AI回复或人工标注。")
            else:
                print(f"上一次AI输出: {last_ai_reply}")
                corrected_reply = input("请输入修正后的答案: ")
                ai.save_memory(last_user_input, corrected_reply)
                print("已保存手动纠正后的答案到记忆。")
                last_ai_reply = corrected_reply
        elif key == 'Q':
            print("已退出。"); break
        else:
            print("无效指令，请输入L/A/K/M/Q。")


        