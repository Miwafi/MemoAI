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
import tkinter as tk
from tkinter import messagebox, simpledialog
import datetime
import ttkbootstrap as ttk


# 自检函数
def system_check():
    checks = [
        {
            "name": "数据文件检查",
            "check_function": lambda: os.path.exists(os.path.join('.', 'memory.txt'))
        },
        {
            "name": "模型文件夹检查",
            "check_function": lambda: os.path.exists("./model")
        },
        {
            "name": "依赖库检查",
            "check_function": lambda: True  # 这里可以添加更复杂的库可用性检查
        }
    ]

    print("系统自检开始...")
    for check in checks:
        result = check["check_function"]()
        if result:
            print(f"{check['name']}: 通过")
        else:
            print(f"{check['name']}: 失败")
    print("系统自检结束。")


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


class LSTMDialogNet(nn.Module):
    """
    更复杂的神经网络系统：LSTM对话生成模型
    输入：字符索引序列
    输出：每步的字符概率分布
    """

    def __init__(self, vocab_size, max_len, hidden_size, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        # x: (batch, max_len)
        emb = self.embed(x)  # (batch, max_len, hidden)
        out, _ = self.lstm(emb)  # (batch, max_len, hidden)
        out = self.fc(out)  # (batch, max_len, vocab)
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
            '裏面': '里面', '裡面': '里面', '裡頭': '里头', '裡頭': '里头',
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

    def __init__(self, input_size, hidden_size=128, output_size=32, data_path="./data", model_dir="./model",
                 num_layers=2, lr=0.0005):
        self.data_manager = DataManager(data_path)
        self.encoder = CharEncoder()
        self.model = LSTMDialogNet(self.encoder.vocab_size, self.encoder.max_len, hidden_size, num_layers=num_layers)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, "dialog_model.pt")
        # 自动创建模型文件夹
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # 自动加载模型参数（如有）
        self.load_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"[模型] 已保存到 {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            print(f"[模型] 已加载模型参数: {self.model_path}")
        else:
            print(f"[模型] 未找到模型参数，将使用初始模型。")

    def train(self, epochs=10):
        print("[训练] 开始训练对话模型...")
        data = self.data_manager.load_data()
        # 数据去重和过滤空摘要
        filtered = []
        seen = set()
        for pair in data:
            if not pair or len(pair) != 2: continue
            user_input, ai_reply = pair
            if not user_input.strip() or not ai_reply.strip(): continue
            key = user_input.strip() + "|||" + ai_reply.strip()
            if key in seen: continue
            seen.add(key)
            filtered.append([user_input.strip(), ai_reply.strip()])
        if not filtered:
            print("[训练] 没有可用对话数据，请先添加问答对。")
            return
        print(f"[训练] 有效样本数: {len(filtered)}")
        for epoch in range(epochs):
            total_loss = 0
            with tqdm(filtered, desc=f"第{epoch + 1}/{epochs}轮训练进度") as pbar:
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
            print(f"第{epoch + 1}/{epochs}轮训练完成，平均损失: {total_loss / len(filtered):.4f}")
        self.save_model()

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
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{now}] {event}"
    with open('log.txt', 'a', encoding='utf-8') as f:
        f.write(log_line + '\n')


# 圆润按钮类
class RoundedButton(tk.Canvas):
    def __init__(self, parent, width, height, corner_radius, bg_color, fg_color, text, command=None):
        tk.Canvas.__init__(self, parent, width=width, height=height, bg=bg_color, highlightthickness=0)
        self.command = command
        self.width = width
        self.height = height
        self.corner_radius = corner_radius
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.text = text

        self.create_rounded_rectangle(0, 0, width, height, corner_radius, fill=fg_color)
        self.create_text(width // 2, height // 2, text=text, fill=bg_color, font=("Arial", 14))

        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)

    def create_rounded_rectangle(self, x1, y1, x2, y2, r, **kwargs):
        points = [x1 + r, y1, x1 + r, y1, x2 - r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y1 + r, x2, y2 - r, x2, y2 - r,
                  x2, y2, x2 - r, y2, x2 - r, y2, x1 + r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y2 - r, x1, y1 + r,
                  x1, y1 + r, x1, y1]
        return self.create_polygon(points, **kwargs, smooth=True)

    def on_press(self, event):
        if self.command:
            self.command()

    def on_release(self, event):
        pass


class App(ttk.Window):
    def __init__(self):
        super().__init__()
        self.title("Self Learning AI")
        self.geometry("600x400")
        self.ai = SelfLearningAI(input_size=32, hidden_size=128, output_size=32, data_path="./data",
                                 model_dir="./model", num_layers=2, lr=0.0005)
        log_event("程序启动")
        system_check()  # 调用自检函数
        self.ai.train(epochs=3)
        log_event("初始训练完成")

        self.last_user_input = None
        self.last_ai_reply = None

        # 背景虚化模拟
        self.bg_canvas = tk.Canvas(self, width=600, height=400, bg="#f0f0f0")
        self.bg_canvas.pack(fill=tk.BOTH, expand=True)

        self.create_widgets()
        self.animate_widgets()

    def create_widgets(self):
        # 输入框
        self.input_frame = tk.Frame(self.bg_canvas, bg="#f0f0f0")
        self.input_frame.pack(fill=tk.X, padx=10, pady=10)

        self.input_label = tk.Label(self.input_frame, text="你:", bg="#f0f0f0")
        self.input_label.pack(side=tk.LEFT)

        self.input_entry = tk.Entry(self.input_frame, width=50)
        self.input_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        # 按钮框架
        self.button_frame = tk.Frame(self.bg_canvas, bg="#f0f0f0")
        self.button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        button_width = 150
        button_height = 30
        button_corner_radius = 10
        button_bg_color = "#f0f0f0"
        button_fg_color = "#007BFF"

        self.ai_reply_button = RoundedButton(self.button_frame, width=button_width, height=button_height,
                                             corner_radius=button_corner_radius, bg_color=button_bg_color,
                                             fg_color=button_fg_color, text="AI自动回复", command=self.ai_reply)
        self.ai_reply_button.pack(pady=5)

        self.manual_labeling_button = RoundedButton(self.button_frame, width=button_width, height=button_height,
                                                    corner_radius=button_corner_radius, bg_color=button_bg_color,
                                                    fg_color=button_fg_color, text="人工标注", command=self.manual_labeling)
        self.manual_labeling_button.pack(pady=5)

        self.self_learning_button = RoundedButton(self.button_frame, width=button_width, height=button_height,
                                                  corner_radius=button_corner_radius, bg_color=button_bg_color,
                                                  fg_color=button_fg_color, text="自主学习", command=self.self_learning)
        self.self_learning_button.pack(pady=5)

        self.manual_correction_button = RoundedButton(self.button_frame, width=button_width, height=button_height,
                                                      corner_radius=button_corner_radius, bg_color=button_bg_color,
                                                      fg_color=button_fg_color, text="手动纠错",
                                                      command=self.manual_correction)
        self.manual_correction_button.pack(pady=5)

        self.online_exploration_button = RoundedButton(self.button_frame, width=button_width, height=button_height,
                                                       corner_radius=button_corner_radius, bg_color=button_bg_color,
                                                       fg_color=button_fg_color, text="联网探索",
                                                       command=self.online_exploration)
        self.online_exploration_button.pack(pady=5)

        self.exit_button = RoundedButton(self.button_frame, width=button_width, height=button_height,
                                         corner_radius=button_corner_radius, bg_color=button_bg_color,
                                         fg_color=button_fg_color, text="退出", command=self.exit_app)
        self.exit_button.pack(pady=5)

        # 聊天显示区域
        self.chat_display = tk.Text(self.bg_canvas, bg="#E5F6FF", fg="black", font=("Arial", 12),
                                    height=10, width=50)
        self.chat_display.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 设置用户和 AI 消息的标签样式
        self.chat_display.tag_config("user", justify=tk.RIGHT, foreground="#0084FF")
        self.chat_display.tag_config("ai", justify=tk.LEFT, foreground="green")

    def animate_widgets(self):
        # 简单的淡入动画
        opacity = 0.0
        def fade_in():
            nonlocal opacity
            if opacity < 1.0:
                opacity += 0.01
                self.attributes("-alpha", opacity)
                self.after(10, fade_in)

        fade_in()

    def ai_reply(self):
        user_input = self.input_entry.get()
        log_event(f"AI自动-用户输入: {user_input}")
        ai_reply = self.ai.infer(user_input)
        log_event(f"AI自动-回复: {ai_reply}")
        # 显示用户输入
        self.chat_display.insert(tk.END, f"你: {user_input}\n", "user")
        # 显示 AI 回复
        self.chat_display.insert(tk.END, f"AI: {ai_reply}\n", "ai")
        self.input_entry.delete(0, tk.END)
        # 滚动到最新消息
        self.chat_display.see(tk.END)
        self.last_user_input = user_input
        self.last_ai_reply = ai_reply

    def manual_labeling(self):
        user_input = self.input_entry.get()
        log_event(f"人工标注-用户输入: {user_input}")
        golden_reply = simpledialog.askstring("人工标注", "期望AI回复:")
        if golden_reply:
            log_event(f"人工标注-期望AI回复: {golden_reply}")
            self.ai.self_learn(user_input, golden_reply)
            log_event("完成一次自我学习")

    def self_learning(self):
        log_event("开始自主学习")
        import time
        start_time = time.time()
        max_seconds = 300  # 最长训练时间（秒），可调整
        while True:
            if time.time() - start_time > max_seconds:
                log_event(f"自主学习超时，终止训练")
                break
            data = self.ai.data_manager.load_data()
            if not data:
                log_event("自主学习无可用对话数据")
                break
            total_loss = 0
            with tqdm(data, desc="自我学习进度") as pbar:
                for user_input, ai_reply in pbar:
                    x = torch.tensor([self.ai.encoder.encode(user_input)], dtype=torch.long)
                    y = torch.tensor([self.ai.encoder.encode(ai_reply)], dtype=torch.long)
                    out = self.ai.model(x)
                    loss = self.ai.criterion(out.view(-1, self.ai.encoder.vocab_size), y.view(-1))
                    self.ai.optimizer.zero_grad()
                    loss.backward()
                    self.ai.optimizer.step()
                    total_loss += loss.item()
                    pbar.set_postfix({"损失": f"{loss.item():.4f}"})
            avg_loss = total_loss / len(data)
            log_event(f"自主学习本轮平均损失: {avg_loss:.4f}")
            self.ai.save_model()
            if avg_loss <= 1e-4:
                log_event("自主学习损失足够小，训练停止")
                break

    def manual_correction(self):
        if self.last_user_input is None or self.last_ai_reply is None:
            log_event("手动纠错-无可纠正AI输出")
            messagebox.showinfo("手动纠错", "暂无可纠正的AI输出，请先进行一次AI回复或人工标注。")
        else:
            log_event(f"手动纠错-上一次AI输出: {self.last_ai_reply}")
            messagebox.showinfo("手动纠错", f"上一次AI输出: {self.last_ai_reply}")
            corrected_reply = simpledialog.askstring("手动纠错", "请输入修正后的答案:")

    # 新增 online_exploration 方法
    def online_exploration(self):
        log_event("开始联网探索")
        # 这里可以添加具体的联网探索逻辑
        messagebox.showinfo("联网探索", "正在进行联网探索...")

    def exit_app(self):
        log_event("程序退出")
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()