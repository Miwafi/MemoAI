
 

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
    def __init__(self, input_size, hidden_size=128, output_size=32, data_path="./data", model_dir="./model", num_layers=2, lr=0.0005):
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
            with tqdm(filtered, desc=f"第{epoch+1}/{epochs}轮训练进度") as pbar:
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
            print(f"第{epoch+1}/{epochs}轮训练完成，平均损失: {total_loss/len(filtered):.4f}")
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
    import datetime
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{now}] {event}"
    with open('log.txt', 'a', encoding='utf-8') as f:
        f.write(log_line + '\n')

if __name__ == "__main__":
    # 启动时清空log.txt
    with open('log.txt', 'w', encoding='utf-8') as f:
        f.write('')
    ai = SelfLearningAI(input_size=32, hidden_size=128, output_size=32, data_path="./data", model_dir="./model", num_layers=2, lr=0.0005)
    log_event("程序启动")
    ai.train(epochs=3)
    log_event("初始训练完成")
    print("\n按L键输入问答对并自我学习，按A键让AI自动回复，按K键自主学习，按M键手动纠错，按I键联网探索，按Q键退出。")
    last_user_input = None
    last_ai_reply = None
    while True:
        key = input("请输入指令(L=人工标注/A=AI自动/K=自主学习/M=手动纠错/I=联网探索/Q): ").strip().upper()
        log_event(f"用户输入指令: {key}")
        if key == 'L':
            user_input = input("你: ")
            log_event(f"人工标注-用户输入: {user_input}")
            golden_reply = input("期望AI回复: ")
            log_event(f"人工标注-期望AI回复: {golden_reply}")
            ai.self_learn(user_input, golden_reply)
            log_event("完成一次自我学习")
            print("已完成一次自我学习。")
            last_user_input = user_input
            last_ai_reply = golden_reply
        elif key == 'A':
            user_input = input("你: ")
            log_event(f"AI自动-用户输入: {user_input}")
            ai_reply = ai.infer(user_input)
            log_event(f"AI自动-回复: {ai_reply}")
            print("AI:", ai_reply)
            last_user_input = user_input
            last_ai_reply = ai_reply
        elif key == 'K':
            import time
            log_event("开始自主学习")
            print("[自主学习] 正在读取全部历史对话并进行模型训练直到损失为0.0000，请稍候...")
            start_time = time.time()
            max_seconds = 300  # 最长训练时间（秒），可调整
            while True:
                if time.time() - start_time > max_seconds:
                    log_event(f"自主学习超时，终止训练")
                    print(f"[自我学习] 训练已超过{max_seconds}秒，自动终止！")
                    break
                data = ai.data_manager.load_data()
                if not data:
                    log_event("自主学习无可用对话数据")
                    print("[自我学习] 没有可用对话数据。"); break
                total_loss = 0
                with tqdm(data, desc="自我学习进度") as pbar:
                    for user_input, ai_reply in pbar:
                        x = torch.tensor([ai.encoder.encode(user_input)], dtype=torch.long)
                        y = torch.tensor([ai.encoder.encode(ai_reply)], dtype=torch.long)
                        out = ai.model(x)
                        loss = ai.criterion(out.view(-1, ai.encoder.vocab_size), y.view(-1))
                        ai.optimizer.zero_grad()
                        loss.backward()
                        ai.optimizer.step()
                        total_loss += loss.item()
                        pbar.set_postfix({"损失": f"{loss.item():.4f}"})
                avg_loss = total_loss / len(data)
                log_event(f"自主学习本轮平均损失: {avg_loss:.4f}")
                print(f"[自我学习] 当前平均损失: {avg_loss:.4f}")
                ai.save_model()
                if avg_loss <= 1e-4:
                    log_event("自主学习损失足够小，训练停止")
                    print("[自我学习] 损失已足够小，训练停止！")
                    break
        elif key == 'I':
            log_event("开始联网探索")
            print("[联网探索] 正在多站点抓取并收集反馈数据...")
            print("提示：本功能将自动在百度、知乎、搜狗、360等站点探索，所有结果与反馈将收集到 netexp.txt。memory.txt 仍用于AI训练。")
            import requests
            from bs4 import BeautifulSoup
            import time
            def fetch_multi_site_data():
                import random
                keywords = ["天气", "人工智能", "健康", "旅游", "美食"]
                sites = [
                    {"name": "百度", "url": lambda kw: f"https://www.baidu.com/s?wd={kw}", "parser": lambda soup: [ab.get_text(strip=True) for ab in soup.find_all(class_="c-abstract")]},
                    {"name": "知乎", "url": lambda kw: f"https://www.zhihu.com/search?type=content&q={kw}", "parser": lambda soup: [ab.get_text(strip=True) for ab in soup.find_all("div", class_="SearchResult-Title")]},
                    {"name": "搜狗", "url": lambda kw: f"https://www.sogou.com/web?query={kw}", "parser": lambda soup: [ab.get_text(strip=True) for ab in soup.find_all(class_="vrTitle")]},
                    {"name": "360", "url": lambda kw: f"https://www.so.com/s?q={kw}", "parser": lambda soup: [ab.get_text(strip=True) for ab in soup.find_all(class_="res-title")]},
                ]
                # 简单代理池（可扩展为从免费代理API获取）
                proxy_list = [
                    None,  # 本地直连
                    # 示例代理格式
                    # "http://123.123.123.123:8080",
                    # "http://111.111.111.111:8000",
                ]
                feedback = []
                results = []
                def is_captcha(text):
                    # 检查常见验证码提示
                    keywords = ["验证码", "人机验证", "请完成安全验证", "请输入验证码", "verify", "captcha"]
                    return any(k in text for k in keywords)
                for kw in keywords:
                    for site in sites:
                        url = site["url"](kw)
                        headers = {"User-Agent": "Mozilla/5.0"}
                        proxy = random.choice(proxy_list)
                        proxies = {"http": proxy, "https": proxy} if proxy else None
                        try:
                            resp = requests.get(url, headers=headers, timeout=10, proxies=proxies)
                            status = resp.status_code
                            # 检查验证码
                            if is_captcha(resp.text):
                                feedback.append({"site": site["name"], "keyword": kw, "url": url, "status": status, "summary": "被验证码拦截，已跳过。"})
                                continue
                            soup = BeautifulSoup(resp.text, "html.parser")
                            items = site["parser"](soup)
                            if items:
                                for summary in items:
                                    results.append([kw, summary])
                                    feedback.append({"site": site["name"], "keyword": kw, "url": url, "status": status, "summary": summary[:100]})
                            else:
                                feedback.append({"site": site["name"], "keyword": kw, "url": url, "status": status, "summary": "未找到摘要/内容"})
                        except Exception as e:
                            feedback.append({"site": site["name"], "keyword": kw, "url": url, "status": "error", "summary": str(e)})
                        time.sleep(2)  # 适当加大间隔，减少被封概率
                return results, feedback
            online_data, net_feedback = fetch_multi_site_data()
            log_event(f"联网探索完成，反馈数: {len(net_feedback)}")
            # 收集所有探索反馈到 netexp.txt
            netexp_path = os.path.join('.', 'netexp.txt')
            with open(netexp_path, 'a', encoding='utf-8') as f:
                for item in net_feedback:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"[联网探索] 已收集{len(net_feedback)}条探索反馈到 netexp.txt。")
            # 仅将有用问答对写入 memory.txt 供AI训练
            if online_data:
                log_event(f"联网学习获取新数据数: {len(online_data)}，开始训练")
                memory_path = os.path.join('.', 'memory.txt')
                with open(memory_path, 'a', encoding='utf-8') as f:
                    for user_input, ai_reply in online_data:
                        f.write(json.dumps([user_input, ai_reply], ensure_ascii=False) + '\n')
                print(f"[联网学习] 已获取{len(online_data)}条新数据，开始训练...")
                # 直接复用K分支训练逻辑
                import time
                start_time = time.time()
                max_seconds = 120  # 最长训练时间（秒），可调整
                while True:
                    if time.time() - start_time > max_seconds:
                        log_event(f"联网学习训练超时，终止训练")
                        print(f"[自我学习] 训练已超过{max_seconds}秒，自动终止！")
                        break
                    data = ai.data_manager.load_data()
                    if not data:
                        log_event("联网学习无可用对话数据")
                        print("[自我学习] 没有可用对话数据。"); break
                    total_loss = 0
                    with tqdm(data, desc="自我学习进度") as pbar:
                        for user_input, ai_reply in pbar:
                            x = torch.tensor([ai.encoder.encode(user_input)], dtype=torch.long)
                            y = torch.tensor([ai.encoder.encode(ai_reply)], dtype=torch.long)
                            out = ai.model(x)
                            loss = ai.criterion(out.view(-1, ai.encoder.vocab_size), y.view(-1))
                            ai.optimizer.zero_grad()
                            loss.backward()
                            ai.optimizer.step()
                            total_loss += loss.item()
                            pbar.set_postfix({"损失": f"{loss.item():.4f}"})
                    avg_loss = total_loss / len(data)
                    log_event(f"联网学习本轮平均损失: {avg_loss:.4f}")
                    print(f"[自我学习] 当前平均损失: {avg_loss:.4f}")
                    ai.save_model()
                    if avg_loss <= 1e-4:
                        log_event("联网学习损失足够小，训练停止")
                        print("[自我学习] 损失已足够小，训练停止！")
                        break
            else:
                print("[联网学习] 未获取到新数据。\n可尝试：\n- 检查网络连接\n- 更换关键词\n- 调试解析逻辑\n- 打印 resp.text 查看页面结构")
        elif key == 'M':
            if last_user_input is None or last_ai_reply is None:
                log_event("手动纠错-无可纠正AI输出")
                print("[手动纠错] 暂无可纠正的AI输出，请先进行一次AI回复或人工标注。")
            else:
                log_event(f"手动纠错-上一次AI输出: {last_ai_reply}")
                print(f"上一次AI输出: {last_ai_reply}")
                corrected_reply = input("请输入修正后的答案: ")
                log_event(f"手动纠错-修正后答案: {corrected_reply}")
                ai.save_memory(last_user_input, corrected_reply)
                log_event("手动纠错-已保存修正答案并开始自我学习")
                print("已保存手动纠正后的答案到记忆。开始自我学习直到损失为0.0000...")
                last_ai_reply = corrected_reply
                # 自我学习直到损失为0.0000
                while True:
                    data = ai.data_manager.load_data()
                    if not data:
                        log_event("手动纠错-自我学习无可用对话数据")
                        print("[自我学习] 没有可用对话数据。"); break
                    total_loss = 0
                    with tqdm(data, desc="自我学习进度") as pbar:
                        for user_input, ai_reply in pbar:
                            x = torch.tensor([ai.encoder.encode(user_input)], dtype=torch.long)
                            y = torch.tensor([ai.encoder.encode(ai_reply)], dtype=torch.long)
                            out = ai.model(x)
                            loss = ai.criterion(out.view(-1, ai.encoder.vocab_size), y.view(-1))
                            ai.optimizer.zero_grad()
                            loss.backward()
                            ai.optimizer.step()
                            total_loss += loss.item()
                            pbar.set_postfix({"损失": f"{loss.item():.4f}"})
                    avg_loss = total_loss / len(data)
                    log_event(f"手动纠错-自我学习本轮平均损失: {avg_loss:.4f}")
                    print(f"[自我学习] 当前平均损失: {avg_loss:.4f}")
                    ai.save_model()
                    if avg_loss <= 1e-4:
                        log_event("手动纠错-自我学习损失足够小，训练停止")
                        print("[自我学习] 损失已足够小，训练停止！")
                        break
        elif key == 'Q':
            log_event("程序退出")
            print("已退出。"); break
        else:
            log_event(f"无效指令: {key}")
            print("无效指令，请输入L/A/K/M/I/Q。")


        