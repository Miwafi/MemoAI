# -*- coding: utf-8 -*-
"""
MemoAI API服务器
这个服务器负责处理前端请求，连接到MemoAI后端服务
"""
import os
import sys
import logging
from flask import Flask, request, jsonify, send_from_directory
import torch
import threading

# 设置项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入MemoAI相关模块
from memoai.config.config import ModelConfig, InferenceConfig
from memoai.core.model import MemoAI
from memoai.utils.vocab import Vocabulary

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoAI-API")

# 初始化Flask应用
app = Flask(__name__, static_folder='../electron-app')

# 全局变量用于存储模型和词汇表
model = None
vocab = None
model_lock = threading.Lock()

# 初始化配置
model_config = ModelConfig()
infer_config = InferenceConfig()

def load_model():
    """加载模型和词汇表

    """
    global model, vocab
    with model_lock:
        if model is None and vocab is None:
            logger.info("正在加载MemoAI模型和词汇表...")
            try:
                # 初始化词汇表
                vocab = Vocabulary()
                logger.info(f"词汇表加载完成，大小: {vocab.vocab_size}")
                
                # 更新模型配置中的词汇表大小
                model_config.vocab_size = vocab.vocab_size
                
                # 初始化模型
                model = MemoAI(model_config).to(torch.device('cpu'))
                logger.info("模型架构初始化完成")
                
                # 加载模型权重
                if os.path.exists(infer_config.model_path):
                    checkpoint = torch.load(infer_config.model_path, map_location=torch.device('cpu'))
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"成功加载模型权重: {infer_config.model_path}")
                else:
                    logger.warning(f"未找到模型权重文件: {infer_config.model_path}，使用未训练模型")
                
                # 设置为评估模式
                model.eval()
                logger.info("模型加载完成，已设置为评估模式")
            except Exception as e:
                logger.error(f"加载模型权重出错: {str(e)}")
                raise

def generate_response(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
    """生成AI响应
    根据用户的提示，生成智能回复

    """
    global model, vocab
    
    # 确保模型已加载
    if model is None or vocab is None:
        load_model()
    
    try:
        with torch.no_grad():
            # 编码提示文本
            input_ids = vocab.text_to_ids(prompt)
            if not input_ids:
                input_ids = vocab.text_to_ids('你')  # 使用'你'作为默认起始字符
            
            input_tensor = torch.tensor(input_ids).unsqueeze(0)
            generated_text = prompt
            
            for _ in range(max_length):
                # 获取模型输出
                output = model(input_tensor)
                # 取最后一个token的预测
                next_token_logits = output[0, -1, :]
                
                # 应用温度 - 控制输出的多样性
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # 应用top-k - 只从概率最高的k个词中选择
                if top_k > 0:
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits[indices] = values
                
                # 应用top-p - 累积概率超过p的词被考虑
                if top_p > 0 and top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # 采样 - 从概率分布中选择下一个词
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1).item()
                
                # 转换为字符
                next_char = vocab.idx_to_char.get(next_token_idx, '<UNK>')
                
                # 添加到生成文本
                generated_text += next_char
                
                # 更新输入
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_idx]])], dim=1)
                
                # 保持序列长度
                if input_tensor.size(1) > model.config.max_seq_len:
                    input_tensor = input_tensor[:, 1:]
                
                # 如果生成了结束符，停止生成
                if next_char == '<EOS>':
                    break
            
            return generated_text
    except Exception as e:
        logger.error(f"生成响应时出错: {str(e)}")
        return f"生成响应时出错: {str(e)}"

@app.route('/api/chat', methods=['POST'])
def chat():
    """处理聊天请求
    接收前端发送的消息，返回AI生成的响应
    """
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.7)
    top_k = data.get('top_k', 50)
    top_p = data.get('top_p', 0.95)
    
    logger.info(f"收到聊天请求: {prompt[:20]}...")
    response = generate_response(prompt, max_length, temperature, top_k, top_p)
    logger.info(f"生成响应: {response[:20]}...")
    
    return jsonify({
        'response': response
    })

# 静态文件路由
@app.route('/')
def serve_index():
    """提供index.html文件

    """
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """提供静态文件
    为用户提供CSS、JavaScript等静态资源
    """
    return send_from_directory(app.static_folder, path)

# 启动服务器
if __name__ == '__main__':
    # 预加载模型
    load_model()
    
    # 启动Flask服务器
    logger.info("启动MemoAI API服务器...")
    app.run(host='0.0.0.0', port=5000, debug=True)