# -*- coding: utf-8 -*-
"""
扩展MemoAI词汇表脚本
从训练数据中提取更多中文字符，解决生成文本中的<UNK>标记问题
"""
import os
import sys
import json
import logging
from collections import Counter

# 将项目根目录添加到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 现在可以正确导入memoai模块
from memoai.utils.vocab import Vocabulary

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoAI-ExtendVocab")

def extend_vocab_from_data(train_data_path, output_vocab_path):
    """从训练数据扩展词汇表
    Args:
        train_data_path: 训练数据文件路径
        output_vocab_path: 输出词汇表文件路径
    """
    # 读取训练数据
    try:
        with open(train_data_path, 'r', encoding='utf-8') as f:
            train_data = f.read()
        logger.info(f"成功读取训练数据，长度: {len(train_data)} 字符")
    except Exception as e:
        logger.error(f"读取训练数据失败: {e}")
        raise

    # 创建词汇表实例
    vocab = Vocabulary()
    logger.info(f"初始词汇表大小: {vocab.vocab_size}")

    # 过滤掉特殊标记和乱码
    special_tokens = {'<PAD>', '<UNK>', '<SOS>', '<EOS>'}
    filtered_data = ''.join([ch for ch in train_data if ch not in special_tokens and ord(ch) < 0xFFFF])

    # 统计字符频率
    char_counter = Counter(filtered_data)
    logger.info(f"发现 {len(char_counter)} 个不同字符")

    # 添加高频字符到词汇表
    # 降低频率阈值以捕获更多字符，1表示只要出现过就添加
    min_freq = 1
    new_chars = 0
    for char, freq in char_counter.items():
        if freq >= min_freq and char not in vocab.char_to_idx:
            # 过滤掉明显的乱码字符（只保留有效的Unicode字符）
            if 0x20 <= ord(char) <= 0xFFFF and not char.isspace():
                vocab.add_char(char)
                new_chars += 1

    logger.info(f"添加了 {new_chars} 个新字符到词汇表")
    logger.info(f"扩展后的词汇表大小: {vocab.vocab_size}")

    # 保存扩展后的词汇表
    vocab.save_vocab(output_vocab_path)
    logger.info(f"词汇表已保存到: {output_vocab_path}")

if __name__ == '__main__':
    # 定义文件路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    train_data_path = os.path.join(project_root, 'data', 'train.txt')
    output_vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab.json')

    # 扩展词汇表
    extend_vocab_from_data(train_data_path, output_vocab_path)
    logger.info("词汇表扩展完成！")