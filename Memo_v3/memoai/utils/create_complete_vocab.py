#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建完整词汇表脚本
从训练数据中提取所有有效字符，彻底解决<UNK>标记问题
作者: Pyro - 让模型认识世界上所有的字！
"""
import os
import json
import logging
from collections import Counter

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoAI-CreateVocab")

def create_complete_vocab(train_data_path, output_vocab_path):
    """从训练数据创建完整词汇表
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

    # 定义特殊标记
    special_tokens = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    logger.info(f"添加特殊标记: {list(special_tokens.keys())}")

    # 过滤掉特殊标记和乱码，只保留有效的Unicode字符
    filtered_data = ''.join([ch for ch in train_data if ch not in special_tokens and 0x20 <= ord(ch) <= 0xFFFF and not ch.isspace()])

    # 统计字符频率
    char_counter = Counter(filtered_data)
    logger.info(f"发现 {len(char_counter)} 个不同字符")

    # 创建字符到索引的映射
    char_to_idx = special_tokens.copy()
    next_idx = len(special_tokens)

    for char in sorted(char_counter.keys()):
        char_to_idx[char] = next_idx
        next_idx += 1

    # 创建索引到字符的映射
    idx_to_char = {str(idx): char for char, idx in char_to_idx.items()}

    # 保存词汇表
    vocab = {
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char
    }

    with open(output_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    logger.info(f"词汇表已保存到: {output_vocab_path}")
    logger.info(f"词汇表大小: {len(char_to_idx)}")

if __name__ == '__main__':
    # 定义文件路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    train_data_path = os.path.join(project_root, 'data', 'train.txt')
    output_vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab.json')

    # 创建完整词汇表
    create_complete_vocab(train_data_path, output_vocab_path)
    logger.info("完整词汇表创建完成！模型现在应该能认识训练数据中的所有字符了！")