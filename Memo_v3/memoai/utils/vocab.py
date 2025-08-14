#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoAI 词汇表工具
这里定义了模型使用的词汇表

"""
import os
import json
import logging
from collections import Counter

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoAI-Vocab")


class Vocabulary:
    """词汇表类 - 负责词与索引的映射"""
    def __init__(self, vocab_path=None):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

        # 预留特殊符号
        self.add_special_tokens()

        # 如果提供了词汇表路径，则加载词汇表
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        else:
            # 否则使用默认词汇表
            logger.warning("Vocabulary file not found. Using default vocabulary.")
            self._create_default_vocab()

    def add_special_tokens(self):
        """添加特殊符号"""
        special_tokens = {
            '<PAD>': 0,   # 填充符号
            '<UNK>': 1,   # 未知符号
            '<SOS>': 2,   # 句子开始符号
            '<EOS>': 3    # 句子结束符号
        }

        for token, idx in special_tokens.items():
            self.char_to_idx[token] = idx
            self.idx_to_char[idx] = token

        self.vocab_size = len(special_tokens)

    def _create_default_vocab(self):
        """创建默认词汇表
        包含常见的中文、英文、数字和符号"""
        # 英文大小写字母
        for ch in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            self.add_char(ch)

        # 数字
        for ch in '0123456789':
            self.add_char(ch)

        # 常见中文标点符号
        for ch in '，。！？；：‘’“”（）【】《》、':
            self.add_char(ch)

        # 常见英文标点符号
        for ch in ',.!?;:\'"\()[]{}<>/|\+-*=%^&@#$¥€£':
            self.add_char(ch)

        # 常见中文字符（基本汉字）
        common_chinese = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
        for ch in common_chinese:
            self.add_char(ch)

        logger.info(f"Default vocabulary created with size: {self.vocab_size}")

    def add_char(self, char):
        """添加字符到词汇表"""
        if char not in self.char_to_idx:
            self.char_to_idx[char] = self.vocab_size
            self.idx_to_char[self.vocab_size] = char
            self.vocab_size += 1

    def load_vocab(self, vocab_path):
        """从文件加载词汇表"""
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
                self.char_to_idx = vocab_data['char_to_idx']
                self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
                self.vocab_size = vocab_data['vocab_size']
                logger.info(f"Loaded vocabulary from {vocab_path} with size: {self.vocab_size}")
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            raise

    def save_vocab(self, vocab_path):
        """保存词汇表到文件"""
        try:
            vocab_data = {
                'char_to_idx': self.char_to_idx,
                'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()},
                'vocab_size': self.vocab_size
            }
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved vocabulary to {vocab_path}")
        except Exception as e:
            logger.error(f"Error saving vocabulary: {e}")
            raise

    def text_to_ids(self, text):
        """将文本转换为索引序列"""
        return [self.char_to_idx.get(ch, self.char_to_idx['<UNK>']) for ch in text]

    def ids_to_text(self, ids):
        """将索引序列转换为文本"""
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in ids])


if __name__ == '__main__':
    # 创建词汇表实例
    vocab = Vocabulary()
    
    # 保存词汇表到文件
    vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab.json')
    vocab.save_vocab(vocab_path)
    
    logger.info(f"词汇表已保存到: {vocab_path}")
    logger.info(f"词汇表大小: {vocab.vocab_size}")