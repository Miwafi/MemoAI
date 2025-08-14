# -*- coding: utf-8 -*-
"""
MemoAI 现代化分词器工具
这里实现了基于Hugging Face tokenizers的现代化分词器
就像给模型配上了更智能的文字解析器，告别原始的字符切割时代
"""
import os
import json
import logging
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFKC
from typing import List, Dict, Union

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoAI-ModernVocab")

class ModernTokenizer:
    """现代化分词器类 - 基于Hugging Face tokenizers
    支持BPE、WordPiece、Unigram等多种分词算法"""
    def __init__(self, tokenizer_path: str = None, vocab_size: int = 50000):
        self._vocab_size = vocab_size
        self.tokenizer = None
        self.special_tokens = {
            '<PAD>': 0,   # 填充符号
            '<UNK>': 1,   # 未知符号
            '<SOS>': 2,   # 句子开始符号
            '<EOS>': 3    # 句子结束符号
        }

        if tokenizer_path and os.path.exists(tokenizer_path):
            self.load_tokenizer(tokenizer_path)
            # 加载后更新词汇表大小
            if self.tokenizer:
                self._vocab_size = len(self.tokenizer.get_vocab())
        else:
            logger.info("Tokenizer not found, initializing a new BPE tokenizer.")
            self._create_bpe_tokenizer()

    def _create_bpe_tokenizer(self):
        """创建BPE分词器
        BPE (Byte Pair Encoding) 是一种常用的子词分词算法，就像把词语拆分成乐高积木一样灵活"""
        # 初始化一个空的BPE模型
        self.tokenizer = Tokenizer(models.BPE())

        # 设置normalizer (标准化文本)
        self.tokenizer.normalizer = NFKC()

        # 设置pre-tokenizer (将文本分割成基本单元)
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        # 设置decoder (将token IDs转换回文本)
        self.tokenizer.decoder = decoders.ByteLevel()

        # 设置post-processor (添加特殊标记)
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="<SOS> $A <EOS>",
            pair="<SOS> $A <EOS> $B <EOS>",
            special_tokens=[
                ("<SOS>", self.special_tokens['<SOS>']),
                ("<EOS>", self.special_tokens['<EOS>']),
            ],
        )

    def train(self, files: List[str]):
        """从文件训练分词器
        就像给分词器喂饱数据，让它学会如何正确切割文本"""
        if not self.tokenizer:
            self._create_bpe_tokenizer()

        # 准备trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=list(self.special_tokens.keys()),
            min_frequency=2,
            continuing_subword_prefix="▁",  # ByteLevel预处理器使用的前缀
        )

        # 训练分词器
        logger.info(f"开始训练分词器，词汇表大小: {self.vocab_size}")
        self.tokenizer.train(files, trainer)
        logger.info("分词器训练完成！")

    def save_tokenizer(self, save_path: str):
        """保存分词器到文件
        就像把学会的知识存入大脑，下次可以直接调用"""
        if not self.tokenizer:
            logger.error("No tokenizer to save!")
            return

        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存分词器
        self.tokenizer.save(save_path)
        logger.info(f"分词器已保存到: {save_path}")

    def load_tokenizer(self, tokenizer_path: str):
        """从文件加载分词器
        就像唤醒沉睡的记忆，让分词器恢复之前学到的知识"""
        try:
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            logger.info(f"成功加载分词器: {tokenizer_path}")
        except Exception as e:
            logger.error(f"加载分词器失败: {e}")
            raise

    def encode(self, text: str) -> List[int]:
        """将文本编码为token IDs
        就像把中文翻译成数字密码，让模型能够理解"""
        if not self.tokenizer:
            logger.error("Tokenizer not initialized!")
            return []

        encoded = self.tokenizer.encode(text)
        return encoded.ids

    def decode(self, ids: List[int]) -> str:
        """将token IDs解码为文本
        就像破解数字密码，把模型的输出转换回人类可读的文字"""
        if not self.tokenizer:
            logger.error("Tokenizer not initialized!")
            return ""

        return self.tokenizer.decode(ids)

    def token_to_id(self, token: str) -> Union[int, None]:
        """获取token对应的ID
        就像查字典，找到单词对应的页码"""
        if not self.tokenizer:
            logger.error("Tokenizer not initialized!")
            return None

        return self.tokenizer.get_vocab().get(token)

    def id_to_token(self, id: int) -> Union[str, None]:
        """获取ID对应的token
        就像根据页码查字典，找到对应的单词"""
        if not self.tokenizer:
            logger.error("Tokenizer not initialized!")
            return None

        vocab = self.tokenizer.get_vocab()
        for token, token_id in vocab.items():
            if token_id == id:
                return token
        return None

    @property
    def pad_token_id(self) -> int:
        """获取填充符号的ID"""
        return self.special_tokens['<PAD>']

    @property
    def unk_token_id(self) -> int:
        """获取未知符号的ID"""
        return self.special_tokens['<UNK>']

    @property
    def sos_token_id(self) -> int:
        """获取句子开始符号的ID"""
        return self.special_tokens['<SOS>']

    @property
    def eos_token_id(self) -> int:
        """获取句子结束符号的ID"""
        return self.special_tokens['<EOS>']

    @property
    def vocab_size(self) -> int:
        """获取词汇表大小"""
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, size: int):
        """设置词汇表大小"""
        self._vocab_size = size

if __name__ == '__main__':
    # 示例用法
    # 创建分词器实例
    tokenizer = ModernTokenizer(vocab_size=30000)

    # 准备训练数据路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    train_data_path = os.path.join(project_root, 'memoai', 'data', 'train.txt')

    # 检查训练数据是否存在
    if os.path.exists(train_data_path):
        # 训练分词器
        tokenizer.train([train_data_path])

        # 保存分词器
        tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modern_tokenizer.json')
        tokenizer.save_tokenizer(tokenizer_path)

        logger.info(f"分词器已训练并保存到: {tokenizer_path}")
    else:
        logger.warning(f"训练数据不存在: {train_data_path}")
        logger.info("请准备训练数据后再训练分词器")