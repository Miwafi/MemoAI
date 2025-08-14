#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
针对中文文本的分词器训练脚本
使用更适合中文的配置
"""
import os
import sys
import logging
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFKC

# 将项目根目录添加到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoAI-ChineseTokenizer")

def train_chinese_tokenizer(train_data_path, output_tokenizer_path, vocab_size=50000):
    """训练适合中文的BPE分词器
    Args:
        train_data_path: 训练数据文件路径
        output_tokenizer_path: 输出分词器文件路径
        vocab_size: 词汇表大小
    """
    # 检查训练数据是否存在
    if not os.path.exists(train_data_path):
        logger.error(f"训练数据不存在: {train_data_path}")
        raise FileNotFoundError(f"训练数据不存在: {train_data_path}")

    # 创建分词器实例
    tokenizer = Tokenizer(models.BPE())
    logger.info(f"创建BPE分词器，目标词汇表大小: {vocab_size}")

    # 设置normalizer (标准化文本)
    tokenizer.normalizer = NFKC()

    # 设置pre-tokenizer (将文本分割成基本单元)
    # 对于中文，我们使用UnicodeScripts预分词器
    tokenizer.pre_tokenizer = pre_tokenizers.UnicodeScripts()
    # 添加一个字节级预分词器作为备选
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.ByteLevel()
    ])

    # 设置decoder (将token IDs转换回文本)
    tokenizer.decoder = decoders.ByteLevel()

    # 特殊标记
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']

    # 设置post-processor (添加特殊标记)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<SOS> $A <EOS>",
        pair="<SOS> $A <EOS> $B <EOS>",
        special_tokens=[
            ("<SOS>", 2),
            ("<EOS>", 3),
        ],
    )

    # 准备trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=1,  # 降低最小频率要求，确保更多中文字符被包含
        continuing_subword_prefix="",  # 中文不需要前缀
    )

    # 训练分词器
    logger.info(f"开始训练分词器，使用训练数据: {train_data_path}")
    tokenizer.train([train_data_path], trainer)
    logger.info("分词器训练完成")

    # 保存分词器
    os.makedirs(os.path.dirname(output_tokenizer_path), exist_ok=True)
    tokenizer.save(output_tokenizer_path)
    logger.info(f"分词器已保存到: {output_tokenizer_path}")

    # 加载保存的分词器，验证词汇表
    try:
        loaded_tokenizer = Tokenizer.from_file(output_tokenizer_path)
        vocab_dict = loaded_tokenizer.get_vocab()
        logger.info(f"验证: 加载的分词器词汇表大小: {len(vocab_dict)}")

        # 检查一些常用中文词汇是否在词汇表中
        test_words = ["你好", "世界", "中国", "人工智能"]
        for word in test_words:
            # 编码文本
            encoding = loaded_tokenizer.encode(word)
            logger.info(f"文本 '{word}' 编码结果: {encoding.tokens}")
            logger.info(f"对应的token IDs: {encoding.ids}")

            # 解码回文本
            decoded_text = loaded_tokenizer.decode(encoding.ids)
            logger.info(f"解码后的文本: '{decoded_text}'")

            # 检查是否包含<UNK>标记
            if "<UNK>" in encoding.tokens:
                logger.warning(f"文本 '{word}' 包含<UNK>标记")
            else:
                logger.info(f"文本 '{word}' 没有包含<UNK>标记")
    except Exception as e:
        logger.error(f"验证分词器失败: {str(e)}")

if __name__ == '__main__':
    # 定义文件路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    train_data_path = os.path.join(project_root, 'data', 'train.txt')
    output_tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modern_tokenizer.json')

    # 训练中文分词器
    train_chinese_tokenizer(train_data_path, output_tokenizer_path, vocab_size=50000)
    logger.info("中文分词器训练完成！")