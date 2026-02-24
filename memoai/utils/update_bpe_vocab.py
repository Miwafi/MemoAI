import os
import sys
import json
import logging
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from memoai.utils.modern_vocab import ModernTokenizer
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoAI-UpdateBPEVocab")
def update_bpe_vocab(train_data_path, output_tokenizer_path, vocab_size=50000):
    if not os.path.exists(train_data_path):
        logger.error(f"训练数据不存在: {train_data_path}")
        raise FileNotFoundError(f"训练数据不存在: {train_data_path}")
    tokenizer = ModernTokenizer(vocab_size=vocab_size)
    logger.info(f"创建BPE分词器，目标词汇表大小: {vocab_size}")
    logger.info(f"开始训练分词器，使用训练数据: {train_data_path}")
    tokenizer.train([train_data_path])
    logger.info("分词器训练完成")
    tokenizer.save_tokenizer(output_tokenizer_path)
    logger.info(f"分词器已保存到: {output_tokenizer_path}")
    loaded_tokenizer = ModernTokenizer(output_tokenizer_path)
    vocab_dict = loaded_tokenizer.tokenizer.get_vocab()
    logger.info(f"验证: 加载的分词器词汇表大小: {len(vocab_dict)}")
    test_words = ["你好", "世界", "中国", "人工智能"]
    for word in test_words:
        tokens = loaded_tokenizer.tokenizer.encode(word).tokens
        logger.info(f"词语 '{word}' 被拆分为: {tokens}")
        all_unk = all(token == "<UNK>" for token in tokens)
        if all_unk:
            logger.warning(f"词语 '{word}' 的所有子词都不在词汇表中")
        else:
            logger.info(f"词语 '{word}' 至少有一个子词在词汇表中")
if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    train_data_path = os.path.join(project_root, 'data', 'train.txt')
    output_tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modern_tokenizer.json')
    update_bpe_vocab(train_data_path, output_tokenizer_path, vocab_size=50000)
    logger.info("BPE分词器词汇表更新完成！")