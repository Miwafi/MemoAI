import os
import sys
import json
import logging
from collections import Counter
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from memoai.utils.vocab import Vocabulary
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoAI-ExtendVocab")
def extend_vocab_from_data(train_data_path, output_vocab_path):
    try:
        with open(train_data_path, 'r', encoding='utf-8') as f:
            train_data = f.read()
        logger.info(f"成功读取训练数据，长度: {len(train_data)} 字符")
    except Exception as e:
        logger.error(f"读取训练数据失败: {e}")
        raise
    vocab = Vocabulary()
    logger.info(f"初始词汇表大小: {vocab.vocab_size}")
    special_tokens = {'<PAD>', '<UNK>', '<SOS>', '<EOS>'}
    filtered_data = ''.join([ch for ch in train_data if ch not in special_tokens and ord(ch) < 0xFFFF])
    char_counter = Counter(filtered_data)
    logger.info(f"发现 {len(char_counter)} 个不同字符")
    min_freq = 1
    new_chars = 0
    for char, freq in char_counter.items():
        if freq >= min_freq and char not in vocab.char_to_idx:
            if 0x20 <= ord(char) <= 0xFFFF and not char.isspace():
                vocab.add_char(char)
                new_chars += 1
    logger.info(f"添加了 {new_chars} 个新字符到词汇表")
    logger.info(f"扩展后的词汇表大小: {vocab.vocab_size}")
    vocab.save_vocab(output_vocab_path)
    logger.info(f"词汇表已保存到: {output_vocab_path}")
if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    train_data_path = os.path.join(project_root, 'data', 'train.txt')
    output_vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab.json')
    extend_vocab_from_data(train_data_path, output_vocab_path)
    logger.info("词汇表扩展完成！")