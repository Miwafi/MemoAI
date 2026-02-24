import os
import json
import logging
from collections import Counter
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoAI-Vocab")
class Vocabulary:
    def __init__(self, vocab_path=None):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.add_special_tokens()
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        else:
            logger.warning("Vocabulary file not found. Using default vocabulary.")
            self._create_default_vocab()
    def add_special_tokens(self):
        special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }
        for token, idx in special_tokens.items():
            self.char_to_idx[token] = idx
            self.idx_to_char[idx] = token
        self.vocab_size = len(special_tokens)
    def _create_default_vocab(self):
        for ch in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            self.add_char(ch)
        for ch in '0123456789':
            self.add_char(ch)
        for ch in '，。！？；：‘’“”（）【】《》、':
            self.add_char(ch)
        for ch in ',.!?;:\'"\()[]{}<>/|\+-*=%^&@#$¥€£':
            self.add_char(ch)
        common_chinese = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
        for ch in common_chinese:
            self.add_char(ch)
        logger.info(f"Default vocabulary created with size: {self.vocab_size}")
    def add_char(self, char):
        if char not in self.char_to_idx:
            self.char_to_idx[char] = self.vocab_size
            self.idx_to_char[self.vocab_size] = char
            self.vocab_size += 1
    def load_vocab(self, vocab_path):
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
        return [self.char_to_idx.get(ch, self.char_to_idx['<UNK>']) for ch in text]
    def ids_to_text(self, ids):
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in ids])
if __name__ == '__main__':
    vocab = Vocabulary()
    vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab.json')
    vocab.save_vocab(vocab_path)
    logger.info(f"词汇表已保存到: {vocab_path}")
    logger.info(f"词汇表大小: {vocab.vocab_size}")