import os
import json
import logging
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFKC
from typing import List, Dict, Union
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoAI-ModernVocab")
class ModernTokenizer:
    def __init__(self, tokenizer_path: str = None, vocab_size: int = 50000):
        self._vocab_size = vocab_size
        self.tokenizer = None
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.load_tokenizer(tokenizer_path)
            if self.tokenizer:
                self._vocab_size = len(self.tokenizer.get_vocab())
        else:
            logger.info("Tokenizer not found, initializing a new BPE tokenizer.")
            self._create_bpe_tokenizer()
    def _create_bpe_tokenizer(self):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.normalizer = NFKC()
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="<SOS> $A <EOS>",
            pair="<SOS> $A <EOS> $B <EOS>",
            special_tokens=[
                ("<SOS>", self.special_tokens['<SOS>']),
                ("<EOS>", self.special_tokens['<EOS>']),
            ],
        )
    def train(self, files: List[str]):
        if not self.tokenizer:
            self._create_bpe_tokenizer()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=list(self.special_tokens.keys()),
            min_frequency=2,
            continuing_subword_prefix="▁",
        )
        logger.info(f"开始训练分词器，词汇表大小: {self.vocab_size}")
        self.tokenizer.train(files, trainer)
        logger.info("分词器训练完成！")
    def save_tokenizer(self, save_path: str):
        if not self.tokenizer:
            logger.error("No tokenizer to save!")
            return
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.tokenizer.save(save_path)
        logger.info(f"分词器已保存到: {save_path}")
    def load_tokenizer(self, tokenizer_path: str):
        try:
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            logger.info(f"成功加载分词器: {tokenizer_path}")
        except Exception as e:
            logger.error(f"加载分词器失败: {e}")
            raise
    def encode(self, text: str) -> List[int]:
        if not self.tokenizer:
            logger.error("Tokenizer not initialized!")
            return []
        encoded = self.tokenizer.encode(text)
        return encoded.ids
    def decode(self, ids: List[int]) -> str:
        if not self.tokenizer:
            logger.error("Tokenizer not initialized!")
            return ""
        return self.tokenizer.decode(ids)
    def token_to_id(self, token: str) -> Union[int, None]:
        if not self.tokenizer:
            logger.error("Tokenizer not initialized!")
            return None
        return self.tokenizer.get_vocab().get(token)
    def id_to_token(self, id: int) -> Union[str, None]:
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
        return self.special_tokens['<PAD>']
    @property
    def unk_token_id(self) -> int:
        return self.special_tokens['<UNK>']
    @property
    def sos_token_id(self) -> int:
        return self.special_tokens['<SOS>']
    @property
    def eos_token_id(self) -> int:
        return self.special_tokens['<EOS>']
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    @vocab_size.setter
    def vocab_size(self, size: int):
        self._vocab_size = size
if __name__ == '__main__':
    tokenizer = ModernTokenizer(vocab_size=30000)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    train_data_path = os.path.join(project_root, 'memoai', 'data', 'train.txt')
    if os.path.exists(train_data_path):
        tokenizer.train([train_data_path])
        tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modern_tokenizer.json')
        tokenizer.save_tokenizer(tokenizer_path)
        logger.info(f"分词器已训练并保存到: {tokenizer_path}")
    else:
        logger.warning(f"训练数据不存在: {train_data_path}")
        logger.info("请准备训练数据后再训练分词器")