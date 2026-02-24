import os
import sys
import logging
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFKC
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoAI-ChineseTokenizer")
def train_chinese_tokenizer(train_data_path, output_tokenizer_path, vocab_size=50000):
    if not os.path.exists(train_data_path):
        logger.error(f"训练数据不存在: {train_data_path}")
        raise FileNotFoundError(f"训练数据不存在: {train_data_path}")
    tokenizer = Tokenizer(models.BPE())
    logger.info(f"创建BPE分词器，目标词汇表大小: {vocab_size}")
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.UnicodeScripts()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.ByteLevel()
    ])
    tokenizer.decoder = decoders.ByteLevel()
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<SOS> $A <EOS>",
        pair="<SOS> $A <EOS> $B <EOS>",
        special_tokens=[
            ("<SOS>", 2),
            ("<EOS>", 3),
        ],
    )
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=1,
        continuing_subword_prefix="",
    )
    logger.info(f"开始训练分词器，使用训练数据: {train_data_path}")
    tokenizer.train([train_data_path], trainer)
    logger.info("分词器训练完成")
    os.makedirs(os.path.dirname(output_tokenizer_path), exist_ok=True)
    tokenizer.save(output_tokenizer_path)
    logger.info(f"分词器已保存到: {output_tokenizer_path}")
    try:
        loaded_tokenizer = Tokenizer.from_file(output_tokenizer_path)
        vocab_dict = loaded_tokenizer.get_vocab()
        logger.info(f"验证: 加载的分词器词汇表大小: {len(vocab_dict)}")
        test_words = ["你好", "世界", "中国", "人工智能"]
        for word in test_words:
            encoding = loaded_tokenizer.encode(word)
            logger.info(f"文本 '{word}' 编码结果: {encoding.tokens}")
            logger.info(f"对应的token IDs: {encoding.ids}")
            decoded_text = loaded_tokenizer.decode(encoding.ids)
            logger.info(f"解码后的文本: '{decoded_text}'")
            if "<UNK>" in encoding.tokens:
                logger.warning(f"文本 '{word}' 包含<UNK>标记")
            else:
                logger.info(f"文本 '{word}' 没有包含<UNK>标记")
    except Exception as e:
        logger.error(f"验证分词器失败: {str(e)}")
if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    train_data_path = os.path.join(project_root, 'data', 'train.txt')
    output_tokenizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modern_tokenizer.json')
    train_chinese_tokenizer(train_data_path, output_tokenizer_path, vocab_size=50000)
    logger.info("中文分词器训练完成！")