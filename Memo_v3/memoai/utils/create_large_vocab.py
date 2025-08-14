#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建大型词汇表脚本
包含更多常用汉字，解决文本生成中的<UNK>标记问题

"""
import os
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoAI-LargeVocab")

# 常用汉字列表 - 前1000个最常用的汉字
COMMON_CHARS = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"

# 扩展汉字列表 - 额外添加一些常用字
EXTENDED_CHARS = "你好谢谢对不起请再见是的不是我是他是她是它是我们你们他们她们它们这那这些那些这里那里现在过去将来今天明天昨天上午下午晚上早晨中午年月份日期小时分钟秒天气冷暖热风雨雪阴晴春夏秋冬东南西北上下左右前后里外高低大小多少长短粗细胖瘦美丑好坏真假对错是非黑白红蓝绿黄紫"


def create_large_vocab(output_vocab_path):
    """创建大型词汇表
    Args:
        output_vocab_path: 输出词汇表文件路径
    """
    # 定义特殊标记
    special_tokens = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    logger.info(f"添加特殊标记: {list(special_tokens.keys())}")

    # 合并所有字符并去重
    all_chars = set(COMMON_CHARS + EXTENDED_CHARS)
    logger.info(f"合并并去重后字符数量: {len(all_chars)}")

    # 创建字符到索引的映射
    char_to_idx = special_tokens.copy()
    next_idx = len(special_tokens)

    for char in sorted(all_chars):
        char_to_idx[char] = next_idx
        next_idx += 1

    # 创建索引到字符的映射
    idx_to_char = {str(idx): char for char, idx in char_to_idx.items()}

    # 保存词汇表
    vocab_size = len(char_to_idx)
    vocab = {
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size
    }

    with open(output_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    logger.info(f"大型词汇表已保存到: {output_vocab_path}")
    logger.info(f"词汇表大小: {len(char_to_idx)}")

if __name__ == '__main__':
    # 定义文件路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab.json')

    # 创建大型词汇表
    create_large_vocab(output_vocab_path)
    logger.info("大型词汇表创建完成！")