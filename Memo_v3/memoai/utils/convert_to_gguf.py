# -*- coding: utf-8 -*-
"""
模型格式转换工具 - 把PyTorch模型变成GGUF格式
"""
import os
import sys
import torch
import struct
import logging
from memoai.core.model import MemoAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelConverter")

def convert_pth_to_gguf(pth_model_path, gguf_model_path):
    """
    将PyTorch模型(.pth)转换为GGUF格式
    
    参数:
    pth_model_path: str - PyTorch模型文件路径
    gguf_model_path: str - 输出的GGUF模型文件路径
    """
    try:
        # 创建模型实例
        model = MemoAI()
        
        # 加载PyTorch模型权重 - 给容器填充内容
        model.load_state_dict(torch.load(pth_model_path, map_location=torch.device('cpu')))
        model.eval()  # 设置为评估模式，避免训练相关的操作
        
        logger.info(f"成功加载PyTorch模型: {pth_model_path}")
        
        # 创建GGUF文件
        with open(gguf_model_path, 'wb') as f:
            # 写入GGUF头部 - 这是文件的"身份证"
            f.write(b"GGUF")
            f.write(struct.pack("<I", 1))  # 版本号
            
            # 写入模型参数 - 每个参数都是模型的"记忆碎片"
            for name, param in model.named_parameters():
                # 转换参数为numpy数组
                param_np = param.cpu().detach().numpy()
                
                # 写入参数名称和形状信息
                name_bytes = name.encode('utf-8')
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                
                # 写入参数形状
                f.write(struct.pack("<I", len(param_np.shape)))
                for dim in param_np.shape:
                    f.write(struct.pack("<I", dim))
                
                # 写入参数数据类型 (0表示float32)
                dtype_code = 0
                f.write(struct.pack("<I", dtype_code))
                
                # 写入参数数据
                param_np.tofile(f)
        
        logger.info(f"模型已成功转换为GGUF格式: {gguf_model_path}")
        return True
    except Exception as e:
        logger.error(f"转换模型为GGUF格式时出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) != 3:
        logger.info("用法: python convert_to_gguf.py <pth_model_path> <gguf_model_path>")
        logger.info("示例: python convert_to_gguf.py memoai/models/Memo-1_final.pth memoai/models/Memo-1_final.gguf")
        sys.exit(1)
    
    pth_path = sys.argv[1]
    gguf_path = sys.argv[2]
    
    # 检查模型文件是否存在
    if not os.path.exists(pth_path):
        logger.error(f"找不到PyTorch模型文件: {pth_path}")
        sys.exit(1)
    
    # 执行转换
    success = convert_pth_to_gguf(pth_path, gguf_path)
    if success:
        logger.info("转换完成！")
    else:
        logger.info("转换失败，请查看错误信息。")