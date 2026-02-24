import os
import sys
import torch
import struct
import logging
from memoai.core.model import MemoAI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelConverter")
def convert_pth_to_gguf(pth_model_path, gguf_model_path):
    try:
        model = MemoAI()
        model.load_state_dict(torch.load(pth_model_path, map_location=torch.device('cpu')))
        model.eval()
        logger.info(f"成功加载PyTorch模型: {pth_model_path}")
        with open(gguf_model_path, 'wb') as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 1))
            for name, param in model.named_parameters():
                param_np = param.cpu().detach().numpy()
                name_bytes = name.encode('utf-8')
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<I", len(param_np.shape)))
                for dim in param_np.shape:
                    f.write(struct.pack("<I", dim))
                dtype_code = 0
                f.write(struct.pack("<I", dtype_code))
                param_np.tofile(f)
        logger.info(f"模型已成功转换为GGUF格式: {gguf_model_path}")
        return True
    except Exception as e:
        logger.error(f"转换模型为GGUF格式时出错: {str(e)}")
        return False
if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.info("用法: python convert_to_gguf.py <pth_model_path> <gguf_model_path>")
        logger.info("示例: python convert_to_gguf.py memoai/models/Memo-1_final.pth memoai/models/Memo-1_final.gguf")
        sys.exit(1)
    pth_path = sys.argv[1]
    gguf_path = sys.argv[2]
    if not os.path.exists(pth_path):
        logger.error(f"找不到PyTorch模型文件: {pth_path}")
        sys.exit(1)
    success = convert_pth_to_gguf(pth_path, gguf_path)
    if success:
        logger.info("转换完成！")
    else:
        logger.info("转换失败，请查看错误信息。")