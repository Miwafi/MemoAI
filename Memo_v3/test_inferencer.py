import os
import torch

# 强制使用CPU的环境变量设置
os.environ['USE_GPU'] = 'False'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# 设置PyTorch默认设备为CPU
torch.set_default_device('cpu')
print("已强制设置为使用CPU")

# 忽略弃用警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

torch.set_default_dtype(torch.float32)
print("已设置环境变量和PyTorch配置强制使用CPU")

# 模型路径 - 使用绝对路径确保正确性
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'test_model_tiny.pth')

# 确保models目录存在
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# 从memoai.inference.infer导入MemoAIInferencer
from memoai.inference.infer import MemoAIInferencer

def test_inferencer():
    print("开始测试MemoAIInferencer...")
    try:
        # 初始化推理器
        inferencer = MemoAIInferencer()
        print("推理器初始化成功")

        # 测试推理
        input_text = "这是一个测试句子"
        result = inferencer.generate_text(input_text)
        print(f"推理结果: {result}")
        return True
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_inferencer()