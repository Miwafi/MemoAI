import os
import subprocess
import sys

# 强制使用CPU的环境变量设置
os.environ['USE_GPU'] = 'False'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

print("已设置环境变量强制使用CPU")

# 运行测试脚本
subprocess.run([sys.executable, 'test_inferencer.py'], check=True)