@echo off
REM 强制使用CPU的环境变量设置
set USE_GPU=False
set CUDA_VISIBLE_DEVICES=
set PYTORCH_ENABLE_MPS_FALLBACK=1

REM 运行测试脚本
python test_inferencer.py