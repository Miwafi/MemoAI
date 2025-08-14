@echo off
REM MemoAI 模型训练批处理文件
REM 这个文件用于运行train_pth_model.py脚本

REM 设置环境变量
set PROJECT_ROOT=%cd%

echo 开始训练MemoAI模型...
echo 项目根目录: %PROJECT_ROOT%

echo 检查Python是否可用...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo Python未找到，请确保Python已安装并添加到环境变量中。
    pause
    exit /b 1
)

echo 安装依赖项...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo 安装依赖项失败。
    pause
    exit /b 1
)

echo 开始训练模型...
python train_pth_model.py
if %ERRORLEVEL% NEQ 0 (
    echo 模型训练失败。
    pause
    exit /b 1
)

echo 模型训练完成！
echo 训练好的模型保存在models目录下。
pause