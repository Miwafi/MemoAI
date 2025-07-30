@echo off
title MemoAI主程序
color 0B
echo.
echo =================================
echo    MemoAI智能对话系统
echo =================================
echo.
echo 正在启动系统...
echo.

:: 切换到当前脚本所在目录
cd /d "%~dp0"

:: 先运行自检确保环境正常
echo 正在检查系统环境...
python run_system_check.py

echo.
echo 正在启动主程序...
echo.

:: 运行主程序
python memoAI_V2_UI.py

echo.
pause