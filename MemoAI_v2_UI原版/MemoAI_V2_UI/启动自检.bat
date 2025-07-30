@echo off
title MemoAI系统自检程序
color 0A
echo.
echo =================================
echo    MemoAI系统自检程序
echo =================================
echo.
echo 正在检查系统状态...
echo.

:: 切换到当前脚本所在目录
cd /d "%~dp0"

:: 运行自检程序
python run_system_check.py

echo.
echo.
pause
echo 按任意键退出...