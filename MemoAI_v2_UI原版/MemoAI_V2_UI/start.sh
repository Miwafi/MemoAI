#!/bin/bash
# MemoAI跨平台启动脚本 (Linux/macOS)

echo "========================================"
echo "    MemoAI智能对话系统"
echo "    跨平台启动器"
echo "========================================"
echo

# 获取脚本所在目录
cd "$(dirname "$0")"
echo "当前工作目录: $(pwd)"
echo

# 检查必要目录
REQUIRED_DIRS=("model" "memory" "log" "settings" "modules")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "创建目录: $dir"
        mkdir -p "$dir"
    fi
done

echo
echo "正在运行系统自检..."

# 运行系统自检
if python3 run_system_check.py; then
    echo "✅ 系统自检通过"
    echo
    echo "正在启动主程序..."
    python3 memoAI_V2_UI.py
else
    echo "❌ 系统自检失败"
    echo "请检查错误信息后重试"
fi

echo
read -p "按回车键退出..." dummy