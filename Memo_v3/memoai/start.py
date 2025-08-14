# -*- coding: utf-8 -*-
"""
MemoAI 启动脚本 (PyQt5 GUI版)
这个脚本会启动MemoAI的图形界面，基于PyQt5构建
"""
import sys
import os
import time

# 添加项目根目录到Python路径
# 使用脚本所在目录的父目录作为项目根目录，这样无论从哪里运行都能找到正确路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
print(f"添加项目根目录到Python路径: {project_root}")

# 尝试导入PyQt5和我们的GUI应用
try:
    from memoai.gui import MemoAIApp
    from PyQt5.QtWidgets import QApplication

except ImportError as e:
    print(f"导入失败: {e}")
    print("请先安装所需依赖:")
    print("pip install -r requirements.txt")
    time.sleep(5)
    sys.exit(1)


def main():
    """主函数 - 启动GUI界面"""
    print("正在启动 MemoAI GUI 界面...")
    print("如果这是你第一次运行，可能需要一点时间加载...")

    # 创建Qt应用
    app = QApplication(sys.argv)

    # 创建主窗口
    window = MemoAIApp()

    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()