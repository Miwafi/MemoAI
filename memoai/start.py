import sys
import os
import time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
print(f"添加项目根目录到Python路径: {project_root}")
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
    print("正在启动 MemoAI GUI 界面...")
    print("如果这是你第一次运行，可能需要一点时间加载...")
    app = QApplication(sys.argv)
    window = MemoAIApp()
    sys.exit(app.exec_())
if __name__ == "__main__":
    main()