"""
MemoAI 基础窗口模块
包含主窗口类和基本UI初始化
"""
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QTabWidget, QStatusBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# 导入其他UI模块
from memoai.ui.chat_tab import ChatTab
from memoai.ui.train_tab import TrainTab
from memoai.ui.examples_tab import ExamplesTab
from memoai.ui.test_tab import TestTab
from memoai.ui.settings_tab import SettingsTab
from memoai.ui.styles import apply_dark_theme, apply_light_theme
from memoai.ui.utils import create_context_menu

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

# 导入MemoAI相关模块
from memoai.inference.infer import MemoAIInferencer


class MemoAIApp(QMainWindow):
    """MemoAI主窗口"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_settings()
        self.apply_styles()
        self.init_context_menu()

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("MemoAI - 智能助手")
        self.setGeometry(100, 100, 1000, 700)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)

        # 创建选项卡
        self.tab_widget = QTabWidget()

        # 创建聊天选项卡
        self.chat_tab = ChatTab(self)
        self.tab_widget.addTab(self.chat_tab, "聊天")

        # 创建训练选项卡
        self.train_tab = TrainTab(self)
        self.tab_widget.addTab(self.train_tab, "训练模型")

        # 创建示例选项卡
        self.examples_tab = ExamplesTab(self)
        self.tab_widget.addTab(self.examples_tab, "快速示例")

        # 创建测试选项卡
        self.test_tab = TestTab(self)
        self.tab_widget.addTab(self.test_tab, "模型测试")

        # 创建设置选项卡
        self.settings_tab = SettingsTab(self)
        self.tab_widget.addTab(self.settings_tab, "设置")

        # 添加选项卡到主布局
        main_layout.addWidget(self.tab_widget)

        # 创建状态栏
        self.statusBar().showMessage("就绪")

    def apply_styles(self):
        """应用样式"""
        # 默认应用深色主题
        apply_dark_theme(self)

    def load_settings(self):
        """加载设置"""
        # 此方法将在settings_tab模块中实现
        pass

    def init_context_menu(self):
        """初始化右键菜单"""
        # 此方法将在utils模块中实现
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MemoAIApp()
    window.show()
    sys.exit(app.exec_())