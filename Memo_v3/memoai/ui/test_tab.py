"""
MemoAI 测试选项卡模块
包含模型测试界面和功能
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QTextEdit)
from PyQt5.QtCore import Qt

from memoai.ui.utils import create_context_menu


class TestTab(QWidget):
    """测试选项卡"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        """初始化测试界面"""
        layout = QVBoxLayout(self)

        # 添加测试按钮
        self.test_button = QPushButton("运行测试")
        self.test_button.clicked.connect(self.run_tests)
        layout.addWidget(self.test_button)

        # 添加测试结果
        self.test_result = QTextEdit()
        self.test_result.setReadOnly(True)
        layout.addWidget(self.test_result)

        # 初始化右键菜单
        self.init_context_menu()

    def init_context_menu(self):
        """初始化右键菜单"""
        create_context_menu(self.test_result)

    def run_tests(self):
        """运行测试"""
        self.test_result.clear()
        self.test_result.append("测试功能已被禁用。")
        self.test_result.append("如需运行测试，请重新创建测试文件。")