from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QTextEdit)
from PyQt5.QtCore import Qt
from memoai.ui.utils import create_context_menu
class TestTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.test_button = QPushButton("运行测试")
        self.test_button.clicked.connect(self.run_tests)
        layout.addWidget(self.test_button)
        self.test_result = QTextEdit()
        self.test_result.setReadOnly(True)
        layout.addWidget(self.test_result)
        self.init_context_menu()
    def init_context_menu(self):
        create_context_menu(self.test_result)
    def run_tests(self):
        self.test_result.clear()
        self.test_result.append("测试功能已被禁用。")
        self.test_result.append("如需运行测试，请重新创建测试文件。")