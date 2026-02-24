import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QTabWidget, QStatusBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from memoai.ui.chat_tab import ChatTab
from memoai.ui.train_tab import TrainTab
from memoai.ui.examples_tab import ExamplesTab
from memoai.ui.test_tab import TestTab
from memoai.ui.settings_tab import SettingsTab
from memoai.ui.styles import apply_dark_theme, apply_light_theme
from memoai.ui.utils import create_context_menu
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))
from memoai.inference.infer import MemoAIInferencer
class MemoAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_settings()
        self.apply_styles()
        self.init_context_menu()
    def init_ui(self):
        self.setWindowTitle("MemoAI - 智能助手")
        self.setGeometry(100, 100, 1000, 700)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.tab_widget = QTabWidget()
        self.chat_tab = ChatTab(self)
        self.tab_widget.addTab(self.chat_tab, "聊天")
        self.train_tab = TrainTab(self)
        self.tab_widget.addTab(self.train_tab, "训练模型")
        self.examples_tab = ExamplesTab(self)
        self.tab_widget.addTab(self.examples_tab, "快速示例")
        self.test_tab = TestTab(self)
        self.tab_widget.addTab(self.test_tab, "模型测试")
        self.settings_tab = SettingsTab(self)
        self.tab_widget.addTab(self.settings_tab, "设置")
        main_layout.addWidget(self.tab_widget)
        self.statusBar().showMessage("就绪")
    def apply_styles(self):
        apply_dark_theme(self)
    def load_settings(self):
        pass
    def init_context_menu(self):
        pass
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MemoAIApp()
    window.show()
    sys.exit(app.exec_())