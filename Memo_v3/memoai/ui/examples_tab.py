"""
MemoAI 示例选项卡模块
包含示例生成界面和功能
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QComboBox, QPushButton,
                            QTextEdit, QLabel, QHBoxLayout, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from memoai.ui.utils import create_context_menu
from memoai.core.chat_thread import ChatThread


class ExamplesTab(QWidget):
    """示例选项卡"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        self.example_thread = None
        self.is_generating = False

    def init_ui(self):
        """初始化示例界面"""
        layout = QVBoxLayout(self)

        # 添加说明标签
        instruction_label = QLabel("选择一个示例提示，然后点击'生成文本'按钮")
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setStyleSheet("color: #666666; font-style: italic;")
        layout.addWidget(instruction_label)

        # 添加示例提示
        self.example_prompts = QComboBox()
        self.example_prompts.addItems([
            "人工智能是",
            "未来的世界会",
            "我喜欢",
            "MemoAI是"
        ])
        layout.addWidget(self.example_prompts)

        # 添加生成按钮
        self.example_button = QPushButton("生成文本")
        self.example_button.clicked.connect(self.generate_example)
        layout.addWidget(self.example_button)

        # 添加结果显示区域
        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.StyledPanel)
        result_layout = QVBoxLayout(result_frame)

        # 添加结果标签
        result_label = QLabel("生成结果:")
        result_layout.addWidget(result_label)

        # 添加提示和续写区域容器
        content_layout = QVBoxLayout()

        # 提示区域
        prompt_frame = QFrame()
        prompt_frame.setStyleSheet("background-color: #f0f0f0; border-radius: 4px; padding: 8px;")
        prompt_layout = QHBoxLayout(prompt_frame)
        prompt_label = QLabel("提示:")
        prompt_label.setStyleSheet("font-weight: bold;")
        self.prompt_text = QLabel()
        prompt_layout.addWidget(prompt_label)
        prompt_layout.addWidget(self.prompt_text, 1)
        content_layout.addWidget(prompt_frame)

        # 续写区域
        continuation_frame = QFrame()
        continuation_frame.setStyleSheet("background-color: #e6f7ff; border: 1px dashed #1890ff; border-radius: 4px; padding: 8px;")
        continuation_layout = QHBoxLayout(continuation_frame)
        continuation_label = QLabel("续写区域:")
        continuation_label.setStyleSheet("font-weight: bold; color: #1890ff;")
        self.continuation_text = QTextEdit()
        self.continuation_text.setReadOnly(True)
        self.continuation_text.setStyleSheet("background-color: transparent; border: none;")
        continuation_layout.addWidget(continuation_label)
        continuation_layout.addWidget(self.continuation_text, 1)
        content_layout.addWidget(continuation_frame)

        result_layout.addLayout(content_layout)
        layout.addWidget(result_frame)

        # 初始化右键菜单
        self.init_context_menu()

    def init_context_menu(self):
        """初始化右键菜单"""
        # 获取当前语言，如果settings_tab未初始化则使用默认值
        try:
            language = self.parent.settings_tab.language_combo.currentText()
        except AttributeError:
            language = '中文'  # 默认语言
        create_context_menu(self.continuation_text, language)

    def generate_example(self):
        """生成示例文本"""
        if self.is_generating:
            return

        prompt = self.example_prompts.currentText()
        self.prompt_text.setText(prompt)
        self.continuation_text.clear()

        # 获取当前选择的语言，如果settings_tab未初始化则使用默认值
        try:
            language = self.parent.settings_tab.language_combo.currentText()
        except AttributeError:
            language = '中文'  # 默认语言
        if language == 'English':
            self.continuation_text.setText("Generating...")
        elif language == '梗体中文':
            self.continuation_text.setText("正在憋...")
        elif language == '日本語':
            self.continuation_text.setText("生成中...")
        else:
            self.continuation_text.setText("正在生成...")

        # 禁用生成按钮和下拉框
        self.example_button.setEnabled(False)
        self.example_prompts.setEnabled(False)
        self.is_generating = True

        # 获取设备选择，如果settings_tab未初始化则使用默认值
        try:
            device = self.parent.settings_tab.device_combo.currentText()
        except AttributeError:
            device = '自动检测'  # 默认设备

        # 获取当前选择的语言，如果settings_tab未初始化则使用默认值
        try:
            language = self.parent.settings_tab.language_combo.currentText()
        except AttributeError:
            language = '中文'  # 默认语言
        # 创建聊天线程
        # 获取最大生成长度，如果settings_tab未初始化则使用默认值
        try:
            max_length = int(self.parent.settings_tab.max_length_input.text())
        except (AttributeError, ValueError):
            max_length = 1000  # 默认最大长度
        self.example_thread = ChatThread(prompt, max_length=max_length, 
                                         device=device, language=language)
        self.example_thread.response_generated.connect(self.on_example_generated)
        self.example_thread.start()

        # 根据语言设置显示状态栏文本
        if language == 'English':
            self.parent.statusBar().showMessage(f"AI is generating example on {device}...")
        elif language == '梗体中文':
            self.parent.statusBar().showMessage(f"AI在{device}上憋示例呢...")
        elif language == '日本語':
            self.parent.statusBar().showMessage(f"AIが{device}で例を生成中...")
        else:
            self.parent.statusBar().showMessage(f"AI正在{device}上生成示例...")

    def on_example_generated(self, response):
        """处理生成的示例"""
        current_text = self.continuation_text.toPlainText()
        language = self.parent.settings_tab.language_combo.currentText()
        # 移除对应的提示文本
        if language == 'English':
            current_text = current_text.replace("Generating...", "")
        elif language == '梗体中文':
            current_text = current_text.replace("正在憋...", "")
        elif language == '日本語':
            current_text = current_text.replace("生成中...", "")
        else:
            current_text = current_text.replace("正在生成...", "")
        self.continuation_text.setPlainText(current_text + response)

        # 启用生成按钮和下拉框
        self.example_button.setEnabled(True)
        self.example_prompts.setEnabled(True)
        self.is_generating = False