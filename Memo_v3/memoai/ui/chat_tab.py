"""
MemoAI 聊天选项卡模块
包含聊天界面和消息处理功能
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
                            QLineEdit, QPushButton, QMessageBox, QLabel)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from memoai.ui.utils import create_context_menu
from memoai.core.chat_thread import ChatThread


class ChatTab(QWidget):
    """聊天选项卡"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        self.chat_thread = None
        self.show_placeholder = True
        self.update_placeholder()

    def init_ui(self):
        """初始化聊天界面"""
        layout = QVBoxLayout(self)

        # 创建聊天区域容器
        chat_container = QWidget()
        chat_container_layout = QVBoxLayout(chat_container)
        chat_container.setLayout(chat_container_layout)

        # 创建聊天历史
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.textChanged.connect(self.on_chat_history_changed)
        chat_container_layout.addWidget(self.chat_history)

        # 创建占位符标签
        self.placeholder_label = QLabel("请输入问题开始聊天...")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("color: #888888; font-style: italic;")
        chat_container_layout.addWidget(self.placeholder_label)

        layout.addWidget(chat_container)

        # 创建输入区域
        input_layout = QHBoxLayout()

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("请输入您的问题...")
        self.chat_input.returnPressed.connect(self.send_message)
        self.chat_input.textChanged.connect(self.on_input_text_changed)
        input_layout.addWidget(self.chat_input)

        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)

        # 初始化右键菜单
        self.init_context_menu()

        # 设置输入限制
        self.max_input_length = 500

    def init_context_menu(self):
        """初始化右键菜单"""
        # 获取当前语言，如果settings_tab未初始化则使用默认值
        try:
            language = self.parent.settings_tab.language_combo.currentText()
        except AttributeError:
            language = '中文'  # 默认语言
        create_context_menu(self.chat_history, language)
        create_context_menu(self.chat_input, language)

    def send_message(self):
        """发送消息"""
        message = self.chat_input.text().strip()
        if message and len(message) <= self.max_input_length:
            # 添加用户消息到聊天历史
            self.chat_history.append(f"<b>你:</b> {message}")
            self.chat_input.clear()

            # 禁用发送按钮和输入框
            self.send_button.setEnabled(False)
            self.chat_input.setEnabled(False)

            # 添加正在输入提示
            try:
                language = self.parent.settings_tab.language_combo.currentText()
                if language == 'English':
                    self.chat_history.append("<b>MemoAI:</b> Thinking...")
                elif language == '梗体中文':
                    self.chat_history.append("<b>MemoAI:</b> 正在琢磨...")
                elif language == '日本語':
                    self.chat_history.append("<b>MemoAI:</b> 考え中...")
                elif language == 'Français':
                    self.chat_history.append("<b>MemoAI:</b> En train de réfléchir...")
                elif language == 'Español':
                    self.chat_history.append("<b>MemoAI:</b> Pensando...")
                else:
                    self.chat_history.append("<b>MemoAI:</b> 正在思考中...")
            except AttributeError:
                self.chat_history.append("<b>MemoAI:</b> 正在思考中...")
            self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

            # 获取设备选择
            try:
                device = self.parent.settings_tab.device_combo.currentText()
            except AttributeError:
                device = '自动检测'

            # 获取当前选择的语言
            try:
                language = self.parent.settings_tab.language_combo.currentText()
            except AttributeError:
                language = '中文'

            # 获取最大生成长度
            try:
                max_length = int(self.parent.settings_tab.max_length_input.text())
            except (AttributeError, ValueError):
                max_length = 1000

            # 根据语言设置显示状态栏文本
            if language == 'English':
                self.parent.statusBar().showMessage(f"AI is thinking on {device}...")
            elif language == '梗体中文':
                self.parent.statusBar().showMessage(f"AI在{device}上琢磨呢...")
            elif language == '日本語':
                self.parent.statusBar().showMessage(f"AIが{device}で考え中...")
            else:
                self.parent.statusBar().showMessage(f"AI正在{device}上思考...")

            try:
                # 创建聊天线程
                self.chat_thread = ChatThread(message, max_length=max_length, 
                                             device=device, language=language)
                self.chat_thread.response_generated.connect(self.on_response_generated)
                self.chat_thread.error_occurred.connect(self.on_thread_error)
                self.chat_thread.start()
            except Exception as e:
                error_msg = f"创建聊天线程失败: {str(e)}"
                logger.error(error_msg)
                # 恢复UI状态
                self.send_button.setEnabled(True)
                self.chat_input.setEnabled(True)
                # 显示错误信息
                if language == 'English':
                    self.chat_history.append(f"<b>Error:</b> Failed to start AI response: {str(e)}")
                    self.parent.statusBar().showMessage("Failed to start AI response")
                else:
                    self.chat_history.append(f"<b>错误:</b> 无法启动AI响应: {str(e)}")
                    self.parent.statusBar().showMessage("无法启动AI响应")
            self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def on_response_generated(self, response):
        """处理生成的响应"""
        try:
            # 移除正在输入提示
            current_text = self.chat_history.toHtml()
            # 获取当前选择的语言
            try:
                language = self.parent.settings_tab.language_combo.currentText()
            except AttributeError:
                language = '中文'

            # 根据当前语言移除对应的提示文本
            if language == 'English':
                current_text = current_text.replace("<b>MemoAI:</b> Thinking...", "")
            elif language == '梗体中文':
                current_text = current_text.replace("<b>MemoAI:</b> 正在琢磨...", "")
            elif language == '日本語':
                current_text = current_text.replace("<b>MemoAI:</b> 考え中...", "")
            else:
                current_text = current_text.replace("<b>MemoAI:</b> 正在思考中...", "")
            self.chat_history.setHtml(current_text)

            # 添加生成的响应
            self.chat_history.append(f"<b>MemoAI:</b> {response}")
            self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

            # 启用发送按钮和输入框
            self.send_button.setEnabled(True)
            self.chat_input.setEnabled(True)

            # 更新状态栏
            if language == 'English':
                self.parent.statusBar().showMessage("AI response completed")
            else:
                self.parent.statusBar().showMessage("AI响应完成")
        except Exception as e:
            error_msg = f"处理AI响应时出错: {str(e)}"
            logger.error(error_msg)
            # 确保UI状态恢复
            self.send_button.setEnabled(True)
            self.chat_input.setEnabled(True)
            # 显示错误信息
            try:
                language = self.parent.settings_tab.language_combo.currentText()
            except AttributeError:
                language = '中文'
            if language == 'English':
                self.chat_history.append(f"<b>Error:</b> Failed to process AI response: {str(e)}")
                self.parent.statusBar().showMessage("Failed to process AI response")
            else:
                self.chat_history.append(f"<b>错误:</b> 无法处理AI响应: {str(e)}")
                self.parent.statusBar().showMessage("无法处理AI响应")
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def on_thread_error(self, error_msg):
        """处理聊天线程错误"""
        try:
            # 获取当前选择的语言
            language = self.parent.settings_tab.language_combo.currentText()
        except AttributeError:
            language = '中文'

        # 移除正在输入提示
        current_text = self.chat_history.toHtml()
        if language == 'English':
            current_text = current_text.replace("<b>MemoAI:</b> Thinking...", "")
        elif language == '梗体中文':
            current_text = current_text.replace("<b>MemoAI:</b> 正在琢磨...", "")
        elif language == '日本語':
            current_text = current_text.replace("<b>MemoAI:</b> 考え中...", "")
        else:
            current_text = current_text.replace("<b>MemoAI:</b> 正在思考中...", "")
        self.chat_history.setHtml(current_text)

        # 显示错误信息
        if language == 'English':
            self.chat_history.append(f"<b>Error:</b> {error_msg}")
            self.parent.statusBar().showMessage("AI response error")
        else:
            self.chat_history.append(f"<b>错误:</b> {error_msg}")
            self.parent.statusBar().showMessage("AI响应错误")

        # 恢复UI状态
        self.send_button.setEnabled(True)
        self.chat_input.setEnabled(True)
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def update_placeholder(self):
        """更新占位符显示"""
        if self.show_placeholder:
            self.placeholder_label.show()
        else:
            self.placeholder_label.hide()

    def on_chat_history_changed(self):
        """聊天历史变化时更新占位符"""
        self.show_placeholder = self.chat_history.toPlainText().strip() == ""
        self.update_placeholder()

    def on_input_text_changed(self):
        """输入文本变化时检查长度限制"""
        current_text = self.chat_input.text()
        if len(current_text) > self.max_input_length:
            # 截断文本
            self.chat_input.setText(current_text[:self.max_input_length])
            # 显示提示
            language = self.parent.settings_tab.language_combo.currentText()
            if language == 'English':
                self.parent.statusBar().showMessage("Input limited to 500 characters")
            elif language == '梗体中文':
                self.parent.statusBar().showMessage("输入限制在500字以内")
            elif language == '日本語':
                self.parent.statusBar().showMessage("入力は500文字以内に制限されています")
            else:
                self.parent.statusBar().showMessage("输入限制在500字以内")
        else:
            # 清除提示
            self.parent.statusBar().clearMessage()