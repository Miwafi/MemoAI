import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QLineEdit, QPushButton, QTextEdit, 
                            QMessageBox, QDialog, QFormLayout)
from memo_ai_V2 import SelfLearningAI, DataManager, CharEncoder

# 初始化 AI 实例
ai = SelfLearningAI(input_size=32, hidden_size=64, output_size=32, data_path="./data")

class MemoAIApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Memo chat')
        self.setGeometry(300, 300, 600, 500)

        layout = QVBoxLayout()

        # 用户输入区域
        self.input_label = QLabel('请输入您的问题:')
        layout.addWidget(self.input_label)
        self.input_entry = QLineEdit()
        layout.addWidget(self.input_entry)

        # AI 答复区域
        self.reply_label = QLabel('AI 答复:')
        layout.addWidget(self.reply_label)
        self.reply_text = QTextEdit()
        self.reply_text.setReadOnly(True)
        layout.addWidget(self.reply_text)

        # 功能按钮区域
        button_layout = QHBoxLayout()

        self.infer_button = QPushButton('获取答复')
        self.infer_button.clicked.connect(self.get_reply)
        button_layout.addWidget(self.infer_button)

        self.train_button = QPushButton('训练模型')
        self.train_button.clicked.connect(self.train_model)
        button_layout.addWidget(self.train_button)

        self.annotation_button = QPushButton('人工标注')
        self.annotation_button.clicked.connect(self.manual_annotation)
        button_layout.addWidget(self.annotation_button)

        self.correct_button = QPushButton('手动纠错')
        self.correct_button.clicked.connect(self.manual_correction)
        button_layout.addWidget(self.correct_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_reply(self):
        user_input = self.input_entry.text()
        if user_input:
            reply = ai.infer(user_input)
            self.reply_text.setPlainText(reply)

    def train_model(self):
        try:
            ai.train()
            QMessageBox.information(self, '提示', '模型训练完成！')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'训练过程中出现错误: {str(e)}')

    def manual_annotation(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('人工标注')
        layout = QFormLayout()

        input_label = QLabel('当你输入这个时:')
        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText('请输入用户输入内容')
        layout.addRow(input_label, self.input_text)

        reply_label = QLabel('memo会告诉你这个:')
        # 修改变量名避免冲突
        self.annotation_reply_text = QLineEdit() 
        self.annotation_reply_text.setPlaceholderText('请输入 memo 的答复内容')
        layout.addRow(reply_label, self.annotation_reply_text)

        submit_button = QPushButton('提交标注')
        submit_button.clicked.connect(lambda: self.submit_annotation(dialog))
        layout.addRow(submit_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def submit_annotation(self, dialog):
        user_input = self.input_text.text()
        # 使用修改后的变量名
        reply = self.annotation_reply_text.text() 
        if user_input and reply:
            try:
                # 调用 save_memory 方法替代不存在的 save_annotation
                ai.save_memory(user_input, reply) 
                QMessageBox.information(self, '提示', '标注保存成功！')
                dialog.close()
            except Exception as e:
                QMessageBox.critical(self, '错误', f'保存标注时出现错误: {str(e)}')
        else:
            QMessageBox.warning(self, '警告', '请输入完整的标注内容！')

    def manual_correction(self):
        reply = self.reply_text.toPlainText()
        if not reply:
            QMessageBox.warning(self, '警告', '请先获取 AI 答复！')
            return

        dialog = QDialog(self)
        dialog.setWindowTitle('手动纠错')
        layout = QFormLayout()

        original_label = QLabel('AI 原始输出:')
        original_text = QTextEdit()
        original_text.setPlainText(reply)
        original_text.setReadOnly(True)
        layout.addRow(original_label, original_text)

        corrected_label = QLabel('你希望 AI 说的话:')
        self.corrected_text = QTextEdit()
        self.corrected_text.setPlaceholderText('请输入修正后的内容')
        self.corrected_text.setPlainText(reply)
        layout.addRow(corrected_label, self.corrected_text)

        submit_button = QPushButton('提交修正')
        submit_button.clicked.connect(lambda: self.submit_correction(dialog))
        layout.addRow(submit_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def submit_correction(self, dialog):
        user_input = self.input_entry.text()
        corrected_reply = self.corrected_text.toPlainText()
        if user_input and corrected_reply:
            try:
                # 这里可以添加将修正内容保存到模型或数据中的逻辑
                ai.save_memory(user_input, corrected_reply)
                self.reply_text.setPlainText(corrected_reply)
                QMessageBox.information(self, '提示', '修正已保存！')
                dialog.close()
            except Exception as e:
                QMessageBox.critical(self, '错误', f'保存修正时出现错误: {str(e)}')
        else:
            QMessageBox.warning(self, '警告', '请输入修正后的内容！')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MemoAIApp()
    ex.show()
    sys.exit(app.exec_())