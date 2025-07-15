# 目前暂时有一些错误，已被禁用
'''
import os
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QProgressBar, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal

class FileCleanerThread(QThread):
    progress_signal = pyqtSignal(int)
    success_count_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.files_to_delete = [
            'app.log',
            'loss_history.png'
        ]

    def run(self):
        total_files = len(self.files_to_delete)
        success_count = 0
        for index, file in enumerate(self.files_to_delete):
            try:
                if os.path.exists(file):
                    os.remove(file)
                    success_count += 1
                # 发送进度信号
                self.progress_signal.emit(int((index + 1) / total_files * 100))
            except Exception as e:
                self.error_signal.emit(file, str(e))
        self.success_count_signal.emit(success_count)

class FileCleanerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('文件清理工具')
        self.setGeometry(300, 300, 400, 150)

        layout = QVBoxLayout()

        self.start_button = QPushButton('开始清理', self)
        self.start_button.clicked.connect(self.start_cleaning)
        layout.addWidget(self.start_button)

        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def start_cleaning(self):
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.thread = FileCleanerThread()
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.success_count_signal.connect(self.show_completion)
        self.thread.error_signal.connect(self.show_error)
        self.thread.finished.connect(self.enable_button)
        self.thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def show_completion(self, success_count):
        QMessageBox.information(self, '完成', f'清理完成！成功删除 {success_count} 个文件。')

    def show_error(self, file, error):
        QMessageBox.critical(self, '错误', f'删除 {file} 时出错: {error}')

    def enable_button(self):
        self.start_button.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FileCleanerApp()
    ex.show()
    sys.exit(app.exec_())
'''