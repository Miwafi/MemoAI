import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                            QLineEdit, QPushButton, QProgressBar, QComboBox,
                            QFrame, QMessageBox, QTextEdit, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from memoai.ui.utils import create_context_menu
from memoai.core.training_thread import TrainingThread
class TrainTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        self.training_thread = None
        self.is_training = False
        self.model_params_panel = None
        self.log_panel = None
        self.setup_animations()
    def setup_animations(self):
        self.model_params_animation = QPropertyAnimation()
        self.log_animation = QPropertyAnimation()
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.model_params_panel = QFrame()
        self.model_params_panel.setFrameShape(QFrame.StyledPanel)
        params_layout = QVBoxLayout(self.model_params_panel)
        model_layout = QHBoxLayout()
        model_label = QLabel("模型选择:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Memo-1"])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        params_layout.addLayout(model_layout)
        epochs_layout = QHBoxLayout()
        epochs_label = QLabel("训练轮次:")
        self.epochs_input = QLineEdit("10")
        epochs_layout.addWidget(epochs_label)
        epochs_layout.addWidget(self.epochs_input)
        params_layout.addLayout(epochs_layout)
        lr_layout = QHBoxLayout()
        lr_label = QLabel("学习率:")
        self.lr_input = QLineEdit("0.001")
        lr_layout.addWidget(lr_label)
        lr_layout.addWidget(self.lr_input)
        params_layout.addLayout(lr_layout)
        layout.addWidget(self.model_params_panel)
        self.train_button = QPushButton("开始训练")
        self.train_button.clicked.connect(self.start_training)
        layout.addWidget(self.train_button)
        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        layout.addWidget(self.train_progress)
        self.log_panel = QScrollArea()
        self.log_panel.setWidgetResizable(True)
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.log_panel.setWidget(self.train_log)
        layout.addWidget(self.log_panel)
        self.model_params_panel.setMaximumHeight(150)
        self.log_panel.setMaximumHeight(200)
        self.init_context_menu()
    def init_context_menu(self):
        create_context_menu(self.train_log)
    def start_training(self):
        if self.is_training:
            return
        self.train_log.clear()
        self.train_log.append("开始训练模型...")
        model_name = self.model_combo.currentText()
        try:
            epochs = int(self.epochs_input.text())
            lr = float(self.lr_input.text())
        except ValueError:
            self.train_log.append("错误: 训练轮次和学习率必须是数字！")
            return
        self.train_button.setEnabled(False)
        self.switch_layout(train_mode=True)
        self.training_thread = TrainingThread(model_name, epochs, lr)
        self.training_thread.progress_updated.connect(self.update_training_progress)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.start()
        self.is_training = True
    def update_training_progress(self, value):
        self.train_progress.setValue(value)
        self.train_log.append(f"训练进度: {value}%")
    def on_training_finished(self, success, message):
        self.train_log.append(message)
        self.train_button.setEnabled(True)
        self.is_training = False
        self.switch_layout(train_mode=False)
        language = self.parent.settings_tab.language_combo.currentText()
        if success:
            if language == 'English':
                QMessageBox.information(self, "Training Completed", "Model training successful!")
            elif language == '梗体中文':
                QMessageBox.information(self, "炼完了", "模型炼好了！")
            elif language == '日本語':
                QMessageBox.information(self, "訓練完了", "モデルの訓練が成功しました！")
            else:
                QMessageBox.information(self, "训练完成", "模型训练成功！")
        else:
            if language == 'English':
                QMessageBox.critical(self, "Training Failed", message)
            elif language == '梗体中文':
                QMessageBox.critical(self, "炼炸了", message)
            elif language == '日本語':
                QMessageBox.critical(self, "訓練失敗", message)
            else:
                QMessageBox.critical(self, "训练失败", message)
    def switch_layout(self, train_mode):
        self.model_params_animation.stop()
        self.log_animation.stop()
        duration = 500
        if train_mode:
            self.model_params_animation = QPropertyAnimation(self.model_params_panel, b"maximumHeight")
            self.model_params_animation.setDuration(duration)
            self.model_params_animation.setStartValue(150)
            self.model_params_animation.setEndValue(50)
            self.model_params_animation.start()
            self.log_animation = QPropertyAnimation(self.log_panel, b"maximumHeight")
            self.log_animation.setDuration(duration)
            self.log_animation.setStartValue(200)
            self.log_animation.setEndValue(400)
            self.log_animation.start()
        else:
            self.model_params_animation = QPropertyAnimation(self.model_params_panel, b"maximumHeight")
            self.model_params_animation.setDuration(duration)
            self.model_params_animation.setStartValue(50)
            self.model_params_animation.setEndValue(150)
            self.model_params_animation.start()
            self.log_animation = QPropertyAnimation(self.log_panel, b"maximumHeight")
            self.log_animation.setDuration(duration)
            self.log_animation.setStartValue(400)
            self.log_animation.setEndValue(200)
            self.log_animation.start()