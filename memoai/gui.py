import sys
import os
import time
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QComboBox, QProgressBar,
    QMessageBox, QTabWidget, QFrame, QSplitter, QMenu, QAction, QSplashScreen
)
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve, Qt, QPoint
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPainter, QLinearGradient, QColor
class DynamicSplashScreen(QSplashScreen):
    def __init__(self, pixmap, parent=None):
        super().__init__(pixmap, Qt.WindowStaysOnTopHint)
        self.gradient_pos = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gradient)
        self.timer.start(80)
    def update_gradient(self):
        self.gradient_pos = (self.gradient_pos + 0.005) % 1.0
        self.update()
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        main_pos = (math.sin(self.gradient_pos * 2 * math.pi) + 1) / 2
        gradient.setColorAt(0.0, QColor(0, 0, 0))
        gradient.setColorAt(min(main_pos * 0.4, 1.0), QColor(30, 30, 40))
        gradient.setColorAt(min(main_pos * 0.5 + 0.1, 1.0), QColor(0, 80, 160))
        gradient.setColorAt(min(main_pos * 0.5 + 0.2, 1.0), QColor(0, 120, 215))
        gradient.setColorAt(min(main_pos * 0.5 + 0.3, 1.0), QColor(100, 180, 255))
        gradient.setColorAt(1.0, QColor(255, 255, 255))
        painter.fillRect(self.rect(), gradient)
        title_font = QFont("Arial", 48, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(255, 255, 255))
        title_rect = painter.boundingRect(0, 0, self.width(), self.height(), Qt.AlignCenter, "Memo AI")
        painter.drawText(title_rect, Qt.AlignCenter, "Memo AI")
        desc_font = QFont("Arial", 12)
        painter.setFont(desc_font)
        painter.drawText(20, self.height() - 40, "An AI Project By Students")
        status_font = QFont("Arial", 10)
        painter.setFont(status_font)
        if hasattr(self, 'status_text'):
            painter.drawText(20, self.height() - 20, self.status_text)
class LoadingThread(QThread):
    progress_updated = pyqtSignal(int, str)
    def run(self):
        statuses = ["加载依赖...", "初始化界面...", "检查模型文件...", "加载设置...", "准备就绪..."]
        for i in range(101):
            if i < 20:
                status_idx = 0
            elif i < 40:
                status_idx = 1
            elif i < 70:
                status_idx = 2
            elif i < 90:
                status_idx = 3
            else:
                status_idx = 4
            self.progress_updated.emit(i, statuses[status_idx])
            if i < 70:
                time.sleep(0.05)
            else:
                time.sleep(0.03)
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from memoai.inference.infer import MemoAIInferencer
has_preloaded_model = False
preloaded_inferencer = None
class TrainingThread(QThread):
    progress_updated = pyqtSignal(int)
    training_finished = pyqtSignal(bool, str)
    def __init__(self, model_name, epochs, lr):
        super().__init__()
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
    def run(self):
        try:
            current_dir = os.getcwd()
            training_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training')
            if not os.path.exists(training_dir):
                self.training_finished.emit(False, f"训练目录 '{training_dir}' 不存在！")
                return
            os.chdir(training_dir)
            sys.path.append(training_dir)
            import train
            try:
                success = train.train_model(self.model_name, self.epochs, self.lr)
                if success:
                    self.training_finished.emit(True, f"模型 {self.model_name} 训练完成！")
                else:
                    self.training_finished.emit(False, f"模型 {self.model_name} 训练失败！")
            except Exception as e:
                self.training_finished.emit(False, f"训练过程中出错: {str(e)}")
            finally:
                os.chdir(current_dir)
        except Exception as e:
            self.training_finished.emit(False, f"训练失败: {str(e)}")
class ChatThread(QThread):
    response_generated = pyqtSignal(str)
    def __init__(self, prompt, max_length=100):
        super().__init__()
        self.prompt = prompt
        self.max_length = max_length
        global preloaded_inferencer
        self.inferencer = preloaded_inferencer
    def run(self):
        try:
            response = self.inferencer.generate_text(self.prompt, max_length=self.max_length)
            self.response_generated.emit(response)
        except Exception as e:
            self.response_generated.emit(f"生成失败: {str(e)}")
class MemoAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.hide()
        self.show_splash_screen()
        global has_preloaded_model, preloaded_inferencer
        if not has_preloaded_model:
            self.statusBar().showMessage("Loading model...")
            preloaded_inferencer = MemoAIInferencer()
            has_preloaded_model = True
            self.statusBar().showMessage("Model loaded successfully")
    def on_loading_complete(self):
        self.init_ui()
        self.load_settings()
        self.apply_styles()
        self.init_context_menu()
        self.show()
    def show_splash_screen(self):
        splash_pix = QPixmap(800, 300)
        splash_pix.fill(Qt.transparent)
        self.splash = DynamicSplashScreen(splash_pix)
        self.splash.status_text = "Loading..."
        self.splash.show()
        self.loading_thread = LoadingThread()
        self.loading_thread.progress_updated.connect(self.update_splash_progress)
        self.loading_thread.finished.connect(self.splash.close)
        self.loading_thread.finished.connect(self.on_loading_complete)
        self.loading_thread.start()
    def update_splash_progress(self, value, status):
        if hasattr(self, 'splash'):
            self.splash.status_text = status
            self.splash.update()
    def init_ui(self):
        self.setWindowTitle("MemoAI")
        self.setGeometry(100, 100, 1000, 700)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.tab_widget.setStyleSheet("QTabWidget::pane { border: 0; }")
        self.tab_animations = {}
        self.current_tab_index = -1
        self.chat_tab = self.create_chat_tab()
        self.tab_widget.addTab(self.chat_tab, "聊天")
        self.train_tab = self.create_train_tab()
        self.tab_widget.addTab(self.train_tab, "训练模型")
        self.examples_tab = self.create_examples_tab()
        self.tab_widget.addTab(self.examples_tab, "快速示例")
        self.test_tab = self.create_test_tab()
        self.tab_widget.addTab(self.test_tab, "模型测试")
        self.settings_tab = self.create_settings_tab()
        self.tab_widget.addTab(self.settings_tab, "设置")
        main_layout.addWidget(self.tab_widget)
        if self.tab_widget.count() > 0:
            self.current_tab_index = 0
            self._init_tab_animation(self.current_tab_index)
        self.statusBar().showMessage("就绪")
    def create_chat_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        layout.addWidget(self.chat_history)
        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("请输入您的问题...")
        self.chat_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.chat_input)
        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)
        return tab
    def create_train_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        params_frame = QFrame()
        params_frame.setFrameShape(QFrame.StyledPanel)
        params_layout = QVBoxLayout(params_frame)
        model_layout = QHBoxLayout()
        model_label = QLabel("模型选择:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Memo-1", "Memo-2", "Custom Model"])
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
        layout.addWidget(params_frame)
        self.train_button = QPushButton("开始训练")
        self.train_button.clicked.connect(self.start_training)
        layout.addWidget(self.train_button)
        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        layout.addWidget(self.train_progress)
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        layout.addWidget(self.train_log)
        return tab
    def create_examples_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.example_prompts = QComboBox()
        self.example_prompts.addItems([
            "人工智能是",
            "未来的世界会",
            "我喜欢",
            "MemoAI是"
        ])
        layout.addWidget(self.example_prompts)
        self.example_button = QPushButton("生成文本")
        self.example_button.clicked.connect(self.generate_example)
        layout.addWidget(self.example_button)
        self.example_result = QTextEdit()
        self.example_result.setReadOnly(True)
        layout.addWidget(self.example_result)
        return tab
    def create_test_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.test_button = QPushButton("运行测试")
        self.test_button.clicked.connect(self.run_tests)
        layout.addWidget(self.test_button)
        self.test_result = QTextEdit()
        self.test_result.setReadOnly(True)
        layout.addWidget(self.test_result)
        return tab
    def create_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        settings_frame = QFrame()
        settings_layout = QVBoxLayout(settings_frame)
        theme_layout = QHBoxLayout()
        theme_label = QLabel("主题选择:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["深色主题", "浅色主题", "温豆丝主题"])
        self.theme_combo.currentIndexChanged.connect(self.change_theme)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        settings_layout.addLayout(theme_layout)
        language_layout = QHBoxLayout()
        language_label = QLabel("语言选择:")
        self.language_combo = QComboBox()
        self.language_combo.addItems(["中文", "English"])
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_combo)
        settings_layout.addLayout(language_layout)
        max_length_layout = QHBoxLayout()
        max_length_label = QLabel("最大生成长度:")
        self.max_length_input = QLineEdit()
        max_length_layout.addWidget(max_length_label)
        max_length_layout.addWidget(self.max_length_input)
        settings_layout.addLayout(max_length_layout)
        temperature_layout = QHBoxLayout()
        temperature_label = QLabel("温度参数:")
        self.temperature_input = QLineEdit()
        temperature_layout.addWidget(temperature_label)
        temperature_layout.addWidget(self.temperature_input)
        settings_layout.addLayout(temperature_layout)
        device_layout = QHBoxLayout()
        device_label = QLabel("设备选择:")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["自动检测", "CPU", "GPU"])
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        settings_layout.addLayout(device_layout)
        self.save_settings_button = QPushButton("保存设置")
        self.save_settings_button.clicked.connect(self.save_settings)
        settings_layout.addWidget(self.save_settings_button)
        layout.addWidget(settings_frame)
        self.settings_info = QTextEdit()
        self.settings_info.setReadOnly(True)
        self.settings_info.append("设置说明:")
        self.settings_info.append("1. 主题选择: 更改应用的外观主题")
        self.settings_info.append("2. 语言选择: 更改应用的界面语言")
        self.settings_info.append("3. 最大生成长度: 控制AI生成文本的最大长度")
        self.settings_info.append("4. 温度参数: 控制AI生成文本的多样性，值越大越多样")
        self.settings_info.append("5. 设备选择: 选择模型运行的设备 (自动检测/CPU/GPU)")
        layout.addWidget(self.settings_info)
        return tab
    def change_theme(self):
        theme = self.theme_combo.currentText()
        if theme == "浅色主题":
            qss =
            self.setStyleSheet(qss)
        elif theme == "深色主题":
            self.apply_styles()
        else:
            self.setStyleSheet("")
    def save_settings(self):
        try:
            theme = self.theme_combo.currentText()
            language = self.language_combo.currentText()
            max_length = self.max_length_input.text()
            temperature = self.temperature_input.text()
            int(max_length)
            float(temperature)
            settings = {
                'theme': theme,
                'language': language,
                'max_length': max_length,
                'temperature': temperature,
                'device': self.device_combo.currentText()
            }
            config_dir = os.path.join(os.path.expanduser('~'), '.memoai')
            os.makedirs(config_dir, exist_ok=True)
            config_file = os.path.join(config_dir, 'settings.json')
            import json
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            self.settings_info.append("\n设置已保存:")
            self.settings_info.append(f"- 主题: {theme}")
            self.settings_info.append(f"- 语言: {language}")
            self.settings_info.append(f"- 最大生成长度: {max_length}")
            self.settings_info.append(f"- 温度参数: {temperature}")
            self.settings_info.append(f"\n设置已保存到: {config_file}")
            self.settings_info.verticalScrollBar().setValue(self.settings_info.verticalScrollBar().maximum())
            QMessageBox.information(self, "保存成功", "设置已成功保存！")
        except ValueError:
            QMessageBox.critical(self, "保存失败", "最大生成长度必须是整数，温度参数必须是数字！")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存设置时出错: {str(e)}")
    def load_settings(self):
        try:
            config_dir = os.path.join(os.path.expanduser('~'), '.memoai')
            config_file = os.path.join(config_dir, 'settings.json')
            if os.path.exists(config_file):
                import json
                with open(config_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                theme = settings.get('theme', '深色主题')
                language = settings.get('language', '中文')
                max_length = settings.get('max_length', '100')
                temperature = settings.get('temperature', '0.7')
                device = settings.get('device', '自动检测')
                self.theme_combo.setCurrentText(theme)
                self.language_combo.setCurrentText(language)
                self.max_length_input.setText(max_length)
                self.temperature_input.setText(temperature)
                self.device_combo.setCurrentText(device)
                self.change_theme()
                self.settings_info.append(f"\n已加载保存的设置:")
                self.settings_info.append(f"- 主题: {theme}")
                self.settings_info.append(f"- 语言: {language}")
                self.settings_info.append(f"- 最大生成长度: {max_length}")
                self.settings_info.append(f"- 温度参数: {temperature}")
                self.settings_info.append(f"- 设备选择: {device}")
                self.settings_info.verticalScrollBar().setValue(self.settings_info.verticalScrollBar().maximum())
            else:
                self.max_length_input.setText('100')
                self.temperature_input.setText('0.7')
                self.settings_info.append("\n首次运行，使用默认设置。")
                self.settings_info.verticalScrollBar().setValue(self.settings_info.verticalScrollBar().maximum())
        except Exception as e:
            self.settings_info.append(f"\n加载设置时出错: {str(e)}")
            self.settings_info.verticalScrollBar().setValue(self.settings_info.verticalScrollBar().maximum())
            self.max_length_input.setText('100')
            self.temperature_input.setText('0.7')
    def init_context_menu(self):
        from PyQt5.QtWidgets import QMenu, QAction
        from PyQt5.QtGui import QFont
        def create_context_menu(widget):
            menu = QMenu(widget)
            font = QFont()
            font.setFamily("Microsoft YaHei")
            menu.setFont(font)
            copy_action = QAction("复制", widget)
            copy_action.triggered.connect(widget.copy)
            cut_action = QAction("剪切", widget)
            cut_action.triggered.connect(widget.cut)
            paste_action = QAction("粘贴", widget)
            paste_action.triggered.connect(widget.paste)
            select_all_action = QAction("全选", widget)
            select_all_action.triggered.connect(widget.selectAll)
            menu.addSeparator()
            menu.addAction(copy_action)
            menu.addAction(cut_action)
            menu.addAction(paste_action)
            menu.addSeparator()
            menu.addAction(select_all_action)
            return menu
        self.chat_history.setContextMenuPolicy(Qt.CustomContextMenu)
        self.chat_history.customContextMenuRequested.connect(
            lambda position: create_context_menu(self.chat_history).exec_(self.chat_history.mapToGlobal(position))
        )
        self.train_log.setContextMenuPolicy(Qt.CustomContextMenu)
        self.train_log.customContextMenuRequested.connect(
            lambda position: create_context_menu(self.train_log).exec_(self.train_log.mapToGlobal(position))
        )
        self.example_result.setContextMenuPolicy(Qt.CustomContextMenu)
        self.example_result.customContextMenuRequested.connect(
            lambda position: create_context_menu(self.example_result).exec_(self.example_result.mapToGlobal(position))
        )
        self.test_result.setContextMenuPolicy(Qt.CustomContextMenu)
        self.test_result.customContextMenuRequested.connect(
            lambda position: create_context_menu(self.test_result).exec_(self.test_result.mapToGlobal(position))
        )
        self.settings_info.setContextMenuPolicy(Qt.CustomContextMenu)
        self.settings_info.customContextMenuRequested.connect(
            lambda position: create_context_menu(self.settings_info).exec_(self.settings_info.mapToGlobal(position))
        )
    def apply_styles(self):
        qss =
        self.setStyleSheet(qss)
    def send_message(self):
        message = self.chat_input.text().strip()
        if message:
            self.chat_history.append(f"<b>你:</b> {message}")
            self.chat_input.clear()
            self.send_button.setEnabled(False)
            self.chat_input.setEnabled(False)
            self.chat_history.append("<b>MemoAI:</b> 正在思考中...")
            self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())
            self.chat_thread = ChatThread(message, max_length=int(self.max_length_input.text()))
            self.chat_thread.response_generated.connect(self.on_response_generated)
            self.chat_thread.start()
            self.statusBar().showMessage("AI正在思考...")
    def on_response_generated(self, response):
        current_text = self.chat_history.toHtml()
        current_text = current_text.replace("<b>MemoAI:</b> 正在思考中...", "")
        self.chat_history.setHtml(current_text)
        self.chat_history.append(f"<b>MemoAI:</b> {response}")
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())
        self.send_button.setEnabled(True)
        self.chat_input.setEnabled(True)
    def start_training(self):
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
        self.training_thread = TrainingThread(model_name, epochs, lr)
        self.training_thread.progress_updated.connect(self.update_training_progress)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.start()
    def update_training_progress(self, value):
        self.train_progress.setValue(value)
        self.train_log.append(f"训练进度: {value}%")
    def _init_tab_animation(self, index):
        tab_widget = self.tab_widget.widget(index)
        if not tab_widget:
            return
        if index != self.current_tab_index:
            tab_widget.setWindowOpacity(0.0)
            tab_widget.move(tab_widget.x() + 30, tab_widget.y())
        else:
            tab_widget.setWindowOpacity(1.0)
        if index not in self.tab_animations:
            from PyQt5.QtCore import QParallelAnimationGroup
            from PyQt5.QtWidgets import QGraphicsOpacityEffect
            animation_group = QParallelAnimationGroup()
            opacity_effect = QGraphicsOpacityEffect(tab_widget)
            tab_widget.setGraphicsEffect(opacity_effect)
            opacity_animation = QPropertyAnimation(opacity_effect, b"opacity")
            opacity_animation.setDuration(800)
            opacity_animation.setEasingCurve(QEasingCurve.OutQuart)
            animation_group.addAnimation(opacity_animation)
            pos_animation = QPropertyAnimation(tab_widget, b"pos")
            pos_animation.setDuration(800)
            pos_animation.setEasingCurve(QEasingCurve.OutQuart)
            animation_group.addAnimation(pos_animation)
            self.tab_animations[index] = animation_group
        else:
            animation_group = self.tab_animations[index]
            animation_group.stop()
        return self.tab_animations[index]
    def on_tab_changed(self, index):
        if self.current_tab_index == -1:
            self.current_tab_index = index
            return
        prev_index = self.current_tab_index
        current_tab = self.tab_widget.widget(index)
        prev_tab = self.tab_widget.widget(prev_index)
        if not current_tab or not prev_tab:
            self.current_tab_index = index
            return
        prev_animation = self._init_tab_animation(prev_index)
        current_animation = self._init_tab_animation(index)
        prev_animation_group = prev_animation
        current_animation_group = current_animation
        prev_opacity_anim = prev_animation_group.animationAt(0)
        prev_pos_anim = prev_animation_group.animationAt(1)
        prev_opacity_anim.setStartValue(1.0)
        prev_opacity_anim.setEndValue(0.0)
        prev_pos_anim.setStartValue(prev_tab.pos())
        prev_pos_anim.setEndValue(prev_tab.pos() - QPoint(30, 0))
        current_opacity_anim = current_animation_group.animationAt(0)
        current_pos_anim = current_animation_group.animationAt(1)
        current_opacity_anim.setStartValue(0.0)
        current_opacity_anim.setEndValue(1.0)
        current_pos_anim.setStartValue(current_tab.pos())
        current_pos_anim.setEndValue(QPoint(self.tab_widget.width() // 2 - current_tab.width() // 2, current_tab.y()))
        def on_animation_finished():
            self.current_tab_index = index
            try:
                current_animation_group.disconnect()
            except:
                pass
            try:
                prev_animation_group.disconnect()
            except:
                pass
        current_animation_group.finished.connect(on_animation_finished)
        prev_animation_group.start()
        current_animation_group.start()
    def on_training_finished(self, success, message):
        self.train_log.append(message)
        self.train_button.setEnabled(True)
        if success:
            QMessageBox.information(self, "训练完成", "模型训练成功！")
        else:
            QMessageBox.critical(self, "训练失败", message)
    def generate_example(self):
        prompt = self.example_prompts.currentText()
        self.example_result.clear()
        self.example_result.append(f"提示: {prompt}")
        self.example_result.append("正在生成...")
        self.example_button.setEnabled(False)
        self.example_prompts.setEnabled(False)
        device = self.device_combo.currentText()
        self.example_thread = ChatThread(prompt, max_length=int(self.max_length_input.text()))
        self.example_thread.response_generated.connect(self.on_example_generated)
        self.example_thread.start()
        self.statusBar().showMessage(f"AI正在{device}上生成示例...")
    def on_example_generated(self, response):
        current_text = self.example_result.toPlainText()
        current_text = current_text.replace("正在生成...", "")
        self.example_result.setPlainText(current_text)
        self.example_result.append(f"生成: {response}")
        self.example_button.setEnabled(True)
        self.example_prompts.setEnabled(True)
    def run_tests(self):
        self.test_result.clear()
        self.test_result.append("测试功能已被禁用。")
        self.test_result.append("如需运行测试，请重新创建测试文件。")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MemoAIApp()
    window.show()
    sys.exit(app.exec_())