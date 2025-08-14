
"""
MemoAI GUI界面 (PyQt5版)
这个界面使用PyQt5构建，并通过QSS实现CSS3样式效果
"""
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
    """动态渐变启动画面"""
    def __init__(self, pixmap, parent=None):
        super().__init__(pixmap, Qt.WindowStaysOnTopHint)
        self.gradient_pos = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gradient)
        self.timer.start(80)  # 降低更新频率，每80毫秒更新一次

    def update_gradient(self):
        """更新渐变位置，使用正弦函数实现更平滑的过渡"""
        self.gradient_pos = (self.gradient_pos + 0.005) % 1.0
        self.update()  # 触发重绘

    def paintEvent(self, event):
        """重绘启动画面，实现动态渐变"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 创建动态线性渐变
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        # 使用正弦函数计算主渐变位置，实现更平滑的移动效果
        main_pos = (math.sin(self.gradient_pos * 2 * math.pi) + 1) / 2

        # 添加更多中间颜色点，实现更平滑的过渡
        gradient.setColorAt(0.0, QColor(0, 0, 0))  # 黑色
        gradient.setColorAt(min(main_pos * 0.4, 1.0), QColor(30, 30, 40))  # 深灰蓝
        gradient.setColorAt(min(main_pos * 0.5 + 0.1, 1.0), QColor(0, 80, 160))  # 深蓝
        gradient.setColorAt(min(main_pos * 0.5 + 0.2, 1.0), QColor(0, 120, 215))  # Memo蓝色 (#0078D7)
        gradient.setColorAt(min(main_pos * 0.5 + 0.3, 1.0), QColor(100, 180, 255))  # 浅蓝
        gradient.setColorAt(1.0, QColor(255, 255, 255))  # 白色

        painter.fillRect(self.rect(), gradient)

        # 绘制标题
        title_font = QFont("Arial", 48, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(255, 255, 255))
        title_rect = painter.boundingRect(0, 0, self.width(), self.height(), Qt.AlignCenter, "Memo AI")
        painter.drawText(title_rect, Qt.AlignCenter, "Memo AI")

        # 绘制介绍文字
        desc_font = QFont("Arial", 12)
        painter.setFont(desc_font)
        painter.drawText(20, self.height() - 40, "智能助手应用程序")

        # 绘制状态标签内容
        status_font = QFont("Arial", 10)
        painter.setFont(status_font)
        if hasattr(self, 'status_text'):
            painter.drawText(20, self.height() - 20, self.status_text)


class LoadingThread(QThread):
    """加载线程，用于显示启动状态"""
    progress_updated = pyqtSignal(int, str)

    def run(self):
        statuses = ["加载依赖...", "初始化界面...", "检查模型文件...", "加载设置...", "准备就绪..."]
        for i in range(101):
            # 调整状态显示
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
            # 根据不同阶段调整睡眠时间，模拟真实加载过程
            if i < 70:
                time.sleep(0.05)
            else:
                time.sleep(0.03)


# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

# 导入MemoAI相关模块
from memoai.inference.infer import MemoAIInferencer

# 预加载模型标志
has_preloaded_model = False
preloaded_inferencer = None


class TrainingThread(QThread):
    """训练模型的线程
    这个线程负责在后台执行模型训练，避免卡住UI"""
    progress_updated = pyqtSignal(int)
    training_finished = pyqtSignal(bool, str)

    def __init__(self, model_name, epochs, lr):
        super().__init__()
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr

    def run(self):
        try:
            # 获取当前目录
            current_dir = os.getcwd()
            # 切换到training目录
            training_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training')
            if not os.path.exists(training_dir):
                self.training_finished.emit(False, f"训练目录 '{training_dir}' 不存在！")
                return
            os.chdir(training_dir)

            # 导入train模块
            sys.path.append(training_dir)
            import train

            # 执行训练
            try:
                # 调用实际的训练函数
                success = train.train_model(self.model_name, self.epochs, self.lr)
                if success:
                    # 训练成功
                    self.training_finished.emit(True, f"模型 {self.model_name} 训练完成！")
                else:
                    # 训练失败
                    self.training_finished.emit(False, f"模型 {self.model_name} 训练失败！")
            except Exception as e:
                self.training_finished.emit(False, f"训练过程中出错: {str(e)}")
            finally:
                # 切换回原目录
                os.chdir(current_dir)


        except Exception as e:
            self.training_finished.emit(False, f"训练失败: {str(e)}")


class ChatThread(QThread):
    """聊天线程"""
    response_generated = pyqtSignal(str)

    def __init__(self, prompt, max_length=100):
        super().__init__()
        self.prompt = prompt
        self.max_length = max_length
        # 使用预加载的模型
        global preloaded_inferencer
        self.inferencer = preloaded_inferencer

    def run(self):
        try:
            response = self.inferencer.generate_text(self.prompt, max_length=self.max_length)
            self.response_generated.emit(response)
        except Exception as e:
            self.response_generated.emit(f"生成失败: {str(e)}")


class MemoAIApp(QMainWindow):
    """MemoAI主窗口"""
    def __init__(self):
        super().__init__()
        # 初始隐藏主窗口
        self.hide()
        # 显示启动画面
        self.show_splash_screen()
        # 预加载模型
        global has_preloaded_model, preloaded_inferencer
        if not has_preloaded_model:
            self.statusBar().showMessage("正在预加载模型...")
            preloaded_inferencer = MemoAIInferencer()
            has_preloaded_model = True
            self.statusBar().showMessage("模型预加载完成")

    def on_loading_complete(self):
        """加载完成后初始化UI"""
        self.init_ui()
        self.load_settings()
        self.apply_styles()
        self.init_context_menu()
        self.show()

    def show_splash_screen(self):
        """显示启动画面
        创建一个8:3比例的窗口，添加动态渐变背景和状态显示"""
        # 创建一个800x300的图像作为初始背景
        splash_pix = QPixmap(800, 300)
        splash_pix.fill(Qt.transparent)  # 透明背景，由DynamicSplashScreen绘制渐变

        # 创建动态启动画面
        self.splash = DynamicSplashScreen(splash_pix)
        self.splash.status_text = "正在加载..."

        # 显示启动画面
        self.splash.show()

        # 创建并启动加载线程
        self.loading_thread = LoadingThread()
        self.loading_thread.progress_updated.connect(self.update_splash_progress)
        self.loading_thread.finished.connect(self.splash.close)
        self.loading_thread.finished.connect(self.on_loading_complete)
        self.loading_thread.start()

    def update_splash_progress(self, value, status):
        """更新启动画面的状态"""
        if hasattr(self, 'splash'):
            self.splash.status_text = status
            self.splash.update()  # 触发重绘

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("MemoAI - 智能助手")
        self.setGeometry(100, 100, 1000, 700)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)

        # 创建选项卡，并启用平滑过渡效果
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        # 设置选项卡样式
        self.tab_widget.setStyleSheet("QTabWidget::pane { border: 0; }")
        # 存储选项卡动画
        self.tab_animations = {}
        # 设置初始透明度
        self.current_tab_index = -1

        # 创建聊天选项卡
        self.chat_tab = self.create_chat_tab()
        self.tab_widget.addTab(self.chat_tab, "聊天")

        # 创建训练选项卡
        self.train_tab = self.create_train_tab()
        self.tab_widget.addTab(self.train_tab, "训练模型")

        # 创建示例选项卡
        self.examples_tab = self.create_examples_tab()
        self.tab_widget.addTab(self.examples_tab, "快速示例")

        # 创建测试选项卡
        self.test_tab = self.create_test_tab()
        self.tab_widget.addTab(self.test_tab, "模型测试")

        # 创建设置选项卡
        self.settings_tab = self.create_settings_tab()
        self.tab_widget.addTab(self.settings_tab, "设置")

        # 添加选项卡到主布局
        main_layout.addWidget(self.tab_widget)
        # 初始化第一个选项卡的动画
        if self.tab_widget.count() > 0:
            self.current_tab_index = 0
            self._init_tab_animation(self.current_tab_index)

        # 创建状态栏
        self.statusBar().showMessage("就绪")

    def create_chat_tab(self):
        """创建聊天选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 创建聊天区域
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        layout.addWidget(self.chat_history)

        # 创建输入区域
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
        """创建训练选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 添加训练参数设置
        params_frame = QFrame()
        params_frame.setFrameShape(QFrame.StyledPanel)
        params_layout = QVBoxLayout(params_frame)

        # 添加模型选择
        model_layout = QHBoxLayout()
        model_label = QLabel("模型选择:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Memo-1", "Memo-2", "Custom Model"])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        params_layout.addLayout(model_layout)

        # 添加训练轮次
        epochs_layout = QHBoxLayout()
        epochs_label = QLabel("训练轮次:")
        self.epochs_input = QLineEdit("10")
        epochs_layout.addWidget(epochs_label)
        epochs_layout.addWidget(self.epochs_input)
        params_layout.addLayout(epochs_layout)

        # 添加学习率
        lr_layout = QHBoxLayout()
        lr_label = QLabel("学习率:")
        self.lr_input = QLineEdit("0.001")
        lr_layout.addWidget(lr_label)
        lr_layout.addWidget(self.lr_input)
        params_layout.addLayout(lr_layout)

        layout.addWidget(params_frame)

        # 添加训练按钮
        self.train_button = QPushButton("开始训练")
        self.train_button.clicked.connect(self.start_training)
        layout.addWidget(self.train_button)

        # 添加进度条
        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        layout.addWidget(self.train_progress)

        # 添加训练日志
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        layout.addWidget(self.train_log)

        return tab

    def create_examples_tab(self):
        """创建示例选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

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

        # 添加结果显示
        self.example_result = QTextEdit()
        self.example_result.setReadOnly(True)
        layout.addWidget(self.example_result)

        return tab

    def create_test_tab(self):
        """创建测试选项卡
        运行模型测试并显示结果"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 添加测试按钮
        self.test_button = QPushButton("运行测试")
        self.test_button.clicked.connect(self.run_tests)
        layout.addWidget(self.test_button)

        # 添加测试结果
        self.test_result = QTextEdit()
        self.test_result.setReadOnly(True)
        layout.addWidget(self.test_result)

        return tab

    def create_settings_tab(self):
        """创建设置选项卡
        让用户可以配置应用的各种参数"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 创建设置框架
        settings_frame = QFrame()
        settings_layout = QVBoxLayout(settings_frame)

        # 添加主题选择
        theme_layout = QHBoxLayout()
        theme_label = QLabel("主题选择:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["深色主题", "浅色主题", "温豆丝主题"])
        self.theme_combo.currentIndexChanged.connect(self.change_theme)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        settings_layout.addLayout(theme_layout)

        # 添加语言选择
        language_layout = QHBoxLayout()
        language_label = QLabel("语言选择:")
        self.language_combo = QComboBox()
        self.language_combo.addItems(["中文", "English"])
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_combo)
        settings_layout.addLayout(language_layout)

        # 添加最大生成长度
        max_length_layout = QHBoxLayout()
        max_length_label = QLabel("最大生成长度:")
        self.max_length_input = QLineEdit()
        max_length_layout.addWidget(max_length_label)
        max_length_layout.addWidget(self.max_length_input)
        settings_layout.addLayout(max_length_layout)

        # 添加温度参数
        temperature_layout = QHBoxLayout()
        temperature_label = QLabel("温度参数:")
        self.temperature_input = QLineEdit()
        temperature_layout.addWidget(temperature_label)
        temperature_layout.addWidget(self.temperature_input)
        settings_layout.addLayout(temperature_layout)

        # 添加设备选择 (GPU/CPU)
        device_layout = QHBoxLayout()
        device_label = QLabel("设备选择:")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["自动检测", "CPU", "GPU"])
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        settings_layout.addLayout(device_layout)

        # 添加保存按钮
        self.save_settings_button = QPushButton("保存设置")
        self.save_settings_button.clicked.connect(self.save_settings)
        settings_layout.addWidget(self.save_settings_button)

        layout.addWidget(settings_frame)

        # 添加设置说明
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
        """更改主题"""
        theme = self.theme_combo.currentText()
        if theme == "浅色主题":
            # 应用浅色主题
            qss = """
            /* 全局样式 */
            QMainWindow, QWidget {
                background-color: #f8fafc;
                color: #0f172a;
            }

            /* 选项卡样式 */
            QTabWidget::pane {
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                background-color: #ffffff;
                margin: 5px;
            }

            QTabBar::tab {
                background-color: #f1f5f9;
                color: #64748b;
                padding: 8px 16px;
                border-radius: 8px 8px 0 0;
                margin-right: 2px;
                min-width: 80px;
                text-align: center;
            }

            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #0f172a;
                border-bottom: 2px solid #6366f1;
            }

            /* 按钮样式 */
            QPushButton {
                background-color: #6366f1;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                border: none;
                transition: all 0.3s ease;
            }

            QPushButton:hover {
                background-color: #8b5cf6;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(99, 102, 241, 0.3);
            }

            QPushButton:pressed {
                background-color: #4f46e5;
                transform: translateY(0);
                box-shadow: none;
            }

            QPushButton:disabled {
                background-color: #cbd5e1;
                color: #94a3b8;
                transform: none;
                box-shadow: none;
            }

            /* 输入框样式 */
            QLineEdit, QTextEdit, QComboBox {
                background-color: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 8px;
                transition: border-color 0.3s ease;
            }

            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border-color: #6366f1;
                outline: none;
            }

            /* 进度条样式 */
            QProgressBar {
                background-color: #f1f5f9;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                text-align: center;
                height: 12px;
            }

            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                              stop:0 #6366f1, stop:1 #8b5cf6);
                border-radius: 7px;
            }

            /* 框架样式 */
            QFrame {
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                background-color: #f1f5f9;
                margin: 5px;
            }

            /* 标签样式 */
            QLabel {
                color: #0f172a;
                padding: 5px;
            }

            /* 文本编辑框滚动条 */
            QTextEdit QScrollBar:vertical {
                background-color: #f1f5f9;
                width: 10px;
                margin: 5px 0 5px 0;
                border-radius: 5px;
            }

            QTextEdit QScrollBar::handle:vertical {
                background-color: #6366f1;
                min-height: 20px;
                border-radius: 5px;
            }

            QTextEdit QScrollBar::handle:vertical:hover {
                background-color: #8b5cf6;
            }
            """
            self.setStyleSheet(qss)
        elif theme == "深色主题":
            # 应用深色主题
            self.apply_styles()
        else:
            # 跟随系统主题
            self.setStyleSheet("")

    def save_settings(self):
        """保存设置
        将用户的设置保存到配置文件
        就像把喜好记录在笔记本上，下次打开时依然记得"""
        try:
            # 获取设置值
            theme = self.theme_combo.currentText()
            language = self.language_combo.currentText()
            max_length = self.max_length_input.text()
            temperature = self.temperature_input.text()

            # 验证数值
            int(max_length)
            float(temperature)

            # 保存设置到文件
            settings = {
                'theme': theme,
                'language': language,
                'max_length': max_length,
                'temperature': temperature,
                'device': self.device_combo.currentText()
            }

            # 确保配置目录存在
            config_dir = os.path.join(os.path.expanduser('~'), '.memoai')
            os.makedirs(config_dir, exist_ok=True)
            config_file = os.path.join(config_dir, 'settings.json')

            # 保存设置
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
        """加载设置
        从配置文件加载用户之前保存的设置
        就像打开笔记本，回忆之前的喜好"""
        try:
            # 检查配置文件是否存在
            config_dir = os.path.join(os.path.expanduser('~'), '.memoai')
            config_file = os.path.join(config_dir, 'settings.json')

            if os.path.exists(config_file):
                # 加载设置
                import json
                with open(config_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)

                # 应用设置
                theme = settings.get('theme', '深色主题')
                language = settings.get('language', '中文')
                max_length = settings.get('max_length', '100')
                temperature = settings.get('temperature', '0.7')
                device = settings.get('device', '自动检测')

                # 设置控件值
                self.theme_combo.setCurrentText(theme)
                self.language_combo.setCurrentText(language)
                self.max_length_input.setText(max_length)
                self.temperature_input.setText(temperature)
                self.device_combo.setCurrentText(device)

                # 应用主题
                self.change_theme()

                self.settings_info.append(f"\n已加载保存的设置:")
                self.settings_info.append(f"- 主题: {theme}")
                self.settings_info.append(f"- 语言: {language}")
                self.settings_info.append(f"- 最大生成长度: {max_length}")
                self.settings_info.append(f"- 温度参数: {temperature}")
                self.settings_info.append(f"- 设备选择: {device}")
                self.settings_info.verticalScrollBar().setValue(self.settings_info.verticalScrollBar().maximum())
            else:
                # 设置默认值
                self.max_length_input.setText('100')
                self.temperature_input.setText('0.7')
                self.settings_info.append("\n首次运行，使用默认设置。")
                self.settings_info.verticalScrollBar().setValue(self.settings_info.verticalScrollBar().maximum())
        except Exception as e:
            self.settings_info.append(f"\n加载设置时出错: {str(e)}")
            self.settings_info.verticalScrollBar().setValue(self.settings_info.verticalScrollBar().maximum())
            # 设置默认值
            self.max_length_input.setText('100')
            self.temperature_input.setText('0.7')

    def init_context_menu(self):
        """初始化右键菜单
        替换默认的英文右键菜单为中文菜单
        就像给应用换了个语言包，界面更亲切"""
        from PyQt5.QtWidgets import QMenu, QAction
        from PyQt5.QtGui import QFont

        # 创建通用的中文右键菜单
        def create_context_menu(widget):
            menu = QMenu(widget)
            
            # 设置菜单字体
            font = QFont()
            font.setFamily("Microsoft YaHei")
            menu.setFont(font)
            
            # 添加菜单项
            copy_action = QAction("复制", widget)
            copy_action.triggered.connect(widget.copy)
            
            cut_action = QAction("剪切", widget)
            cut_action.triggered.connect(widget.cut)
            
            paste_action = QAction("粘贴", widget)
            paste_action.triggered.connect(widget.paste)
            
            select_all_action = QAction("全选", widget)
            select_all_action.triggered.connect(widget.selectAll)
            
            # 添加分隔符
            menu.addSeparator()
            
            # 添加菜单项到菜单
            menu.addAction(copy_action)
            menu.addAction(cut_action)
            menu.addAction(paste_action)
            menu.addSeparator()
            menu.addAction(select_all_action)
            
            return menu

        # 为聊天历史添加右键菜单
        self.chat_history.setContextMenuPolicy(Qt.CustomContextMenu)
        self.chat_history.customContextMenuRequested.connect(
            lambda position: create_context_menu(self.chat_history).exec_(self.chat_history.mapToGlobal(position))
        )

        # 为训练日志添加右键菜单
        self.train_log.setContextMenuPolicy(Qt.CustomContextMenu)
        self.train_log.customContextMenuRequested.connect(
            lambda position: create_context_menu(self.train_log).exec_(self.train_log.mapToGlobal(position))
        )

        # 为示例结果添加右键菜单
        self.example_result.setContextMenuPolicy(Qt.CustomContextMenu)
        self.example_result.customContextMenuRequested.connect(
            lambda position: create_context_menu(self.example_result).exec_(self.example_result.mapToGlobal(position))
        )

        # 为测试结果添加右键菜单
        self.test_result.setContextMenuPolicy(Qt.CustomContextMenu)
        self.test_result.customContextMenuRequested.connect(
            lambda position: create_context_menu(self.test_result).exec_(self.test_result.mapToGlobal(position))
        )

        # 为设置说明添加右键菜单
        self.settings_info.setContextMenuPolicy(Qt.CustomContextMenu)
        self.settings_info.customContextMenuRequested.connect(
            lambda position: create_context_menu(self.settings_info).exec_(self.settings_info.mapToGlobal(position))
        )

    def apply_styles(self):
        """应用QSS样式
        给应用穿上漂亮的衣服，让它看起来更吸引人"""
        qss = """
        /* 全局样式 */
        QMainWindow, QWidget {
            background-color: #0f172a;
            color: #f8fafc;
            font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
        }

        /* 选项卡样式 */
        QTabWidget::pane {
            border: 1px solid #334155;
            border-radius: 8px;
            background-color: #1e293b;
            margin: 5px;
        }

        QTabBar::tab {
            background-color: #0f172a;
            color: #94a3b8;
            padding: 8px 16px;
            border-radius: 8px 8px 0 0;
            margin-right: 2px;
            min-width: 80px;
            text-align: center;
        }

        QTabBar::tab:selected {
            background-color: #1e293b;
            color: #f8fafc;
            border-bottom: 2px solid #6366f1;
        }

        /* 按钮样式 */
        QPushButton {
            background-color: #6366f1;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: bold;
            border: none;
        }

        QPushButton:hover {
            background-color: #8b5cf6;
        }

        QPushButton:pressed {
            background-color: #4f46e5;
        }

        QPushButton:disabled {
            background-color: #334155;
            color: #64748b;
        }

        /* 输入框样式 */
        QLineEdit, QTextEdit, QComboBox {
            background-color: #1e293b;
            color: #f8fafc;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 8px;
        }

        QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
            border-color: #6366f1;
            outline: none;
        }

        /* 进度条样式 */
        QProgressBar {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            text-align: center;
            height: 12px;
        }

        QProgressBar::chunk {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                          stop:0 #6366f1, stop:1 #8b5cf6);
            border-radius: 7px;
        }

        /* 框架样式 */
        QFrame {
            border: 1px solid #334155;
            border-radius: 8px;
            background-color: #0f172a;
            margin: 5px;
        }

        /* 标签样式 */
        QLabel {
            color: #f8fafc;
            padding: 5px;
        }

        /* 文本编辑框滚动条 */
        QTextEdit QScrollBar:vertical {
            background-color: #1e293b;
            width: 10px;
            margin: 5px 0 5px 0;
            border-radius: 5px;
        }

        QTextEdit QScrollBar::handle:vertical {
            background-color: #6366f1;
            min-height: 20px;
            border-radius: 5px;
        }

        QTextEdit QScrollBar::handle:vertical:hover {
            background-color: #8b5cf6;
        }
        """
        self.setStyleSheet(qss)

    def send_message(self):
        """发送消息
        将用户的问题发送给AI，并显示正在输入的状态"""
        message = self.chat_input.text().strip()
        if message:
            # 添加用户消息到聊天历史
            self.chat_history.append(f"<b>你:</b> {message}")
            self.chat_input.clear()

            # 禁用发送按钮和输入框
            self.send_button.setEnabled(False)
            self.chat_input.setEnabled(False)

            # 添加正在输入提示
            self.chat_history.append("<b>MemoAI:</b> 正在思考中...")
            self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

            # 创建聊天线程
            self.chat_thread = ChatThread(message, max_length=int(self.max_length_input.text()))
            self.chat_thread.response_generated.connect(self.on_response_generated)
            self.chat_thread.start()

            self.statusBar().showMessage("AI正在思考...")

    def on_response_generated(self, response):
        """处理生成的响应
        接收AI的回答并显示在聊天窗口中"""
        # 移除正在输入提示
        current_text = self.chat_history.toHtml()
        current_text = current_text.replace("<b>MemoAI:</b> 正在思考中...", "")
        self.chat_history.setHtml(current_text)

        # 添加生成的响应
        self.chat_history.append(f"<b>MemoAI:</b> {response}")
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

        # 启用发送按钮和输入框
        self.send_button.setEnabled(True)
        self.chat_input.setEnabled(True)

    def start_training(self):
        """开始训练
        启动训练线程，传入模型名称、训练轮次和学习率"""
        # 清空训练日志
        self.train_log.clear()
        self.train_log.append("开始训练模型...")

        # 获取训练参数
        model_name = self.model_combo.currentText()
        try:
            epochs = int(self.epochs_input.text())
            lr = float(self.lr_input.text())
        except ValueError:
            self.train_log.append("错误: 训练轮次和学习率必须是数字！")
            return

        # 禁用训练按钮
        self.train_button.setEnabled(False)

        # 创建训练线程
        self.training_thread = TrainingThread(model_name, epochs, lr)
        self.training_thread.progress_updated.connect(self.update_training_progress)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.start()

    def update_training_progress(self, value):
        """更新训练进度"""
        self.train_progress.setValue(value)
        self.train_log.append(f"训练进度: {value}%")

    def _init_tab_animation(self, index):
        """初始化选项卡动画"""
        # 获取选项卡部件
        tab_widget = self.tab_widget.widget(index)
        if not tab_widget:
            return

        # 设置初始属性
        if index != self.current_tab_index:
            tab_widget.setWindowOpacity(0.0)
            tab_widget.move(tab_widget.x() + 30, tab_widget.y())  # 更大的偏移
        else:
            tab_widget.setWindowOpacity(1.0)

        # 创建动画组
        if index not in self.tab_animations:
            from PyQt5.QtCore import QParallelAnimationGroup
            from PyQt5.QtWidgets import QGraphicsOpacityEffect

            # 创建动画组
            animation_group = QParallelAnimationGroup()

            # 创建透明度效果
            opacity_effect = QGraphicsOpacityEffect(tab_widget)
            tab_widget.setGraphicsEffect(opacity_effect)

            # 透明度动画
            opacity_animation = QPropertyAnimation(opacity_effect, b"opacity")
            opacity_animation.setDuration(800)
            opacity_animation.setEasingCurve(QEasingCurve.OutQuart)
            animation_group.addAnimation(opacity_animation)

            # 位置动画
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
        """选项卡切换时的处理"""
        # 忽略初始加载时的索引变化
        if self.current_tab_index == -1:
            self.current_tab_index = index
            return

        # 获取当前和前一个选项卡
        prev_index = self.current_tab_index
        current_tab = self.tab_widget.widget(index)
        prev_tab = self.tab_widget.widget(prev_index)

        if not current_tab or not prev_tab:
            self.current_tab_index = index
            return

        # 初始化动画
        prev_animation = self._init_tab_animation(prev_index)
        current_animation = self._init_tab_animation(index)

        # 获取动画组
        prev_animation_group = prev_animation
        current_animation_group = current_animation

        # 配置前一个选项卡的动画
        prev_opacity_anim = prev_animation_group.animationAt(0)
        prev_pos_anim = prev_animation_group.animationAt(1)

        prev_opacity_anim.setStartValue(1.0)
        prev_opacity_anim.setEndValue(0.0)

        prev_pos_anim.setStartValue(prev_tab.pos())
        prev_pos_anim.setEndValue(prev_tab.pos() - QPoint(30, 0))  # 更大的移动距离

        # 配置当前选项卡的动画
        current_opacity_anim = current_animation_group.animationAt(0)
        current_pos_anim = current_animation_group.animationAt(1)

        current_opacity_anim.setStartValue(0.0)
        current_opacity_anim.setEndValue(1.0)

        current_pos_anim.setStartValue(current_tab.pos())
        current_pos_anim.setEndValue(QPoint(self.tab_widget.width() // 2 - current_tab.width() // 2, current_tab.y()))

        # 连接动画完成信号
        def on_animation_finished():
            # 更新当前选项卡索引
            self.current_tab_index = index
            # 清理动画组连接
            try:
                current_animation_group.disconnect()
            except:
                pass
            try:
                prev_animation_group.disconnect()
            except:
                pass

        # 连接动画完成信号
        current_animation_group.finished.connect(on_animation_finished)

        # 启动动画
        prev_animation_group.start()
        current_animation_group.start()

    def on_training_finished(self, success, message):
        """处理训练完成"""
        self.train_log.append(message)
        self.train_button.setEnabled(True)

        if success:
            QMessageBox.information(self, "训练完成", "模型训练成功！")
        else:
            QMessageBox.critical(self, "训练失败", message)

    def generate_example(self):
        """生成示例文本
        根据选择的提示词生成文本示例"""
        prompt = self.example_prompts.currentText()
        self.example_result.clear()
        self.example_result.append(f"提示: {prompt}")
        self.example_result.append("正在生成...")

        # 禁用生成按钮和下拉框
        self.example_button.setEnabled(False)
        self.example_prompts.setEnabled(False)

        # 获取设备选择
        device = self.device_combo.currentText()

        # 创建聊天线程
        self.example_thread = ChatThread(prompt, max_length=int(self.max_length_input.text()))
        self.example_thread.response_generated.connect(self.on_example_generated)
        self.example_thread.start()

        self.statusBar().showMessage(f"AI正在{device}上生成示例...")

    def on_example_generated(self, response):
        """处理生成的示例
        显示AI生成的示例文本"""
        current_text = self.example_result.toPlainText()
        current_text = current_text.replace("正在生成...", "")
        self.example_result.setPlainText(current_text)
        self.example_result.append(f"生成: {response}")

        # 启用生成按钮和下拉框
        self.example_button.setEnabled(True)
        self.example_prompts.setEnabled(True)

    def run_tests(self):
        """运行测试"""
        self.test_result.clear()
        self.test_result.append("测试功能已被禁用。")
        self.test_result.append("如需运行测试，请重新创建测试文件。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MemoAIApp()
    window.show()
    sys.exit(app.exec_())