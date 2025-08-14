"""
MemoAI 设置选项卡模块
包含应用设置界面和功能
"""
import os
import json
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                            QComboBox, QLineEdit, QPushButton, QFrame, QTextEdit,
                            QMessageBox, QMenu, QAction, QCheckBox, QInputDialog)
from PyQt5.QtCore import Qt

from memoai.ui.styles import apply_dark_theme, apply_light_theme
from memoai.ui.utils import create_context_menu
from memoai.utils.security_utils import SecurityUtils


class SettingsTab(QWidget):
    """设置选项卡"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        """初始化设置界面"""
        layout = QVBoxLayout(self)

        # 创建设置框架
        settings_frame = QFrame()
        settings_layout = QVBoxLayout(settings_frame)

        # 添加主题选择
        theme_layout = QHBoxLayout()
        theme_label = QLabel("主题选择:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["深色主题", "浅色主题", "跟随系统"])
        self.theme_combo.currentIndexChanged.connect(self.change_theme)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        settings_layout.addLayout(theme_layout)

        # 添加语言选择
        language_layout = QHBoxLayout()
        language_label = QLabel("语言选择:")
        self.language_combo = QComboBox()
        self.language_combo.addItems(["中文", "English", "梗体中文", "日本語", "Français", "Español"])
        self.language_combo.currentIndexChanged.connect(self.apply_language)
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

        # 添加开发者模式选项
        dev_mode_layout = QHBoxLayout()
        self.dev_mode_label = QLabel("开发者模式:")
        self.dev_mode_checkbox = QCheckBox()
        self.dev_mode_checkbox.stateChanged.connect(self.toggle_dev_mode)
        dev_mode_layout.addWidget(self.dev_mode_label)
        dev_mode_layout.addWidget(self.dev_mode_checkbox)
        settings_layout.addLayout(dev_mode_layout)

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

        # 初始化右键菜单
        self.init_context_menu()

        # 加载保存的设置
        self.load_settings()

    def load_settings(self):
        """加载保存的设置"""
        try:
            config_dir = os.path.join(os.path.expanduser('~'), '.memoai')
            config_file = os.path.join(config_dir, 'settings.json')

            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)

                # 设置开发者模式状态
                if 'developer_mode' in settings:
                    self.dev_mode_checkbox.setChecked(settings['developer_mode'])            
        except Exception as e:
            # 加载设置失败时静默处理
            pass

    def toggle_dev_mode(self, state):
        """切换开发者模式状态"""
        if state == Qt.Checked:
            # 请求密码
            password, ok = QInputDialog.getText(self, '开发者模式', '请输入开发者密码:', echo=QInputDialog.Password)
            if not ok:
                self.dev_mode_checkbox.setChecked(False)
                return

            # 验证密码
            credentials_path = SecurityUtils.get_credentials_path()
            credentials = SecurityUtils.load_credentials(credentials_path)

            if not credentials:
                # 首次设置密码
                confirm_password, ok = QInputDialog.getText(self, '设置开发者密码', '请再次输入密码:', echo=QInputDialog.Password)
                if not ok or password != confirm_password:
                    QMessageBox.warning(self, '密码不匹配', '两次输入的密码不匹配，请重试。')
                    self.dev_mode_checkbox.setChecked(False)
                    return

                # 生成盐和哈希密码
                salt = SecurityUtils.generate_salt()
                hashed_password = SecurityUtils.hash_password(password, salt)

                # 保存凭据
                SecurityUtils.save_credentials(credentials_path, hashed_password, salt)
                QMessageBox.information(self, '密码设置成功', '开发者密码已设置成功。')
            else:
                # 验证现有密码
                if not SecurityUtils.verify_password(password, credentials['salt'], credentials['hashed_password']):
                    QMessageBox.warning(self, '密码错误', '输入的开发者密码错误，请重试。')
                    self.dev_mode_checkbox.setChecked(False)
                    return
        # 无需处理取消选中的情况

    def init_context_menu(self):
        """初始化右键菜单"""
        create_context_menu(self.settings_info)
        
        # 为主题选择框添加右键菜单
        create_context_menu(self.theme_combo)
        
        # 为语言选择框添加右键菜单
        create_context_menu(self.language_combo)
        
        # 为最大生成长度输入框添加右键菜单
        create_context_menu(self.max_length_input)
        
        # 为温度参数输入框添加右键菜单
        create_context_menu(self.temperature_input)
        
        # 为设备选择框添加右键菜单
        create_context_menu(self.device_combo)
        
        # 为保存按钮添加右键菜单
        create_context_menu(self.save_settings_button)

    def change_theme(self):
        """更改主题"""
        try:
            theme = self.theme_combo.currentText()
            if theme == "浅色主题" or (theme == "Light Theme" and self.language_combo.currentText() == 'English'):
                # 应用浅色主题
                apply_light_theme(self.parent)
            elif theme == "深色主题" or (theme == "Dark Theme" and self.language_combo.currentText() == 'English'):
                # 应用深色主题
                apply_dark_theme(self.parent)
            else:
                # 跟随系统主题
                import sys
                if sys.platform == 'darwin':  # macOS
                    self.parent.setStyleSheet("")
                elif sys.platform == 'win32':  # Windows
                    self.parent.setStyleSheet("")
                else:  # Linux
                    # 对于Linux，使用系统默认样式
                    from PyQt5.QtWidgets import QStyleFactory
                    self.parent.setStyle(QStyleFactory.create("Fusion"))
                    self.parent.setStyleSheet("")
        except Exception as e:
            error_msg = f"更改主题时发生错误: {str(e)}"
            QMessageBox.critical(self, "主题更改错误", error_msg)

    def save_settings(self):
        """保存设置"""
        try:
            # 获取设置值
            theme = self.theme_combo.currentText()
            language = self.language_combo.currentText()
            max_length = self.max_length_input.text()
            temperature = self.temperature_input.text()
            device = self.device_combo.currentText()
            dev_mode = self.dev_mode_checkbox.isChecked()

            # 验证数值
            try:
                max_length_value = int(max_length)
                if max_length_value <= 0:
                    raise ValueError("最大生成长度必须大于0")
            except ValueError:
                raise ValueError("最大生成长度必须是正整数")

            try:
                temperature_value = float(temperature)
                if temperature_value < 0 or temperature_value > 2:
                    raise ValueError("温度参数必须在0到2之间")
            except ValueError:
                raise ValueError("温度参数必须是有效的数字")

            # 保存设置到文件
            settings = {
                'theme': theme,
                'language': language,
                'max_length': max_length_value,
                'temperature': temperature_value,
                'device': device,
                'developer_mode': dev_mode
            }

            # 确保配置目录存在
            config_dir = os.path.join(os.path.expanduser('~'), '.memoai')
            try:
                os.makedirs(config_dir, exist_ok=True)
            except OSError as e:
                raise Exception(f"无法创建配置目录: {str(e)}")

            config_file = os.path.join(config_dir, 'settings.json')

            # 保存设置
            try:
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(settings, f, ensure_ascii=False, indent=2)
            except IOError as e:
                raise Exception(f"保存设置文件失败: {str(e)}")

            # 应用语言设置
            self.apply_language()

            # 更新设置信息显示
            self.settings_info.append("\n设置已保存:")
            self.settings_info.append(f"- 主题: {theme}")
            self.settings_info.append(f"- 语言: {language}")
            self.settings_info.append(f"- 最大生成长度: {max_length}")
            self.settings_info.append(f"- 温度参数: {temperature}")
            self.settings_info.append(f"- 设备: {device}")
            self.settings_info.append(f"- 开发者模式: {'开启' if dev_mode else '关闭'}")
            self.settings_info.append(f"\n设置已保存到: {config_file}")
            self.settings_info.verticalScrollBar().setValue(self.settings_info.verticalScrollBar().maximum())

            # 显示成功消息
            if language == 'English':
                QMessageBox.information(self, "Save Successful", "Settings have been saved successfully!")
            elif language == '梗体中文':
                QMessageBox.information(self, "保存完毕", "配置已经存好辣！")
            elif language == '日本語':
                QMessageBox.information(self, "設定保存", "設定が正常に保存されました！")
            else:
                QMessageBox.information(self, "保存成功", "设置已成功保存！")
        except ValueError as e:
            # 处理数值验证错误
            error_msg = str(e)
            if language == 'English':
                if "最大生成长度" in error_msg:
                    QMessageBox.critical(self, "Save Failed", "Max generation length must be a positive integer!")
                else:
                    QMessageBox.critical(self, "Save Failed", "Temperature parameter must be a valid number between 0 and 2!")
            elif language == '梗体中文':
                if "最大生成长度" in error_msg:
                    QMessageBox.critical(self, "保存失败", "最长唠多少得是正整数！")
                else:
                    QMessageBox.critical(self, "保存失败", "整活程度得在0到2之间！")
            elif language == '日本語':
                if "最大生成长度" in error_msg:
                    QMessageBox.critical(self, "設定失敗", "最大生成長は正の整数でなければなりません！")
                else:
                    QMessageBox.critical(self, "設定失敗", "温度パラメータは0から2までの有効な数値でなければなりません！")
            else:
                QMessageBox.critical(self, "保存失败", error_msg)
        except Exception as e:
            # 处理其他异常
            error_msg = f"保存设置时发生错误: {str(e)}"
            if language == 'English':
                QMessageBox.critical(self, "Save Failed", f"An error occurred while saving settings: {str(e)}")
            elif language == '梗体中文':
                QMessageBox.critical(self, "保存失败", f"存配置的时候出错了: {str(e)}")
            elif language == '日本語':
                QMessageBox.critical(self, "設定失敗", f"設定の保存中にエラーが発生しました: {str(e)}")
            else:
                QMessageBox.critical(self, "保存失败", error_msg)
            if self.language_combo.currentText() == 'English':
                QMessageBox.critical(self, "Save Failed", f"Error saving settings: {str(e)}")
            elif self.language_combo.currentText() == '梗体中文':
                QMessageBox.critical(self, "保存失败", f"存配置的时候出错了: {str(e)}")
            elif self.language_combo.currentText() == '日本語':
                QMessageBox.critical(self, "設定失敗", f"設定の保存中にエラーが発生しました: {str(e)}")
            else:
                QMessageBox.critical(self, "保存失败", f"保存设置时出错: {str(e)}")

    def apply_language(self):
        try:
            language = self.language_combo.currentText()
            # 更新界面元素文本
            if language == 'English':
                self.theme_combo.setItemText(0, "Dark Theme")
                self.theme_combo.setItemText(1, "Light Theme")
                self.theme_combo.setItemText(2, "Follow System")
                self.save_settings_button.setText("Save Settings")
                self.dev_mode_label.setText("Developer Mode:")
                self.settings_info.clear()
                self.settings_info.append("Settings Description:")
                self.settings_info.append("1. Theme Selection: Change the appearance theme of the application")
                self.settings_info.append("2. Language Selection: Change the interface language of the application")
                self.settings_info.append("3. Max Generation Length: Control the maximum length of AI-generated text")
                self.settings_info.append("4. Temperature Parameter: Control the diversity of AI-generated text, higher values are more diverse")
                self.settings_info.append("5. Device Selection: Choose the device on which the model runs (Auto-detect/CPU/GPU)")
                self.settings_info.append("6. Developer Mode: Enable advanced features and debugging options")
            elif language == '中文' or language == '梗体中文':
                self.theme_combo.setItemText(0, "深色主题")
                self.theme_combo.setItemText(1, "浅色主题")
                self.theme_combo.setItemText(2, "跟随系统")
                self.save_settings_button.setText("保存设置")
                self.dev_mode_label.setText("开发者模式:")
                self.settings_info.clear()
                self.settings_info.append("设置说明:")
                self.settings_info.append("1. 主题选择: 更改应用的外观主题")
                self.settings_info.append("2. 语言选择: 更改应用的界面语言")
                self.settings_info.append("3. 最大生成长度: 控制AI生成文本的最大长度")
                self.settings_info.append("4. 温度参数: 控制AI生成文本的多样性，值越大越多样")
                self.settings_info.append("5. 设备选择: 选择模型运行的设备 (自动检测/CPU/GPU)")
                self.settings_info.append("6. 开发者模式: 启用高级功能和调试选项")
            elif language == '日本語':
                self.theme_combo.setItemText(0, "ダークテーマ")
                self.theme_combo.setItemText(1, "ライトテーマ")
                self.theme_combo.setItemText(2, "システムに追従")
                self.save_settings_button.setText("設定を保存")
                self.dev_mode_label.setText("開発者モード:")
                self.settings_info.clear()
                self.settings_info.append("設定の説明:")
                self.settings_info.append("1. テーマ選択: アプリケーションの外観テーマを変更します")
                self.settings_info.append("2. 言語選択: アプリケーションのインターフェース言語を変更します")
                self.settings_info.append("3. 最大生成長: AIが生成するテキストの最大長を制御します")
                self.settings_info.append("4. 温度パラメータ: AIが生成するテキストの多様性を制御します。値が大きいほど多様になります")
                self.settings_info.append("5. デバイス選択: モデルを実行するデバイスを選択します (自動検出/CPU/GPU)")
                self.settings_info.append("6. 開発者モード: 高度な機能とデバッグオプションを有効にします")
            elif language == 'Français':
                # 更新为法语界面
                self.parent.setWindowTitle("MemoAI - Assistant Intelligent")
                # 更新状态栏文本
                self.parent.statusBar().showMessage("Prêt")
                self.parent.tab_widget.setTabText(0, "Discussion")
                self.parent.tab_widget.setTabText(1, "Entraîner Modèle")
                self.parent.tab_widget.setTabText(2, "Exemples Rapides")
                self.parent.tab_widget.setTabText(3, "Test de Modèle")
                self.parent.tab_widget.setTabText(4, "Paramètres")
                self.parent.chat_tab.chat_input.setPlaceholderText("Veuillez entrer votre question...")
                self.parent.chat_tab.send_button.setText("Envoyer")
                self.parent.train_tab.train_button.setText("Démarrer l'Entraînement")
                self.parent.examples_tab.example_button.setText("Générer Texte")
            elif language == 'Español':
                # 更新为西班牙语界面
                self.parent.setWindowTitle("MemoAI - Asistente Inteligente")
                # 更新状态栏文本
                self.parent.statusBar().showMessage("Listo")
                self.parent.tab_widget.setTabText(0, "Chat")
                self.parent.tab_widget.setTabText(1, "Entrenar Modelo")
                self.parent.tab_widget.setTabText(2, "Ejemplos Rápidos")
                self.parent.tab_widget.setTabText(3, "Prueba de Modelo")
                self.parent.tab_widget.setTabText(4, "Configuración")
                self.parent.chat_tab.chat_input.setPlaceholderText("Por favor, ingrese su pregunta...")
                self.parent.chat_tab.send_button.setText("Enviar")
                self.parent.train_tab.train_button.setText("Iniciar Entrenamiento")
                self.parent.examples_tab.example_button.setText("Generar Texto")
                self.parent.test_tab.test_button.setText("Run Tests")
                self.save_settings_button.setText("Save Settings")
                self.settings_info.clear()
                self.settings_info.append("Settings Instructions:")
                self.settings_info.append("1. Theme Selection: Change the application's appearance theme")
                self.settings_info.append("2. Language Selection: Change the application's interface language")
                self.settings_info.append("3. Max Generation Length: Control the maximum length of AI-generated text")
                self.settings_info.append("4. Temperature Parameter: Control the diversity of AI-generated text, higher values are more diverse")
            # 添加更多语言的处理...
            elif language == '梗体中文':
                # 更新为梗体中文界面
                self.parent.setWindowTitle("MemoAI - 能工智人")
                # 更新状态栏文本
                self.parent.statusBar().showMessage("稳了")
                self.parent.tab_widget.setTabText(0, "吹水")
                self.parent.tab_widget.setTabText(1, "炼丹")
                self.parent.tab_widget.setTabText(2, "整活示例")
                self.parent.tab_widget.setTabText(3, "这里没有模型测试")
                self.parent.tab_widget.setTabText(4, "调参")
                self.parent.chat_tab.chat_input.setPlaceholderText("有啥事儿尽管问...")
                self.parent.chat_tab.send_button.setText("冲")
                self.parent.train_tab.train_button.setText("开炼")
                self.parent.examples_tab.example_button.setText("按下开始整活")
                self.parent.test_tab.test_button.setText("开测")
                self.save_settings_button.setText("保存配置")
                self.settings_info.clear()
                self.settings_info.append("设置说明:")
                self.settings_info.append("1. 主题选择: 换皮肤")
                self.settings_info.append("2. 语言选择: 换个说法")
            elif language == '日本語':
                # 更新为日语界面
                self.parent.setWindowTitle("MemoAI - 知的アシスタント")
                # 更新状态栏文本
                self.parent.statusBar().showMessage("準備完了")
                self.parent.tab_widget.setTabText(0, "チャット")
                self.parent.tab_widget.setTabText(1, "モデル訓練")
                self.parent.tab_widget.setTabText(2, "クイック例")
                self.parent.tab_widget.setTabText(3, "モデルテスト")
                self.parent.tab_widget.setTabText(4, "設定")
                self.parent.chat_tab.chat_input.setPlaceholderText("質問を入力してください...")
                self.parent.chat_tab.send_button.setText("送信")
                self.parent.train_tab.train_button.setText("訓練開始")
                self.parent.examples_tab.example_button.setText("テキスト生成")
                self.parent.test_tab.test_button.setText("テスト実行")
                self.save_settings_button.setText("設定を保存")
                self.settings_info.clear()
        except Exception as e:
            error_msg = f"应用语言设置时发生错误: {str(e)}"
            QMessageBox.critical(self, "语言设置错误", error_msg)
            # 默认回退到英语界面
            self.parent.setWindowTitle("MemoAI - Intelligent Assistant")
            self.parent.statusBar().showMessage("Ready")
            self.parent.tab_widget.setTabText(0, "Chat")
            self.parent.tab_widget.setTabText(1, "Train Model")
            self.parent.tab_widget.setTabText(2, "Quick Examples")
            self.parent.tab_widget.setTabText(3, "Model Test")
            self.parent.tab_widget.setTabText(4, "Settings")
            self.parent.chat_tab.chat_input.setPlaceholderText("Please enter your question...")
            self.parent.chat_tab.send_button.setText("Send")
            self.parent.train_tab.train_button.setText("Start Training")
            self.parent.examples_tab.example_button.setText("Generate Text")
            self.settings_info.append("4. Temperature Parameter: Control the diversity of AI-generated text, higher values are more diverse")
            self.settings_info.append("5. Device Selection: Choose the device to run the model (Auto-detect/CPU/GPU)")
            self.settings_info.append("3. 最大生成长度: 管管AI别唠太狠")
            self.settings_info.append("4. 温度参数: 控制AI整活程度，越高越离谱")
            self.settings_info.append("5. 设备选择: 选个地方跑模型 (自动/嬉皮优/鸡皮优)")
            self.settings_info.append("設定の説明:")
            self.settings_info.append("1. テーマ選択: アプリケーションの外観テーマを変更")
            self.settings_info.append("2. 言語選択: アプリケーションのインターフェース言語を変更")
            self.settings_info.append("3. 最大生成長: AIが生成するテキストの最大長を制御")
            self.settings_info.append("4. 温度パラメータ: AIが生成するテキストの多様性を制御、値が大きいほど多様性が増す")
            self.settings_info.append("5. デバイス選択: モデルを実行するデバイスを選択 (自動検出/CPU/GPU)")
        else:
            # 更新为中文界面
            self.parent.setWindowTitle("MemoAI - 智能助手")
            # 更新状态栏文本
            self.parent.statusBar().showMessage("就绪")
            self.parent.tab_widget.setTabText(0, "聊天")
            self.parent.tab_widget.setTabText(1, "训练模型")
            self.parent.tab_widget.setTabText(2, "快速示例")
            self.parent.tab_widget.setTabText(3, "模型测试")
            self.parent.tab_widget.setTabText(4, "设置")
            # 检查并更新聊天选项卡
            if hasattr(self.parent, 'chat_tab') and self.parent.chat_tab is not None:
                self.parent.chat_tab.chat_input.setPlaceholderText("请输入您的问题...")
                self.parent.chat_tab.send_button.setText("发送")
            
            # 检查并更新训练选项卡
            if hasattr(self.parent, 'train_tab') and self.parent.train_tab is not None:
                self.parent.train_tab.train_button.setText("开始训练")
            
            # 检查并更新示例选项卡
            if hasattr(self.parent, 'examples_tab') and self.parent.examples_tab is not None:
                self.parent.examples_tab.example_button.setText("生成文本")
            
            # 检查并更新测试选项卡
            if hasattr(self.parent, 'test_tab') and self.parent.test_tab is not None:
                self.parent.test_tab.test_button.setText("运行测试")
            self.save_settings_button.setText("保存设置")
            self.settings_info.clear()
            self.settings_info.append("设置说明:")
            self.settings_info.append("1. 主题选择: 更改应用的外观主题")
            self.settings_info.append("2. 语言选择: 更改应用的界面语言")
            self.settings_info.append("3. 最大生成长度: 控制AI生成文本的最大长度")
            self.settings_info.append("4. 温度参数: 控制AI生成文本的多样性，值越大越多样")
            self.settings_info.append("5. 设备选择: 选择模型运行的设备 (自动检测/CPU/GPU)")
            self.settings_info.append("6. 开发者模式: 启用高级功能和调试选项")

    def load_settings(self):
        """加载设置"""
        try:
            # 检查配置文件是否存在
            config_dir = os.path.join(os.path.expanduser('~'), '.memoai')
            config_file = os.path.join(config_dir, 'settings.json')

            if os.path.exists(config_file):
                # 加载设置
                with open(config_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)

                # 应用设置
                theme = settings.get('theme', '深色主题')
                language = settings.get('language', '中文')
                self.language_combo.setCurrentText(language)
                max_length = settings.get('max_length', '100')
                temperature = settings.get('temperature', '0.7')
                device = settings.get('device', '自动检测')

                # 设置控件值
                self.theme_combo.setCurrentText(theme)
                self.language_combo.setCurrentText(language)
                self.max_length_input.setText(max_length)
                self.temperature_input.setText(temperature)
                self.device_combo.setCurrentText(device)

                # 应用主题和语言
                self.change_theme()
                self.apply_language()

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
                # 应用默认语言
                self.apply_language()
        except Exception as e:
            self.settings_info.append(f"\n加载设置时出错: {str(e)}")
            self.settings_info.verticalScrollBar().setValue(self.settings_info.verticalScrollBar().maximum())
            # 设置默认值
            self.max_length_input.setText('100')
            self.temperature_input.setText('0.7')
            # 应用默认语言
            self.apply_language()