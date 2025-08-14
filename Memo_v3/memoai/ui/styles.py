"""
MemoAI 样式模块
包含应用的主题样式定义
"""
from PyQt5.QtWidgets import QApplication


def apply_dark_theme(window):
    """应用深色主题"""
    qss = """
    /* 全局样式 */
    QWidget {
        background-color: #2d2d2d;
        color: #e0e0e0;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 16px;
    }

    /* 按钮样式 */
    QPushButton {
        background-color: #4a6fa5;
        color: white;
        border-radius: 8px;
        padding: 10px 18px;
        border: none;
        transition: all 0.2s ease;
    }

    QPushButton:hover {
        background-color: #5b7fb9;
        transform: translateY(-2px);
    }

    QPushButton:pressed {
        background-color: #3a5a89;
        transform: translateY(0);
    }

    /* 输入框样式 */
    QLineEdit {
        background-color: #3a3a3a;
        color: #e0e0e0;
        border: 1px solid #555555;
        border-radius: 8px;
        padding: 8px;
    }

    QLineEdit:focus {
        border: 2px solid #4a6fa5;
        outline: none;
        background-color: #3f3f3f;
    }

    /* 下拉框样式 */
    QComboBox {
        background-color: #3a3a3a;
        color: #e0e0e0;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 4px;
    }

    QComboBox::drop-down {
        border: none;
        background-color: #444444;
        border-radius: 0 4px 4px 0;
    }

    QComboBox::down-arrow {
        color: #e0e0e0;
    }

    /* 文本编辑框样式 */
    QTextEdit {
        background-color: #3a3a3a;
        color: #e0e0e0;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 4px;
        transition: all 0.2s ease;
    }

    QTextEdit:focus {
        border: 2px solid #4a6fa5;
        outline: none;
        background-color: #3f3f3f;
    }

    /* 标签页样式 */
    QTabWidget::pane {
        background-color: #333333;
        border: 1px solid #555555;
        border-radius: 8px;
        margin-top: 2px;
        padding: 10px;
    }

    QTabBar::tab {
        background-color: #3a3a3a;
        color: #e0e0e0;
        padding: 8px 16px;
        margin-right: 4px;
        border-radius: 6px 6px 0 0;
        min-width: 100px;
        text-align: center;
        transition: all 0.3s ease;
    }

    QTabBar::tab:selected {
        background-color: #4a6fa5;
        color: white;
        font-weight: bold;
        transform: translateY(-2px);
    }

    QTabBar::tab:hover:not(selected) {
        background-color: #444444;
        color: #ffffff;
        border-bottom: 2px solid #4a6fa5;
    }

    /* 框架样式 */
    QFrame {
        border: 1px solid #555555;
        border-radius: 8px;
        padding: 12px;
        background-color: #333333;
    }

    /* 进度条样式 */
    QProgressBar {
        background-color: #3a3a3a;
        border: 1px solid #555555;
        border-radius: 10px;
        text-align: center;
        height: 14px;
    }

    QProgressBar::chunk {
        background-color: #4a6fa5;
        border-radius: 10px;
        background-image: linear-gradient(to right, #4a6fa5, #5b7fb9);
    }

    /* 状态栏样式 */
    QStatusBar {
        background-color: #3a3a3a;
        color: #e0e0e0;
        padding: 4px 10px;
        border-top: 1px solid #555555;
    }

    /* 菜单样式 */
    QMenuBar {
        background-color: #3a3a3a;
        color: #e0e0e0;
        border-bottom: 1px solid #555555;
    }

    QMenuBar::item {
        background-color: #3a3a3a;
        color: #e0e0e0;
        padding: 6px 10px;
    }

    QMenuBar::item:selected {
        background-color: #4a6fa5;
        color: white;
    }

    QMenu {
        background-color: #3a3a3a;
        color: #e0e0e0;
        border: 1px solid #555555;
        border-radius: 4px;
    }

    QMenu::item {
        padding: 6px 20px;
    }

    QMenu::item:selected {
        background-color: #4a6fa5;
        color: white;
    }
    """
    window.setStyleSheet(qss)


def apply_light_theme(window):
    """应用浅色主题"""
    qss = """
    /* 全局样式 */
    QWidget {
        background-color: #f0f0f0;
        color: #333333;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 16px;
    }

    /* 按钮样式 */
    QPushButton {
        background-color: #4a6fa5;
        color: white;
        border-radius: 6px;
        padding: 8px 16px;
        border: none;
    }

    QPushButton:hover {
        background-color: #5b7fb9;
    }

    QPushButton:pressed {
        background-color: #3a5a89;
    }

    /* 输入框样式 */
    QLineEdit {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #cccccc;
        border-radius: 8px;
        padding: 8px;
    }

    QLineEdit:focus {
        border: 2px solid #4a6fa5;
        outline: none;
        background-color: #f5f5f5;
    }

    /* 下拉框样式 */
    QComboBox {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #cccccc;
        border-radius: 6px;
        padding: 6px;
    }

    QComboBox::drop-down {
        border: none;
        background-color: #e0e0e0;
        border-radius: 0 6px 6px 0;
    }

    QComboBox::down-arrow {
        color: #333333;
    }

    /* 文本编辑框样式 */
    QTextEdit {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #cccccc;
        border-radius: 6px;
        padding: 6px;
    }

    QTextEdit:focus {
        border: 1px solid #4a6fa5;
        outline: none;
    }

    /* 标签页样式 */
    QTabWidget::pane {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 8px;
        margin-top: 2px;
        padding: 10px;
    }

    QTabBar::tab {
        background-color: #e0e0e0;
        color: #333333;
        padding: 8px 16px;
        margin-right: 4px;
        border-radius: 6px 6px 0 0;
        min-width: 100px;
        text-align: center;
    }

    QTabBar::tab:selected {
        background-color: #4a6fa5;
        color: white;
        font-weight: bold;
    }

    QTabBar::tab:hover:not(selected) {
        background-color: #d0d0d0;
    }

    /* 框架样式 */
    QFrame {
        border: 1px solid #cccccc;
        border-radius: 8px;
        padding: 12px;
        background-color: #f5f5f5;
    }

    /* 进度条样式 */
    QProgressBar {
        background-color: #e0e0e0;
        border: 1px solid #cccccc;
        border-radius: 10px;
        text-align: center;
        height: 14px;
    }

    QProgressBar::chunk {
        background-color: #4a6fa5;
        border-radius: 10px;
        background-image: linear-gradient(to right, #4a6fa5, #5b7fb9);
    }

    /* 状态栏样式 */
    QStatusBar {
        background-color: #e0e0e0;
        color: #333333;
        padding: 4px 10px;
        border-top: 1px solid #cccccc;
    }

    /* 菜单样式 */
    QMenuBar {
        background-color: #e0e0e0;
        color: #333333;
        border-bottom: 1px solid #cccccc;
    }

    QMenuBar::item {
        background-color: #e0e0e0;
        color: #333333;
        padding: 6px 10px;
    }

    QMenuBar::item:selected {
        background-color: #4a6fa5;
        color: white;
    }

    QMenu {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #cccccc;
        border-radius: 6px;
    }

    QMenu::item {
        padding: 6px 20px;
    }

    QMenu::item:selected {
        background-color: #4a6fa5;
        color: white;
    }

    QTabBar::tab {
        background-color: #e0e0e0;
        color: #333333;
        padding: 6px 12px;
        margin-right: 2px;
        border-radius: 4px 4px 0 0;
    }

    QTabBar::tab:selected {
        background-color: #ffffff;
        color: #4a6fa5;
        border: 1px solid #cccccc;
        border-bottom: none;
    }

    QTabBar::tab:hover:not(selected) {
        background-color: #d0d0d0;
    }

    /* 框架样式 */
    QFrame {
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 8px;
    }

    /* 进度条样式 */
    QProgressBar {
        background-color: #e0e0e0;
        border: 1px solid #cccccc;
        border-radius: 4px;
        text-align: center;
    }

    QProgressBar::chunk {
        background-color: #4a6fa5;
        border-radius: 2px;
    }

    /* 状态栏样式 */
    QStatusBar {
        background-color: #e0e0e0;
        color: #333333;
    }
    """
    window.setStyleSheet(qss)