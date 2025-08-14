"""
MemoAI 工具模块
包含应用的通用工具函数
"""
from PyQt5.QtWidgets import QMenu, QAction
from PyQt5.QtCore import Qt


def create_context_menu(widget, language='中文'):
    """为控件创建右键菜单"""
    # 启用右键菜单
    widget.setContextMenuPolicy(Qt.CustomContextMenu)

    # 创建菜单
    def show_context_menu(position):
        context_menu = QMenu(widget)

        # 根据语言设置菜单文本
        if language == '中文' or language == '梗体中文':
            copy_text = "复制"
            cut_text = "剪切"
            paste_text = "粘贴"
            select_all_text = "全选"
        elif language == 'English':
            copy_text = "Copy"
            cut_text = "Cut"
            paste_text = "Paste"
            select_all_text = "Select All"
        elif language == '日本語':
            copy_text = "コピー"
            cut_text = "カット"
            paste_text = "貼り付け"
            select_all_text = "すべて選択"
        elif language == 'Français':
            copy_text = "Copier"
            cut_text = "Couper"
            paste_text = "Coller"
            select_all_text = "Sélectionner tout"
        elif language == 'Español':
            copy_text = "Copiar"
            cut_text = "Cortar"
            paste_text = "Pegar"
            select_all_text = "Seleccionar todo"
        else:
            copy_text = "复制"
            cut_text = "剪切"
            paste_text = "粘贴"
            select_all_text = "全选"

        # 添加复制、剪切、粘贴等操作
        copy_action = QAction(copy_text, widget)
        copy_action.triggered.connect(widget.copy)

        cut_action = QAction(cut_text, widget)
        cut_action.triggered.connect(widget.cut)

        paste_action = QAction(paste_text, widget)
        paste_action.triggered.connect(widget.paste)

        # 添加分隔线
        context_menu.addSeparator()

        # 添加全选操作
        select_all_action = QAction(select_all_text, widget)
        select_all_action.triggered.connect(widget.selectAll)

        # 添加操作到菜单
        context_menu.addAction(copy_action)
        context_menu.addAction(cut_action)
        context_menu.addAction(paste_action)
        context_menu.addSeparator()
        context_menu.addAction(select_all_action)

        # 显示菜单
        context_menu.exec_(widget.mapToGlobal(position))

    # 连接信号
    widget.customContextMenuRequested.connect(show_context_menu)

    return show_context_menu


def setup_language_menu(main_window):
    """设置语言菜单"""
    language_menu = QMenu("语言", main_window)

    # 添加语言选项
    chinese_action = QAction("中文", main_window)
    chinese_action.triggered.connect(lambda: main_window.settings_tab.language_combo.setCurrentText("中文"))

    english_action = QAction("English", main_window)
    english_action.triggered.connect(lambda: main_window.settings_tab.language_combo.setCurrentText("English"))

    meme_chinese_action = QAction("梗体中文", main_window)
    meme_chinese_action.triggered.connect(lambda: main_window.settings_tab.language_combo.setCurrentText("梗体中文"))

    japanese_action = QAction("日本語", main_window)
    japanese_action.triggered.connect(lambda: main_window.settings_tab.language_combo.setCurrentText("日本語"))

    korean_action = QAction("한국어", main_window)
    korean_action.triggered.connect(lambda: main_window.settings_tab.language_combo.setCurrentText("한국어"))

    french_action = QAction("Français", main_window)
    french_action.triggered.connect(lambda: main_window.settings_tab.language_combo.setCurrentText("Français"))

    spanish_action = QAction("Español", main_window)
    spanish_action.triggered.connect(lambda: main_window.settings_tab.language_combo.setCurrentText("Español"))

    # 添加到菜单
    language_menu.addAction(chinese_action)
    language_menu.addAction(english_action)
    language_menu.addAction(meme_chinese_action)
    language_menu.addAction(japanese_action)
    language_menu.addAction(french_action)
    language_menu.addAction(spanish_action)

    return language_menu