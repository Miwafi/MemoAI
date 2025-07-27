import tkinter as tk
from ttkbootstrap import ttk
from language.translations import Translations
from modules.logger import log_event

class UIManager:
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.translations = Translations()
        self.current_language = '中文'
        self.create_widgets()

    def create_widgets(self):
        """创建UI组件"""
        # 设置样式
        self.style = ttk.Style()
        self._setup_fonts()
        self.style.configure('ChatFrame.TFrame', background='#f0f0f0')
        self.style.configure('UserMessage.TLabel', background='#0078d7', foreground='white')
        self.style.configure('AIMessage.TLabel', background='#e6e6e6', foreground='black')
        self.style.configure('SystemMessage.TLabel', background='#A7A7A7', foreground='black')  # 浅灰色背景，黑色文字
        self.style.configure('SystemSuccess.TLabel', background='#00cc66', foreground='white')   # 保留成功状态的绿色
        
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)


        # 右侧内容框架 - 垂直排列聊天区域和控制区域
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 聊天框架
        chat_frame = ttk.Frame(right_frame)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 聊天历史滚动区域
        self.chat_canvas = tk.Canvas(chat_frame)
        self.chat_scrollbar = ttk.Scrollbar(chat_frame, orient=tk.VERTICAL, command=self.chat_canvas.yview)
        self.chat_history = ttk.Frame(self.chat_canvas, style='ChatFrame.TFrame')
        
        self.chat_history.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        )
        
        self.chat_canvas.create_window((0, 0), window=self.chat_history, anchor="nw")
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)
        
        self.chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.chat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # 绑定鼠标滚轮事件
        self.chat_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # 输入框架
        input_frame = ttk.Frame(right_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        # 就绪状态标签
        self.status_label = ttk.Label(input_frame, text=self.get_text('READY_STATUS'), foreground="green")
        self.status_label.pack(side=tk.LEFT, padx=(0, 10))

        # 用户输入框（缩短并居中）
        self.user_input = ttk.Entry(input_frame, font=('SimHei', 10), width=40)  # 设置固定宽度
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", lambda e: self.send_message() if self.user_input['state'] == 'normal' else None)

        # 发送按钮
        self.send_btn = RoundedButton(input_frame, text=self.get_text('SEND_BTN'), command=lambda: self.send_message() if self.ai_state == 'idle' else None)
        self.send_btn.pack(side=tk.LEFT)
        
        # 状态和控制框架
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 进度条
        self.progress_bar = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # 功能按钮框架
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X)
        
        self.learn_btn = RoundedButton(btn_frame, text=self.get_text('SELF_LEARN_BTN'), command=self.start_self_learning)
        self.learn_btn.pack(side=tk.LEFT, padx=5)
        
        self.online_learn_btn = RoundedButton(btn_frame, text=self.get_text('ONLINE_LEARN_BTN'), command=self.start_online_learning)
        self.online_learn_btn.pack(side=tk.LEFT, padx=5)
        
        self.correct_btn = RoundedButton(btn_frame, text="手动纠错", command=self.open_correction_window)
        self.correct_btn.pack(side=tk.LEFT, padx=5)
        
        # 添加复制AI输出按钮
        self.copy_ai_btn = RoundedButton(btn_frame, text="复制AI输出", command=self.copy_last_ai_response)
        self.copy_ai_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = RoundedButton(btn_frame, text="清除对话", command=self.clear_chat)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.settings_btn = RoundedButton(btn_frame, text="设置", command=self.open_settings_window)
        self.settings_btn.pack(side=tk.LEFT, padx=5)
        self.quit_btn = RoundedButton(btn_frame, text="退出", command=self.quit_app)
        self.quit_btn.pack(side=tk.RIGHT, padx=5)
    


    def update_language(self, lang):
        self.current_language = lang
        self.translations.set_language(lang)
        self.update_ui_texts()
        self.current_language = lang
        self.translations.set_language(lang)
        self.update_ui_texts()

    # 添加鼠标滚轮事件处理方法
    def _on_mousewheel(self, event):
        self.chat_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        return "break"