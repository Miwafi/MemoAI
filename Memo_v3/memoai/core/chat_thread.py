"""
MemoAI 聊天线程模块
用于在后台线程中处理聊天请求，避免阻塞UI
"""
from PyQt5.QtCore import QThread, pyqtSignal
import logging
import time
from memoai.inference.infer import MemoAIInferencer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoAI-ChatThread")

class ChatThread(QThread):
    """聊天线程类
    在后台线程中处理聊天请求，生成响应
    """
    response_generated = pyqtSignal(str)

    def __init__(self, prompt, max_length=100, device="cpu", language="中文"):
        """初始化聊天线程
        Args:
            prompt: 用户输入的提示文本
            max_length: 生成文本的最大长度
            device: 运行设备 (cpu/gpu)
            language: 语言设置
        """
        super().__init__()
        self.prompt = prompt
        self.max_length = max_length
        self.device = device
        self.language = language

    def run(self):
        """运行聊天线程，处理请求并生成响应
        使用优化的推理接口进行实际推理
        """
        try:
            logger.info(f"开始处理聊天请求: {self.prompt[:20]}...")

            # 创建推理器实例并生成响应
            inferencer = MemoAIInferencer()
            response = inferencer.generate_text(
                prompt=self.prompt,
                max_length=self.max_length
            )

            logger.info(f"聊天请求处理完成，生成响应: {response[:20]}...")
            # 发送生成的响应
            self.response_generated.emit(response)
        except Exception as e:
            logger.error(f"处理聊天请求时出错: {str(e)}")
            # 发送错误信息
            error_msg = f"处理请求时出错: {str(e)}"
            self.response_generated.emit(error_msg)