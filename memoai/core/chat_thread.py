from PyQt5.QtCore import QThread, pyqtSignal
import logging
import time
from memoai.inference.infer import MemoAIInferencer
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoAI-ChatThread")
class ChatThread(QThread):
    response_generated = pyqtSignal(str)
    def __init__(self, prompt, max_length=100, device="cpu", language="中文"):
        super().__init__()
        self.prompt = prompt
        self.max_length = max_length
        self.device = device
        self.language = language
    def run(self):
        try:
            logger.info(f"开始处理聊天请求: {self.prompt[:20]}...")
            inferencer = MemoAIInferencer()
            response = inferencer.generate_text(
                prompt=self.prompt,
                max_length=self.max_length
            )
            logger.info(f"聊天请求处理完成，生成响应: {response[:20]}...")
            self.response_generated.emit(response)
        except Exception as e:
            logger.error(f"处理聊天请求时出错: {str(e)}")
            error_msg = f"处理请求时出错: {str(e)}"
            self.response_generated.emit(error_msg)