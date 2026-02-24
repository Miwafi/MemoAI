from PyQt5.QtCore import QThread, pyqtSignal
import logging
import time
import random
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoAI-TrainingThread")
class TrainingThread(QThread):
    progress_updated = pyqtSignal(int)
    training_finished = pyqtSignal(bool, str)
    def __init__(self, model_name, epochs, lr):
        super().__init__()
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.is_running = True
    def run(self):
        try:
            logger.info(f"开始训练模型: {self.model_name}, 训练轮次: {self.epochs}, 学习率: {self.lr}")
            total_steps = self.epochs * 10
            for step in range(total_steps):
                if not self.is_running:
                    logger.info("训练已停止")
                    self.training_finished.emit(False, "训练已停止")
                    return
                progress = int((step + 1) / total_steps * 100)
                self.progress_updated.emit(progress)
                time.sleep(0.2)
                if random.random() < 0.01:
                    raise Exception("模拟训练过程中出现随机错误")
            logger.info(f"模型 {self.model_name} 训练完成！")
            self.training_finished.emit(True, f"模型 {self.model_name} 训练成功完成！\n训练轮次: {self.epochs}\n学习率: {self.lr}")
        except Exception as e:
            logger.error(f"训练模型时出错: {str(e)}")
            self.training_finished.emit(False, f"训练出错: {str(e)}")
    def stop_training(self):
        self.is_running = False