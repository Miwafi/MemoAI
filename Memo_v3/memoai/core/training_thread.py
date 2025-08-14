"""
MemoAI 训练线程模块
用于在后台线程中处理训练请求，避免阻塞UI
"""
from PyQt5.QtCore import QThread, pyqtSignal
import logging
import time
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoAI-TrainingThread")

class TrainingThread(QThread):
    """训练线程类
    在后台线程中处理训练请求，执行模型训练
    """
    progress_updated = pyqtSignal(int)
    training_finished = pyqtSignal(bool, str)

    def __init__(self, model_name, epochs, lr):
        """初始化训练线程
        Args:
            model_name: 模型名称
            epochs: 训练轮次
            lr: 学习率
        """
        super().__init__()
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.is_running = True

    def run(self):
        """运行训练线程，执行训练过程
        """
        try:
            logger.info(f"开始训练模型: {self.model_name}, 训练轮次: {self.epochs}, 学习率: {self.lr}")

            # 模拟训练过程
            total_steps = self.epochs * 10  # 假设每个epoch有10个steps
            for step in range(total_steps):
                if not self.is_running:
                    logger.info("训练已停止")
                    self.training_finished.emit(False, "训练已停止")
                    return

                # 模拟训练进度
                progress = int((step + 1) / total_steps * 100)
                self.progress_updated.emit(progress)

                # 模拟训练耗时
                time.sleep(0.2)  # 每步耗时0.2秒

                # 随机模拟训练失败
                if random.random() < 0.01:  # 1%的概率失败
                    raise Exception("模拟训练过程中出现随机错误")

            logger.info(f"模型 {self.model_name} 训练完成！")
            self.training_finished.emit(True, f"模型 {self.model_name} 训练成功完成！\n训练轮次: {self.epochs}\n学习率: {self.lr}")

        except Exception as e:
            logger.error(f"训练模型时出错: {str(e)}")
            self.training_finished.emit(False, f"训练出错: {str(e)}")

    def stop_training(self):
        """停止训练
        """
        self.is_running = False