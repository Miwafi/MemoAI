import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.logger import log_event
import json

class SystemChecker:
    @staticmethod
    def get_config():
        config_path = os.path.join(os.path.dirname(__file__), '..', 'settings', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def run_full_check():
        config = SystemChecker.get_config()
        log_event("开始系统自检")
        checks = [
            SystemChecker.check_directories,
            SystemChecker.check_dependencies,
            SystemChecker.check_model_files
        ]
        
        for check in checks:
            if not check():
                log_event("系统自检失败", 'error')
                return False
        
        log_event("系统自检完成")
        return True

    @staticmethod
    def check_directories():
        config = SystemChecker.get_config()
        required_dirs = [
            config.get('data_dir', 'data'),
            config.get('model_dir', 'models'),
            config.get('log_dir', 'logs')
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    log_event(f"创建目录成功: {dir_path}")
                except Exception as e:
                    log_event(f"创建目录失败: {dir_path}, 错误: {str(e)}", 'error')
                    return False
            if not os.path.isdir(dir_path):
                log_event(f"路径不是目录: {dir_path}", 'error')
                return False
        return True

    @staticmethod
    def check_dependencies():
        required_packages = [
            ('torch', 'torch'),
            ('tkinter', 'tkinter'),
            ('ttkbootstrap', 'ttkbootstrap'),
            ('json', 'json'),
            ('os', 'os')
        ]
        
        for package_name, import_name in required_packages:
            try:
                __import__(import_name)
            except ImportError:
                log_event(f"缺少依赖包: {package_name}", 'error')
                return False
        return True
    
    @staticmethod
    def check_model_files():
        config = SystemChecker.get_config()
        model_path = config.get('model_path', 'model/dialog_model.pth')
        vocab_path = config.get('vocab_path', 'model/vocab.json')
        
        if not os.path.exists(model_path):
            log_event(f"模型文件不存在: {model_path}", 'error')
            return False
        if not os.path.exists(vocab_path):
            log_event(f"词汇表文件不存在: {vocab_path}", 'error')
            return False
        return True