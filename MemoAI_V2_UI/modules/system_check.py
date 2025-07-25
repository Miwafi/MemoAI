from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
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

    # ... 实现具体检查方法 ...
from modules.logger import log_event
import os
import json

class SystemChecker:
    @staticmethod
    def get_config():
        config_path = os.path.join(os.path.dirname(__file__), '..')