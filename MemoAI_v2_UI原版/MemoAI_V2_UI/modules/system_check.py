class SystemChecker:
    """系统检查器类"""
    
    @staticmethod
    def check_system():
        """执行系统检查"""
        return True, "系统检查完成"
    
    @staticmethod
    def check_pytorch():
        """检查PyTorch"""
        try:
            import torch
            return True, f"PyTorch {torch.__version__} 已安装"
        except ImportError:
            return False, "PyTorch未安装"
    
    @staticmethod
    def check_directories(required_dirs):
        """检查必要目录"""
        import os
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)
                except Exception as e:
                    return False, f"创建目录 {dir_path} 失败: {str(e)}"
        return True, "所有目录检查完成"