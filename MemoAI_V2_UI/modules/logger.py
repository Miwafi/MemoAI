import datetime
import os

def log_event(message, level='info'):
    """记录事件到日志"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f'[{timestamp}] [{level.upper()}] {message}\n'
    
    # 获取配置中的日志目录（从配置文件读取）
    config_path = os.path.join(os.path.dirname(__file__), '..', 'settings', 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        import json
        config = json.load(f)
        log_dir = config.get('log_dir', 'log')
    
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 写入日志文件
    log_file = os.path.join(log_dir, 'app.log')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)
    
    # 打印到控制台
    print(log_entry.strip())