import requests
from bs4 import BeautifulSoup
from modules.logger import log_event

class NetworkManager:
    @staticmethod
    def fetch_url(url, timeout=10):
        # 检查是否允许从HuggingFace获取数据
        from settings.config import config
        if 'huggingface.co' in url and not config.fetch_from_huggingface:
            log_event("已阻止从HuggingFace获取数据（用户已禁用）", 'warning')
            return None
        
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            log_event(f"网络请求失败: {str(e)}", 'error')
            return None

    @staticmethod
    def parse_html(html, selector=None):
        if not html:
            return None
        soup = BeautifulSoup(html, 'html.parser')
        return soup.select(selector) if selector else soup