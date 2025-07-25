import json
import os

class Translations:
    def __init__(self):
        self.translations = {}
        self.current_language = '中文'
        self.load_translations()

    def load_translations(self):
        """加载翻译文件内容"""
        json_path = os.path.join(os.path.dirname(__file__), 'translations.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            self.translations = json.load(f)

    def set_language(self, lang):
        """设置当前语言"""
        if lang in self.translations:
            self.current_language = lang
            return True
        return False

    def get(self, key, default=None):
        """获取翻译文本"""
        lang_dict = self.translations.get(self.current_language, {})
        return lang_dict.get(key, default or key)