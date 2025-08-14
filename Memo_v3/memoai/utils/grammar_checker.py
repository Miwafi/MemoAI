# -*- coding: utf-8 -*-
"""
语法检查工具
使用language-tool-python库进行文本语法检查和修正
"""
import logging
import time
import atexit
from language_tool_python import LanguageTool

# 跟踪所有创建的语法检查器实例
_checker_instances = []

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoAI-GrammarChecker")

class GrammarChecker:
    """语法检查器
    使用language-tool-python库提供语法检查和修正功能
    支持多种语言，默认使用中文
    """
    def __init__(self, language='zh-CN'):
        """初始化语法检查器
        Args:
            language: 语言代码，默认为'zh-CN'(简体中文)
        """
        self.tool = None
        self.language = None
        self.closed = False
        try:
            self.tool = LanguageTool(language)
            self.language = language
            logger.info(f"成功加载语法检查工具，语言: {language}")
            # 将实例添加到全局跟踪列表
            global _checker_instances
            _checker_instances.append(self)
        except Exception as e:
            logger.error(f"加载语法检查工具失败: {str(e)}")
            self.tool = None
            self.language = None

    def close(self):
        """显式关闭语法检查器，释放资源"""
        if not self.closed and self.tool is not None:
            try:
                # 直接设置tool为None而不调用其close方法
                # 这避免了LanguageTool内部可能导致线程问题的清理逻辑
                self.tool = None
                self.closed = True
                logger.info("语法检查工具资源已手动释放")
                # 从全局跟踪列表中移除
                global _checker_instances
                if self in _checker_instances:
                    _checker_instances.remove(self)
            except Exception as e:
                logger.error(f"释放语法检查工具资源时出错: {str(e)}")

    def __del__(self):
        """析构函数，确保资源正确释放"""
        # 避免在解释器关闭时尝试创建新线程
        try:
            # 检查是否已经关闭
            if not self.closed and self.tool is not None:
                # 不尝试调用close()，因为这可能会创建新线程
                # 只是简单地设置为None
                self.tool = None
                self.closed = True
                logger.info("语法检查工具资源已在析构时释放")
                # 从全局跟踪列表中移除
                global _checker_instances
                if self in _checker_instances:
                    _checker_instances.remove(self)
        except Exception as e:
            # 在析构函数中，我们不能做太多事情，只能记录错误
            pass

    def check_grammar(self, text):
        """检查文本语法
        Args:
            text: 要检查的文本
        Returns:
            list: 语法错误列表，每个错误包含详细信息
        """
        if not self.tool:
            logger.warning("语法检查工具未初始化，无法检查语法")
            return []

        try:
            start_time = time.time()
            matches = self.tool.check(text)
            end_time = time.time()
            logger.info(f"发现 {len(matches)} 个语法问题，耗时: {end_time - start_time:.4f}秒")
            return matches
        except Exception as e:
            logger.error(f"检查语法时出错: {str(e)}")
            return []

    def correct_text(self, text):
        """修正文本语法错误
        Args:
            text: 要修正的文本
        Returns:
            str: 修正后的文本
        """
        if not self.tool:
            logger.warning("语法检查工具未初始化，无法修正文本")
            return text

        try:
            start_time = time.time()
            # 检查语法错误
            matches = self.tool.check(text)
            # 从后往前修正，避免索引偏移
            corrections = []
            for match in reversed(matches):
                # 只使用第一个建议的修正
                if match.replacements:
                    start = match.offset
                    end = start + match.errorLength
                    corrections.append((start, end, match.replacements[0]))

            # 应用修正
            corrected_text = list(text)
            for start, end, replacement in corrections:
                corrected_text[start:end] = list(replacement)

            corrected_text = ''.join(corrected_text)
            end_time = time.time()
            logger.info(f"文本语法修正完成，应用了 {len(corrections)} 处修正，耗时: {end_time - start_time:.4f}秒")
            return corrected_text
        except Exception as e:
            logger.error(f"修正文本时出错: {str(e)}")
            return text

# 注册程序退出时的清理函数
def _cleanup_checkers():
    """程序退出时清理所有语法检查器实例"""
    global _checker_instances
    for checker in _checker_instances[:]:  # 使用副本避免修改迭代中的列表
        try:
            checker.close()
        except Exception as e:
            logger.error(f"清理语法检查器时出错: {str(e)}")
    _checker_instances = []

atexit.register(_cleanup_checkers)

# 单例模式，避免重复初始化
_grammar_checker = None

def get_grammar_checker(language='zh-CN'):
    """获取语法检查器实例
        Args:
            language: 语言代码
        Returns:
            GrammarChecker: 语法检查器实例
        单例模式，确保全局只有一个语法检查器实例
        """
    global _grammar_checker
    if _grammar_checker is None or _grammar_checker.language != language:
        _grammar_checker = GrammarChecker(language)
    return _grammar_checker