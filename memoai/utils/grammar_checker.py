import logging
import time
import atexit
from language_tool_python import LanguageTool
_checker_instances = []
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoAI-GrammarChecker")
class GrammarChecker:
    def __init__(self, language='zh-CN'):
        self.tool = None
        self.language = None
        self.closed = False
        try:
            self.tool = LanguageTool(language)
            self.language = language
            logger.info(f"成功加载语法检查工具，语言: {language}")
            global _checker_instances
            _checker_instances.append(self)
        except Exception as e:
            logger.error(f"加载语法检查工具失败: {str(e)}")
            self.tool = None
            self.language = None
    def close(self):
        if not self.closed and self.tool is not None:
            try:
                self.tool = None
                self.closed = True
                logger.info("语法检查工具资源已手动释放")
                global _checker_instances
                if self in _checker_instances:
                    _checker_instances.remove(self)
            except Exception as e:
                logger.error(f"释放语法检查工具资源时出错: {str(e)}")
    def __del__(self):
        try:
            if not self.closed and self.tool is not None:
                self.tool = None
                self.closed = True
                logger.info("语法检查工具资源已在析构时释放")
                global _checker_instances
                if self in _checker_instances:
                    _checker_instances.remove(self)
        except Exception as e:
            pass
    def check_grammar(self, text):
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
        if not self.tool:
            logger.warning("语法检查工具未初始化，无法修正文本")
            return text
        try:
            start_time = time.time()
            matches = self.tool.check(text)
            corrections = []
            for match in reversed(matches):
                if match.replacements:
                    start = match.offset
                    end = start + match.errorLength
                    corrections.append((start, end, match.replacements[0]))
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
def _cleanup_checkers():
    global _checker_instances
    for checker in _checker_instances[:]:
        try:
            checker.close()
        except Exception as e:
            logger.error(f"清理语法检查器时出错: {str(e)}")
    _checker_instances = []
atexit.register(_cleanup_checkers)
_grammar_checker = None
def get_grammar_checker(language='zh-CN'):
    global _grammar_checker
    if _grammar_checker is None or _grammar_checker.language != language:
        _grammar_checker = GrammarChecker(language)
    return _grammar_checker