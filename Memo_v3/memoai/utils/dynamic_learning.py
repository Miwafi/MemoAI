# -*- coding: utf-8 -*-
"""
动态学习模块
负责记录用户交互模式并调整AI的响应逻辑
"""
import os
import json
import logging
import numpy as np
import time
import threading
import importlib
import copy
from collections import defaultdict, deque

# 用于存储动态加载的模块
_dynamic_modules = {}
# 线程锁，确保动态加载模块时的线程安全
_module_lock = threading.Lock()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemoAI-DynamicLearning")

class DynamicLearner:
    """动态学习器
    记录用户输入模式, 分析交互数据, 并调整AI的响应策略
    """
    def __init__(self, learning_rate=0.1, history_size=1000):
        """初始化动态学习器
        Args:
            learning_rate: 学习率, 控制参数调整的幅度
            history_size: 历史记录大小
        """
        self.learning_rate = learning_rate
        self.history_size = history_size
        
        # 存储交互历史
        self.interaction_history = deque(maxlen=history_size)
        
        # 存储用户输入模式分析结果
        self.input_patterns = defaultdict(int)
        
        # 存储调整后的参数
        self.adjusted_params = {
            'temperature': 0.7,  # 初始温度参数
            'top_k': 50,         # 初始top_k参数
            'top_p': 0.95        # 初始top_p参数
        }
        
        # 学习标志, 控制是否进行学习
        self.is_learning = True
        
        # 加载之前保存的学习数据（如果存在）
        self.load_learning_data()
        
        logger.info("动态学习器初始化完成")
    
    def record_interaction(self, user_input, ai_response):
        """记录用户交互
        Args:
            user_input: 用户输入
            ai_response: AI响应
        """
        if not self.is_learning:
            return
            
        # 记录交互时间
        timestamp = time.time()
        
        # 分析用户输入模式
        self._analyze_input_pattern(user_input)
        
        # 保存交互历史
        interaction = {
            'timestamp': timestamp,
            'user_input': user_input,
            'ai_response': ai_response,
            'params': self.adjusted_params.copy()
        }
        self.interaction_history.append(interaction)
        
        # 每10次交互进行一次参数调整
        if len(self.interaction_history) % 10 == 0:
            self._adjust_parameters()
        
        # 定期保存学习数据
        if len(self.interaction_history) % 50 == 0:
            self.save_learning_data()
    
    def _analyze_input_pattern(self, user_input):
        """分析用户输入模式
        Args:
            user_input: 用户输入
        多维度分析用户输入，提取模式特征
        """
        # 多维度模式分析: 统计关键词、问题类型、情感倾向等
        
        # 1. 统计输入长度 - 短小精悍还是长篇大论？
        input_length = len(user_input)
        length_category = 'short' if input_length < 10 else ('medium' if input_length < 50 else 'long')
        self.input_patterns[f'length_{length_category}'] += 1
        
        # 2. 问题检测 - 用户是不是十万个为什么？
        has_question = '?' in user_input or '？' in user_input
        if has_question:
            self.input_patterns['question'] += 1
            
            # 细分问题类型 - 是what/why/how/when/where？
            question_types = {
                'what': ['什么', '哪些', '哪', '哪个', '哪些'],
                'why': ['为什么', '为何', '为啥', '何故'],
                'how': ['怎么', '如何', '怎样', '咋样'],
                'when': ['什么时候', '何时', '啥时候'],
                'where': ['哪里', '哪儿', '何处']
            }
            
            for q_type, keywords in question_types.items():
                for keyword in keywords:
                    if keyword in user_input:
                        self.input_patterns[f'question_{q_type}'] += 1
                        break
        
        # 3. 情感倾向分析 - 用户今天心情如何？
        positive_words = ['好', '不错', '优秀', '开心', '高兴', '满意', '喜欢']
        negative_words = ['不好', '糟糕', '失望', '生气', '讨厌', '不满意']
        
        for word in positive_words:
            if word in user_input:
                self.input_patterns['sentiment_positive'] += 1
                break
        
        for word in negative_words:
            if word in user_input:
                self.input_patterns['sentiment_negative'] += 1
                break
        
        # 4. 实体识别 - 用户在谈论什么？
        entities = {
            'person': ['我', '你', '他', '她', '我们', '你们', '他们'],
            'technology': ['AI', '人工智能', '模型', '算法', '程序', '代码'],
            'product': ['MemoAI', '助手', '工具', '软件', '应用'],
        }
        
        for entity_type, entity_words in entities.items():
            for word in entity_words:
                if word in user_input:
                    self.input_patterns[f'entity_{entity_type}'] += 1
                    break
        
        # 5. 统计常见关键词 - 高频词汇大发现
        keywords = ['你好', '名字', '是什么', '为什么', '怎么', '如何', '哪里', '什么时候', '谢谢']
        for keyword in keywords:
            if keyword in user_input:
                self.input_patterns[f'keyword_{keyword}'] += 1

    def _deep_analytics(self):
        """深度数据分析
        对存储的交互历史进行深度分析，提取有价值的模式和趋势
        """
        if not self.interaction_history or len(self.interaction_history) < 20:
            return
            
        logger.info("开始深度数据分析")
        
        # 1. 分析响应时间与用户满意度的关系
        response_times = []
        positive_responses = []
        
        for interaction in self.interaction_history:
            # 假设响应时间越短，用户越满意
            # 这里简化处理，实际应用中可以结合用户反馈
            response_time = 1.0  # 占位，实际应计算真实响应时间
            response_times.append(response_time)
            
            # 简单判断用户是否满意
            is_positive = '好' in interaction['user_input'] or '满意' in interaction['user_input']
            positive_responses.append(is_positive)
        
        # 计算相关性 (这里简化处理)
        if len(response_times) > 1 and len(positive_responses) > 1:
            # 实际应用中可以使用更复杂的统计方法
            avg_response_time = sum(response_times) / len(response_times)
            avg_positive = sum(positive_responses) / len(positive_responses)
            
            logger.info(f"深度分析结果: 平均响应时间 {avg_response_time:.2f}秒, 用户满意度 {avg_positive:.2%}")
        
        # 2. 分析参数调整与响应质量的关系
        param_quality = defaultdict(list)
        
        for interaction in self.interaction_history:
            # 假设生成的文本越长，质量越高
            # 实际应用中可以使用更复杂的评估指标
            quality_score = len(interaction['ai_response']) / 100
            
            for param, value in interaction['params'].items():
                param_quality[param].append((value, quality_score))
        
        # 分析参数与质量的关系
        for param, values in param_quality.items():
            if len(values) < 5:
                continue
                
            # 计算平均值
            avg_value = sum(v for v, _ in values) / len(values)
            avg_quality = sum(q for _, q in values) / len(values)
            
            logger.info(f"参数 '{param}' 分析: 平均值 {avg_value:.2f}, 平均质量分 {avg_quality:.2f}")
        
        logger.info("深度数据分析完成")
    
    def _adjust_parameters(self):
        """调整模型参数
        根据分析结果调整生成文本的参数
        """
        if not self.is_learning:
            return
            
        logger.info("开始调整模型参数")
        
        # 先执行深度数据分析
        self._deep_analytics()
        
        # 基于多维度分析结果调整参数
        
        # 1. 根据问题类型调整温度
        question_ratio = self.input_patterns.get('question', 0) / len(self.interaction_history) if self.interaction_history else 0
        
        if question_ratio > 0.5:
            # 问题类输入较多，增加温度以提高回答的多样性
            self.adjusted_params['temperature'] = min(1.0, self.adjusted_params['temperature'] + self.learning_rate * 0.2)
            logger.info(f"问题类输入较多 ({question_ratio:.2f}), 提高温度参数至 {self.adjusted_params['temperature']:.2f}")
        elif question_ratio < 0.2:
            # 问题类输入较少，降低温度以提高回答的确定性
            self.adjusted_params['temperature'] = max(0.5, self.adjusted_params['temperature'] - self.learning_rate * 0.1)
            logger.info(f"问题类输入较少 ({question_ratio:.2f}), 降低温度参数至 {self.adjusted_params['temperature']:.2f}")
        
        # 2. 根据问题细分类型调整top_p
        why_ratio = self.input_patterns.get('question_why', 0) / self.input_patterns.get('question', 1) if self.input_patterns.get('question', 0) > 0 else 0
        
        if why_ratio > 0.3:
            # '为什么'类问题较多，降低top_p以提高回答的精确性
            self.adjusted_params['top_p'] = max(0.7, self.adjusted_params['top_p'] - self.learning_rate * 0.1)
            logger.info(f"'为什么'类问题较多 ({why_ratio:.2f}), 降低top_p参数至 {self.adjusted_params['top_p']:.2f}")
        
        how_ratio = self.input_patterns.get('question_how', 0) / self.input_patterns.get('question', 1) if self.input_patterns.get('question', 0) > 0 else 0
        
        if how_ratio > 0.3:
            # '如何'类问题较多，提高top_p以提高回答的全面性
            self.adjusted_params['top_p'] = min(0.95, self.adjusted_params['top_p'] + self.learning_rate * 0.1)
            logger.info(f"'如何'类问题较多 ({how_ratio:.2f}), 提高top_p参数至 {self.adjusted_params['top_p']:.2f}")
        
        # 3. 根据输入长度调整top_k
        long_input_ratio = self.input_patterns.get('length_long', 0) / len(self.interaction_history) if self.interaction_history else 0
        short_input_ratio = self.input_patterns.get('length_short', 0) / len(self.interaction_history) if self.interaction_history else 0
        
        if long_input_ratio > 0.3:
            # 长文本输入较多，增加top_k以提高连贯性
            self.adjusted_params['top_k'] = min(100, self.adjusted_params['top_k'] + int(self.learning_rate * 10))
            logger.info(f"长文本输入较多 ({long_input_ratio:.2f}), 提高top_k参数至 {self.adjusted_params['top_k']}")
        elif short_input_ratio > 0.5:
            # 短文本输入较多，降低top_k以提高响应速度
            self.adjusted_params['top_k'] = max(20, self.adjusted_params['top_k'] - int(self.learning_rate * 10))
            logger.info(f"短文本输入较多 ({short_input_ratio:.2f}), 降低top_k参数至 {self.adjusted_params['top_k']}")
        
        # 4. 根据情感倾向调整重复惩罚
        positive_ratio = self.input_patterns.get('sentiment_positive', 0) / len(self.interaction_history) if self.interaction_history else 0
        negative_ratio = self.input_patterns.get('sentiment_negative', 0) / len(self.interaction_history) if self.interaction_history else 0
        
        if negative_ratio > 0.2:
            # 负面情绪较多，增加重复惩罚以避免啰嗦
            self.adjusted_params['repetition_penalty'] = min(2.0, self.adjusted_params.get('repetition_penalty', 1.2) + self.learning_rate * 0.2)
            logger.info(f"负面情绪较多 ({negative_ratio:.2f}), 提高重复惩罚参数至 {self.adjusted_params['repetition_penalty']:.2f}")
        elif positive_ratio > 0.5:
            # 正面情绪较多，降低重复惩罚以保持自然
            self.adjusted_params['repetition_penalty'] = max(1.0, self.adjusted_params.get('repetition_penalty', 1.2) - self.learning_rate * 0.1)
            logger.info(f"正面情绪较多 ({positive_ratio:.2f}), 降低重复惩罚参数至 {self.adjusted_params['repetition_penalty']:.2f}")
        
        logger.info(f"参数调整完成: {self.adjusted_params}")
    
    def get_adjusted_params(self):
        """获取调整后的参数
        Returns:
            dict: 调整后的参数
        """
        return self.adjusted_params.copy()
    
    def get_current_params(self):
        """获取当前参数
        Returns:
            dict: 当前参数
        为兼容旧代码添加的方法
        """
        return self.get_adjusted_params()
    
    def _get_data_path(self):
        """获取数据保存路径
        Returns:
            str: 数据保存路径
        """
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'learning')
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, 'learning_data.json')

    @staticmethod
    def load_dynamic_module(module_path):
        """动态加载模块
        支持运行时加载和更新分析模块
        
        Args:
            module_path: 模块路径
            
        Returns:
            module: 加载的模块
        """
        global _dynamic_modules, _module_lock
        module_name = os.path.basename(module_path).replace('.py', '')
        
        with _module_lock:
            try:
                # 如果模块已加载，先卸载
                if module_name in _dynamic_modules:
                    importlib.reload(_dynamic_modules[module_name])
                else:
                    # 动态导入模块
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    _dynamic_modules[module_name] = module
                    spec.loader.exec_module(module)
                return _dynamic_modules[module_name]
            except Exception as e:
                logging.error(f"动态加载模块失败: {str(e)}")
                return None

    def update_analytics_logic(self, module_path):
        """更新分析逻辑
        在运行时更新数据分析方法，实现动态自我升级
        
        Args:
            module_path: 包含新分析方法的模块路径
            
        Returns:
            bool: 更新是否成功
        """
        try:
            module = self.load_dynamic_module(module_path)
            if module is None:
                return False
            
            # 保存旧方法的引用，用于回滚
            old_methods = {
                '_analyze_input_pattern': getattr(self, '_analyze_input_pattern', None),
                '_deep_analytics': getattr(self, '_deep_analytics', None)
            }
            
            # 尝试更新方法
            if hasattr(module, 'new_analyze_input_pattern'):
                self._analyze_input_pattern = module.new_analyze_input_pattern.__get__(self)
                logging.info("成功更新输入模式分析方法")
            
            if hasattr(module, 'new_deep_analytics'):
                self._deep_analytics = module.new_deep_analytics.__get__(self)
                logging.info("成功更新深度分析方法")
            
            return True
        except Exception as e:
            logging.error(f"更新分析逻辑失败: {str(e)}")
            # 回滚到旧方法
            for name, method in old_methods.items():
                if method is not None:
                    setattr(self, name, method)
            return False

    def save_learning_data(self):
        """保存学习数据
        将交互历史、分析结果和优化模型保存到文件
        """
        data = {
            'interaction_history': list(self.interaction_history),
            'input_patterns': dict(self.input_patterns),
            'adjusted_params': self.adjusted_params,
            'learning_rate': self.learning_rate,
            'history_size': self.history_size,
            'last_optimization_time': time.time(),
            'total_interactions': len(self.interaction_history)
        }
        
        try:
            # 确保目录存在
            data_dir = os.path.dirname(self._get_data_path())
            os.makedirs(data_dir, exist_ok=True)
            
            # 保存主数据文件
            with open(self._get_data_path(), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"学习数据保存成功: {self._get_data_path()}")
            
            # 创建备份 - 防止数据损坏
            backup_path = os.path.join(data_dir, f'learning_data_backup_{int(time.time())}.json')
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"学习数据备份成功: {backup_path} - 存档已创建！")
            
            # 保存优化模型参数到单独文件，便于快速加载
            params_path = os.path.join(data_dir, 'optimized_params.json')
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(self.adjusted_params, f, ensure_ascii=False, indent=2)
            logger.info(f"优化参数保存成功: {params_path} - 最佳配置已导出！")
        except Exception as e:
            logger.error(f"保存学习数据时出错: {str(e)} - 日记写入失败！")
    
    def load_learning_data(self):
        """加载学习数据
        从文件加载之前保存的交互历史和分析结果
        就像AI在'复习日记', 回忆之前的学习历程 - 作者: Pyro"""
        data_path = self._get_data_path()
        if not os.path.exists(data_path):
            logger.info(f"学习数据文件不存在: {data_path} - 没有旧日记可看！")
            return
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 恢复数据
            self.interaction_history = deque(data.get('interaction_history', []), maxlen=self.history_size)
            self.input_patterns = defaultdict(int, data.get('input_patterns', {}))
            self.adjusted_params = data.get('adjusted_params', self.adjusted_params)
            self.learning_rate = data.get('learning_rate', self.learning_rate)
            self.history_size = data.get('history_size', self.history_size)
            
            logger.info(f"学习数据加载成功: {data_path} - 成功复习了旧日记！")
        except Exception as e:
            logger.error(f"加载学习数据时出错: {str(e)} - 复习日记失败！")
    
    def _get_data_path(self):
        """获取数据文件路径
        Returns:
            str: 数据文件路径
        就像知道日记放在哪个抽屉里 - 作者: Pyro"""
        return os.path.join(os.path.dirname(__file__), 'learning_data.json')
    
    def toggle_learning(self, enable):
        """切换学习功能
        Args:
            enable: 是否启用学习
        就像AI的'学习开关' - 作者: Pyro"""
        self.is_learning = enable
        status = "启用" if enable else "禁用"
        logger.info(f"动态学习功能已{status} - 学习开关已{status}！")
        return self.is_learning

# 单例模式, 确保全局只有一个动态学习器实例
_dynamic_learner = None

def get_dynamic_learner(learning_rate=0.1, history_size=1000):
    """
    获取动态学习器实例

    Args:
        learning_rate: 学习率
        history_size: 历史记录大小

    Returns:
        DynamicLearner: 动态学习器实例

    单例模式, 就像动态学习器的专属管家, 确保只有一个学习器在工作 - 作者: Pyro
    """
    global _dynamic_learner
    if _dynamic_learner is None:
        _dynamic_learner = DynamicLearner(learning_rate, history_size)
    return _dynamic_learner