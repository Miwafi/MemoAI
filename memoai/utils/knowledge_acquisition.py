import os
import json
import logging
import requests
import threading
import time
import re
from datetime import datetime
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional
import networkx as nx
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("memoai.log"), logging.StreamHandler()]
)
logger = logging.getLogger("knowledge_acquisition")
RELIABLE_SOURCES = [
    "https://en.wikipedia.org",
    "https://zh.wikipedia.org",
    "https://www.scientificamerican.com",
    "https://www.nature.com",
    "https://arxiv.org",
    "https://developer.mozilla.org",
    "https://docs.python.org"
]
class KnowledgeGraph:
    def __init__(self, data_path: str = None):
        self.graph = nx.DiGraph()
        self.data_path = data_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'knowledge', 'knowledge_graph.json')
        self.lock = threading.RLock()
        self._load_from_disk()
    def _load_from_disk(self):
        try:
            if os.path.exists(self.data_path):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for node in data.get('nodes', []):
                        self.graph.add_node(node['id'], **node['attributes'])
                    for edge in data.get('edges', []):
                        self.graph.add_edge(edge['source'], edge['target'], **edge['attributes'])
                logger.info(f"成功加载知识图谱，包含 {len(self.graph.nodes)} 个节点和 {len(self.graph.edges)} 条边")
            else:
                logger.info("知识图谱文件不存在，创建新的知识图谱")
        except Exception as e:
            logger.error(f"加载知识图谱失败: {str(e)}")
    def save_to_disk(self):
        try:
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            data = {
                'nodes': [],
                'edges': []
            }
            for node_id, attributes in self.graph.nodes(data=True):
                data['nodes'].append({
                    'id': node_id,
                    'attributes': attributes
                })
            for source, target, attributes in self.graph.edges(data=True):
                data['edges'].append({
                    'source': source,
                    'target': target,
                    'attributes': attributes
                })
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"成功保存知识图谱到 {self.data_path}")
        except Exception as e:
            logger.error(f"保存知识图谱失败: {str(e)}")
    def add_entity(self, entity_id: str, attributes: Dict[str, Any] = None):
        with self.lock:
            attributes = attributes or {}
            attributes['last_updated'] = datetime.now().isoformat()
            self.graph.add_node(entity_id, **attributes)
            logger.debug(f"添加实体: {entity_id}")
            return True
    def add_relation(self, source_id: str, target_id: str, relation_type: str, attributes: Dict[str, Any] = None):
        with self.lock:
            attributes = attributes or {}
            attributes['type'] = relation_type
            attributes['last_updated'] = datetime.now().isoformat()
            self.graph.add_edge(source_id, target_id, **attributes)
            logger.debug(f"添加关系: {source_id} -[{relation_type}]-> {target_id}")
            return True
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            if entity_id in self.graph.nodes:
                return dict(self.graph.nodes[entity_id])
            return None
    def search_entities(self, query: str) -> List[Dict[str, Any]]:
        results = []
        with self.lock:
            for node_id, attributes in self.graph.nodes(data=True):
                if query.lower() in node_id.lower() or any(query.lower() in str(value).lower() for value in attributes.values()):
                    results.append({
                        'id': node_id,
                        'attributes': attributes
                    })
        return results
class KnowledgeAcquisition:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.lock = threading.RLock()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    def _is_reliable_source(self, url: str) -> bool:
        for source in RELIABLE_SOURCES:
            if url.startswith(source):
                return True
        return False
    def _extract_text_from_html(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        for script in soup(['script', 'style']):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        return text
    def acquire_knowledge(self, url: str) -> bool:
        try:
            if not self._is_reliable_source(url):
                logger.warning(f"忽略不可靠来源: {url}")
                return False
            logger.info(f"开始从 {url} 获取知识")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            text = self._extract_text_from_html(response.text)
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else url
            doc_id = f"doc:{hash(url) % 1000000}"
            self.knowledge_graph.add_entity(
                doc_id,
                {
                    'type': 'document',
                    'title': title,
                    'url': url,
                    'content': text[:5000],
                    'full_content_length': len(text),
                    'acquisition_date': datetime.now().isoformat()
                }
            )
            url_id = f"url:{url}"
            self.knowledge_graph.add_entity(
                url_id,
                {
                    'type': 'url',
                    'url': url,
                    'last_visited': datetime.now().isoformat()
                }
            )
            self.knowledge_graph.add_relation(doc_id, url_id, 'source')
            self.knowledge_graph.save_to_disk()
            logger.info(f"成功从 {url} 获取知识并添加到知识图谱")
            return True
        except Exception as e:
            logger.error(f"从 {url} 获取知识失败: {str(e)}")
            return False
    def auto_learn(self, topics: List[str], max_depth: int = 2):
        logger.info(f"开始自动学习主题: {', '.join(topics)}")
        for topic in topics:
            topic_id = f"topic:{topic.lower().replace(' ', '_')}"
            self.knowledge_graph.add_entity(
                topic_id,
                {
                    'type': 'topic',
                    'name': topic,
                    'last_studied': datetime.now().isoformat()
                }
            )
            logger.info(f"添加学习主题: {topic}")
        self.knowledge_graph.save_to_disk()
        logger.info("自动学习完成")
_knowledge_graph = None
_knowledge_acquisition = None
def get_knowledge_graph() -> KnowledgeGraph:
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph()
    return _knowledge_graph
def get_knowledge_acquisition() -> KnowledgeAcquisition:
    global _knowledge_acquisition
    if _knowledge_acquisition is None:
        _knowledge_acquisition = KnowledgeAcquisition(get_knowledge_graph())
    return _knowledge_acquisition
class BackgroundLearningThread(threading.Thread):
    def __init__(self, interval: int = 3600):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = False
        self.knowledge_acquisition = get_knowledge_acquisition()
    def run(self):
        self.running = True
        logger.info(f"启动后台学习线程，间隔 {self.interval} 秒")
        while self.running:
            try:
                topics = ["人工智能最新进展", "Python编程技巧", "机器学习新算法"]
                self.knowledge_acquisition.auto_learn(topics)
            except Exception as e:
                logger.error(f"后台学习出错: {str(e)}")
            for _ in range(self.interval):
                if not self.running:
                    break
                time.sleep(1)
    def stop(self):
        self.running = False
        logger.info("停止后台学习线程")
_background_learning_thread = None
def start_background_learning(interval: int = 3600):
    global _background_learning_thread
    if _background_learning_thread is None or not _background_learning_thread.is_alive():
        _background_learning_thread = BackgroundLearningThread(interval)
        _background_learning_thread.start()
        logger.info("后台学习已启动")
    else:
        logger.info("后台学习已经在运行中")
def stop_background_learning():
    global _background_learning_thread
    if _background_learning_thread is not None and _background_learning_thread.is_alive():
        _background_learning_thread.stop()
        _background_learning_thread.join(timeout=5)
        logger.info("后台学习已停止")
    else:
        logger.info("后台学习没有运行")