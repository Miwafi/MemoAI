#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path

class TranslationManager:
    
    def __init__(self):
        self.translations = {
            "中文": {
                "APP_TITLE": "MemoAI V2 - 自学习对话系统",
                "APP_SUBTITLE": "智能对话助手",
                "SYSCheck": "系统自检",
                "CREATMenu": "创建必要目录",
                "SEND_BTN": "发送",
                "SELF_LEARN_BTN": "自主学习",
                "ONLINE_LEARN_BTN": "联网学习",
                "CORRECT_BTN": "手动纠错",
                "COPY_AI_BTN": "复制AI输出",
                "CLEAR_BTN": "清除对话",
                "SETTINGS_BTN": "设置",
                "QUIT_BTN": "退出",
                "APPLY_BTN": "应用",
                "READY_STATUS": "就绪",
                "THINKING_STATUS": "思考中...",
                "LEARNING_STATUS": "学习中...",
                "RESPONDING_STATUS": "生成回复中...",
                "SETTINGS_TITLE": "设置",
                "LANG_SETTING": "语言设置",
                "FUNC_SETTING": "功能设置",
                "NETWORK_ROAMING": "网络漫游",
                "GPU_ACCEL": "GPU加速",
                "HF_DATA": "HuggingFace数据获取",
                "TEMP_SETTING": "推理温度设置:",
                "USER": "用户",
                "AI": "AI助手",
                "SYSTEM": "系统",
                "WELCOME_MSG": "欢迎使用 MemoAI V2 智能对话系统！\n\n系统功能：\n• 智能对话\n• 自主学习\n• 联网知识获取\n• 记忆管理",
                "AI_GREETING": "你好！我是MemoAI智能助手，很高兴为你服务。我可以进行智能对话、自主学习新知识，还可以通过联网获取最新信息。有什么我可以帮助你的吗？",
                "SYSTEM_START_LEARNING": "开始自主学习新知识...",
                "SYSTEM_START_NETWORK": "开始联网获取知识...",
                "SYSTEM_COMPLETE": "处理完成，等待下一个任务...",
                "ERROR_TITLE": "错误",
                "WARNING_TITLE": "警告",
                "INFO_TITLE": "提示",
                "SAVE_SUCCESS": "保存成功",
                "SAVE_FAILED": "保存失败",
                "LOAD_SUCCESS": "加载成功",
                "LOAD_FAILED": "加载失败",
                "NETWORK_CONNECTED": "网络已连接",
                "NETWORK_DISCONNECTED": "网络未连接",
                "FETCHING_DATA": "获取数据中...",
                "LEARNING_COMPLETE": "学习完成",
                "LEARNING_FAILED": "学习失败",
                "KNOWLEDGE_UPDATED": "知识已更新",
                "BUTTON_ENABLED": "启用",
                "BUTTON_DISABLED": "禁用",
                "COPY_SUCCESS": "已复制到剪贴板",
                "COPY_FAILED": "复制失败",
                "CLEAR_CONFIRM": "确定要清除所有对话记录吗？",
                "CLEAR_SUCCESS": "对话已清除",
                "SETTINGS_APPLIED": "设置已应用",
                "SETTINGS_RESET": "设置已重置",
                "TEMP_LOW": "低",
                "TEMP_MEDIUM": "中",
                "TEMP_HIGH": "高",
                "MODEL_LOADING": "模型加载中...",
                "MODEL_LOADED": "模型已加载",
                "MODEL_FAILED": "模型加载失败",
                "MEMORY_SAVING": "保存记忆中...",
                "MEMORY_LOADING": "加载记忆中...",
                "MEMORY_CLEARED": "记忆已清除",
                "CORRECTION_TITLE": "手动纠错",
                "CORRECTION_PROMPT": "请输入纠错内容：",
                "CORRECTION_SAVED": "纠错内容已保存",
                "HELP_TITLE": "帮助",
                "HELP_CONTENT": "使用说明：\n1. 在输入框中输入问题或对话内容\n2. 点击发送按钮或按Enter键发送\n3. 可以使用自主学习功能让AI学习新知识\n4. 联网学习可以获取最新的网络信息\n5. 设置中可以调整语言、温度等参数\n"
            },
            
            "ENG": {
                "APP_TITLE": "MemoAI V2 - Self-Learning Dialog System",
                "APP_SUBTITLE": "Intelligent Assistant",
                "SYSCheck": "System Check",
                "CREATMenu": "Create Necessary Directories",
                "SEND_BTN": "Send",
                "SELF_LEARN_BTN": "Self-Learning",
                "ONLINE_LEARN_BTN": "Online Learning",
                "CORRECT_BTN": "Manual Correction",
                "COPY_AI_BTN": "Copy AI Output",
                "CLEAR_BTN": "Clear Chat",
                "SETTINGS_BTN": "Settings",
                "QUIT_BTN": "Quit",
                "APPLY_BTN": "Apply",
                "READY_STATUS": "Ready",
                "THINKING_STATUS": "Thinking...",
                "LEARNING_STATUS": "Learning...",
                "RESPONDING_STATUS": "Generating response...",
                "SETTINGS_TITLE": "Settings",
                "LANG_SETTING": "Language Settings",
                "FUNC_SETTING": "Function Settings",
                "NETWORK_ROAMING": "Network Roaming",
                "GPU_ACCEL": "GPU Acceleration",
                "HF_DATA": "HuggingFace Data Fetch",
                "TEMP_SETTING": "Inference Temperature:",
                "USER": "User",
                "AI": "AI Assistant",
                "SYSTEM": "System",
                "WELCOME_MSG": "Welcome to MemoAI V2 Intelligent Dialog System!\n\nSystem Features:\n• Intelligent conversation\n• Self-learning\n• Online knowledge acquisition\n• Memory management",
                "AI_GREETING": "Hello! I'm MemoAI intelligent assistant, glad to serve you. I can engage in intelligent conversations, learn new knowledge autonomously, and obtain the latest information through networking. How can I help you?",
                "SYSTEM_START_LEARNING": "Starting self-learning new knowledge...",
                "SYSTEM_START_NETWORK": "Starting online knowledge acquisition...",
                "SYSTEM_COMPLETE": "Processing complete, waiting for next task...",
                "ERROR_TITLE": "Error",
                "WARNING_TITLE": "Warning",
                "INFO_TITLE": "Information",
                "SAVE_SUCCESS": "Save successful",
                "SAVE_FAILED": "Save failed",
                "LOAD_SUCCESS": "Load successful",
                "LOAD_FAILED": "Load failed",
                "NETWORK_CONNECTED": "Network connected",
                "NETWORK_DISCONNECTED": "Network disconnected",
                "FETCHING_DATA": "Fetching data...",
                "LEARNING_COMPLETE": "Learning complete",
                "LEARNING_FAILED": "Learning failed",
                "KNOWLEDGE_UPDATED": "Knowledge updated",
                "BUTTON_ENABLED": "Enabled",
                "BUTTON_DISABLED": "Disabled",
                "COPY_SUCCESS": "Copied to clipboard",
                "COPY_FAILED": "Copy failed",
                "CLEAR_CONFIRM": "Are you sure you want to clear all conversation records?",
                "CLEAR_SUCCESS": "Conversation cleared",
                "SETTINGS_APPLIED": "Settings applied",
                "SETTINGS_RESET": "Settings reset",
                "TEMP_LOW": "Low",
                "TEMP_MEDIUM": "Medium",
                "TEMP_HIGH": "High",
                "MODEL_LOADING": "Model loading...",
                "MODEL_LOADED": "Model loaded",
                "MODEL_FAILED": "Model loading failed",
                "MEMORY_SAVING": "Saving memory...",
                "MEMORY_LOADING": "Loading memory...",
                "MEMORY_CLEARED": "Memory cleared",
                "CORRECTION_TITLE": "Manual Correction",
                "CORRECTION_PROMPT": "Please enter correction content:",
                "CORRECTION_SAVED": "Correction content saved",
                "HELP_TITLE": "Help",
                "HELP_CONTENT": "Instructions:\n1. Enter questions or conversation content in the input box\n2. Click send button or press Enter to send\n3. Use self-learning feature to let AI learn new knowledge\n4. Online learning can get the latest network information\n5. Adjust language, temperature and other parameters in settings\n"
            },
            
            "日本語": {
                "APP_TITLE": "MemoAI V2 - 自己学習対話システム",
                "APP_SUBTITLE": "インテリジェントアシスタント",
                "SYSCheck": "システムチェック",
                "CREATMenu": "必要なディレクトリを作成",
                "SEND_BTN": "送信",
                "SELF_LEARN_BTN": "自主学習",
                "ONLINE_LEARN_BTN": "オンライン学習",
                "CORRECT_BTN": "手動修正",
                "COPY_AI_BTN": "AI出力をコピー",
                "CLEAR_BTN": "チャットをクリア",
                "SETTINGS_BTN": "設定",
                "QUIT_BTN": "終了",
                "APPLY_BTN": "適用",
                "READY_STATUS": "準備完了",
                "THINKING_STATUS": "考え中...",
                "LEARNING_STATUS": "学習中...",
                "RESPONDING_STATUS": "応答生成中...",
                "SETTINGS_TITLE": "設定",
                "LANG_SETTING": "言語設定",
                "FUNC_SETTING": "機能設定",
                "NETWORK_ROAMING": "ネットワークローミング",
                "GPU_ACCEL": "GPU加速",
                "HF_DATA": "HuggingFaceデータ取得",
                "TEMP_SETTING": "推論温度:",
                "USER": "ユーザー",
                "AI": "AIアシスタント",
                "SYSTEM": "システム",
                "WELCOME_MSG": "MemoAI V2 インテリジェント対話システムへようこそ！\n\nシステム機能:\n• インテリジェント会話\n• 自己学習\n• オンライン知識取得\n• メモリ管理",
                "AI_GREETING": "こんにちは！MemoAIインテリジェントアシスタントです。インテリジェント会話、新しい知識の自己学習、ネットワークを通じた最新情報の取得が可能です。どのようにお手伝いできますか？",
                "SYSTEM_START_LEARNING": "新しい知識の自己学習を開始...",
                "SYSTEM_START_NETWORK": "オンライン知識取得を開始...",
                "SYSTEM_COMPLETE": "処理完了、次のタスクを待機中...",
                "ERROR_TITLE": "エラー",
                "WARNING_TITLE": "警告",
                "INFO_TITLE": "情報",
                "SAVE_SUCCESS": "保存成功",
                "SAVE_FAILED": "保存失敗",
                "LOAD_SUCCESS": "読み込み成功",
                "LOAD_FAILED": "読み込み失敗",
                "NETWORK_CONNECTED": "ネットワーク接続済み",
                "NETWORK_DISCONNECTED": "ネットワーク未接続",
                "FETCHING_DATA": "データ取得中...",
                "LEARNING_COMPLETE": "学習完了",
                "LEARNING_FAILED": "学習失敗",
                "KNOWLEDGE_UPDATED": "知識更新済み",
                "BUTTON_ENABLED": "有効",
                "BUTTON_DISABLED": "無効",
                "COPY_SUCCESS": "クリップボードにコピー",
                "COPY_FAILED": "コピー失敗",
                "CLEAR_CONFIRM": "すべての会話記録をクリアしてもよろしいですか？",
                "CLEAR_SUCCESS": "会話をクリアしました",
                "SETTINGS_APPLIED": "設定を適用しました",
                "SETTINGS_RESET": "設定をリセットしました",
                "TEMP_LOW": "低",
                "TEMP_MEDIUM": "中",
                "TEMP_HIGH": "高",
                "MODEL_LOADING": "モデル読み込み中...",
                "MODEL_LOADED": "モデル読み込み完了",
                "MODEL_FAILED": "モデル読み込み失敗",
                "MEMORY_SAVING": "メモリ保存中...",
                "MEMORY_LOADING": "メモリ読み込み中...",
                "MEMORY_CLEARED": "メモリをクリアしました",
                "CORRECTION_TITLE": "手動修正",
                "CORRECTION_PROMPT": "修正内容を入力してください:",
                "CORRECTION_SAVED": "修正内容を保存しました",
                "HELP_TITLE": "ヘルプ",
                "HELP_CONTENT": "使用説明:\n1. 入力ボックスに質問や会話内容を入力\n2. 送信ボタンをクリックまたはEnterキーで送信\n3. 自己学習機能を使用してAIに新しい知識を学習させる\n4. オンライン学習で最新のネットワーク情報を取得\n5. 設定で言語、温度などのパラメータを調整\n"
            }
        }
        
        self.current_language = "中文"
    
    def set_language(self, language):
        if language in self.translations:
            self.current_language = language
    
    def get_text(self, key, language=None):
        lang = language or self.current_language
        return self.translations.get(lang, {}).get(key, key)
    
    def get_available_languages(self):
        return list(self.translations.keys())
    
    def get_all_texts(self, language):
        return self.translations.get(language, {})

laun = TranslationManager()

def get_text(key, language=None):
    return laun.get_text(key, language)

def load_additional_translations():
    try:
        json_path = Path(__file__).parent / "language" / "translations.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                additional_translations = json.load(f)
                for lang, texts in additional_translations.items():
                    if lang in laun.translations:
                        laun.translations[lang].update(texts)
                    else:
                        laun.translations[lang] = texts
    except Exception as e:
        print(f"加载额外翻译失败: {e}")

load_additional_translations()

if __name__ == "__main__":
    print("=== 翻译测试 ===")
    
    for lang in laun.get_available_languages():
        print(f"\n{lang}:")
        print(f"  SEND_BTN: {laun.get_text('SEND_BTN', lang)}")
        print(f"  READY_STATUS: {laun.get_text('READY_STATUS', lang)}")
        print(f"  AI_GREETING: {laun.get_text('AI_GREETING', lang)}")