from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    ChatResult, HumanMessage, AIMessage, ChatMessage, ChatGeneration, SystemMessage
)
import os
from typing import ClassVar
from typing import List, Optional
import torch
import re
from langchain_deepseek import ChatDeepSeek
class DeepSeekChat(BaseChatModel):
    """支持清理格式标记和截断Human内容的deepseek聊天模型"""
    llm: ClassVar[ChatDeepSeek] = ChatDeepSeek(
        base_url="https://api.deepseek.com",
        api_key=os.getenv(key='api_key'),
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=4096,
        timeout=120,
        max_retries=2,
    )
    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"
    

    
    def _truncate_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """只保留最近5轮对话，避免历史过长"""
        if len(messages) > 5:
            return messages[-5:]
        return messages
    
    
    def _generate(self, messages: List[ChatMessage], stop: Optional[List[str]] = None) -> ChatResult:
        """生成回复并严格清理格式"""
        # 1. 截断过长历史
        prompt = self._truncate_messages(messages)
        
        # 2. 调用模型生成
        response = self.llm.invoke(prompt, stop=stop)
        
        return ChatResult(generations=[ChatGeneration(message=response)])
    
    def _call(self, messages: List[ChatMessage], stop: Optional[List[str]] = None) -> ChatResult:
        return self._generate(messages, stop)

