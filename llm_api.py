import os
from openai import OpenAI
from typing import Dict, Any, Optional, List

class LLMAPIBase:
    """LLM API的基类，定义统一接口"""

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """生成回复的统一接口"""
        raise NotImplementedError

class OpenAIAPI(LLMAPIBase):
    """OpenAI API封装"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return f"Error: {str(e)}"

class DeepSeekAPI(LLMAPIBase):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=base_url or "https://api.deepseek.com/v1"
        )

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        try:
            # 根据DeepSeek文档使用合适的模型
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # 假设使用的是deepseek-chat模型
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API error: {e}")
            return f"Error: {str(e)}"

class LLMAPIFactory:
    """LLM API工厂类，用于创建不同的LLM API实例"""

    @staticmethod
    def create_api(provider: str, **kwargs) -> LLMAPIBase:
        if provider.lower() == "openai":
            return OpenAIAPI(**kwargs)
        elif provider.lower() == "deepseek":
            return DeepSeekAPI(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")