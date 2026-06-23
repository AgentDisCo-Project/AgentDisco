import requests
import asyncio
import httpx
import logging

from openai import AsyncOpenAI
from tqdm import tqdm
from typing import Callable, Any
from api.utils.key_operator import ApiKeyCycler



class CustomizeToolGenerator:
    def __init__(
        self,
        model_name: str,
        tool_corpus: list = None,
        max_retries: int = 5,
        retry_delay: int = 3,
        max_tokens: int = 32768,
        temperature: float = 0.1,
        top_p: float = 0.5,
        timeout: int = 60,
        use_customize_url: bool = False,
        customize_url: str = ""
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        
        self.use_customize_url = use_customize_url
        self.customize_url = customize_url
        
        self.tool_corpus = []
        if "rednote_text_search" in tool_corpus:
            self.tool_corpus.append(self.add_rednote_text_search())
        if "rednote_image_search" in tool_corpus:
            self.tool_corpus.append(self.add_rednote_image_search())
        if "web_text_search" in tool_corpus:
            self.tool_corpus.append(self.add_web_text_search())
    
    
    def add_rednote_text_search(self):
        pass
    
    def add_rednote_image_search(self):
        pass
    
    def add_web_text_search(self):
        pass
    
    
    async def chat_qwen_or_deepseek(
        self,
        system_prompt: list,
        user_prompt: list,
        check_func: Callable[[str], Any],
        cycler: ApiKeyCycler,
        return_cot: bool = False
    ):
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
        
        basic_url = self.customize_url if self.use_customize_url and self.customize_url else "http://redservingapi.devops.xiaohongshu.com/v1"
        for attempt in range(self.max_retries):
            try:
                api_key = await cycler.get_key()
                client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=basic_url
                )
                completion = await client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=False,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                response = completion.choices[0].message.content
                response = check_func(response)
                if return_cot and "reasoning_content" in completion.choices[0].message:
                    cot = completion.choices[0].message.reasoning_content
                    return (response, cot)
                else:
                    return response
            
            except (
                requests.exceptions.RequestException,
                Exception,
            ) as e:
                tqdm.write(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
                continue
            else:
                tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                return ""
    
    
    async def chat(self):
        pass