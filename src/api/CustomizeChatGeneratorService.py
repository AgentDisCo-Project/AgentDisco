import os
import asyncio
import httpx
import logging
import gin
import sys
import requests
import json
sys.path.append('.')

from api.utils.string_operator import json_fix
from typing import List
from openai import AsyncOpenAI
from tqdm import tqdm
from typing import Callable, Any, List, Union
from api.utils.key_operator import ApiKeyCycler
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_PROGRAM = os.environ.get("DIRECTLLM_API_KEY_PROGRAM", "{}")
DIRECTLLM_API_KEY_PROGRAM = json.loads(DIRECTLLM_API_KEY_PROGRAM)
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
DIRECTLLM_APIS = os.environ.get("DIRECTLLM_APIS", "{}")
DIRECTLLM_APIS = json.loads(DIRECTLLM_APIS)

GEMINI_2_5_PRO_API_KEY = os.environ.get("GEMINI_2_5_PRO_API_KEY", "")
GEMINI_2_5_FLASH_API_KEY = os.environ.get("GEMINI_2_5_FLASH_API_KEY", "")
GEMINI_3_PRO_API_KEY = os.environ.get("GEMINI_3_PRO_API_KEY", "")
GEMINI_3_1_PRO_API_KEY = os.environ.get("GEMINI_3_1_PRO_API_KEY", "")
GEMINI_3_FLASH_API_KEY = os.environ.get("GEMINI_3_FLASH_API_KEY", "")
GEMINI_IMAGE_API_KEY = os.environ.get("GEMINI_IMAGE_API_KEY", "")
GPT_5_API_KEY = os.environ.get("GPT_5_API_KEY", "")
GPT_4_API_KEY = os.environ.get("GPT_4_API_KEY", "")
CLAUDE_OPUS_46_API_KEY = os.environ.get("CLAUDE_OPUS_46_API_KEY", "")
CLAUDE_SONNET_46_API_KEY = os.environ.get("CLAUDE_SONNET_46_API_KEY", "")


@gin.configurable()
class CustomizeChatGenerator:
    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        retry_delay: int = 3,
        max_tokens: int = -1,
        temperature: float = 0.1,
        top_p: float = 0.5,
        timeout: int = 1024,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = False,
        need_cycle_key: bool = True,
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
        self.use_api_key = use_api_key
        self.cycler = ApiKeyCycler(api_key_list=list(DIRECTLLM_APIS.values()))
    
    
    async def chat_gpt(
        self,
        system_prompt: str,
        user_prompt: str,
        check_func: Callable[[str], Any],
    ):
        if "gpt-5" in self.model_name:
            gpt_api_key = GPT_5_API_KEY
        elif "gpt-4.1" in self.model_name:
            gpt_api_key = GPT_4_API_KEY
        else:
            gpt_api_key = ""
        
        attempt = 0
        while attempt < self.max_retries:
            try:
                headers = {
                    "api-key": f"{gpt_api_key}",  # gpt4o的token
                    "Content-Type": "application/json"
                }
                if self.max_tokens > 0:
                    request = {
                        "max_tokens": self.max_tokens,
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ],
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    }
                else:
                    request = {
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ],
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    }

                async with httpx.AsyncClient() as client:
                    if self.timeout > 0:
                        response = await client.post(
                            "https://runway.devops.xiaohongshu.com/openai/chat/completions?api-version=2024-12-01-preview",
                            headers=headers,
                            json=request,
                            timeout=self.timeout,  # 可根据需求调整
                        )
                    else:
                        response = await client.post(
                            "https://runway.devops.xiaohongshu.com/openai/chat/completions?api-version=2024-12-01-preview",
                            headers=headers,
                            json=request,
                        )
                
                if response.status_code != 200:
                    logging.info(f"Fail to call API {response.status_code} - {response.text}")
                    response.raise_for_status()
                response = response.json()["choices"][0]["message"]["content"]
                response = check_func(response)
                return response
            
            except (
                requests.exceptions.RequestException,
                Exception,
            ) as e:
                error_str = str(e)
                
                # 检查是否是 429 错误（Too Many Requests）或配额超限错误
                if ("429" in error_str or
                    "Too Many Requests" in error_str or
                    "exceed quota" in error_str or
                    "E429" in error_str):
                    tqdm.write(f"请求超限 (不计入重试次数): {error_str}")
                    await asyncio.sleep(self.retry_delay * 3)  # 等待更长时间
                    continue  # 不增加 attempt，直接重试
                
                # 其他错误计入重试次数
                attempt += 1
                tqdm.write(f"请求失败 (尝试 {attempt}/{self.max_retries}): {error_str}")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                    return ""
    
    
    async def chat_gpt_oss(
        self,
        system_prompt: str,
        user_prompt: str,
        check_func: Callable[[str], Any],
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
        
        attempt = 0
        while attempt < self.max_retries:
            try:
                # if self.use_api_key and self.model_name in DIRECTLLM_API_KEY_PROGRAM:
                #     api_key = DIRECTLLM_API_KEY_PROGRAM[self.model_name]
                #     basic_url = "https://maas.devops.xiaohongshu.com/v1"
                # else:
                #     api_key = await self.cycler.get_key()
                #     basic_url = "http://redservingapi.devops.xiaohongshu.com/v1"
                api_key = await self.cycler.get_key()
                basic_url = "http://redservingapi.devops.xiaohongshu.com/v1"
                    
                client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=basic_url
                )
                if self.max_tokens > 0:
                    completion = await client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        stream=False,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                else:
                    completion = await client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        stream=False,
                        temperature=self.temperature,
                    )
                response = completion.choices[0].message.content
                response = check_func(response)
                return response
            
            except (
                requests.exceptions.RequestException,
                Exception,
            ) as e:
                error_str = str(e)
                
                # 检查是否是 429 错误（Too Many Requests）或配额超限错误
                if ("429" in error_str or
                    "Too Many Requests" in error_str or
                    "exceed quota" in error_str or
                    "E429" in error_str):
                    tqdm.write(f"请求超限 (不计入重试次数): {error_str}")
                    await asyncio.sleep(self.retry_delay * 3)  # 等待更长时间
                    continue  # 不增加 attempt，直接重试
                
                # 其他错误计入重试次数
                attempt += 1
                tqdm.write(f"请求失败 (尝试 {attempt}/{self.max_retries}): {error_str}")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                    return ""
    
    
    async def chat_gemini(
        self,
        system_prompt: List,
        user_prompt: List,
        check_func: Callable[[str], Any],
        return_cot: bool = False,
    ):
        if "gemini-2.5-pro" in self.model_name:
            gemini_api_key = GEMINI_2_5_PRO_API_KEY
        elif "gemini-2.5-flash" in self.model_name:
            gemini_api_key = GEMINI_2_5_FLASH_API_KEY
        elif "gemini-3-pro" in self.model_name:
            gemini_api_key = GEMINI_3_PRO_API_KEY
        elif "gemini-3-flash" in self.model_name:
            gemini_api_key = GEMINI_3_FLASH_API_KEY
        elif "gemini-3.1-pro" in self.model_name:
            gemini_api_key = GEMINI_3_1_PRO_API_KEY
        else:
            gemini_api_key = ""

        attempt = 0
        while attempt < self.max_retries:
            try:
                headers = {
                    'content-type': 'application/json',
                    'api-key': f'{gemini_api_key}'
                }
                if self.max_tokens > 0:
                    request = {
                        "system_instruction": {
                            "parts": system_prompt,
                        },
                        "contents": [
                            {"role": "user", "parts": user_prompt},
                        ],
                        "generationConfig": {
                            "maxOutputTokens": self.max_tokens,
                            "thinkingConfig": {
                                "includeThoughts": return_cot
                            }
                        }
                    }
                else:
                    request = {
                        "system_instruction": {
                            "parts": system_prompt,
                        },
                        "contents": [
                            {"role": "user", "parts": user_prompt},
                        ],
                        "generationConfig": {
                            "thinkingConfig": {
                                "includeThoughts": return_cot
                            }
                        }
                    }

                async with httpx.AsyncClient() as client:
                    if self.timeout > 0:
                        response = await client.post(
                            "https://runway.devops.rednote.life/openai/google/v1:generateContent",
                            headers=headers,
                            json=request,
                            timeout=self.timeout,
                        )
                    else:
                        response = await client.post(
                            "https://runway.devops.rednote.life/openai/google/v1:generateContent",
                            headers=headers,
                            json=request,
                        )
                if response.status_code != 200:
                    print(f"Status: {response.status_code}")
                    print(f"Response body: {response.text}")
                    print(f"Request sent: {request}")
                    response.raise_for_status()
                response = response.json()["candidates"][0]["content"]['parts'][0]['text']
                response = check_func(response)
                return response

            except (
                requests.exceptions.RequestException,
                Exception,
            ) as e:
                error_str = str(e)

                if ("429" in error_str or
                    "Too Many Requests" in error_str or
                    "exceed quota" in error_str or
                    "E429" in error_str):
                    tqdm.write(f"请求超限 (不计入重试次数): {error_str}")
                    await asyncio.sleep(self.retry_delay * 3)
                    continue

                attempt += 1
                tqdm.write(f"请求失败 (尝试 {attempt}/{self.max_retries}): {error_str}")

                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                    return ""


    async def chat_gemini_image(
        self,
        system_prompt: List,
        user_prompt: List,
        aspect_ratio: str = "9:16",
        mime_type: str = "image/png",
    ):
        """Gemini 原生图片生成，使用 GEMINI_IMAGE_API_KEY。

        Args:
            system_prompt: Gemini parts 格式，会合并到 user_prompt 开头
            user_prompt: Gemini parts 格式（text + inlineData）
            aspect_ratio: 输出图片比例，如 "9:16", "3:4", "16:9", "4:3"
            mime_type: 输出图片格式
        Returns:
            {"text": str, "image_bytes": bytes|None, "mime_type": str}
        """
        import base64

        merged_prompt = list(system_prompt) + list(user_prompt)

        attempt = 0
        while attempt < self.max_retries:
            try:
                headers = {
                    'content-type': 'application/json',
                    'api-key': GEMINI_IMAGE_API_KEY,
                }
                request = {
                    "contents": [
                        {"role": "user", "parts": merged_prompt},
                    ],
                    "generationConfig": {
                        "responseModalities": ["TEXT", "IMAGE"],
                        "temperature": 0.6,
                        "imageConfig": {
                            "aspectRatio": aspect_ratio,
                            "imageOutputOptions": {"mimeType": mime_type},
                        },
                    },
                }

                async with httpx.AsyncClient() as client:
                    if self.timeout > 0:
                        response = await client.post(
                            "https://runway.devops.rednote.life/openai/google/v1:generateContent",
                            headers=headers,
                            json=request,
                            timeout=self.timeout,
                        )
                    else:
                        response = await client.post(
                            "https://runway.devops.rednote.life/openai/google/v1:generateContent",
                            headers=headers,
                            json=request,
                        )
                if response.status_code != 200:
                    print(f"Status: {response.status_code}")
                    print(f"Response body: {response.text}")
                    response.raise_for_status()

                resp_json = response.json()
                if "candidates" not in resp_json:
                    error_info = resp_json.get("Error", resp_json.get("error", str(resp_json)[:300]))
                    raise RuntimeError(f"Gemini image API error: {error_info}")

                parts = resp_json["candidates"][0]["content"]["parts"]
                text_parts = []
                image_bytes = None
                resp_mime = mime_type
                for part in parts:
                    if "text" in part:
                        text_parts.append(part["text"])
                    elif "inlineData" in part and part["inlineData"].get("data"):
                        image_bytes = base64.b64decode(part["inlineData"]["data"])
                        resp_mime = part["inlineData"].get("mimeType", mime_type)
                return {
                    "text": "\n".join(text_parts),
                    "image_bytes": image_bytes,
                    "mime_type": resp_mime,
                }

            except (
                requests.exceptions.RequestException,
                Exception,
            ) as e:
                error_str = str(e)

                if ("429" in error_str or
                    "Too Many Requests" in error_str or
                    "exceed quota" in error_str or
                    "E429" in error_str):
                    tqdm.write(f"请求超限 (不计入重试次数): {error_str}")
                    await asyncio.sleep(self.retry_delay * 3)
                    continue

                attempt += 1
                tqdm.write(f"请求失败 (尝试 {attempt}/{self.max_retries}): {error_str}")

                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                    return {"text": "", "image_bytes": None, "mime_type": mime_type}
    
    
    async def chat_qwen_or_deepseek(
        self,
        system_prompt: Union[List, str],
        user_prompt: Union[List, str],
        check_func: Callable[[str], Any],
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
        
        attempt = 0
        while attempt < self.max_retries:
            try:
                # if self.use_api_key and self.model_name in DIRECTLLM_API_KEY_PROGRAM:
                #     api_key = DIRECTLLM_API_KEY_PROGRAM[self.model_name]
                #     basic_url = "https://maas.devops.xiaohongshu.com/v1"
                # else:
                #     api_key = await self.cycler.get_key()
                #     basic_url = "https://maas.devops.xiaohongshu.com/v1"
                api_key = await self.cycler.get_key()
                basic_url = "https://maas.devops.xiaohongshu.com/v1"
                client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=basic_url
                )
                if self.max_tokens > 0:
                    completion = await client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        stream=False,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                else:
                    completion = await client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        stream=False,
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
                error_str = str(e)
                
                # 检查是否是 429 错误（Too Many Requests）或配额超限错误
                if ("429" in error_str or
                    "Too Many Requests" in error_str or
                    "exceed quota" in error_str or
                    "E429" in error_str):
                    tqdm.write(f"请求超限 (不计入重试次数): {error_str}")
                    await asyncio.sleep(self.retry_delay * 3)  # 等待更长时间
                    continue  # 不增加 attempt，直接重试
                
                # 其他错误计入重试次数
                attempt += 1
                tqdm.write(f"请求失败 (尝试 {attempt}/{self.max_retries}): {error_str}")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                    return ""

    
    async def chat_claude(
        self,
        system_prompt: List,
        user_prompt: List,
        check_func: Callable[[str], Any],
        return_cot: bool = False,
    ):
        if "claude-opus-4.6" in self.model_name:
            claude_api_key = CLAUDE_OPUS_46_API_KEY
        elif "claude-sonnet-4.6" in self.model_name:
            claude_api_key = CLAUDE_SONNET_46_API_KEY
        else:
            claude_api_key = ""

        attempt = 0
        while attempt < self.max_retries:
            try:
                headers = {
                    'content-type': 'application/json',
                    'api-key': f'{claude_api_key}'
                }
                if self.max_tokens > 0:
                    request = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "system": system_prompt,
                        "messages": [
                            {"role": "user", "content": user_prompt},
                        ],
                        "max_tokens": self.max_tokens,
                    }
                else:
                    request = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "system": system_prompt,
                        "messages": [
                            {"role": "user", "content": user_prompt},
                        ],
                    }
                
                async with httpx.AsyncClient() as client:
                    if self.timeout > 0:
                        response = await client.post(
                            'https://runway.devops.rednote.life/openai/bedrock_runtime/model/invoke',
                            headers=headers,
                            json=request,
                            timeout=self.timeout,  # 可根据需求调整
                        )
                    else:
                        response = await client.post(
                            'https://runway.devops.rednote.life/openai/bedrock_runtime/model/invoke',
                            headers=headers,
                            json=request,
                        )

                if response.status_code != 200:
                    # 打印完整错误信息
                    print(f"Status: {response.status_code}")
                    print(f"Response body: {response.text}")   # ← 这里才是关键！
                    print(f"Request sent: {request}")           # ← 确认发了什么
                    response.raise_for_status()
                response = response.json()["content"][0]["text"] 
                response = check_func(response)
                return response
            
            except (
                requests.exceptions.RequestException,
                Exception,
            ) as e:
                error_str = str(e)
                
                # 检查是否是 429 错误（Too Many Requests）或配额超限错误
                if ("429" in error_str or
                    "Too Many Requests" in error_str or
                    "exceed quota" in error_str or
                    "E429" in error_str):
                    tqdm.write(f"请求超限 (不计入重试次数): {error_str}")
                    await asyncio.sleep(self.retry_delay * 3)  # 等待更长时间
                    continue  # 不增加 attempt，直接重试
                
                # 其他错误计入重试次数
                attempt += 1
                tqdm.write(f"请求失败 (尝试 {attempt}/{self.max_retries}): {error_str}")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                    return ""



if __name__ == "__main__":
    async def main():
        # model_name = "qwen3-next-80b-a3b-instruct"
        # model_name = "qwen2.5-72b-instruct"
        model_name = "claude-opus-4.6"
        service = CustomizeChatGenerator(
            model_name=model_name,
            max_retries=5,
            retry_delay=1,
            temperature=0.1,
            max_tokens=4096,
        )
        
        def check_func(response: str):
            return response
        _system_prompt = "你需要扮演一个智能回答助手。"
        _user_prompt = "你是谁？"
        if "qwen" in model_name or "deepseek" in model_name or "qwq" in model_name:
            system_prompt = [
                {
                    "type": "text",
                    "text": _system_prompt
                }
            ]
            user_prompt = [
                {
                    "type": "text",
                    "text": _user_prompt
                }
            ]

            response = await service.chat_qwen_or_deepseek(
                system_prompt=_system_prompt,
                user_prompt=_user_prompt,
                check_func=check_func,
                return_cot=False,
            )
            
            print(response)
        
        elif "gemini" in model_name:
            system_prompt = [
                {
                    "text": _system_prompt
                }
            ]
            user_prompt = [
                {
                    "text": _user_prompt
                }
            ]

            response = await service.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=check_func,
                return_cot=False,
            )
            print(response)
        
        elif "claude" in model_name:
            system_prompt = [
                {
                    "text": _system_prompt
                }
            ]
            user_prompt = [
                {
                    "text": _user_prompt
                }
            ]

            breakpoint()
            response = await service.chat_claude(
                system_prompt=_system_prompt,
                user_prompt=_user_prompt,
                check_func=check_func,
                return_cot=False,
            )
            print(response)
        

    asyncio.run(main())