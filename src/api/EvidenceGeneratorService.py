import os
import gin
import jinja2
import asyncio
import re
import httpx
import logging
import json
import time
import sys
sys.path.append('.')

from tqdm import tqdm
from typing import Union, Optional, List, Dict
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.key_operator import ApiKeyCycler
from api.EvidenceMergeService import EvidenceMerger
from api.utils.string_operator import json_fix
from agent.BaseAgent import BasicAgent
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")


@gin.configurable()
class EvidenceGenerator:
    def __init__(
        self,
        model_name: str = "",
        use_zh: bool = False,
        max_retries: int = 5,
        retry_delay: int = 3,
        use_customize_url: bool = False,
        customize_url: str = "",
        max_summary_len: int = 256,
        max_evidence_len: int = 512,
        max_evidence_num: int = 10,
        use_api_key: bool = True,
        include_user_query: bool = False,
        include_outline: bool = False,
        include_search_query: bool = False,
        use_evidence_as_key: bool = False,
        system_template_dir: str = "./template",
        system_template_en_file: str = "EvidenceGenerator_EN.jinja2",
        system_template_zh_file: str = "EvidenceGenerator_ZH.jinja2",
        system_template_en_file_evidence: str = "EvidenceGeneratorEvidence_EN.jinja2",
        system_template_zh_file_evidence: str = "EvidenceGeneratorEvidence_ZH.jinja2",
        max_concurrent: int = 50,
    ):
        self.model = CustomizeChatGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
        )
        self.use_zh = use_zh
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        if use_evidence_as_key:
            self.jinja_file = system_template_en_file_evidence if not self.use_zh else system_template_zh_file_evidence
        else:
            self.jinja_file = system_template_en_file if not self.use_zh else system_template_zh_file
        
        self.max_summary_len = max_summary_len
        self.max_evidence_len = max_evidence_len
        self.max_evidence_num = max_evidence_num
        self.include_user_query = include_user_query
        self.include_outline = include_outline
        self.include_search_query = include_search_query
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_evidence_as_key = use_evidence_as_key
        self.max_concurrent = max_concurrent
    
    
    def parser_response(
        self,
        response: str,
    ):
        response = json_fix(response)
        response = json.loads(response)
        
        if not isinstance(response, Dict):
            raise ValueError()
        
        if self.use_evidence_as_key:
            if "summary" not in response or "evidences" not in response:
                raise ValueError()
            if len(response["summary"]) > self.max_summary_len:
                raise ValueError()
            for evidence in response["evidences"]:
                if len(evidence) > self.max_evidence_len:
                    raise ValueError()
        else:
            if "summary" not in response or "evidence" not in response:
                raise ValueError()
            if len(response["summary"]) > self.max_summary_len or len(response["evidence"]) > self.max_evidence_len:
                raise ValueError()
        
        return response
    
    
    def get_system_prompt(
        self,
    ):
        if self.use_evidence_as_key:
            template_vars = {
                "include_user_query": self.include_user_query,
                "max_summary_len": self.max_summary_len,
                "max_evidence_len": self.max_evidence_len,
                "max_evidence_num": self.max_evidence_num
            }
        else:
            template_vars = {
                "include_user_query": self.include_user_query,
                "max_summary_len": self.max_summary_len,
                "max_evidence_len": self.max_evidence_len
            }
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    def check_func(
        self,
        response: str,
    ):
        return self.parser_response(response)
    
    
    async def get_detail(
        self,
        url: str,
    ):
        for attempt in range(self.max_retries):
            try:
                jina_url = "https://r.jina.ai/"
                headers = {
                    "Authorization": f"Bearer {JINA_API_KEY}",
                    "Content-Type": "application/json",
                    "X-Engine": "browser",
                    "X-Timeout": "100"
                }
                data = {
                    "url": url
                }
                async with httpx.AsyncClient(timeout=100) as client:
                    response = await client.post(jina_url, headers=headers, json=data)
                    response.raise_for_status()  # 抛出HTTP错误
                    return response.text
            except Exception as e:
                tqdm.write(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
            
            if attempt < self.max_retries - 1:  # 如果不是最后一次尝试
                time.sleep(self.retry_delay)  # 等待一段时间再重试
                continue
            else:
                tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                return ""
    
    
    async def post_request(
        self,
        query_text: str,
        search_query: str,
        document: Dict,
        outline: str = "",
    ):
        
        if "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            system_prompt = [
                {
                    "type": "text",
                    "text": self.get_system_prompt()
                }
            ]
            user_prompt = []
            
            if self.include_outline:
                if self.use_zh:
                    _user_prompt = f"""
# 报告大纲
{outline}
"""
                else:
                    _user_prompt = f"""
# Report Outline
{outline}
"""
                
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt
                    }
                )
            
            
            idx, title, content, search_from = document["id"], document["title"], document["content"], document["search_from"]
            if "google" in search_from:
                detail = await self.get_detail(url=document["url"])
                if self.use_zh:
                    _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
摘要：{content}
内容：{detail}
"""
                else:
                    _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Abstract: {content}
Content: {detail}
"""
            
            else:
                detail = ""
                if self.use_zh:
                    _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
内容：{content}
"""
                else:
                    _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Content: {content}
"""
            
            user_prompt.append(
                {
                    "type": "text",
                    "text": _user_prompt,
                }
            )
            
            if self.include_search_query:
                if self.use_zh:
                    _user_prompt = f"""
# 搜索词
{search_query}
"""
                else:
                    _user_prompt = f"""
# Search Query Terms
{search_query}
"""
                
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt
                    }
                )
            
            
            if self.include_user_query:
                if self.use_zh:
                    _user_prompt = f"""
# 用户提问
{query_text}
"""
                else:
                    _user_prompt = f"""
# User Question
{query_text}
"""
                
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt
                    }
                )
            
            cycler = ApiKeyCycler(api_key_list=list(DIRECTLLM_API_KEY_USER.values()))
            response = await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                cycler=cycler,
                return_cot=False,
            )
        
        elif "gemini" in self.model.model_name:
            system_prompt = [{"text": self.get_system_prompt()}]
            user_prompt = []
            
            if self.include_outline:
                if self.use_zh:
                    _user_prompt = f"""
# 报告大纲
{outline}
"""
                else:
                    _user_prompt = f"""
# Report Outline
{outline}
"""
                
                user_prompt.append({"text": _user_prompt})
            
            idx, title, content, search_from = document["id"], document["title"], document["content"], document["search_from"]
            if "google" in search_from:
                detail = await self.get_detail(url=document["url"])
                if self.use_zh:
                    _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
摘要：{content}
内容：{detail}
"""
                else:
                    _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Abstract: {content}
Content: {detail}
"""
            
            else:
                detail = ""
                if self.use_zh:
                    _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
内容：{content}
"""
                else:
                    _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Content: {content}
"""
            
            user_prompt.append({"text": _user_prompt})
            
            if self.include_search_query:
                if self.use_zh:
                    _user_prompt = f"""
# 搜索词
{search_query}
"""
                else:
                    _user_prompt = f"""
# Search Query Terms
{search_query}
"""
                
                user_prompt.append({"text": _user_prompt})
            
            
            if self.include_user_query:
                if self.use_zh:
                    _user_prompt = f"""
# 用户提问
{query_text}
"""
                else:
                    _user_prompt = f"""
# User Question
{query_text}
"""
                user_prompt.append({"text": _user_prompt})
            
            response = await self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                gemini_api_key=GEMINI_API_KEY,
                directllm_api_key=DIRECTLLM_API_KEY_USER["tusen"],
                return_cot=False,
            )
        
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        if len(response) > 0:
            document["detail"] = detail
            document["summary"] = response["summary"]
            if self.use_evidence_as_key:
                document["evidences"] = response["evidences"]
            else:
                document["evidence"] = response["evidence"]
        return document
    
    
    async def act(
        self,
        input_dict: Dict,
        turn_id: int,
        print_concurrent: bool = False,
    ):
        sem = asyncio.Semaphore(self.max_concurrent)
        num_active_task = 0
        active_lock = asyncio.Lock()
        
        query_text = input_dict["query_text"]
        outline = input_dict.get(f"outline_turn_{turn_id}", "")
        search_query = input_dict[f"search_query_turn_{turn_id}"]
        
        async def post_and_generate(
            doc_id: int,
            document: Dict,
        ):
            nonlocal num_active_task
            async with sem:
                async with active_lock:
                    num_active_task += 1
                if print_concurrent:
                    print(f"number of active tasks: {num_active_task}")
                try:
                    document = await self.post_request(
                        query_text=query_text,
                        outline=outline,
                        search_query=search_query,
                        document=document
                    )
                    input_dict[f"search_result_turn_{turn_id}"][doc_id] = document
                finally:
                    async with active_lock:
                        num_active_task -= 1
        
        tasks = [
            asyncio.create_task(
                post_and_generate(\
                    doc_id=doc_id,
                    document=document
                )
            ) for doc_id, document in enumerate(input_dict[f"search_result_turn_{turn_id}"])
        ]
        await asyncio.gather(*tasks)
        
        return input_dict
    
