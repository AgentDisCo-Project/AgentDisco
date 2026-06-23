import os
import gin
import jinja2
import asyncio
import re
import httpx
import logging
import json
import time
import copy
import numpy as np
import sys
sys.path.append('.')

from tqdm import tqdm
from typing import Union, Optional, List, Dict
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.NoteCaptionService import NoteCaption
from api.utils.key_operator import ApiKeyCycler
from api.EvidenceMergeService import EvidenceMerger
from api.utils.string_operator import json_fix
from agent.BaseAgent import BasicAgent
from tool.DocParserService import WebParser
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
JINA_API_KEYS = os.environ.get("JINA_API_KEYS", "{}")
JINA_API_KEYS = json.loads(JINA_API_KEYS)
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")


@gin.configurable()
class JudgeSearchQueryDiversityAPI:
    def __init__(
        self,
        model_name: str = "",
        max_retries: int = 5,
        retry_delay: int = 3,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        use_zh: bool = False,
        system_template_dir: str = "./template",
        system_template_en_file: str = "JudgeSearchQueryDiversity_EN.jinja2",
        system_template_zh_file: str = "JudgeSearchQueryDiversity_ZH.jinja2",
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
        self.jinja_file = system_template_en_file if not self.use_zh else system_template_zh_file
    

    def get_system_prompt(
        self,
    ):
        template_vars = {}
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    

    def parser_response(
        self,
        response: str,
    ):
        response = json_fix(response)
        response = json.loads(response)
        return response

    
    def check_func(
        self,
        response: str,
    ):
        return self.parser_response(response)
    

    async def post_request(
        self,
        query_text: str,
        search_docs: Dict,
        outline: str = "",
        blueprints: List = "",
    ):
        judge_search_query_st = time.time()
        if "gemini" in self.model.model_name:
            system_prompt = [{"text": self.get_system_prompt()}]
            user_prompt = []

            if len(outline) > 0:
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
            
            if self.use_zh:
                _user_prompt = f"""
# 大纲要点列表
{blueprints}
"""
            else:
                _user_prompt = f"""
# Report Outline Blueprints
{blueprints}
"""
            user_prompt.append({"text": _user_prompt})

            if self.use_zh:
                _user_prompt = f"""
# 下面是所有的搜索query
"""
            else:
                _user_prompt = f"""
# Below are all search queries.
"""
            user_prompt.append({"text": _user_prompt})

            for idx, search_doc in enumerate(search_docs):
                search_query = search_doc["query"]

                judge_scores = []
                snippets = []
                for doc in search_doc["docs"]:
                    judge_scores.append(doc["judge"])
                    snippets.append(doc["summary"][:50])
                
                if self.use_zh:
                    _user_prompt = f"""
## 搜索query-{idx}
query: {search_query}
返回的文档数量(最大是10): {len(judge_scores)}
文档的平均相似度: {np.mean(judge_scores)}
文档的相似度标准差: {np.std(judge_scores)}
文档的最大相似度: {np.max(judge_scores)}
文档的最小相似度: {np.min(judge_scores)}
文档的所有相似度数据: {judge_scores}
文档的一些snippets: {snippets}
"""
                else:
                    _user_prompt = f"""
## Search query-{idx}
Query: {search_query}
Number of returned documents (maximum is 10): {len(judge_scores)}
Average similarity of documents: {np.mean(judge_scores)}
Standard deviation of document similarity: {np.std(judge_scores)}
Maximum similarity of documents: {np.max(judge_scores)}
Minimum similarity of documents: {np.min(judge_scores)}
All similarity scores of documents: {judge_scores}
Some snippets of the documents: {snippets}
"""

                user_prompt.append({"text": _user_prompt})

            
            response = await self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                return_cot=False,
            )

        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        judge_search_query_et = time.time()
        logging.info(f"judge_search_query costs: {judge_search_query_et-judge_search_query_st}")

        return response
            

    async def act(
        self,
        input_dict: Dict,
        turn_id: int,
    ):
        query_text = input_dict["query_text"]
        outline = input_dict.get(f"outline_turn_{turn_id}", "")
        blueprints = input_dict.get(f"blueprints_turn_{turn_id}", "")
        search_docs = input_dict.get(f"search_result_turn_{turn_id}", "")

        response = await self.post_request(
            query_text=query_text,
            search_docs=search_docs,
            outline=outline,
            blueprints=blueprints,
        )
        input_dict[f"judge_search_query_turn_{turn_id}"] = response
        score = response["evaluation"]["overall"]["score"]
        return score, input_dict
        