import os
import gin
import asyncio
import jinja2
import json
import logging
import time
import sys
sys.path.append('.')

from typing import List,Dict
from dotenv import load_dotenv
from typing import Optional
from run_workflow import select_search
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.NoteRerankerService import NoteReranker
from api.NoteJudgeService import NoteJudge
from api.utils.url_operator import compress_and_convert_base64, compress_url
from api.utils.key_operator import ApiKeyCycler
from api.utils.string_operator import json_fix

load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


@gin.configurable()
class NoteFilter:
    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        retry_delay: int = 3,
        use_zh: bool = True,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file: str = "SearchFilter_EN.jinja2",
        system_template_zh_file: str = "SearchFilter_ZH.jinja2",
        system_template_en_file_return_score: str = "SearchFilterScore_EN.jinja2",
        system_template_zh_file_return_score: str = "SearchFilterScore_ZH.jinja2",
        top_k: int = 50,
        include_search_query: bool = False,
        return_score: bool = True,
        use_flash_filter: bool = True,
        search_engine: str = "google",
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.model = CustomizeChatGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
        )
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        if return_score:
            self.jinja_file = system_template_zh_file_return_score if use_zh else system_template_en_file_return_score
        else:
            self.jinja_file = system_template_zh_file if use_zh else system_template_en_file
        self.return_score = return_score
        self.use_flash_filter = use_flash_filter
        
        self.include_search_query = include_search_query
        self.top_k = top_k
        self.use_zh = use_zh
        
        assert search_engine in ("google", "xiaohongshu", "combine", "knowledge", "sandbox"), f"Unsupported search engine {search_engine}"
        self.search_engine = search_engine
        
        
    def get_system_prompt(
        self,
        include_search_query: bool = False,
    ):
        template_vars = {
            "include_search_query": include_search_query
        }
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    async def post_request(
        self,
        query_text: str,
        search_query: List[str],
        results: List[Dict],
    ):
        if "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            system_prompt = [
                {
                    "type": "text",
                    "text": self.get_system_prompt()
                }
            ]
            user_prompt = []
            
            if self.use_zh:
                _user_prompt = """
# 外源搜索结果
"""
            
            else:
                _user_prompt = """
# External Search Results
"""
            
            user_prompt.append(
                {
                    "type": "text",
                    "text": _user_prompt,
                }
            )
            
            results_dict = {}
            for idx, doc in enumerate(results):
                title, content = doc["title"], doc["content"]
                if self.use_zh:
                    _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
标题：{title}
摘要：{content}
"""
                else:
                    _user_prompt = f"""
## External Search Document {idx}
title: {title}
content: {content}
"""
                
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt,
                    }
                )
                
                results_dict[idx] = doc
            
            include_search_query = self.include_search_query and len(search_query) > 0
            if include_search_query:
                search_query = json.dumps(search_query, ensure_ascii=False)
                if self.use_zh:
                    _user_prompt = f"""
# 查询词
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
            
            results = self.parser_response(
                response=response,
                results_dict=results_dict,
            )
        
        
        elif "gemini" in self.model.model_name:
            system_prompt = [{"text": self.get_system_prompt()}]
            user_prompt = []
            
            if self.use_zh:
                _user_prompt = """
# 外源搜索结果
"""
            
            else:
                _user_prompt = """
# External Search Results
"""
            
            user_prompt.append({"text": _user_prompt})
            results_dict = {}
            for idx, doc in enumerate(results):
                title, content = doc["title"], doc["content"]
                if self.use_zh:
                    _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
标题：{title}
摘要：{content}
"""
                else:
                    _user_prompt = f"""
## External Search Document {idx}
title: {title}
content: {content}
"""
                
                user_prompt.append({"text": _user_prompt})
                results_dict[idx] = doc
            
            include_search_query = self.include_search_query and len(search_query) > 0
            if include_search_query:
                search_queries = json.dumps(search_query, ensure_ascii=False)
                if self.use_zh:
                    _user_prompt = f"""
# 查询词
{search_queries}
"""
                
                else:
                    _user_prompt = f"""
# Search Query Terms
{search_queries}
"""
                user_prompt.append({"text": _user_prompt})
            
            
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
            
            results = self.parser_response(
                response=response,
                results_dict=results_dict,
            )
        
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        return results
    
    
    def parser_response(
        self,
        response: List[Dict],
        results_dict: Optional[Dict],
    ):
        if not self.return_score:
            results_list = []
            for _response in response:
                idx, relevance = _response["id"], _response["relevance"]
                if relevance.lower() in ("low", "medium"):
                    continue
                if idx in results_dict:
                    results_list.append(results_dict[idx])
                else:
                    logging.info(f"Unavailable search results idx: {idx}")
            if len(results_list) >= self.top_k:
                logging.info("high is enough")
                return results_list[:self.top_k]
            
            logging.info("need high and medium")
            for _response in response:
                idx, relevance = _response["id"], _response["relevance"]
                if relevance.lower() in ("low", "high"):
                    continue
                if idx in results_dict:
                    results_list.append(results_dict[idx])
                else:
                    logging.info(f"Unavailable search results idx: {idx}")
                return results_list[:self.top_k]
        else:
            # 首先收集所有有效的响应数据
            valid_responses = []
            for _response in response:
                idx, relevance = _response["id"], _response["score"]
                if idx in results_dict:
                    valid_responses.append({
                        "id": idx,
                        "score": relevance,
                        "result": results_dict[idx]
                    })
                else:
                    logging.info(f"Unavailable search results idx: {idx}")
            
            # 按score从大到小排序并取前50个
            sorted_responses = sorted(valid_responses, key=lambda x: x["score"], reverse=True)[:self.top_k]
            
            # 构建最终的results_list
            results_list = [item["result"] for item in sorted_responses]
            return results_list
         
         
    def check_func(
        self,
        response: str,
    ):
        response = json_fix(response)
        response = json.loads(response)
        return response
        
        
    async def post_request_flash(
        self,
        query_text: str,
        search_query: List[str],
        results: List[Dict],
        use_search_queries_as_query: bool = False,
    ):
        if use_search_queries_as_query:
            query_text = ",".join([query_text] + search_query)
        note_judge = NoteJudge(
            use_query_modality="text",
            use_note_modality="text",
        )
        input_dict = {
            "query_text": query_text,
            "search_results": results
        }
        input_dict = await note_judge.act(input_dict=input_dict)
        note_reranker = NoteReranker(
            use_query_modality="text",
            use_note_modality="text"
        )
        input_dict = await note_reranker.act(input_dict=input_dict)
        input_dict = select_search(input_dict=input_dict)
        return input_dict["selected_search_results"]
    
    
    async def act(
        self,
        input_dict: Dict,
        turn_id: int,
    ):
        filter_results_st = time.time()
        xiaohongshu_search_query = input_dict.get("xiaohongshu_search_query", [])
        google_search_query = input_dict.get("google_search_query", [])
        if self.search_engine == "combine":
            search_query = xiaohongshu_search_query + google_search_query
        elif self.search_engine == "google":
            search_query = google_search_query
        else:
            search_query = xiaohongshu_search_query
        
        query_text = input_dict.get("query_text", "") or input_dict.get("query", "")
        results = input_dict[f"search_result_turn_{turn_id}"]
        if self.use_flash_filter:
            results = await self.post_request_flash(
                query_text=query_text,
                search_query=search_query,
                results=results,
            )
        else:
            results = await self.post_request(
                query_text=query_text,
                search_query=search_query,
                results=results,
            )
        filter_results_et = time.time()
        logging.info(f"number of search results after filtering is {len(results)}, top-k is {self.top_k}")
        logging.info(f"finish filter search results costs: {filter_results_et-filter_results_st}")
        input_dict[f"record_search_result_turn_before_filter_{turn_id}"] = input_dict[f"search_result_turn_{turn_id}"]
        input_dict[f"search_result_turn_{turn_id}"] = results
        input_dict[f"search_query_turn_{turn_id}"] = search_query
        return input_dict