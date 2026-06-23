import gin
import jinja2
import re
import os
import json
import time
import logging
import asyncio
import sys
sys.path.append('.')

from functools import partial
from datetime import datetime
from typing import Optional, Union, Dict, List
from agent.BaseAgent import BasicAgent
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.NoteFilterService import NoteFilter
from api.utils.key_operator import ApiKeyCycler
from api.utils.string_operator import json_fix
from tool.WebSearchService import WebSearch
from dotenv import load_dotenv

load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)


@gin.configurable()
class DisentangledOutlineJudge:
    def __init__(
        self,
        model_name: str = "",
        max_retries: int = 5,
        retry_delay: int = 3,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file: str = "DisentangledOutlineJudgeQA_EN.jinja2",
        system_template_zh_file: str = "DisentangledOutlineJudgeQA_ZH.jinja2",
        system_template_en_file_combine: str = "DisentangledOutlineJudgeCombineQA_EN.jinja2",
        system_template_zh_file_combine: str = "DisentangledOutlineJudgeCombineQA_ZH.jinja2",
        system_template_en_file_evidence: str = "DisentangledOutlineJudgeEvidenceQA_EN.jinja2",
        system_template_zh_file_evidence: str = "DisentangledOutlineJudgeEvidenceQA_ZH.jinja2",
        system_template_en_file_evidence_combine: str = "DisentangledOutlineJudgeEvidenceCombineQA_EN.jinja2",
        system_template_zh_file_evidence_combine: str = "DisentangledOutlineJudgeEvidenceCombineQA_ZH.jinja2",
        include_prev_query: bool = False,
        num_searches: int = 10,
        llm_filter_model_name: str = "",
        search_engine: str = "google",
        top_k: int = 50,
        return_score: bool = True,
        use_flash_filter: bool = False,
        use_evidence_as_key: bool = False,
        use_zh: bool = False,
        add_origin_query: bool = True,
        need_filter: bool = False,
        outline_judge_threshold: int = -1,
    ):
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
        self.use_zh = use_zh
        if search_engine == "combine":
            if not use_evidence_as_key:
                self.jinja_file = system_template_zh_file_combine if self.use_zh else system_template_en_file_combine
            else:
                self.jinja_file = system_template_zh_file_evidence_combine if self.use_zh else system_template_en_file_evidence_combine
        else:
            if not use_evidence_as_key:
                self.jinja_file = system_template_zh_file if self.use_zh else system_template_en_file
            else:
                self.jinja_file = system_template_zh_file_evidence if self.use_zh else system_template_en_file_evidence
        self.include_prev_query = include_prev_query
        
        self.searcher = WebSearch(
            use_zh=use_zh,
            num_searches=num_searches,
            search_engine=search_engine,
        )
        self.filter = NoteFilter(
            model_name=llm_filter_model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            include_search_query=True,
            search_engine=search_engine,
            top_k=top_k,
            return_score=return_score,
            use_flash_filter=use_flash_filter,
        )
        self.search_engine = search_engine
        self.use_evidence_as_key = use_evidence_as_key
        self.add_origin_query = add_origin_query
        self.num_searches = num_searches
        self.need_filter = need_filter
        self.outline_judge_threshold = outline_judge_threshold
        
    
    def get_system_prompt(
        self,
    ):
        template_vars = {
            "curr_date": datetime.now().strftime("%Y年%m月%d日"),
            "search_engine": self.search_engine,
        }
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    def check_func(
        self,
        response: str,
    ):
        return self.parser_response(response=response)
    
    
    def parser_response(
        self,
        response: str,
    ):
        response = json_fix(response)
        response = json.loads(response)
        
        def smart_split_query(search_query: List[str]):
            """对列表中包含逗号的元素进行进一步切分"""
            result = []
            for item in search_query:
                if isinstance(item, str) and (',' in item or '，' in item):
                    # 如果元素包含逗号，进一步切分
                    sub_items = re.split('[,，]', item)
                    result.extend([sub_item.strip() for sub_item in sub_items if sub_item.strip()])
                else:
                    # 如果不包含逗号，直接添加
                    result.append(item.strip() if isinstance(item, str) else item)
            return result
        
        if not isinstance(response, Dict):
            raise ValueError()
        
        if "rating" not in response or "justification" not in response:
            raise ValueError()
        if self.search_engine == "combine":
            if "xiaohongshu_search_query" not in response or "google_search_query" not in response:
                raise ValueError()
            response["xiaohongshu_search_query"] = smart_split_query(response["xiaohongshu_search_query"])
            response["google_search_query"] = smart_split_query(response["google_search_query"])
        else:
            if "search_query" not in response:
                raise ValueError()
            response["search_query"] = smart_split_query(response["search_query"])
        if not (isinstance(response["rating"], float) or isinstance(response["rating"], int)) or not isinstance(response["justification"], str) or not isinstance(response["search_query"], List):
            raise ValueError()
        if response["rating"] > 10. or response["rating"] < 0.:
            raise ValueError()
        return response
    
    
    async def act(
        self,
        input_dict: Dict,
        turn_id: int,
    ):
        outline_judge_st = time.time()
        query_text = input_dict["query_text"]
        include_prev_query = (turn_id > 0) and self.include_prev_query
        if "qwen3-next-80b-a3b-instruct" in self.model.model_name:
            system_prompt = self.get_system_prompt()
            user_prompt = ""
            
            if include_prev_query:
                prev_queries = []
                for t in range(turn_id):
                    prev_queries.extend(input_dict[f"search_query_turn_{t}"])
                if self.use_zh:
                    _user_prompt = f"""
# 历史的搜索词
{prev_queries}
"""
                
                else:
                    _user_prompt = f"""
# Previous Search Query Terms
{prev_queries}
"""
                
                user_prompt += _user_prompt
            
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
            
            user_prompt += _user_prompt
            
            outline = input_dict.get(f"outline_turn_{turn_id-1}", "")
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
            user_prompt += _user_prompt
            
            cycler = ApiKeyCycler(api_key_list=list(DIRECTLLM_API_KEY_USER.values()))
            
            response = await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                cycler=cycler,
                return_cot=False,
            )
            
        
        elif "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            system_prompt = self.get_system_prompt()
            user_prompt = ""
            
            if include_prev_query:
                prev_queries = []
                for t in range(turn_id):
                    prev_queries.extend(input_dict[f"search_query_turn_{t}"])
                if self.use_zh:
                    _user_prompt = f"""
# 历史的搜索词
{prev_queries}
"""
                
                else:
                    _user_prompt = f"""
# Previous Search Query Terms
{prev_queries}
"""
                
                user_prompt += _user_prompt
            
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
            
            user_prompt += _user_prompt

            outline = input_dict.get(f"outline_turn_{turn_id-1}", "")
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
            user_prompt += _user_prompt
            
            cycler = ApiKeyCycler(api_key_list=list(DIRECTLLM_API_KEY_USER.values()))
            
            response = await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                cycler=cycler,
                return_cot=False,
            )
        
        elif "gemini" in self.model.model_name:
            system_prompt = [
                {
                    "text": self.get_system_prompt()
                }
            ]
            user_prompt = []
            
            if include_prev_query:
                prev_queries = []
                for t in range(turn_id):
                    prev_queries.extend(input_dict[f"search_query_turn_{t}"])
                if self.use_zh:
                    _user_prompt = f"""
# 历史的搜索词
{prev_queries}
"""
                
                else:
                    _user_prompt = f"""
# Previous Search Query Terms
{prev_queries}
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
            
            outline = input_dict.get(f"outline_turn_{turn_id-1}", "")
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
            
            response = await self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.parser_response,
                return_cot=False,
            )
        
        elif "gpt-oss" in self.model.model_name:
            system_prompt = self.get_system_prompt()
            user_prompt = ""
            
            if include_prev_query:
                prev_queries = []
                for t in range(turn_id):
                    prev_queries.extend(input_dict[f"search_query_turn_{t}"])
                if self.use_zh:
                    _user_prompt = f"""
# 历史的搜索词
{prev_queries}
"""
                
                else:
                    _user_prompt = f"""
# Previous Search Query Terms
{prev_queries}
"""
                
                user_prompt += _user_prompt
            
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
            
            user_prompt += _user_prompt
            
            outline = input_dict.get(f"outline_turn_{turn_id-1}", "")
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
            user_prompt += _user_prompt
            
            cycler = ApiKeyCycler(api_key_list=list(DIRECTLLM_API_KEY_USER.values()))
            
            response = await self.model.chat_gpt_oss(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                cycler=cycler,
            )
        
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        outline_judge_et = time.time()
        logging.info(f"outline judge costs: {outline_judge_et-outline_judge_st}")
        
        if self.add_origin_query:
            if self.search_engine == "combine":
                response["xiaohongshu_search_query"].append(query_text)
                response["google_search_query"].append(query_text)
            else:
                response["search_query"].append(query_text)
        
        if self.search_engine == "sandbox":
            response["search_query"].append(query_text)
        
        input_dict[f"judge_turn_{turn_id}"] = response
        if self.search_engine == "combine":
            input_dict[f"xiaohongshu_search_query_turn_{turn_id}"] = response["xiaohongshu_search_query"]
            input_dict[f"google_search_query_turn_{turn_id}"] = response["google_search_query"]
        else:
            input_dict[f"search_query_turn_{turn_id}"] = response["search_query"]
        
        if (self.outline_judge_threshold > 0) and (input_dict[f"judge_turn_{turn_id}"]["rating"] >= self.outline_judge_threshold):
            return input_dict
        
        if len(response.get("search_query", [])) > 0 or len(response.get("xiaohongshu_search_query", [])) > 0 and len(response.get("google_search_query", [])) > 0:
            query_search_st = time.time()
            search_docs = await self.searcher.call({
                "query_text": input_dict["query_text"],
                "search_query": response.get("search_query", []),
                "turn_id": f"turn_{turn_id}",
                "xiaohongshu_search_query": response.get("xiaohongshu_search_query", []),
                "google_search_query": response.get("google_search_query", [])
            })
            input_dict[f"search_result_turn_{turn_id}"] = search_docs
            logging.info(f"number of total search results at turn {turn_id} is: {len(search_docs)}")
            query_search_et = time.time()
            logging.info(f"query search costs: {query_search_et-query_search_st}")
            if self.need_filter:
                filter_st = time.time()
                input_dict = await self.filter.act(input_dict=input_dict, turn_id=turn_id)
                filter_et = time.time()
                logging.info(f"filter out search results costs: {filter_et-filter_st}")
                
        return input_dict
    


if __name__ == "__main__":
    async def main():
        service = DisentangledOutlineJudge(
            model_name="gemini-3-pro",
            llm_filter_model_name="gemini-3-pro",
            num_searches=10,
            use_zh=True,
            search_engine="sandbox",
            top_k=50,
            return_score=True,
            use_flash_filter=False,
            use_evidence_as_key=False,
            add_origin_query=False,
            need_filter=False,
        )
        sample_data_path = "/mnt/tidalfs-bdsz01/usr/tusen/search-agent-dev/dev/0109/ragengine/data/sample_data.json"
       
        with open(sample_data_path, 'r', encoding='utf-8') as f:
            input_dict = json.load(f)
        
        input_dict["query"] = "收集整理目前中国9阶层实际收入和财务状况，特别研究得出中国的中产有哪些特点，实际中产人数，财力等等"
        input_dict = await service.act(
            input_dict=input_dict,
            turn_id=1,
        )
        breakpoint()
        print(input_dict)
    
    asyncio.run(main())
    