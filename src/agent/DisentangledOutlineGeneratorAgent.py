import gin
import jinja2
import re
import time
import os
import json
import logging
import sys
sys.path.append('.')

from functools import partial
from datetime import datetime
from typing import Optional, Union, Dict, List
from agent.BaseAgent import BasicAgent
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.DisentangledOutlineJudgeService import DisentangledOutlineJudge
from api.DisentangledOutlineGeneratorService import DisentangledOutlineGenerator
from api.utils.key_operator import ApiKeyCycler
from api.utils.string_operator import json_fix
from tool.WebSearchService import WebSearch
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


@gin.configurable()
class DisentangledOutlineGenerator:
    def __init__(
        self,
        use_zh: bool = False,
        model_name: str = "",
        max_retries: int = 5,
        retry_delay: int = 3,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        include_prev_query: bool = True,
        include_prev_outline: bool = True,
        include_prev_judge: bool = True,
        include_summary: bool = False,
        num_searches: int = 10,
        search_engine: str = "google",
        llm_filter_model_name: str = "",
        top_k: int = 50,
        return_score: bool = True,
        use_flash_filter: bool = False,
        use_evidence_as_key: bool = False,
        max_outline_generator_turns: int = 5,
        outline_judge_threshold: int = 8,
        add_origin_query: bool = True,
    ):
        self.outline_judge = DisentangledOutlineJudge(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
            include_prev_query=include_prev_query,
            num_searches=num_searches,
            llm_filter_model_name=llm_filter_model_name,
            search_engine=search_engine,
            top_k=top_k,
            return_score=return_score,
            use_flash_filter=use_flash_filter,
            use_evidence_as_key=use_evidence_as_key,
            use_zh=use_zh,
            add_origin_query=add_origin_query,
        )
        
        self.outline_generator = DisentangledOutlineGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
            include_prev_outline=include_prev_outline,
            include_prev_judge=include_prev_judge,
            include_summary=include_summary,
        )
        
        self.use_zh = use_zh
        self.max_outline_generator_turns = max_outline_generator_turns
        self.outline_judge_threshold = outline_judge_threshold
    
    
    @staticmethod
    def divide_outline_into_chunks(outline: str):
        def split_by_header_level(content: str, header_pattern: str):
            """按照指定的标题级别分割内容"""
            if not content.strip():
                return []
            
            lines = content.split('\n')
            chunks = []
            curr_chunk = {'content': []}  # 初始化curr_chunk
            
            for line in lines:
                # 检查是否是指定级别的标题
                if re.match(header_pattern, line.strip()):
                    # 如果当前chunk有内容，先保存
                    if curr_chunk['content']:
                        curr_chunk['content'] = '\n'.join(curr_chunk['content'])
                        chunks.append(curr_chunk)
                    
                    # 开始新的chunk
                    curr_chunk = {
                        'content': [line]  # 将标题行加入content
                    }
                else:
                    # 将内容添加到当前chunk
                    curr_chunk['content'].append(line)
            
            # 添加最后一个chunk
            if curr_chunk['content']:
                curr_chunk['content'] = '\n'.join(curr_chunk['content'])
                chunks.append(curr_chunk)
            
            return chunks
        
        def contains_level2_title(content: str):
            """检查chunk是否有二级标题（## ）"""
            lines = content.split('\n')
            for line in lines:
                if re.match(r'^##\s+', line.strip()):
                    return True
            return False
        
        if not outline or not outline.strip():
            return []
        
        # 首先按一级标题分割
        chunks = split_by_header_level(outline, r'^#\s+')
        
        # 如果只有一个chunk，按二级标题重新分割
        if len(chunks) <= 1:
            chunks = split_by_header_level(outline, r'^##\s+')
            
            if len(chunks) >= 2 and not contains_level2_title(chunks[0]['content']):
                # 将第一个chunk与第二个chunk合并
                merged_content = chunks[0]['content'] + '\n' + chunks[1]['content']
                merged_chunk = {
                    'content': merged_content
                }
                # 更新chunks列表：第一个chunk变成合并后的chunk，剩余的chunk从第三个开始
                chunks = [merged_chunk] + chunks[2:]
        
        for idx, chunk in enumerate(chunks):
            chunk["id"] = idx
        return chunks
    

    async def act(
        self,
        input_dict: Dict,
    ):
        pass

