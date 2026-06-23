import os
import jinja2
import json
import logging
import time
import asyncio
import sys
sys.path.append('.')
sys.path.append('./api')

from typing import Union, Optional, Dict, List
from api.RednoteTextSearchService import RedNoteTextSearch
from api.GoogleSearchService import GoogleTextImageSearch
from api.KnowledgeBaseSearchService import KnowledgeBaseSearch
from api.SandboxSearchService import SandboxSearch
from tool.BaseToolService import register_tool, BasicTool
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


@register_tool("web_search", allow_overwrite=True)
class WebSearch(BasicTool):
    name = 'web_search'
    description_en = 'Search for information from the internet.'
    description_zh = '从互联网中搜索信息。'
    parameters = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
            },
            'searcher': {
                'type': 'string',
            },
            'num_searches': {
                'type': 'integer',
            },
        },
        'required': ['query'],
    }
    
    def __init__(
        self,
        cfg: Optional[Dict] = None,
        use_zh: bool = False,
        num_searches: int = 10,
        search_engine: str = "google",
        use_auto_language: bool = True,
    ):
        super().__init__(
            cfg=cfg,
            use_zh=use_zh,
        )
        assert search_engine in ("google", "xiaohongshu", "combine", "knowledge", "sandbox"), f"Unsupported search engine {search_engine}"
        self.search_engine = search_engine
        self.num_searches = num_searches
        self.use_auto_language = use_auto_language
        
        
    @staticmethod
    def contains_chinese_basic(
        text: str
    ):
        return any('\u4E00' <= char <= '\u9FFF' for char in text)

        
    async def search_query(
        self,
        query: List[str] = None,
        idx_prefix: str = "",
        xiaohongshu_query: List[str] = None,
        google_query:  List[str] = None,
    ):
        logging.info(f"number of search per query {self.num_searches}")
        if self.search_engine == "google":
            if not self.use_auto_language:
                searcher = GoogleTextImageSearch(
                    num_searches=self.num_searches,
                    search_type="text",
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay,
                    country="us" if not self.use_zh else "cn",
                    language="en" if not self.use_zh else "zh-cn",
                    encode_idx_with_prefix=True,
                    return_search_results=True,
                )
            
            else:
                searcher = GoogleTextImageSearch(
                    num_searches=self.num_searches,
                    search_type="text",
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay,
                    country="us" if not self.contains_chinese_basic(",".join(query)) else "cn",
                    language="en" if not self.contains_chinese_basic(",".join(query)) else "zh-cn",
                    encode_idx_with_prefix=True,
                    return_search_results=True,
                )
            
            results = await searcher.act(
                input_dict={
                    "query_text": query,
                },
                input_key="query_text",
                idx_prefix=idx_prefix,
            )
        
        # elif searcher == "baidu":
        #     self.searcher = BaiduImageSearch(
        #         num_searches=num_searches or 10,
        #         search_type="text",
        #         max_retries=self.max_retries,
        #         retry_delay=self.retry_delay,
        #         encode_idx_with_prefix=True,
        #         return_search_results=True,
        #     )
        
        # elif searcher == "bocha":
        #     self.searcher = BochaTextSearch(
        #         num_searches=self.num_searches,
        #         max_retries=self.max_retries,
        #         retry_delay=self.retry_delay,
        #         web_search_type="web-search",
        #         encode_idx_with_prefix=True,
        #     )
        
        elif self.search_engine == "xiaohongshu":
            searcher = RedNoteTextSearch(
                user_id="life",
                num_searches=self.num_searches,
                disable_comment=True,
                disable_video=True,
                use_note_modality="all_images",
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                encode_idx_with_prefix=True,
                return_search_results=True,
            )
            
            results = await searcher.act(
                input_dict={
                    "query_text": query,
                },
                input_key="query_text",
                idx_prefix=idx_prefix,
            )
        
        elif self.search_engine == "knowledge":
            searcher = KnowledgeBaseSearch(
                num_searches=self.num_searches,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                encode_idx_with_prefix=True,
                return_search_results=True,
            )
            
            results = await searcher.act(
                input_dict={
                    "query_text": query,
                },
                input_key="query_text",
                idx_prefix=idx_prefix,
            )
        
        elif self.search_engine == "sandbox":
            searcher = SandboxSearch()
            
            results = await searcher.act(
                input_dict={
                    "query_text": query,
                },
                input_key="query_text",
            )
        
        elif self.search_engine == "combine":
            searcher = RedNoteTextSearch(
                user_id="life",
                num_searches=self.num_searches,
                disable_comment=True,
                disable_video=True,
                use_note_modality="text",
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                encode_idx_with_prefix=True,
                return_search_results=True,
            )
        
            xhs_results = await searcher.act(
                input_dict={
                    "query_text": xiaohongshu_query,
                },
                input_key="query_text",
                idx_prefix=idx_prefix,
            )
            
            searcher = GoogleTextImageSearch(
                num_searches=self.num_searches,
                search_type="text",
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                country="us" if not self.use_zh else "cn",
                language="en" if not self.use_zh else "zh-cn",
                encode_idx_with_prefix=True,
                return_search_results=True,
            )
        
            google_results = await searcher.act(
                input_dict={
                    "query_text": google_query,
                },
                input_key="query_text",
                idx_prefix=idx_prefix,
            )
            
            results = []
            merged_results = google_results + xhs_results
            for idx, res in enumerate(merged_results):
                res['id'] = f"{idx_prefix}_{idx}"
                results.append(res)
        
        else:
            raise NotImplementedError
                    
        return results
    

    async def call(
        self,
        params: Union[str, Dict],
    ):
        params = self.verify_json_format_args(params)
        idx_prefix = params.get("turn_id", "")
        
        search_query = params.get("search_query", [])
        xiaohongshu_search_query = params.get("xiaohongshu_search_query", [])
        google_search_query = params.get("google_search_query", [])
        logging.info(f"number of search search_query is {len(search_query)}")
        logging.info(f"number of xiaohongshu search query is {len(xiaohongshu_search_query)}")
        logging.info(f"number of google search query is {len(google_search_query)}")
        logging.info(f"number of search results per query: {self.num_searches}")
        search_query_st = time.time()
        results = await self.search_query(
            query=search_query,
            idx_prefix=idx_prefix,
            xiaohongshu_query=xiaohongshu_search_query,
            google_query=google_search_query,
        )
        logging.info(f"number of search results is {len(results)}")
        search_query_et = time.time()
        logging.info(f"finish search query costs: {search_query_et-search_query_st}")
        return results




if __name__ == "__main__":
    async def main():
        service = WebSearch(
            use_zh=True,
            num_searches=10,
            search_engine="xiaohongshu",
        )
        params = {
            "query_text": '湿气重怎么调理？',
            "turn_id": "turn_1",
            "search_query": ['脾虚湿气重 现代医学解读', '痰湿体质 代谢综合征 关系', '祛湿茶饮 副作用 禁忌', '艾灸祛湿 禁忌人群', '拔罐祛湿 穴位图', '压力大 脾胃不好 怎么调理', '肝郁脾虚 调理顺序', '健脾祛湿 懒人食谱', '办公室祛湿神器', '运动排湿 正确方法', '四神汤 功效与禁忌', '不同体质 祛湿方法 对比']
        }
        results = await service.call(
            params=params
        )
        breakpoint()
        print(results)
    
    asyncio.run(main())
    

    


