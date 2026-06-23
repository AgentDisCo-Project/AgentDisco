import os
import aiohttp
import asyncio
import json
import time
import gin
import logging
import httpx
import sys
sys.path.append('.')

from typing import Dict
from tqdm import tqdm
from dotenv import load_dotenv
from tool.DocParserService import WebParser


load_dotenv('./api/utils/keys.env')


@gin.configurable()
class SandboxSearch:
    def __init__(
        self,
        num_searches: int = 10,
        max_retries: int = 5,
        retry_delay: int = 3,
        encode_idx_with_prefix: bool = False,
        return_search_results: bool = False,
        reference_citation_path: str = "./sandbox/deep_research_bench_zh_citations.jsonl"
    ):
        self.citations_dict = self.load_citations(reference_citation_path)
        self.webparser = WebParser()
    
    
    @staticmethod
    def load_citations(
        citation_path: str,
    ):
        citations_dict = {}
        with open(citation_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                citations_dict[data['query']] = data['citations']
        return citations_dict
    
    
    async def act(
        self,
        input_dict: Dict,
        input_key: str = "query_text",
    ):
        query = input_dict.get(input_key, [])[-1] or input_dict.get("query_text", [])[-1] or input_dict.get("query", "")[-1]
        response = self.citations_dict.get(query, [])
        # 并发执行所有网页内容获取任务
        # 直接创建异步任务列表
        detail_tasks = [
            self.webparser.call(params={"url": res.get("link", "")})
            for res in response
        ]
        
        details = await asyncio.gather(*detail_tasks)
        
        # 构建docs
        docs = []
        for idx, (res, detail) in enumerate(zip(response, details)):
            image_list = []
            if len(res.get("imageUrl", "")) > 0:
                image_meta = {
                    "fileId": -1,
                    "url": res.get("imageUrl"),
                    "width": -1,
                    "height": -1,
                }
                image_list.append(image_meta)
            
            doc = {
                "id": f"sandbox_{query[:10]}_{idx}",
                "search_from": f"sandbox_{query[:10]}",
                "content": res.get("snippet", "") or detail,
                "title": res.get("title", ""),
                "url": res.get("link", ""),
                "date": res.get("date", ""),
                "note_type": "images",
                "video": "",
                "images": image_list,
                "like_count": -1,
                "collect_count": -1,
                "view_count": -1,
                "comments": [],
                "confidence": -1,
                "detail": detail,
            }
            docs.append(doc)
        
        return docs
    
    

if __name__ == "__main__":
    async def main():
        service = SandboxSearch(
            reference_citation_path="./sandbox/deep_research_bench_zh_citations.jsonl"
        )
        input_dict = dict()
        input_dict["query_text"] = ["收集整理目前国际综合实力前十的保险公司的相关资料，横向比较各公司的融资情况、信誉度、过往五年的增长幅度、实际分红、未来在中国发展潜力等维度，并为我评估出最有可能在未来资产排名靠前的2-3家公司"]
        results = await service.act(
            input_dict=input_dict
        )
        cnt404 = []
        cnt403 = []
        for res in results:
            if "404" in res["content"]:
                cnt404.append(res["url"])
            if "403" in res["content"]:
                cnt403.append(res["url"])
        good_results = []
        for res in results:
            if "404" in res["content"] or "403" in res["content"]:
                continue 
            good_results.append(res["url"])
            
        breakpoint()
        print(results)
    
    asyncio.run(main())
    