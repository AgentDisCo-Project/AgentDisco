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


load_dotenv('./api/utils/keys.env')
SEARCH_KNOWLEDGE_BUSINESS_NAME = os.environ.get("SEARCH_KNOWLEDGE_BUSINESS_NAME", "")
SEARCH_KNOWLEDGE_APP_ID = os.environ.get("SEARCH_KNOWLEDGE_APP_ID", "")
SEARCH_KNOWLEDGE_APP_KEY = os.environ.get("SEARCH_KNOWLEDGE_APP_KEY", "")



@gin.configurable()
class KnowledgeBaseSearch:
    def __init__(
        self,
        num_searches: int,
        max_retries: int = 5,
        retry_delay: int = 3,
        encode_idx_with_prefix: bool = False,
        return_search_results: bool = False,
    ):
        # export https_proxy=10.140.24.177:3128
        # export https_proxy=10.140.15.68:3128
        # httpx版本小于0.28.0以支持proxies
        # pip install httpx==0.27.2
        self.num_searches = num_searches
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.encode_idx_with_prefix = encode_idx_with_prefix
        self.return_search_results = return_search_results
    
    
    async def post_request(
        self,
        query: str,
    ):
        # https://ai.devops.xiaohongshu.com/app/6731?tab=2&key=1&wid=81892
        for attempt in range(self.max_retries):
            try:
                proxies = {
                    "http://": "http://10.140.24.177:3128",
                    "https://": "http://10.140.24.177:3128",
                }
                # 服务配置
                host = "https://aiplat-gateway.devops.beta.xiaohongshu.com"
                business_name = SEARCH_KNOWLEDGE_BUSINESS_NAME

                SEARCH_KNOWLEDGE_APP_ID = "sigmaragtest"
                SEARCH_KNOWLEDGE_APP_KEY = "qdn2J2Hm+Pty7wCda9EcJJaEGxnPAAZT+MVrsQdSaE0="
                business_name = 'allin-workflow-sigmaragtest'

                async with httpx.AsyncClient(timeout=100, trust_env=False, proxies=proxies) as client:
                    response = await client.post(
                        url=f"{host}/{business_name}/pipelines/main",
                        headers={"APP_ID": SEARCH_KNOWLEDGE_APP_ID, "APP_KEY": SEARCH_KNOWLEDGE_APP_KEY, "Content-Type": "application/json"},
                        json={"query": query, "chat_history": []},
                    )
                response.raise_for_status()
                searched_knowledge = response.json()['replies']
                return searched_knowledge

            except (httpx.HTTPError, Exception) as e:
                tqdm.write(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                    return []
    
    
    async def act(
        self,
        input_dict: Dict,
        input_key: str = "knowledge_subquery",
        output_key: str = "search_knowledge",
        idx_prefix: str = ""
    ):
        subquery = input_dict[input_key]
        tasks = [
            self.post_request(
                query=q
            ) for q in subquery
        ]
        responses = await asyncio.gather(*tasks)
        docs = []
        knowledge_id_set = set()
        for response in responses:
            for knowledge in response:
                knowledge_id = knowledge["id"]
                if knowledge_id in knowledge_id_set:
                    continue
                else:
                    knowledge_id_set.add(knowledge_id)
                    doc = {
                        "id": knowledge_id,
                        "search_from": "search_knowledge",
                        "content": knowledge.get("content", ""),
                        "title": knowledge.get("title", ""),
                        "url": knowledge.get("filename", ""),
                        "date": knowledge_id,
                        "note_type": "images",
                        "video": {},
                        "images": [],
                        "like_count": -1,
                        "collect_count": -1,
                        "view_count": -1,
                        "comments": [],
                        "confidence": -1,
                    }
                    docs.append(doc)
        docs = docs[::-1]
        if self.encode_idx_with_prefix:
            for idx, doc in enumerate(docs):
                doc["id"] = f"{idx_prefix}_{idx}"
        
        if self.return_search_results:
            return docs
        else:
            input_dict[output_key] = docs
            return input_dict




# test case
if __name__ == "__main__":
    async def main():
        service = KnowledgeBaseSearch(
            num_searches=10,
            max_retries=5,
            retry_delay=3,
            encode_idx_with_prefix=False,
            return_search_results=True
        )
        input_dict = dict()
        input_dict["query_image"] = "/mnt/tidalfs-bdsz01/usr/tusen/search-agent-dev/data/image/KIKO.png"
        input_dict["query_text"] = "AI是什么"
        input_dict["knowledge_subquery"] = ["AI是什么", "大模型是什么"]
        
        results = await service.act(
            input_dict=input_dict,
        )
        print(results)
    
    asyncio.run(main())
    
