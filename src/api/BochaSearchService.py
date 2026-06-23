import os
import aiohttp
import asyncio
import json
import time
import gin
import logging
import sys
sys.path.append('.')

from typing import Dict
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
BOCHA_API_KEY = os.environ.get("BOCHA_API_KEY", "")


@gin.configurable()
class BochaTextSearch:
    def __init__(
        self,
        num_searches: int,
        max_retries: int = 5,
        retry_delay: int = 3,
        web_search_type: str = "web-search",
        encode_idx_with_prefix: bool = False,
        return_search_results: bool = False,
    ):
        assert web_search_type in ("web-search", "ai-search"), f"Unsupported {web_search_type}"
        self.web_search_type = web_search_type
        self.num_searches = num_searches
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.encode_idx_with_prefix = encode_idx_with_prefix
        self.return_search_results = return_search_results
    
    
    async def post_request(
        self,
        query: str,
    ):
        for attempt in range(self.max_retries):
            try:
                url = f'https://runway.devops.xiaohongshu.com/openai/bocha/{self.web_search_type}'
                headers = {
                    "Authorization": f"Bearer {BOCHA_API_KEY}",
                    'Content-Type': 'application/json',
                }
                
                request = {
                    "query": query,
                    "freshness": "noLimit",
                    "summary": True,
                    "count": self.num_searches,
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, headers=headers, data=json.dumps(request), timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        response.raise_for_status()
                        response_json = await response.json()
                        searched_webs = response_json["data"]["webPages"]["value"]
                        return searched_webs
            
            except (
                aiohttp.ClientError,
                Exception,
            ) as e:
                tqdm.write(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                
                if attempt < self.max_retries - 1:  # 如果不是最后一次尝试
                    time.sleep(self.retry_delay)  # 等待一段时间再重试
                    continue
                else:
                    tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                    return []  # 重试多次后仍失败，返回None
    
    
    async def act(
        self,
        input_dict: Dict,
        input_key: str = "web_subquery",
        output_key: str = "search_bocha",
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
        web_url_set = set()
        for response_id, response in enumerate(responses):
            for web_id, web in enumerate(response):
                web_url = web.get("url", {})
                if not web_url or web_url in web_url_set:
                    continue
                else:
                    web_url_set.add(web_url)
                
                doc = {
                    "id": f"{response_id}_{web_id}",
                    "search_from": "search_web",
                    "content": web.get("summary"),
                    "title": web.get("name"),
                    "url": web_url,
                    "date": web.get("datePublished", {}),
                    "note_type": "",
                    "video": {},
                    "images": [],
                    "like_count": -1,
                    "collect_count": -1,
                    "view_count": -1,
                    "comments": [],
                    "confidence": -1,
                    "detail": "",
                }
                docs.append(doc)
        
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
        service = BochaTextSearch(
            num_searches=10,
            max_retries=5,
            retry_delay=3,
            web_search_type="web-search",
            encode_idx_with_prefix=False,
            return_search_results=True
        )
        input_dict = dict()
        input_dict["query_image"] = "/mnt/tidalfs-bdsz01/usr/tusen/search-agent-dev/data/image/KIKO.png"
        input_dict["query_text"] = "AI是什么"
        input_dict["web_subquery"] = ["AI是什么", "大模型是什么"]
        
        results = await service.act(
            input_dict=input_dict,
        )
        print(results)
    
    asyncio.run(main())