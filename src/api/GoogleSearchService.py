import os
import gin
import requests
import json
import time
import asyncio
import aiohttp 
import math
import re
import logging
import sys
sys.path.append('.')

from dotenv import load_dotenv
from tqdm import tqdm
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from tool.DocParserService import WebParser


load_dotenv('./api/utils/keys.env')
GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
COS_SECRET_ID = os.environ.get("COS_SECRET_ID", "")
COS_SECRET_KEY = os.environ.get("COS_SECRET_KEY", "")
BUCKET_KEY = os.environ.get("BUCKET_KEY", "")
SERP_API_KEY = os.environ.get("SERP_API_KEY", "")


@gin.configurable()
class GoogleTextImageSearch:
    def __init__(
        self,
        num_searches: int,
        search_type: str = "text",
        max_retries: int = 5,
        retry_delay: int = 3,
        max_retries_jina: int = 1,
        # country: str = "us",
        country: str = "cn",
        # language: str = "en",
        language: str = "zh-cn",
        encode_idx_with_prefix: bool = False,
        return_search_results: bool = False,
        use_serp_api: bool = False,
        fetch_detail: bool = False,
        return_images: bool = False,
        use_jina_as_backup: bool = False,
    ):
        assert search_type in ("text", "image", "scholar"), f"Unsupported {search_type}"
        self.search_type = search_type
        self.num_searches = num_searches
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.country = country
        self.language = language
        
        self.client = self.initialize_client()
        self.encode_idx_with_prefix = encode_idx_with_prefix
        self.return_search_results = return_search_results
        self.use_serp_api = use_serp_api
        self.webparser = WebParser(max_retries=max_retries_jina,use_jina_as_backup=use_jina_as_backup)
        self.fetch_detail = fetch_detail
        self.return_images = return_images
    
    
    @staticmethod
    def initialize_client():
        region = 'ap-shanghai'      # 替换为用户的 region，已创建桶归属的 region 可以在控制台查看，https://console.cloud.tencent.com/cos5/bucket
        # COS 支持的所有 region 列表参见 https://cloud.tencent.com/document/product/436/6224
        token = None               # 如果使用永久密钥不需要填入 token，如果使用临时密钥需要填入，临时密钥生成和使用指引参见 https://cloud.tencent.com/document/product/436/14048
        scheme = 'https'           # 指定使用 http/https 协议来访问 COS，默认为 https，可不填
        config = CosConfig(Region=region, SecretId=COS_SECRET_ID, SecretKey=COS_SECRET_KEY, Token=token, Scheme=scheme)
        return CosS3Client(config)
    
    
    async def upload_image(
        self,
        image: str,
        image_name: str
    ):
        def _sync_upload():
            with open(image, 'rb') as fp:
                _ = self.client.put_object(
                    Bucket=BUCKET_KEY,
                    Body=fp,
                    Key=f'active/{image_name}',
                    StorageClass='STANDARD',
                    EnableMD5=False
                )
            logging.info(f"Successfully upload {image}")
            return True
        return await asyncio.to_thread(_sync_upload)
    
    
    async def check_image(
        self,
        image_name: str
    ):
        def _sync_check():
            response = self.client.list_objects(
                Bucket=BUCKET_KEY,
                Prefix=f"active/{image_name}"
            )
            return "Contents" in response
        return await asyncio.to_thread(_sync_check)
    
    
    async def download_image(
        self,
        image: str,
        image_name: str
    ):
        if not await self.check_image(image_name=image_name):
            raise ValueError(f"{image_name} is not exist")
        
        def _download_body():
            response = self.client.get_object(
                Bucket=BUCKET_KEY,
                Key=f'active/{image_name}',
            )
            response['Body'].get_stream_to_file(image)
            logging.info(f"Successfully download {image}")
            return True
        
        return await asyncio.to_thread(_download_body)


    async def post_request(
        self,
        query_text: str = "",
        image_name: str = "",
    ):
        search_google_st = time.time()
        
        # 创建异步 Session (建议在类初始化时创建 session，这里为了改动最小在函数内创建)
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries):
                try:
                    if self.search_type == "text":
                        assert len(query_text) > 0, "No available query_text"
                        if not self.use_serp_api:
                            # ==================== 核心修改：使用 aiohttp ====================
                            if not self.return_images:
                                url = "https://google.serper.dev/search"
                                headers = {
                                    'X-API-KEY': GOOGLE_SEARCH_API_KEY,
                                    'Content-Type': 'application/json'
                                }
                                num_pages = math.ceil(self.num_searches / 10)
                                response = []
                                for num_page in range(num_pages):
                                    payload = {
                                        "q": query_text,
                                        "gl": self.country,
                                        "hl": self.language,
                                        "page": num_page+1,
                                        "location": "China" if self.country == "cn" else "United States",
                                    }
                                    # 异步请求
                                    async with session.post(url, headers=headers, json=payload) as resp:
                                        resp_json = await resp.json()
                                        _response = resp_json.get("organic", [])
                                        response.extend(_response)
                                response = response[:self.num_searches]

                            else:
                                url = "https://google.serper.dev/images"
                                headers = {
                                    'X-API-KEY': GOOGLE_SEARCH_API_KEY,
                                    'Content-Type': 'application/json'
                                }
                                num_pages = math.ceil(self.num_searches / 10)
                                response = []
                                for num_page in range(num_pages):
                                    payload = {
                                        "q": query_text,
                                        "gl": self.country,
                                        "hl": self.language,
                                        "page": num_page+1,
                                        "location": "China" if self.country == "cn" else "United States"
                                    }
                                    # 异步请求
                                    async with session.post(url, headers=headers, json=payload) as resp:
                                        resp_json = await resp.json()
                                        _response = resp_json.get("images", [])
                                        response.extend(_response)
                                response = response[:self.num_searches]
                            # ==============================================================
                            
                        else:
                            # SerpAPI 官方库是同步的，必须用 to_thread 包装，否则会阻塞
                            def _sync_serp_search():
                                params = {
                                    "engine": "google",
                                    "q": query_text,
                                    "gl": self.country,
                                    "hl": self.language,
                                    "num": self.num_searches,
                                    "api_key": SERP_API_KEY,
                                }
                                from serpapi import GoogleSearch
                                search = GoogleSearch(params)
                                results = search.get_dict()
                                return results.get("organic_results", [])[:self.num_searches]
                            
                            response = await asyncio.to_thread(_sync_serp_search)
                    
                    elif self.search_type == "image":
                        # ... (此处逻辑同上，需将 requests 换成 session.post) ...
                        assert len(image_name) > 0, "No available query_image"
                        image_url = self.get_image_url(image_name=image_name)
                        
                        url = "https://google.serper.dev/lens"
                        headers = {
                            'X-API-KEY': GOOGLE_SEARCH_API_KEY,
                            'Content-Type': 'application/json'
                        }
                        num_pages = math.ceil(self.num_searches / 10)
                        response = []
                        for num_page in range(num_pages):
                            payload = {
                                "url": image_url,
                                "gl": self.country,
                                "hl": self.language,
                                "page": num_page+1,
                                "location": "China" if self.country == "cn" else "United States"
                            }
                            async with session.post(url, headers=headers, json=payload) as resp:
                                resp_json = await resp.json()
                                _response = resp_json.get("organic", [])
                                response.extend(_response)
                        response = response[:self.num_searches]

                    elif self.search_type == "scholar":
                        # ... (此处逻辑同上，需将 requests 换成 session.post) ...
                        url = "https://google.serper.dev/scholar"
                        headers = {
                            'X-API-KEY': GOOGLE_SEARCH_API_KEY,
                            'Content-Type': 'application/json'
                        }
                        num_pages = math.ceil(self.num_searches / 10)
                        response = []
                        for num_page in range(num_pages):
                            payload = {
                                "q": query_text,
                                "gl": self.country,
                                "hl": self.language,
                                "page": num_page+1,
                                "location": "China" if self.country == "cn" else "United States",
                            }
                            async with session.post(url, headers=headers, json=payload) as resp:
                                resp_json = await resp.json()
                                _response = resp_json.get("organic", [])
                                response.extend(_response)
                        response = response[:self.num_searches]
                        
                    else:
                        raise ValueError(f"Unsupported {self.search_type}")
                    
                    break # 成功则跳出重试循环
                
                except Exception as e:
                    tqdm.write(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                
                    if attempt < self.max_retries - 1:
                        # 核心修改：使用非阻塞的睡眠
                        await asyncio.sleep(self.retry_delay) 
                        continue
                    else:
                        tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                        response = []

        search_google_et = time.time()
        logging.info(f"finish search google costs: {search_google_et-search_google_st}")
        
        # 原有的 fetch detail 部分 (保持逻辑不变，但需确认 webparser 也是异步的)
        fetch_detail_st = time.time()
        if self.fetch_detail and self.search_type in ("text", "scholar"):
            detail_tasks = [
                self.webparser.call(params={"url": res.get("link", "")})
                for res in response
            ]
            details = await asyncio.gather(*detail_tasks)
        else:
            details = [""] * len(response)

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
                "id": f"google_{self.search_type}_{idx}",
                "search_from": f"google_{self.search_type}",
                "content": res.get("snippet", ""),
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
        
        fetch_detail_et = time.time()
        logging.info(f"finish fetch detail costs: {fetch_detail_et-fetch_detail_st}")

        return docs

    
    
    # @staticmethod
    # async def get_image_url_request(
    #     image_name: str,
    # ):
    #     from redrpc.client.client import Client
    #     from redrpc.utils.context_helper import get_context
    #     from multicdn import Executor
    #     from multicdn import GetCDNResourceRequest
    #     from multicdn import GetCDNResourceResponse
    #     client = Client(Executor, service_name="multicdn-service-executor")
    #     context = get_context()
    #
    #     file_key = f"active/{image_name}"
    #     file_keys = [file_key]
    #     tags = {"auth_offset": "7776000"}
    #     req = GetCDNResourceRequest(business="sns", scene="ecological_cooperation", file_keys=file_keys, tags=tags)
    #     result = client.GetCDNResource(context, req)
    #     urls = result.resource.urls
    #     image_url = urls.get(file_key).master
    #     return image_url
    
    
    @staticmethod
    def get_image_url(
        image_name: str = "",
    ):
        image_url = f"https://{BUCKET_KEY}.cos.ap-shanghai.myqcloud.com/active/{image_name}"
        return image_url
    
    
    async def act(
        self,
        input_dict: dict,
        input_key: str = "web_subquery",
        output_key: str = "search_google",
        idx_prefix: str = ""
    ):
        search_google_st = time.time()
        if self.search_type == "image":
            query_image = input_dict.get("query_image", "")
            if query_image != "":
                match = re.search(r'[^/]+$', query_image)
                image_name = match.group() if match else query_image
                upload_status = await self.check_image(image_name=image_name)
                if not upload_status:
                    await self.upload_image(image=query_image, image_name=image_name)
                upload_status = await self.check_image(image_name=image_name)
            else:
                image_name = query_image
                upload_status = False
            
            docs = []
            if upload_status:
                responses = await self.post_request(image_name=image_name)
                
                doc_id_set = set()
                for response in responses:
                    doc_id = response.get("url")
                    if doc_id in doc_id_set:
                        continue
                    else:
                        doc_id_set.add(doc_id)
                    docs.append(response)
        
        elif self.search_type in ("text", "scholar"):
            subquery = input_dict[input_key]
            tasks = [
                self.post_request(
                    query_text=q,
                ) for q in subquery
            ]
            responses = await asyncio.gather(*tasks)
            
            docs = []
            doc_id_set = set()
            for response in responses:
                for doc in response:
                    doc_id = doc.get("url")
                    if doc_id in doc_id_set:
                        continue
                    else:
                        doc_id_set.add(doc_id)
                    docs.append(doc)
        else:
            raise ValueError(f"Unsupported {self.search_type}")
        
        search_google_et = time.time()
        logging.info(f"search google costs: {search_google_et-search_google_st}")

        if self.encode_idx_with_prefix:
            for idx, doc in enumerate(docs):
                doc["id"] = f"{idx_prefix}_{idx}"
        
        if self.return_search_results:
            return docs
        else:
            input_dict[output_key] = docs
            return input_dict



if __name__ == "__main__":
    async def main():
        service = GoogleTextImageSearch(
            num_searches=10,
            max_retries=5,
            retry_delay=3,
            search_type="text",
            encode_idx_with_prefix=False,
            return_search_results=True,
            return_images=False,
            use_serp_api=False,
            fetch_detail=False,
        )
        input_dict = dict()
        input_dict["query_image"] = "/mnt/tidalfs-bdsz01/usr/tusen/search-agent-dev/data/image/KIKO.png"
        input_dict["query_text"] = "大模型Agent的最近技术"
        input_dict["web_subquery"] = ["大模型Agent的最近技术"]
        
        results = await service.act(
            input_dict=input_dict,
        )
        breakpoint()
        print(results)
        
    
    asyncio.run(main())