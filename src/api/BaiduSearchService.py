import os
import aiohttp
import gin
import requests
import json
import time
import asyncio
import re
import logging
import base64
import sys
sys.path.append('.')

from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client


load_dotenv('./api/utils/keys.env')
BAIDU_SEARCH_IMAGE_KEY = os.environ.get("BAIDU_SEARCH_IMAGE_KEY", "")
COS_SECRET_ID = os.environ.get("COS_SECRET_ID", "")
COS_SECRET_KEY = os.environ.get("COS_SECRET_KEY", "")
BUCKET_KEY = os.environ.get("BUCKET_KEY", "")
BAIDU_SEARCH_IMAGE_API_KEY = os.environ.get("BAIDU_SEARCH_IMAGE_API_KEY", "")
BAIDU_SEARCH_IMAGE_SECRET_KEY = os.environ.get("BAIDU_SEARCH_IMAGE_SECRET_KEY", "")


@gin.configurable()
class BaiduImageSearch:
    def __init__(
        self,
        max_retries: int = 5,
        retry_delay: int = 3,
        encode_idx_with_prefix: bool = False,
        return_search_results: bool = False,
        image_key_type: bool = "path",
        require_summary: bool = False,
        require_base64: bool = False,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.encode_idx_with_prefix = encode_idx_with_prefix
        self.return_search_results = return_search_results
        self.client = self.initialize_client()
        
        assert image_key_type in ("path", "url"), f"Unsupported image_key_type {image_key_type}"
        self.image_key_type = image_key_type
        self.require_summary = require_summary
        self.require_base64 = require_base64
    
    
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
    
    
    @staticmethod
    def smart_decode(
        text: str
    ):
        if not isinstance(text, str) or not text:
            return text
        # 如果已经是正常中文/非转义，直接返回
        try:
            text.encode('ascii')
            return text
        except Exception as e:
            logging.info(f"smart decode exception info is {e}")
            pass
        # 如果开头是'\x'或多数内容不是有效utf-8，则尝试解码
        try:
            return text.encode('latin1').decode('utf-8')
        except Exception as e:
            logging.info(f"smart decode exception info is {e}")
            return text
        
    
    @staticmethod
    async def get_image_url_request(
        image_name: str,
    ):
        # TODO: support CDN request
        from redrpc.client.client import Client
        from redrpc.utils.context_helper import get_context
        from multicdn import Executor
        from multicdn import GetCDNResourceRequest
        from multicdn import GetCDNResourceResponse
        client = Client(Executor, service_name="multicdn-service-executor")
        context = get_context()
        
        file_key = f"active/{image_name}"
        file_keys = [file_key]
        tags = {"auth_offset": "7776000"}
        req = GetCDNResourceRequest(business="sns", scene="ecological_cooperation", file_keys=file_keys, tags=tags)
        result = client.GetCDNResource(context, req)
        urls = result.resource.urls
        image_url = urls.get(file_key).master
        return image_url
    
    
    @staticmethod
    def get_image_url(
        image_name: str = "",
    ):
        image_url = f"https://{BUCKET_KEY}.cos.ap-shanghai.myqcloud.com/active/{image_name}"
        return image_url
    
    
    @staticmethod
    async def download_image_base64(
        image_url: str = "",
        chunk_size: int = 4096,
    ):
        # 流式下载，边读边编码，节省 40-60% 时间
        image = requests.get(image_url, stream=True)
        image.raise_for_status()
        # 用 BytesIO 当缓冲区，避免多次 realloc
        buffer = BytesIO()
        for chunk in image.iter_content(chunk_size=chunk_size):
            if chunk:
                buffer.write(chunk)
        return base64.b64encode(buffer.getvalue()).decode()
        
    
    @staticmethod
    async def load_image_base64(
        image_path: str = ""
    ):
        image = Image.open(image_path).convert("RGB")
        image_buffer = BytesIO()
        image.save(image_buffer, format="JPEG", quality=80, optimize=True)
        image_str = base64.b64encode(image_buffer.getvalue()).decode()
        return image_str
    
    
    async def post_request(
        self,
        query_text: str = "这个是什么？",
        image_name: str = "",
        image_path: str = "",
    ):
        for attempt in range(self.max_retries):
            try:
                assert len(image_name) > 0, "No available query_image"
                url = "https://aip.baidubce.com/oauth/2.0/token"
                params = {"grant_type": "client_credentials", "client_id": BAIDU_SEARCH_IMAGE_API_KEY, "client_secret": BAIDU_SEARCH_IMAGE_SECRET_KEY}
                access_token = str(requests.post(url, params=params).json().get("access_token"))
                
                url = "https://aip.baidubce.com/stream/2.0/image-classify/v1/object_recognition?access_token=" + access_token
                
                if self.image_key_type == "path":
                    image = "data:image/jpeg;base64," + await self.load_image_base64(image_path=image_path)
                else:
                    if not self.require_base64:
                        image = image_name
                    else:
                        image = "data:image/jpeg;base64," + await self.download_image_base64(image_url=image_name)
                payload = json.dumps({
                    "url": image,
                    "question": query_text,
                    "search_mode": "required",
                    "search_result": True,
                    "baike_result": True,
                    "res_data_result": True
                }, ensure_ascii=False)
                headers = {
                    'Content-Type': 'application/json'
                }
                
                # 使用流式请求，只读取第一个数据包
                response = requests.post(url, headers=headers, data=payload.encode("utf-8"), stream=True)
                response.encoding = 'utf-8'
                
                # 读取响应内容的第一行（第一个数据包）
                first_line = None
                for line in response.iter_lines(decode_unicode=True):
                    if line and line.startswith('data:'):
                        first_line = line
                        break  # 找到第一个数据包后就退出
                
                if first_line:
                    # 去除"data:"前缀并解析JSON
                    json_str = first_line[5:]  # 去掉"data:"前缀
                    result_data = json.loads(json_str)
                    result_data = result_data["result"]["image_result"]["res_data_result"]
                    docs = []
                    for idx, _result in enumerate(result_data):
                        image_meta = {

                        }

                        doc = {
                            "id": f"baidu_image_{idx}",
                            "search_from": "baidu_image",
                            "content": "",
                            "title": _result.get("title", ""),
                            "url": _result.get("objurl", ""),
                            "date": _result.get("update_time", ""),
                            "note_type": "images",
                            "video": "",
                        }
                    
                    
                    
                    return docs
                else:
                    return {"error": "未找到有效的数据包"}
            
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
                
    
    async def post_request_require_summary(
        self,
        query_text: str = "图里是什么？",
        image_name: str = "",
        image_path: str = "",
    ):
        for attempt in range(self.max_retries):
            try:
                assert len(image_name) > 0, "No available query_image"
                image_url = self.get_image_url(image_name=image_name)
                url = "https://runway.devops.xiaohongshu.com/openai/baidu/rest/2.0/image_classify/v1/object_recognition"
                headers = {
                    "api-key": BAIDU_SEARCH_IMAGE_KEY,
                    "Content-Type": "application/json"
                }
                data = {
                    "search_mode": "required",
                    "guessword_result": True,
                    "search_result": True,
                    "baike_result": True,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": query_text
                                }
                            ]
                        }
                    ]
                }
                
                response = requests.post(url, headers=headers, json=data)
                response = response.content.decode('utf-8', errors='replace')
                docs = []
                description = []
                for idx, line in enumerate(response.splitlines()):
                    line = line.strip()
                    if not line or not line.startswith('data:'):
                        continue
                    # 取出JSON字符串部分
                    json_str = line[5:]
                    try:
                        obj = json.loads(json_str)
                    except Exception as e:
                        print("JSON解析失败:", e)
                        continue
                    
                    # 优先处理 result/description
                    result = obj.get('result', {})
                    desc = self.smart_decode(result.get('description', ''))
                    description.append(desc)
                    if 'baike_result' in result and isinstance(result['baike_result'], list):
                        for _result in result['baike_result']:
                            image_meta = {

                            }

                            doc = {
                                "id": f"baidu_image_{idx}",
                                "search_from": "baidu_image",
                                "content": _result.get("baike_summary", ""),
                                "title": _result.get("baike_title", ""),
                                "url": _result.get("baike_url", ""),
                                "date": "",
                                "note_type": "images",
                                "video": "",
                                "images": [_result.get("image_url", "")] if len(_result.get("image_url", "")) > 0 else [],
                                "like_count": -1,
                                "collect_count": -1,
                                "view_count": -1,
                                "comments": [],
                                "confidence": -1,
                                "score": -1,
                                "detail": "",
                            }
                            docs.append(doc)
                
                description = ''.join(description)
                return description, docs
                
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
                    return "", []  # 重试多次后仍失败，返回None
        
    
    async def act(
        self,
        input_dict: dict,
        output_key: str = "search_baidu",
        idx_prefix: str = ""
    ):
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
            
        if upload_status:
            if self.require_summary:
                description, docs = await self.post_request_require_summary(image_name=image_name, image_path=query_image)
                
                if self.encode_idx_with_prefix:
                    for idx, doc in enumerate(docs):
                        doc["id"] = f"{idx_prefix}_{idx}"
            
                if self.return_search_results:
                    return description, docs
                else:
                    input_dict[output_key] = docs
                    input_dict["description"] = description
                    return input_dict

            else:
                docs = await self.post_request(image_name=image_name, image_path=query_image)
                if self.return_search_results:
                    return docs 
                else:
                    input_dict[output_key] = docs
                    return input_dict

        else:
            if self.return_search_results:
                return "", []
            else:
                return input_dict



# test case
if __name__ == "__main__":
    async def main():
        service = BaiduImageSearch(
            max_retries=5,
            retry_delay=3,
            encode_idx_with_prefix=False,
            return_search_results=True,
            require_summary=False,
        )
        input_dict = dict()
        input_dict["query_image"] = "/mnt/tidalfs-bdsz01/usr/tusen/search-agent-dev/data/image/KIKO.png"
        
        results = await service.act(
            input_dict=input_dict,
        )
        print(results)
    
    asyncio.run(main())