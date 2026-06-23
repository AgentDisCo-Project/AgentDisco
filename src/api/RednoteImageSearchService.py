import os
import time
import datetime
import requests
import gin
import re
import asyncio
import logging
import sys
sys.path.append('.')

from tqdm import tqdm
from poi_rpc.infra.rpc.base.ttypes import Context
from poi_rpc.thrift_rpc.red_rpc_util import create_thrift_client
from searchimage.search_image import SearchImageService
from searchimage.dto.ttypes import SearchImageReq
from RednoteSearchService import RedNoteSearch
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from api.utils.urls import REDNOTE_EXPLORE_URL_TEMPLATE
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
COS_SECRET_ID = os.environ.get("COS_SECRET_ID", "")
COS_SECRET_KEY = os.environ.get("COS_SECRET_KEY", "")
BUCKET_KEY = os.environ.get("BUCKET_KEY", "")
SEARCH_IMAGE_SOURCE_TYPE = os.environ.get("SEARCH_IMAGE_SOURCE_TYPE", "")
SEARCH_IMAGE_SCENE_TYPE = os.environ.get("SEARCH_IMAGE_SCENE_TYPE", "")


@gin.configurable()
class RedImageSearch(RedNoteSearch):
    def __init__(
        self,
        user_id: str,
        num_searches: int,
        disable_comment: bool,
        disable_video: bool,
        use_note_modality: str,
        max_retries: int = 5,
        retry_delay: int = 3,
        encode_idx_with_prefix: bool = False,
        return_search_results: bool = False,
    ):
        super().__init__(
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        user_id_map = {
            "travel": "61751b7d0000000002023e08",
            "outdoor": "62b2c862000000001501cbb5",
            "life": "56826634cb35fb7e671d6bfc",
            "photo": "610e28f30000000020029293",
            "sport": "59b3829850c4b4197d115edf",
        }
        assert user_id in user_id_map, f"Unsupported {user_id}"
        self.user_id = user_id_map[user_id]
        self.num_searches = num_searches
        
        self.disable_comment = disable_comment
        self.disable_video = disable_video
        if use_note_modality == "text":
            self.disable_images = True
            self.disable_multi_images = True
        elif use_note_modality == "one_image":
            self.disable_images = False
            self.disable_multi_images = True
        elif use_note_modality == "all_images":
            self.disable_images = False
            self.disable_multi_images = False
        else:
            raise ValueError(f"Unsupported {use_note_modality}")
        
        self.client = self.initialize_client()
        self.encode_idx_with_prefix = encode_idx_with_prefix
        self.return_search_results = return_search_results
    
    
    async def post_request(
        self,
        image_name: str,
    ):
        for attempt in range(self.max_retries):
            try:
                client2 = create_thrift_client(
                    service_name="searchimage-service-default",
                    client_class=SearchImageService,
                )
                req_params = {
                    "noteId":"",
                    "imageFileId":f'active/{image_name}',
                    "cursor": 0,
                    "size":  self.num_searches,
                    "userId": self.user_id,
                    "source": SEARCH_IMAGE_SOURCE_TYPE,
                    "scene": SEARCH_IMAGE_SCENE_TYPE,
                    "ip": "43.156.12.221",
                    "searchId": "2eusdz7dk2sb6yeskprdw",
                }
                request = SearchImageReq(**req_params)
                response = client2.searchImage(Context(), request)
                return response
            
            except (
                Exception,
            ) as e:  # 捕获网络相关异常和其他异常
                tqdm.write(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
            
            if attempt < self.max_retries - 1:  # 如果不是最后一次尝试
                time.sleep(self.retry_delay)  # 等待一段时间再重试
                continue
            else:
                tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                return ""  # 重试多次后仍失败，返回None
    
    
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
    
    
    async def act(
        self,
        input_dict: dict,
        idx_prefix: str = ""
    ):
        query_image = input_dict["query_image"]
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
            response = await self.post_request(
                image_name=image_name,
            )
            
            note_images = []
            note_id_set = set()
            for note_image in response.imageItemList:
                note_id = note_image.id
                if note_id in note_id_set:
                    continue
                else:
                    note_id_set.add(note_id)
                note_images.append(note_image)
            
            tasks = [
                self.post_request_detail(
                    note_image=note_image
                ) for note_image in note_images
            ]
            docs = await asyncio.gather(*tasks)
            
            if self.encode_idx_with_prefix:
                for idx, doc in enumerate(docs):
                    doc["id"] = f"{idx_prefix}_{idx}"
        
            if self.return_search_results:
                return docs
            else:
                input_dict["search_images"] = docs
                return input_dict
                
        return input_dict
    
    
    async def post_request_detail(
        self,
        note_image,
    ):
        note_id = note_image.id
        detail = await self.get_note_details(note_id)
        note_type = "video" if detail.type == 2 else "images"
        
        video = ""
        if note_type == "video" and not self.disable_video:
            # 用aiohttp异步请求 + await
            video_url = await self.convert_note_id_to_video_id(note_id)
            response = await requests.get(video_url).json()
            video_dict = response["data"]["video"]["play"]
            stream_types = ["258", "259", "4610", "720", "0"]
            for stream_type in stream_types:
                if stream_type in video_dict and "endpointUrl" in video_dict[stream_type]:
                    video = video_dict[stream_type]["endpointUrl"]
                    break
        video_meta = {
            "noteId": note_id,
            "url": video,
        }
        
        image_list = []
        if not self.disable_images:
            if note_id in detail.noteDetails:
                for img in detail.noteDetails[note_id].imagesList:
                    image_meta = {
                        "fileId": img.fileId,
                        "url": img.url,
                        "width": img.width,
                        "height": img.height,
                    }
                    image_list.append(image_meta)
                    if self.disable_multi_images:
                        break
        
        comment_list = []
        if not self.disable_comment:
            note_long_id = await self.convert_oid_to_long(note_id)
            # async版本
            comments = await self.get_comment_info(note_long_id) if note_long_id != -1 else []
            if comments != []:
                comments = comments.l1Comments
                if comments:
                    for comment in comments:
                        comment_meta = {
                            "content": comment.comment.content,
                            "like_count": comment.comment.likeCount,
                            "subcomment_count": comment.comment.subCommentCount,
                        }
                        comment_list.append(comment_meta)
        
        date = detail.time.createTime
        date = int(date) / 1000
        date = datetime.fromtimestamp(date)
        date = date.strftime("%Y-%m-%d")
        doc = {
            "id": note_id,
            "search_from": "search_image",
            "content": detail.content,
            "title": detail.title,
            "url": REDNOTE_EXPLORE_URL_TEMPLATE.format(note_id=note_id),
            "date": date,
            "note_type": note_type,
            "video": video_meta,
            "images": image_list,
            "like_count": -1,
            "collect_count": -1,
            "view_count": -1,
            "comments": comment_list,
            "confidence": -1,
            "detail": "",
        }
        return doc
    
    