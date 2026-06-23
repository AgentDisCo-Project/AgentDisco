import os
import aiohttp
import asyncio
import json
import time
import gin
import requests
import traceback
import sys
sys.path.append('.')

from dotenv import load_dotenv
from tqdm import tqdm
from api.utils.urls import REDNOTE_EXPLORE_URL_TEMPLATE
from api.RednoteSearchService import RedNoteSearch


load_dotenv('./api/utils/keys.env')
SEARCH_TEXT_BUSINESS_TYPE = os.environ.get("SEARCH_TEXT_BUSINESS_TYPE", "")


@gin.configurable()
class RedNoteTextSearch(RedNoteSearch):
    def __init__(
        self,
        user_id: str,
        num_searches: int,
        disable_comment: bool,
        disable_video: bool,
        use_note_modality: str,
        max_retries: int = 5,
        retry_delay: int = 3,
        timeout: int = 1024,
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
        
        self.encode_idx_with_prefix = encode_idx_with_prefix
        self.return_search_results = return_search_results
        self.timeout = timeout
        
    
    async def post_request(
        self,
        query: str,
    ):
        for attempt in range(self.max_retries):
            try:
                url = "http://droom.devops.xiaohongshu.com/api/debug/do_debug"
                headers = {
                    "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
                    "Content-Type": "application/json",
                    "Accept": "*/*",
                    "Host": "droom.shadow.devops.xiaohongshu.com",
                    "Connection": "keep-alive",
                }
                
                request = {
                    "sceneName": "note_search",
                    "sampleFlag": True,
                    "requestParams": {
                        "query": query,
                        "user_id": self.user_id,
                        "dp_test_env": "DEBUG",
                        "dp_base_env": "DEBUG",
                        "debug_type": "test",
                        "platform": "ios",
                        "rewritten_flag": "false",
                        "from": 0,
                        "size": self.num_searches,
                        "business_type": SEARCH_TEXT_BUSINESS_TYPE,
                        "debug": 0,
                    },
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, headers=headers, data=json.dumps(request), timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        response.raise_for_status()
                        response_json = await response.json()
                        searched_notes = response_json["data"]["debugInfoList"][0]["docList"]
                        searched_notes = searched_notes[:self.num_searches] # 防止搜索服务抽风
                        return searched_notes
            
            except (
                Exception,
            ) as e:
                tqdm.write(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                tqdm.write(traceback.format_exc())
                
                if attempt < self.max_retries - 1:  # 如果不是最后一次尝试
                    time.sleep(self.retry_delay)  # 等待一段时间再重试
                    continue
                else:
                    tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                    return []  # 重试多次后仍失败，返回None
    
    
    async def act(
        self,
        input_dict: dict,
        input_key: str = "note_subquery",
        output_key: str = "search_notes",
        idx_prefix: str = ""
    ):
        subquery = input_dict[input_key]
        tasks = [
            self.post_request(
                query=q,
            ) for q in subquery
        ]
        responses = await asyncio.gather(*tasks)
        
        notes = []
        note_id_set = set()
        for response in responses:
            for note in response:
                note_id = note.get("docId")
                if note_id in note_id_set:
                    continue
                else:
                    note_id_set.add(note_id)
                notes.append(note)
        
        tasks = [
            self.post_request_detail(
                note=note
            ) for note in notes
        ]
        docs = await asyncio.gather(*tasks)
        
        
        if self.encode_idx_with_prefix:
            for idx, doc in enumerate(docs):
                doc["id"] = f"{idx_prefix}_{idx}"
        
        if self.return_search_results:
            return docs
        else:
            input_dict[output_key] = docs
            return input_dict
    
    
    async def post_request_detail(
        self,
        note: dict,
    ):
        info = note.get("baseInfo", {})
        content = info.get("content", "")
        note_id = note.get("docId", "")
        
        note_type = note.get("debugInfo", {}).get("NOTE_TYPE", {})
        note_type = "video" if note_type == "视频类型" else "images"
        
        video = ""
        if note_type == "video" and not self.disable_video:
            # 用aiohttp异步请求 + await
            video_url = await self.convert_note_id_to_video_id(note_id)
            response = requests.get(video_url).json()
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
            detail = await self.get_note_details(note_id)
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
        
        doc = {
            "id": note_id,
            "search_from": "search_note",
            "content": content,
            "title": info.get("title", ""),
            "url": REDNOTE_EXPLORE_URL_TEMPLATE.format(note_id=note_id),
            "date": info.get("publishTime", ""),
            "note_type": note_type,
            "video": video_meta,
            "images": image_list,
            "like_count": info.get("likeCount", -1),
            "collect_count": info.get("collectCount", -1),
            "view_count": info.get("viewCount", -1),
            "comments": comment_list,
            "confidence": -1,
            "detail": "",
        }
        return doc


if __name__ == "__main__":
    async def main():
        service = RedNoteTextSearch(
            user_id="life",
            num_searches=10,
            disable_comment=False,
            disable_video=False,
            use_note_modality="text",
            max_retries=5,
            retry_delay=1,
            encode_idx_with_prefix=True,
            return_search_results=True,
        )
        input_dict = dict()
        input_dict["query_image"] = "/mnt/tidalfs-bdsz01/usr/tusen/search-agent-dev/data/image/KIKO.png"
        input_dict["query_text"] = "AI是什么"
        input_dict["note_subquery"] = ["AI是什么", "大模型是什么"]
        
        results = await service.act(
            input_dict=input_dict,
        )
        breakpoint()
        print(results)

    asyncio.run(main())