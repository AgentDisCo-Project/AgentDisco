import gin
import requests
import datetime
import asyncio
import jinja2

from typing import Optional, List, Union, Dict
from api.RednoteSearchService import RedNoteSearch
from agent.BaseAgent import BasicAgent
from api.utils.urls import REDNOTE_EXPLORE_URL_TEMPLATE
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.key_operator import ApiKeyCycler
from api.utils.keys import DIRECTLLM_API_KEY_USER, GEMINI_API_KEY


class TemplateExtractor:
    def __init__(
        self,
        model_name: str = "",
        max_retries: int = 5,
        retry_delay: int = 3,
        max_concurrent: int = 50,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file: str = "TemplateExtractor_EN.jinja2",
        system_template_zh_file: str = "TemplateExtractor_ZH.jinja2",
        disable_video: bool = False,
        disable_images: bool = False,
        disable_multi_images: bool = False,
        disable_comment: bool = False,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.disable_video = disable_video
        self.disable_images = disable_images
        self.disable_multi_images = disable_multi_images
        self.disable_comment = disable_comment
        
        self.model = CustomizeChatGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
        )
        self.max_concurrent = max_concurrent
        
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.jinja_file = system_template_zh_file if self.use_zh else system_template_en_file
        
        self.note_manager = RedNoteSearch(
            max_retries=self.max_retries,
            retry_delay=self.retry_delay
        )
    

    def get_system_prompt(self):
        template_vars = {
        }
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    async def get_note_details(
        self,
        note_id: str,
    ):
        detail = await self.note_manager.get_note_details(note_id)
        note_type = "video" if detail.type == 2 else "images"
        
        video = ""
        if note_type == "video" and not self.disable_video:
            # 用aiohttp异步请求 + await
            video_url = await self.note_manager.convert_note_id_to_video_id(note_id)
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
            note_long_id = await self.note_manager.convert_oid_to_long(note_id)
            # async版本
            comments = await self.note_manager.get_comment_info(note_long_id) if note_long_id != -1 else []
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
        }
        
        return doc
    
    
    def check_func(
        self,
        response: str
    ):
        return response


    async def post_request(
        self,
        notes: List[Dict],
    ):
        if "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            system_prompt = [
                {
                    "type": "text",
                    "text": self.get_system_prompt()
                }
            ]
            
            user_prompt = []
            if self.use_zh:
                _user_prompt = """
# 高热笔记列表
"""
            else:
                _user_prompt = """
# Most Popular Notes
"""
            
            user_prompt.append(
                {
                    "type": "text",
                    "text": _user_prompt,
                }
            )
            
            for idx, note in enumerate(notes):
                title, content = note["title"], note["content"]

                if self.use_zh:
                    _user_prompt = f"""
## 高热笔记{idx}的内容如下：
标题：{title}
内容：{content}
"""
                    
                else:
                    _user_prompt = f"""
# Most Popular Notes {idx}
Title: {title}
Content: {content}
"""
                    
                    user_prompt.append(
                        {
                            "type": "text",
                            "text": _user_prompt,
                        }
                    )
            
            
            cycler = ApiKeyCycler(api_key_list=list(DIRECTLLM_API_KEY_USER.values()))
            response = await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                cycler=cycler,
                return_cot=False,
            )
        
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        return response

    
    async def act(
        self,
        note_ids: List,
    ):
        tasks = [
            asyncio.create_task(self.get_note_details(note_id))
            for note_id in note_ids
        ]
        notes = await asyncio.gather(*tasks)
        
        response = await self.post_request(notes)
        return response
        
    