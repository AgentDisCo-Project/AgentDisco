import gin
import requests
import asyncio
import jinja2
import json
import sys 
import aiohttp

sys.path.append('.')
sys.path.append('./api')

from datetime import datetime
from typing import Optional, List, Union, Dict
from api.RednoteSearchService import RedNoteSearch
from agent.BaseAgent import BasicAgent
from api.utils.urls import REDNOTE_EXPLORE_URL_TEMPLATE
from api.utils.string_operator import json_fix
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.key_operator import ApiKeyCycler
from api.utils.keys import DIRECTLLM_API_KEY_USER, GEMINI_API_KEY



@gin.configurable()
class QueryMiner(BasicAgent):
    def __init__(
        self,
        name: Optional[str] = "",
        description_en: Optional[str] = "",
        description_zh: Optional[str] = "",
        tool_bank: Optional[List[Union[str, Dict]]] = "",
        use_zh: bool = False,
        model_name: str = "",
        max_retries: int = 5,
        retry_delay: int = 3,
        max_concurrent: int = 50,
        use_customize_url: bool = False,
        customize_url: str = "",
        max_query_len: int = 5,
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file: str = "QueryMiner_EN.jinja2",
        system_template_zh_file: str = "QueryMiner_ZH.jinja2",
        system_template_en_file_action: str = "QueryMinerAction_EN.jinja2",
        system_template_zh_file_action: str = "QueryMinerAction_ZH.jinja2",
        disable_video: bool = True,
        disable_images: bool = True,
        disable_multi_images: bool = True,
        disable_comment: bool = True,
        use_input_query: bool = True,
        include_summary: bool = True,
        from_file: bool = True,
        query_type_is_action: bool = False,
    ):
        super().__init__(
            name=name,
            description_en=description_en,
            description_zh=description_zh,
            tool_bank=tool_bank,
            use_zh=use_zh
        )
        self.use_input_query = use_input_query
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
        self.max_query_len = max_query_len
        self.include_summary = include_summary
        
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )

        if query_type_is_action:
            self.jinja_file = system_template_zh_file_action if self.use_zh else system_template_zh_file
        else:
            self.jinja_file = system_template_zh_file if self.use_zh else system_template_en_file
        

    async def get_note_details(
        self,
        note_manager: RedNoteSearch,
        note_id: str,
    ):
        detail = await note_manager.get_note_details(note_id)
        
        # ✅ 修复：从 noteDetails 字典中取出具体笔记对象
        if note_id not in detail.noteDetails:
            return None
        note_detail = detail.noteDetails[note_id]
        
        note_type = "video" if note_detail.type == 2 else "images"
        
        video = ""
        if note_type == "video" and not self.disable_video:
            video_url = await note_manager.convert_note_id_to_video_id(note_id)
            # ✅ 修复：使用 aiohttp 进行异步 HTTP 请求
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url) as response:
                    response_data = await response.json(content_type=None)
            video_dict = response_data["data"]["video"]["play"]
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
            # ✅ 修复：使用 note_detail 对象，并对 imagesList 做空值保护
            images = note_detail.imagesList or []
            for img in images:
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
            note_long_id = await note_manager.convert_oid_to_long(note_id)
            comments = await note_manager.get_comment_info(note_long_id) if note_long_id != -1 else []
            if comments:
                l1_comments = comments.l1Comments or []
                for comment in l1_comments:
                    comment_meta = {
                        "content": comment.comment.content,
                        "like_count": comment.comment.likeCount,
                        "subcomment_count": comment.comment.subCommentCount,
                    }
                    comment_list.append(comment_meta)
        
        # ✅ 修复：从 note_detail 中读取时间、内容、标题
        date = int(note_detail.time.createTime) / 1000
        date = datetime.fromtimestamp(date).strftime("%Y-%m-%d")

        doc = {
            "id": note_id,
            "search_from": "search_image",
            "content": note_detail.content,
            "title": note_detail.title,
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

    
    def get_system_prompt(self):
        template_vars = {
            "max_query_len": self.max_query_len,
            "include_summary": self.include_summary
        }
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    def check_func(
        self,
        response: str
    ):
        response = json_fix(response)
        response = json.loads(response)
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
# 用户浏览文档内容
"""
            else:
                _user_prompt = """
# User Browsed Documents
"""
            
            user_prompt.append(
                {
                    "type": "text",
                    "text": _user_prompt,
                }
            )
            
            for idx, note in enumerate(notes):
                title, content = note["title"], note["content"]
                
                if self.include_summary and "summary" in note:
                    summary = note["summary"]
                    
                    if self.use_zh:
                        _user_prompt = f"""
# 用户浏览文档{idx}内容如下
总结：{summary}
"""
                    
                    else:
                        _user_prompt = f"""
# User Browsed Document {idx}
Summary: {summary}
"""
                    
                    user_prompt.append(
                        {
                            "type": "text",
                            "text": _user_prompt,
                        }
                    )
                
                else:
                    if self.use_zh:
                        _user_prompt = f"""
## 用户浏览文档{idx}的内容如下：
标题：{title}
内容：{content}
"""
                    
                    else:
                        _user_prompt = f"""
# User Browsed Document {idx}
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
        
        elif "gemini" in self.model.model_name:
            system_prompt = [
                {
                    "text": self.get_system_prompt()
                }
            ]
            user_prompt = [] 

            if self.use_zh:
                _user_prompt = """
# 用户浏览文档内容
"""
            else:
                _user_prompt = """
# User Browsed Documents
"""
            
            user_prompt.append({"text": _user_prompt})
            
            for idx, note in enumerate(notes):
                title, content = note["title"], note["content"]
                
                if self.include_summary and "summary" in note:
                    summary = note["summary"]
                    
                    if self.use_zh:
                        _user_prompt = f"""
# 用户浏览文档{idx}内容如下
总结：{summary}
"""
                    
                    else:
                        _user_prompt = f"""
# User Browsed Document {idx}
Summary: {summary}
"""
                    
                    user_prompt.append({"text": _user_prompt})
                
                else:
                    if self.use_zh:
                        _user_prompt = f"""
## 用户浏览文档{idx}的内容如下：
标题：{title}
内容：{content}
"""
                    
                    else:
                        _user_prompt = f"""
# User Browsed Document {idx}
Title: {title}
Content: {content}
"""
                    
                    user_prompt.append({"text": _user_prompt})

            
            response = await self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                return_cot=False,
            )


        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        return response
    
    
    async def act(
        self,
        input_dict: Dict,
        response_key: str = "query"
    ):
        if self.use_input_query:
            assert "query" in input_dict or "query_text" in input_dict, f"Not available query in {input_dict.keys()}"
            input_dict[response_key] = input_dict.get("query", "") or input_dict.get("query_text", "")
            return input_dict
        
        else:
            assert "user_id" in input_dict and "note_ids" in input_dict, f"Not available user id {input_dict.keys()}"
            note_manager = RedNoteSearch(
                max_retries=self.max_retries,
                retry_delay=self.retry_delay
            )
            
            note_ids = input_dict["note_ids"]
            tasks = [
                asyncio.create_task(self.get_note_details(note_manager, note_id))
                for note_id in note_ids
            ]
            notes = await asyncio.gather(*tasks)
            
            response = await self.post_request(notes)
            input_dict[response_key] = response
            return input_dict



if __name__ == "__main__":
    async def main():
        service = QueryMiner(
            model_name="gemini-3-flash",
            use_zh=True,
            use_input_query=False,
        )
        uid_note_dict_path = "/mnt/tidalfs-bdsz01/usr/tusen/uid_note_dict.jsonl"

        # 读取所有记录
        records = []
        with open(uid_note_dict_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        print(f"总记录数: {len(records)}")

        # 取第一条记录
        first_record = records[0]

        uid = first_record["uid"]
        note_ids = first_record["note_ids"]

        input_dict = dict()
        input_dict["user_id"] = uid
        input_dict["note_ids"] = note_ids
        results = await service.act(input_dict=input_dict)

    asyncio.run(main())