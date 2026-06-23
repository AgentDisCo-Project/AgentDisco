import os
import gin
import asyncio
import jinja2
import json
import sys
sys.path.append('.')

from dotenv import load_dotenv
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.url_operator import compress_and_convert_base64, compress_url
from api.utils.key_operator import ApiKeyCycler


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)


@gin.configurable()
class NoteJudge:
    def __init__(
        self,
        model_name: str = "qwen2.5-7b-instruct",
        max_retries: int = 5,
        retry_delay: int = 3,
        max_concurrent: int = 50,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_query_modality: str = "",
        use_note_modality: str = "",
        image_key_type: str = "url",
        max_num_images: int = 14,
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_file: str = "NoteJudge.jinja2"
    ):
        self.model = CustomizeChatGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
        )
        assert image_key_type in ("path", "url"), f"Unsupported image_key_type {image_key_type}"
        self.image_key_type = image_key_type
        assert use_query_modality in ("text", "both"), f"Unsupported {use_query_modality}"
        self.use_query_modality = use_query_modality
        assert use_note_modality in ("one_image", "all_images", "text")
        self.use_note_modality = use_note_modality
        self.max_num_images = max_num_images
        self.max_concurrent = max_concurrent
        
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.jinja_file = system_template_file
    
    
    def get_system_prompt(self):
        template_vars = {
            'use_query_modality': self.use_query_modality,
            'use_note_modality': self.use_note_modality
        }
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    @staticmethod
    def check_func(
        response: str
    ):
        response = float(response)
        return response


    async def act(
        self,
        input_dict: dict,
        input_key: str = "search_results",
        output_key: str = "judge",
        print_concurrent: bool = False
    ):
        query_text = input_dict["query_text"]
        query_image = input_dict.get("query_image", "")
        
        sem = asyncio.Semaphore(self.max_concurrent)
        num_active_task = 0
        active_lock = asyncio.Lock()
        
        async def post_and_judge(note):
            nonlocal num_active_task
            async with sem:
                async with active_lock:
                    num_active_task += 1
                if print_concurrent:
                    print(f"number of active tasks: {num_active_task}")
                try:
                    response = await self.post_request(
                        query_text=query_text,
                        query_image=query_image,
                        note=note
                    )
                    note[output_key] = response
                finally:
                    async with active_lock:
                        num_active_task -= 1
            return note
        
        # 修复：正确创建任务列表
        tasks = [
            asyncio.create_task(post_and_judge(note=note))
            for note in input_dict[input_key]
        ]
        
        await asyncio.gather(*tasks)
        return input_dict
    
    
    async def post_request(
        self,
        query_text: str,
        query_image: str,
        note: dict,
    ):
        if "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            system_prompt = [
                {
                    "type": "text",
                    "text": self.get_system_prompt()
                }
            ]
            
            user_prompt = []
            title, content, url, search_from = note["title"], note["content"], note["url"], note["search_from"]
            if search_from == "search_note":
                search_from = "文本搜索小红书笔记"
            elif search_from == "search_image":
                search_from = "图片搜索小红书笔记"
            elif search_from == "search_web":
                search_from = "文本搜索网页"
            else:
                raise ValueError(f"Unsupported search from type {search_from}")
            
            if self.use_note_modality == "text" or len(note["images"]) == 0:
                _user_prompt = f"""
# 外源搜索文档内容
标题：{title}
摘要：{content}
链接：{url}
来源：{search_from}
"""
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt,
                    }
                )
            
            else:
                _user_prompt = f"""
# 外源搜索文档内容
标题：{title}
摘要：{content}
链接：{url}
来源：{search_from}
图片：
"""
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt,
                    }
                )
                
                for image in note["images"][:self.max_num_images]:
                    if self.image_key_type == "url":
                        image_url = compress_url(image["url"])
                        user_prompt.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                },
                            }
                        )
                    
                    elif self.image_key_type == "path":
                        if image["status"] != "valid":
                            continue
                        image = compress_and_convert_base64(path=image["path"])
                        user_prompt.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image}",
                                },
                            }
                        )
                    
                    else:
                        raise ValueError(f"Unsupported {self.image_key_type}")
            
            if self.use_query_modality == "both" and len(query_image) > 0:
                _user_prompt = f"""
# 用户提问
文本：{query_text}
图片：
"""
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt
                    }
                )
                image = compress_and_convert_base64(path=query_image)
                user_prompt.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}",
                        },
                    }
                )
            
            else:
                _user_prompt = f"""
# 用户提问
{query_text}
"""
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt
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
            title, content, url, search_from = note["title"], note["content"], note["url"], note["search_from"]
            if search_from == "search_note":
                search_from = "文本搜索小红书笔记"
            elif search_from == "search_image":
                search_from = "图片搜索小红书笔记"
            elif search_from == "search_web":
                search_from = "文本搜索网页"
            else:
                raise ValueError(f"Unsupported search from type {search_from}")
            
            if self.use_note_modality == "text" or len(note["images"]) == 0:
                _user_prompt = f"""
# 外源搜索文档内容
标题：{title}
摘要：{content}
链接：{url}
来源：{search_from}
"""
                user_prompt.append({"text": _user_prompt})
            
            else:
                _user_prompt = f"""
# 外源搜索文档内容
标题：{title}
摘要：{content}
链接：{url}
来源：{search_from}
图片：
"""
                user_prompt.append({"text": _user_prompt})
                
                for image in note["images"][:self.max_num_images]:
                    if self.image_key_type == "url":
                        image_url = compress_url(image["url"])
                        user_prompt.append(
                            {
                                "inlineData": {
                                    "mimeType": "image/jpg",
                                    "data": image_url,
                                }
                            }
                        )
                    
                    elif self.image_key_type == "path":
                        if image["status"] != "valid":
                            continue
                        image = compress_and_convert_base64(path=image["path"])
                        user_prompt.append(
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": image,
                                }
                            }
                        )
                    
                    else:
                        raise ValueError(f"Unsupported {self.image_key_type}")
            
            if self.use_query_modality == "both" and len(query_image) > 0:
                _user_prompt = f"""
# 用户提问
文本：{query_text}
图片：
"""
                user_prompt.append({"text": _user_prompt})
                image = compress_and_convert_base64(path=query_image)
                user_prompt.append(
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": image,
                        }
                    }
                )
            
            else:
                _user_prompt = f"""
# 用户提问
{query_text}
"""
                user_prompt.append({"text": _user_prompt})
            
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
                
