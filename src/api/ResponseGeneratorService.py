import os
import asyncio
import gin
import jinja2
import json
import sys
sys.path.append('.')

from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.url_operator import compress_and_convert_base64, compress_url
from api.utils.key_operator import ApiKeyCycler
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


@gin.configurable()
class ResponseGenerator:
    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        retry_delay: int = 3,
        use_query_modality: str = "text",
        use_note_modality: str = "text",
        use_customize_url: bool = False,
        customize_url: str = "",
        include_search: bool = True,
        include_summary: bool = True,
        include_comment: bool = False,
        image_key_type: str = "url",
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_file: str = "ResponseGenerator.jinja2"
    ):
        self.model = CustomizeChatGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
        )
        assert use_query_modality in ("text", "both"), f"Unsupported use_query_modality {use_query_modality}"
        self.use_query_modality = use_query_modality
        assert use_note_modality in ("text", "one_image", "all_images"), f"Unsupported {use_note_modality}"
        self.use_note_modality = use_note_modality
        self.image_key_type = image_key_type
        
        self.include_search = include_search
        self.include_summary = include_summary
        
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.jinja_file = system_template_file
        self.include_comment = include_comment
    
    
    def get_system_prompt(self):
        def _determine_search_content_type():
            # 确定搜索内容类型
            if not self.include_search:
                return None
            if (self.use_query_modality == "text" and self.include_search and self.include_summary) or \
                (self.use_query_modality == "text" and self.include_search and self.use_note_modality == "text"):
                return "text_only"
            elif self.use_query_modality == "text" and self.include_search and \
                self.use_note_modality in ("one_image", "all_images") and not self.include_summary:
                return "text_and_image"
            elif (self.use_query_modality == "both" and self.include_search and self.include_summary) or \
                (self.use_query_modality == "both" and self.include_search and self.use_note_modality == "text"):
                return "text_only"
            elif self.use_query_modality == "both" and self.include_search and \
                self.use_note_modality in ("one_image", "all_images") and not self.include_summary:
                return "text_and_image"
            else:
                return "text_and_image"  # 默认值
        
        def _has_input_description():
            # 判断是否需要输入说明部分
            # 当查询模式为both或者包含搜索且搜索内容包含图片时，需要输入说明
            return (self.use_query_modality == "both") or \
                (self.include_search and _determine_search_content_type() == "text_and_image")
        
        
        template_vars = {
            'use_query_modality': self.use_query_modality,
            'include_search': self.include_search,
            'include_summary': getattr(self, 'include_summary', False),
            'use_note_modality': getattr(self, 'use_note_modality', None),
            'search_content_type': _determine_search_content_type(),
            'has_input_description': _has_input_description(),
            "include_comment": self.include_comment,
        }
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    @staticmethod
    def check_func(
        response: str,
    ):
        return response
    
    
    async def act(
        self,
        input_dict: dict,
        response_key: str = "response"
    ):
        query_text = input_dict["query_text"]
        query_image = input_dict["query_image"]
        notes = input_dict["selected_search_results"]
        if "qwen3-next-80b-a3b-instruct" in self.model.model_name:
            system_prompt = self.get_system_prompt()
            user_prompt = ""
            
            if self.include_search:
                _user_prompt = """
# 外源搜索文档内容
"""
                user_prompt += _user_prompt

                for idx, note in enumerate(notes):
                    title, content, url, search_from = note["title"], note["content"], note["url"], note["search_from"]
                    if search_from == "search_note":
                        search_from = "文本搜索小红书笔记"
                    elif search_from == "search_image":
                        search_from = "图片搜索小红书笔记"
                    elif search_from == "search_web":
                        search_from = "文本搜索网页"
                    else:
                        raise ValueError(f"unsupported search from type {search_from}")
                    
                    _user_prompt = f"""
## 外源搜索文档{idx}的内容如下：
文档ID：{note["id"]}
标题：{title}
摘要：{content}
链接：{url}
来源：{search_from}
"""
                    user_prompt += _user_prompt
                    
                    if self.include_comment:
                        comments = note["comments"]
                        for comment_id, comment in enumerate(comments):
                            _comment = comment["content"]
                            _user_prompt = f"""
第{comment_id}条评论内容：{_comment}
"""
                            user_prompt += _user_prompt
                
            
            _user_prompt = f"""
# 用户提问
{query_text}
"""
            user_prompt += _user_prompt
            cycler = ApiKeyCycler(api_key_list=list(DIRECTLLM_API_KEY_USER.values()))
            response = await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                cycler=cycler,
                return_cot=False,
            )



        elif "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            system_prompt = [
                {
                    "type": "text",
                    "text": self.get_system_prompt()
                }
            ]
            user_prompt = []
            
            if self.include_search:
                _user_prompt = """
# 外源搜索文档内容
"""
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt,
                    }
                )
                
                for idx, note in enumerate(notes):
                    title, content, url, search_from = note["title"], note["content"], note["url"], note["search_from"]
                    if search_from == "search_note":
                        search_from = "文本搜索小红书笔记"
                    elif search_from == "search_image":
                        search_from = "图片搜索小红书笔记"
                    elif search_from == "search_web":
                        search_from = "文本搜索网页"
                    elif search_from == "search_knowledge":
                        search_from = "外源知识数据库"
                    else:
                        raise ValueError(f"unsupported search from type {search_from}")
                    
                    if self.include_summary and "summary" in note:
                        summary = note["summary"]
                        _user_prompt = f"""
## 外源搜索文档{idx}的内容如下：
总结：{summary}
"""
                        
                        user_prompt.append(
                            {
                                "type": "text",
                                "text": _user_prompt,
                            }
                        )
                    
                    elif self.use_note_modality in ("one_image", "all_images") and len(note["images"]) > 0:
                        _user_prompt = f"""
## 外源搜索文档{idx}的内容如下：
文档ID：{note["id"]}
标题：{title}
内容：{content}
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
                        
                        
                        for image in note["images"][:1]:
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
                    
                    else:
                        _user_prompt = f"""
## 外源搜索文档{idx}的内容如下：
文档ID：{note["id"]}
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
            
                    if self.include_comment:
                        comments = note["comments"]
                        for comment_id, comment in enumerate(comments):
                            _comment = comment["content"]
                            _user_prompt = f"""
第{comment_id}条评论内容：{_comment}
"""
                            
                            user_prompt.append(
                                {
                                    "type": "text",
                                    "text": _user_prompt
                                }
                            )
            
            if self.use_query_modality == "both" and len(query_image) != 0:
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
            if self.include_search:
                _user_prompt = f"""
# 外源搜索文档内容
"""
                user_prompt.append({"text": _user_prompt})
                
                for idx, note in enumerate(notes):
                    title, content, url, search_from = note["title"], note["content"], note["url"], note["search_from"]
                    if search_from == "search_note":
                        search_from = "文本搜索小红书笔记"
                    elif search_from == "search_image":
                        search_from = "图片搜索小红书笔记"
                    elif search_from == "search_web":
                        search_from = "文本搜索网页"
                    elif search_from == "search_knowledge":
                        search_from = "外源知识数据库"
                    else:
                        raise ValueError(f"unsupported search from type {search_from}")
                    
                    if self.include_summary and "summary" in note:
                        summary = note["summary"]
                        _user_prompt = {
                            "text": f"""
## 外源搜索文档{idx}的内容如下：
总结：{summary}
"""
                        }
                        user_prompt.append(_user_prompt)
                    
                    elif self.use_note_modality in ("one_image", "all_images") and len(note["images"]) > 0:
                        _user_prompt = {
                            "text": f"""
## 外源搜索文档{idx}的内容如下：
文档ID：{note["id"]}
标题：{title}
摘要：{content}
链接：{url}
来源：{search_from}
图片：
"""
                        }
                        user_prompt.append(_user_prompt)
                        for image in note["images"][:1]:
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
                    
                    else:
                        _user_prompt = {
                            "text": f"""
## 外源搜索文档{idx}的内容如下：
文档ID：{note["id"]}
标题：{title}
摘要：{content}
链接：{url}
来源：{search_from}
"""
                        }
                        user_prompt.append(_user_prompt)
                    
                    if self.include_comment and len(note["comments"]) > 0:
                        comments = note["comments"]
                        _user_prompt = f"""
评论区内容：
"""
                        user_prompt.append({"text": _user_prompt})
                        for comment_id, comment in enumerate(comments):
                            _comment = comment["content"]
                            _user_prompt = f"""
第{comment_id}条评论内容：{_comment}
"""
                            user_prompt.append({"text": _user_prompt})
            
            
            if self.use_query_modality == "both" and len(query_image) != 0:
                _user_prompt = f"""
# 用户提问
文本：{query_text}
图片：
"""
                user_prompt.append(
                    {
                        "text": _user_prompt
                    }
                )
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
            
            response = await self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                gemini_api_key=GEMINI_API_KEY,
                directllm_api_key=DIRECTLLM_API_KEY_USER["tusen"],
                return_cot=False,
            )
        
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        input_dict[response_key] = response
        # reference_map = dict()
        # for idx, note in enumerate(notes):
        #         reference_map[f"{idx}"] = note
        # input_dict["reference_map"] = reference_map
        return input_dict


if __name__ == "__main__":
    async def main():
        service = ResponseGenerator(
            model_name="gemini-2.5-pro",
            max_retries=5,
            retry_delay=3,
        )
        input_dict = dict()
        results = await service.act(
            input_dict=input_dict
        )
        print(results)
    
    asyncio.run(main())
 