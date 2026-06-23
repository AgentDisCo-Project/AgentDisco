import json
import os
import gin
import jinja2
import json
import sys
sys.path.append('.')

from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.url_operator import compress_and_convert_base64
from api.utils.key_operator import ApiKeyCycler
from dotenv import load_dotenv

load_dotenv('./api/utils/keys.env')
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)


@gin.configurable()
class ResponseJudge:
    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        retry_delay: int = 3,
        use_query_modality: str = "text",
        generate_type: str = "note",
        use_customize_url: bool = False,
        customize_url: str = "",
        max_query_len: int = 5,
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_file: str = "ResponseJudge.jinja2"
    ):
        self.model = CustomizeChatGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key
        )
        assert use_query_modality in ("text", "both"), f"Unsupported use_query_modality {use_query_modality}"
        self.use_query_modality = use_query_modality
        self.generate_type = generate_type
        self.max_query_len = max_query_len
        
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.jinja_file = system_template_file
    
    
    def get_system_prompt(self):
        template_vars = {
            'generate_type': self.generate_type,
            'use_query_modality': self.use_query_modality,
        }
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        
        return system_prompt
    
    
    def check_func(
        self,
        response: str,
    ):
        response = json.loads(response)
        assert len(response) <= self.max_query_len, f"Unsupported {self.max_query_len}"
        return response
    
    
    async def act(
        self,
        input_dict: dict,
        response_key: str,
        response_queue: list = None,
        note_subquery_queue: list = None,
        web_subquery_queue: list = None,
    ):
        query_text = input_dict.get("query_text", "")
        query_image = input_dict.get("query_image", "")
        response = input_dict.get(response_key, "")
        note_subquery = input_dict.get("note_subquery", [])
        web_subquery = input_dict.get("web_subquery", [])
        
        if "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            system_prompt = [
                {
                    "type": "text",
                    "text": self.get_system_prompt()
                }
            ]
            
            if self.use_query_modality == "both" and len(query_image) != 0:
                user_prompt = []
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
                user_prompt = []
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
            
            _user_prompt = f"""
# 当前回答
{response}

"""
            user_prompt.append(
                {
                    "type": "text",
                    "text": _user_prompt
                }
            )
            
            if self.generate_type == "note":
                _user_prompt = f"""
# 当前小红书笔记搜索词列表
{".".join(note_subquery)}

"""
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt
                    }
                )
            
            elif self.generate_type == "web":
                _user_prompt = f"""
# 当前互联网搜索词列表
{".".join(web_subquery)}

"""
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt
                    }
                )
            
            if self.generate_type == "note" and len(response_queue) == len(note_subquery_queue):
                for idx, (response, note_subquery) in enumerate(zip(response_queue, note_subquery_queue)):
                    _user_prompt = f"""
# 第{idx}轮回答
{response}

# 第{idx}轮小红书笔记搜索词列表
{".".join(note_subquery)}

"""
                    user_prompt.append(
                        {
                            "type": "text",
                            "text": _user_prompt
                        }
                    )
            
            if self.generate_type == "web" and len(response_queue) == len(web_subquery_queue):
                for idx, (response, web_subquery) in enumerate(zip(response_queue, web_subquery_queue)):
                    _user_prompt = f"""
# 第{idx}轮回答
{response}

# 第{idx}轮互联网搜索词列表
{".".join(web_subquery)}

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
            
            if self.use_query_modality == "both" and len(query_image) != 0:
                user_prompt = []
                _user_prompt = f"""
# 用户提问
文本：{query_text}
图片：
"""
                user_prompt.append({"text": user_prompt})
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
                user_prompt = []
                _user_prompt = f"""
# 用户提问
{query_text}
"""
                user_prompt.append({"text": _user_prompt})
            
            
            _user_prompt = f"""
# 当前回答
{response}
"""
            user_prompt.append({"text": _user_prompt})
            
            _user_prompt = f"""
# 当前小红书笔记搜索词列表
{".".join(note_subquery)}
"""
            user_prompt.append({"text": _user_prompt})
            
            _user_prompt = f"""
# 当前互联网搜索词列表
{".".join(web_subquery)}
"""
            user_prompt.append({"text": _user_prompt})
            
            if self.generate_type == "note" and len(response_queue) == len(note_subquery_queue) > 0:
                for idx, (response, note_subquery) in enumerate(zip(response_queue, note_subquery_queue)):
                    _user_prompt = f"""
# 第{idx}轮回答
{response}

# 第{idx}轮小红书笔记搜索词列表
{".".join(note_subquery)}

"""
                    user_prompt.append({"text": _user_prompt})
            
            if self.generate_type == "web" and len(response_queue) == len(web_subquery_queue) > 0:
                for idx, (response, web_subquery) in enumerate(zip(response_queue, web_subquery_queue)):
                    _user_prompt = f"""
# 第{idx}轮回答
{response}

# 第{idx}轮互联网搜索词列表
{".".join(web_subquery)}

"""
                    user_prompt.append({"text": _user_prompt})
            
            
            
            response = self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                gemini_api_key=GEMINI_API_KEY,
                directllm_api_key=DIRECTLLM_API_KEY_USER["tusen"],
                return_cot=False,
            )
        
        
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        if self.generate_type == "note":
            input_dict["note_subquery_judge"] = response
        elif self.generate_type == "web":
            input_dict["web_subquery_judge"] = response
        else:
            raise ValueError(f"Unsupported {self.generate_type}")
        
        return input_dict