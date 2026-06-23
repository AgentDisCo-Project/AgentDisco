import os
import json
import gin
import jinja2
import sys
sys.path.append('.')

from dotenv import load_dotenv
from datetime import datetime
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.url_operator import compress_and_convert_base64
from api.utils.key_operator import ApiKeyCycler


load_dotenv('./api/utils/keys.env')
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER) 


@gin.configurable()
class QueryGenerator:
    def __init__(
        self,
        model_name: str,
        note_model_name: str = None,
        web_model_name: str = None,
        max_retries: int = 5,
        retry_delay: int = 3,
        max_concurrent: int = 50,
        use_query_modality: str = "text",
        generate_type: str = "note",
        use_customize_url: bool = False,
        customize_url: str = "",
        max_query_len: int = 10,
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_file: str = "QueryGenerator.jinja2"
    ):
        assert generate_type in ("note", "web"), f"Unsupported generate_type {generate_type}"
        self.generate_type = generate_type
        if generate_type == "note":
            model_name = note_model_name if note_model_name is not None else model_name
        elif generate_type == "web":
            model_name = web_model_name if web_model_name is not None else model_name
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
        self.max_concurrent = max_concurrent
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
            'use_query_modality': self.use_query_modality
        }
        if self.generate_type == "web":
            template_vars['curr_date'] = datetime.now().strftime("%Y年%m月%d日")
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    @staticmethod
    def json_fix(
        response: str
    ):
        if len(response) == 0:
            return response
        
        response = response.strip()
        # 移除markdown代码块标记
        if response.startswith('```json'):
            response = response[7:]
        elif response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        return response.strip()
    
    
    def check_func(
        self,
        response: str,
    ):
        response = self.json_fix(response)
        response = json.loads(response)
        assert len(response) <= self.max_query_len, f"Unsupported {self.max_query_len}"
        return response
    
    
    async def act(
        self,
        input_dict: dict,
    ):
        query_text = input_dict.get("query_text", "")
        query_image = input_dict.get("query_image", "")

        if "qwen3-next-80b-a3b-instruct" in self.model.model_name:
            system_prompt = self.get_system_prompt()
            user_prompt = ""
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
                user_prompt = []
                _user_prompt = f"""
# 用户提问
{query_text}
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
        
        response += [query_text]
        if  self.generate_type == "note":
            input_dict["note_subquery"] = response
        elif self.generate_type == "web":
            input_dict["web_subquery"] = response
        else:
            raise ValueError(f"Unsupported {self.generate_type}")
        return input_dict
        