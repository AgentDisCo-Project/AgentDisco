import os
import gin
import jinja2
import json
import sys
sys.path.append('.')

from dotenv import load_dotenv
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.url_operator import compress_and_convert_base64
from api.utils.key_operator import ApiKeyCycler


load_dotenv('./api/utils/keys.env')
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)


@gin.configurable()
class EntryJudge:
    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        retry_delay: int = 3,
        use_query_modality: str = "text",
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_file: str = "EntryJudge.jinja2"
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
        
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.jinja_file = system_template_file
    
    
    def get_system_prompt(self):
        template_vars = {
            'use_query_modality': self.use_query_modality
        }
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
        response: str
    ):
        response = self.json_fix(response)
        response = response.replace("\n", "")
        response = response.lstrip("。").rstrip('。')
        response = float(response)
        return response
    
    
    async def act(
        self,
        input_dict: dict,
    ):
        query_text = input_dict.get("query_text", "")
        query_image = input_dict.get("query_image", "")
        
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
            
            response = await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
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
                user_prompt.append({"text": user_prompt})
            
            breakpoint()
            response = await  self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                return_cot=False,
            )
            breakpoint()
        
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        input_dict["entry_judge"] = response
        return input_dict

