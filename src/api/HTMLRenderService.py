import os
import gin
import asyncio
import jinja2
import json
import sys
sys.path.append('.')

from typing import Dict, List
from api.utils.key_operator import ApiKeyCycler
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.string_operator import json_fix
from api.utils.url_operator import compress_and_convert_base64, compress_url
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


@gin.configurable()
class HTMLRender:
    def __init__(
        self,
        model_name: str,
        use_zh: bool = True,
        max_retries: int = 5,
        retry_delay: int = 3,
        max_concurrent: int = 50,
        use_customize_url: bool = False,
        customize_url: str = "",
        max_summary_len: int = 256,
        image_key_type: str = "url",
        max_num_images: int = 14,
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file: str = "HTMLRender_EN.jinja2",
        system_template_zh_file: str = "HTMLRender_ZH.jinja2",
        render_template_dir: str = "./gallery",
        render_template_en_file: str = "KIMIVertical.txt",
        render_template_zh_file: str = "KIMIVertical.txt",
        render_with_image: bool = True,
    ):
        self.model = CustomizeChatGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
        )

        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.jinja_file = system_template_en_file if not use_zh else system_template_zh_file
        self.render_file = os.path.join(render_template_dir, render_template_en_file) if not use_zh else os.path.join(render_template_dir, render_template_zh_file)
        self.use_zh = use_zh
        self.render_with_image = render_with_image


    def get_system_prompt(
        self,
        render_with_image: bool,
    ):
        template_vars = {
            "render_with_image": render_with_image
        }
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt

    def get_render_template(self):
        if not os.path.isfile(self.render_file):
            raise ValueError(self.render_file)
        with open(self.render_file, "r", encoding="utf-8") as f:
            return f.read()
    
    def check_func(
        self,
        response: str,
    ):
        return response

    async def post_request(
      self,
      query_text: str,
      images: List,
      report: str,      
    ):
        template = self.get_render_template()

        if "gemini" in self.model.model_name:
            render_with_image = self.render_with_image and len(images) > 0
            system_prompt = [{"text": self.get_system_prompt(render_with_image=render_with_image)}]
            user_prompt = []

            if self.use_zh:
                _user_prompt = f"""
# 用户提问
{query_text}
"""
            
            else:
                _user_prompt = f"""
# User Question
{query_text}
"""
            user_prompt.append({"text": _user_prompt})
        

            if self.use_zh:
                _user_prompt = f"""
# 报告内容
{report}
"""
            
            else:
                _user_prompt = f"""
# Report
{report}
"""
            user_prompt.append({"text": _user_prompt})

            if self.use_zh:
                _user_prompt = f"""
# HTML模版
{template}
"""
            else:
                _user_prompt = f"""
# HTML Template
{template}
"""
            user_prompt.append({"text": _user_prompt})
            
            for idx, image in enumerate(images):
                image_url, image_path = image["url"], image["path"]

                if self.use_zh:
                    _user_prompt = f"""
# 图片{idx}
图片的链接（HTML使用图片请引用）: {image_url}
图片如下：
"""
                else:
                    _user_prompt = f"""
# Picture {idx}
Image link (please reference this when using the image in HTML): {image_url}
The image is shown below:
"""

                user_prompt.append({"text": _user_prompt})
                image = compress_and_convert_base64(path=image_path)
                user_prompt.append(
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": image,
                        }
                    }
                )
            
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
        input_key: str = "rendered_report"
    ):
        query_text = input_dict.get("query_text", "") or input_dict.get("query", "")
        report = input_dict[input_key]
        if self.render_with_image:
            images = input_dict.get("selected_images", [])
        else:
            images = []
        
        response = await self.post_request(
            query_text=query_text,
            images=images,
            report=report,
        )
        input_dict["rendered_html"] = response 
        return input_dict
    



