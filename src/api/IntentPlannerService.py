import os
import gin
import asyncio
import jinja2
import json
import sys
sys.path.append('.')

from typing import Dict
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
class IntentPlanner:
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
        system_template_en_file: str = "IntentPlanner_EN.jinja2",
        system_template_zh_file: str = "IntentPlanner_ZH.jinja2"
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
        self.use_zh = use_zh
    

    def get_system_prompt(self):
        template_vars = {}
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt

    
    def parser_response(
        self,
        response: str,
    ):
        response = json_fix(response)
        response = json.loads(response)

        if not isinstance(response, Dict):
            raise ValueError()
        if "intent" not in response or "response_style" not in response:
            raise ValueError()
        if self.use_zh:
            if response["intent"] not in (
                "对比选择",
                "推荐建议",
                "操作指南",
                "旅游规划",
                "购买决策",
                "事实查询",
                "状态进展",
                "新闻资讯",
                "探索深入",
                "资源定位"
            ):
                raise ValueError(f"Unsupported intent type {response['intent']}")
        else:
            if response["intent"] not in (
                "Comparison & Selection",
                "Recommendations & Suggestions",
                "How-to Guide",
                "Travel Planning",
                "Purchase Decision",
                "Fact Query",
                "Status & Progress",
                "News & Information",
                "Deep Exploration",
                "Resource Locating"
            ):
                raise ValueError(f"Unsupported intent type {response['intent']}")
        return response

    
    def check_func(
        self,
        response: str
    ):
        response = self.parser_response(response)
        return response

    
    async def act(
        self,
        input_dict: dict,
    ):
        query_text = input_dict.get("query_text", "") or input_dict.get("query", "")

        if "gemini" in self.model.model_name:
            system_prompt = [{"text": self.get_system_prompt()}]

            user_prompt = []
            if self.use_zh:
                _user_prompt =f"""
# 用户查询
{query_text}
"""
            else:
                _user_prompt = f"""
# User Query
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
        
        input_dict["intent"] = response["intent"]
        input_dict['response_style'] = response['response_style']
        
        return input_dict 


if __name__ == "__main__":
    async def main():
        service = IntentPlanner(
            model_name="gemini-3-flash",
            use_zh=True,
        )

        input_dict = dict()
        input_dict["query_text"] = "去廊坊水世界周围玩点什么比较好，抖音上定了团购券，但不能退了，听说廊坊水世界水不干净"
        input_dict = await service.act(
            input_dict=input_dict,
        )
        breakpoint()
        
    asyncio.run(main())
       