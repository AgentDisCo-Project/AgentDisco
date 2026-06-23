import os
import gin
import jinja2
import re
import json
import asyncio
import sys
sys.path.append('.')

from typing import List, Dict
from api.utils.key_operator import ApiKeyCycler
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from dotenv import load_dotenv
from api.utils.string_operator import json_fix


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


@gin.configurable()
class CriteriaGenerator:
    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        max_retries: int = 50,
        retry_delay: int = 3,
        max_concurrent: int = 50,
        use_zh: bool = False,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file_for_dimension: str = "CriteriaGeneratorDimension_EN.jinja2",
        system_template_en_file_for_comprehensiveness: str = "CriteriaGeneratorComprehensiveness_EN.jinja2",
        system_template_en_file_for_insight: str = "CriteriaGeneratorInsight_EN.jinja2",
        system_template_en_file_for_instruction_following: str = "CriteriaGeneratorInstructionFollowing_EN.jinja2",
        system_template_en_file_for_readability: str = "CriteriaGeneratorReadability_EN.jinja2",
        system_template_zh_file_for_dimension: str = "CriteriaGeneratorDimension_ZH.jinja2",
        system_template_zh_file_for_comprehensiveness: str = "CriteriaGeneratorComprehensiveness_ZH.jinja2",
        system_template_zh_file_for_insight: str = "CriteriaGeneratorInsight_ZH.jinja2",
        system_template_zh_file_for_instruction_following: str = "CriteriaGeneratorInstructionFollowing_ZH.jinja2",
        system_template_zh_file_for_readability: str = "CriteriaGeneratorReadability_ZH.jinja2",
        num_samples_for_dimension: int = 1,
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
        self.jinja_file_for_dimension = system_template_zh_file_for_dimension if use_zh else system_template_en_file_for_dimension
        self.jinja_file_for_comprehensiveness = system_template_zh_file_for_comprehensiveness if use_zh else system_template_en_file_for_comprehensiveness
        self.jinja_file_for_insight = system_template_zh_file_for_insight if use_zh else system_template_en_file_for_insight
        self.jinja_file_for_instruction_following = system_template_zh_file_for_instruction_following if use_zh else system_template_en_file_for_instruction_following
        self.jinja_file_for_readability = system_template_zh_file_for_readability if use_zh else system_template_en_file_for_readability
        self.use_zh = use_zh
        self.max_concurrent = max_concurrent
        self.num_samples_for_dimension = num_samples_for_dimension
    
    
    def get_system_prompt(
        self,
        user_question: str,
        request_type: str,
    ):
        template_vars = {
            "user_question": user_question
        }
        if request_type == "dimension":
            jinja_file = self.jinja_file_for_dimension
        elif request_type == "comprehensiveness":
            jinja_file = self.jinja_file_for_comprehensiveness
        elif request_type == "insight":
            jinja_file = self.jinja_file_for_insight
        elif request_type == "instruction_following":
            jinja_file = self.jinja_file_for_instruction_following
        elif request_type == "readability":
            jinja_file = self.jinja_file_for_readability
        else:
            raise ValueError(f"Unsupported {request_type}")
        template = self.jinja_env.get_template(jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    @staticmethod
    def parser_response(
        response: str,
        return_type: str = "",
        check_weights: bool = False,
    ):
        response = json_fix(response)
        parsed_data = json.loads(response)
        
        if return_type == "dict":
            assert isinstance(parsed_data, Dict)
            if check_weights:
                sum_weights = sum(float(value) for value in parsed_data.values())
                assert sum_weights == 1.
            
        if return_type == "list":
            assert isinstance(parsed_data, List)
        return parsed_data
    
    
    
    async def act(
        self,
        input_dict: Dict,
    ):
        user_question = input_dict["query_text"]
        weight_comprehensiveness, weight_insight, weight_instruction_following, weight_readability = 0., 0., 0., 0.
        for _ in range(self.num_samples_for_dimension):
            dimension_weights = await self.post_request(user_question=user_question, request_type="dimension")
            weight_comprehensiveness += dimension_weights["comprehensiveness"]
            weight_insight += dimension_weights["insight"]
            weight_instruction_following += dimension_weights["instruction_following"]
            weight_readability += dimension_weights["readability"]
        weight_comprehensiveness = weight_comprehensiveness / self.num_samples_for_dimension
        weight_insight = weight_insight / self.num_samples_for_dimension
        weight_instruction_following = weight_instruction_following / self.num_samples_for_dimension
        weight_readability = weight_readability / self.num_samples_for_dimension
        
        # round_weights_and_adjust
        if weight_comprehensiveness + weight_insight + weight_instruction_following + weight_readability != 1:
            weight_readability += 1 - (weight_comprehensiveness + weight_insight + weight_instruction_following + weight_readability)
        
        criteria = dict()
        for request_type in ("comprehensiveness", "insight", "instruction_following", "readability"):
            criteria[request_type] = await self.post_request(user_question=user_question, request_type=request_type)
                
        input_dict["race_criteria"] = criteria
        input_dict["race_weight"] = {
            "comprehensiveness": weight_comprehensiveness,
            "insight": weight_insight,
            "instruction_following": weight_instruction_following,
            "readability": weight_readability
        }
        return input_dict
    
    
    async def post_request(
        self,
        user_question: str,
        request_type: str,
    ):
        if "qwen" in self.model.model_name or "deepseek" in self.model.model_name or "qwq" in self.model.model_name:
            system_prompt = self.get_system_prompt(user_question=user_question, request_type=request_type)

            user_prompt = ""
            if self.use_zh:
                _user_prompt = f"""
请严格遵循以上说明和方法，现在针对以下具体任务开始你的工作：
<task>
    "{user_question}"
</task>
请输出你的 `<analysis>` 和 `<json_output>`。
"""
            else:
                _user_prompt = f"""
Please strictly follow the above instructions and methods. Now, begin your work on the following specific task:
<task>
    "{user_question}"
</task>
Please output your `<analysis>` and `<json_output>`.
"""
            user_prompt += _user_prompt
            response = await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.parser_response,
                return_cot=False,
            )
        
        elif "gemini" in self.model.model_name:
            system_prompt = [
                {
                    "text": self.get_system_prompt(user_question=user_question, request_type=request_type)
                }
            ]
            user_prompt = []
            if self.use_zh:
                _user_prompt = f"""
请严格遵循以上说明和方法，现在针对以下具体任务开始你的工作：
<task>
    "{user_question}"
</task>
请输出你的 `<analysis>` 和 `<json_output>`。
"""
            else:
                _user_prompt = f"""
Please strictly follow the above instructions and methods. Now, begin your work on the following specific task:
<task>
    "{user_question}"
</task>
Please output your `<analysis>` and `<json_output>`.
"""
            user_prompt.append(
                {
                    "text": _user_prompt,
                }
            )
            
            response = await self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.parser_response,
                return_cot=False,
            )
            
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        return response



if __name__ == "__main__":
    async def main():
        file_path = "/mnt/tidalfs-bdsz01/usr/tusen/search-agent-dev/data/data1010/测试样本_2.md"
        print(file_path)
        with open(file_path, "r", encoding='utf-8') as file:
            report = file.read()
        
        service = CriteriaGenerator(
            use_zh=True,
            num_samples_for_dimension=1,
        )
        input_dict = {
            "query_text": "文生图（Diffusion Models）领域知名学者",
            "report": report,
        }
        
        input_dict = await service.act(
            input_dict=input_dict
        )
        breakpoint()
        print(input_dict)
    
    asyncio.run(main())
    