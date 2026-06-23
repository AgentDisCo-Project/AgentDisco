import os
import gin
import jinja2
import re
import json
import asyncio
import numpy as np
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
class ReportScorer:
    def __init__(
        self,
        model_name: str,
        max_retries: int = 50,
        retry_delay: int = 3,
        use_zh: bool = False,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file: str = "ReportScoreGenerator_EN.jinja2",
        system_template_zh_file: str = "ReportScoreGenerator_ZH.jinja2",
        system_template_en_file_for_detail: str = "ReportScoreGeneratorDetail_EN.jinja2",
        system_template_zh_file_for_detail: str = "ReportScoreGeneratorDetail_ZH.jinja2",
        system_template_en_file_for_detail_pairwise: str = "ReportScoreGeneratorPairwiseDetail_EN.jinja2",
        system_template_zh_file_for_detail_pairwise: str = "ReportScoreGeneratorPairwiseDetail_ZH.jinja2",
        system_template_en_file_pairwise: str = "ReportScoreGeneratorPairwise_EN.jinja2",
        system_template_zh_file_pairwise: str = "ReportScoreGeneratorPairwise_ZH.jinja2",
        use_detail: bool = False,
        use_pairwise: bool = False,
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
        self.use_zh = use_zh
        self.use_detail = use_detail
        self.use_pairwise = use_pairwise
        
        self.system_template_zh_file = system_template_zh_file
        self.system_template_en_file = system_template_en_file
        self.system_template_zh_file_for_detail = system_template_zh_file_for_detail
        self.system_template_en_file_for_detail = system_template_en_file_for_detail
        self.system_template_zh_file_for_detail_pairwise = system_template_zh_file_for_detail_pairwise
        self.system_template_en_file_for_detail_pairwise = system_template_en_file_for_detail_pairwise
        self.system_template_zh_file_pairwise = system_template_zh_file_pairwise
        self.system_template_en_file_pairwise = system_template_en_file_pairwise
        
    
    def get_system_prompt(
        self,
        user_question: str,
        report: str,
        race_criteria: Dict = None,
        reference_report: str = "",
    ):
        self.use_detail = self.use_detail and race_criteria is not None
        self.use_pairwise = self.use_pairwise and len(reference_report) > 0
        if not self.use_detail:
            if not self.use_pairwise:
                jinja_file = self.system_template_zh_file if self.use_zh else self.system_template_en_file
                template_vars = {
                    "user_question": user_question,
                    "report": report
                }
            else:
                jinja_file = self.system_template_zh_file_pairwise if self.use_zh else self.system_template_en_file_pairwise
                template_vars = {
                    "user_question": user_question,
                    "report_1": report,
                    "report_2": reference_report,
                }
        else:
            if not self.use_pairwise:
                jinja_file = self.system_template_zh_file_for_detail if self.use_zh else self.system_template_en_file_for_detail
                template_vars = {
                    "user_question": user_question,
                    "report": report,
                    "criteria_list": race_criteria
                }
            else:
                jinja_file = self.system_template_zh_file_for_detail_pairwise if self.use_zh else self.system_template_en_file_for_detail_pairwise
                template_vars = {
                    "user_question": user_question,
                    "report_1": report,
                    "report_2": reference_report,
                    "criteria_list": race_criteria
                }
        
        template = self.jinja_env.get_template(jinja_file)
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
        user_question: str,
        report: str,
        reference_report: str = "",
        race_criteria: List = None,
    ):
        if "qwen" in self.model.model_name or "deepseek" in self.model.model_name or "qwq" in self.model.model_name:
            system_prompt = self.get_system_prompt(user_question=user_question, report=report, reference_report=reference_report, race_criteria=race_criteria)
            user_prompt = ""
            response = await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                return_cot=False,
            )
        
        elif "gemini" in self.model.model_name:
            system_prompt = [
                {
                    "text": self.get_system_prompt(user_question=user_question, report=report, reference_report=reference_report, race_criteria=race_criteria)
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
                check_func=self.check_func,
                return_cot=False,
            )

        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        return response


    async def act(
        self,
        input_dict: Dict,
        report_key: str = "report"
    ):
        user_question = input_dict["query_text"]
        report = input_dict[f"{report_key}"]
        reference_report = input_dict.get("reference_report", "")
        race_criteria = input_dict.get("race_criteria", None)
        race_weight = input_dict.get("race_weight", {})
        if self.use_detail and race_criteria and race_weight:
            if self.use_pairwise:
                response = await self.post_request(user_question=user_question, report=report, reference_report=reference_report, race_criteria=race_criteria)
                report_score, reference_report_score = dict(), dict()
                overall_report_score, overall_reference_report_score = 0., 0.
                for request_type in ("comprehensiveness", "insight", "instruction_following", "readability"):
                    report_score[request_type], reference_report_score[request_type] = 0., 0.
                    for res in response[request_type]:
                        report_score[request_type] += res["article_1_score"]
                        reference_report_score[request_type] += res['article_2_score']
                    overall_report_score += float(np.mean(report_score[request_type])) * race_weight[request_type]
                    overall_reference_report_score += float(np.mean(reference_report_score[request_type])) * race_weight[request_type]
                input_dict["report_score"] = report_score
                input_dict["reference_report_score"] = reference_report_score
                input_dict["overall_report_score"] = overall_report_score
                input_dict["overall_reference_report_score"] = overall_reference_report_score
                input_dict["justification"] = response
                input_dict["overall_report_norm_score"] = overall_report_score / (overall_report_score + overall_reference_report_score)
                input_dict["overall_reference_report_norm_score"] = overall_reference_report_score / (overall_report_score + overall_reference_report_score)
                
            else:
                response = await self.post_request(user_question=user_question, report=report, reference_report=reference_report, race_criteria=race_criteria)
                report_score = dict()
                overall_report_score = 0.
                for request_type in ("comprehensiveness", "insight", "instruction_following", "readability"):
                    report_score[request_type] = 0.
                    for res in response[request_type]:
                        report_score[request_type] += res["target_score"]
                    overall_report_score += float(np.mean(report_score[request_type])) * race_weight[request_type]
                input_dict["report_score"] = report_score
                input_dict["overall_report_score"] = overall_report_score
                input_dict["justification"] = response

        else:
            if self.use_pairwise:
                response = await self.post_request(user_question=user_question, report=report, reference_report=reference_report, race_criteria=race_criteria)
                report_score, reference_report_score = dict(), dict()
                overall_report_score, overall_reference_report_score = 0., 0.
                for request_type in ("comprehensiveness", "insight", "instruction_following", "readability"):
                    report_score[request_type], reference_report_score[request_type] = 0., 0.
                    for res in response[request_type]:
                        report_score[request_type] += res["article_1_score"]
                        reference_report_score[request_type] += res['article_2_score']
                    overall_report_score += float(np.mean(report_score[request_type])) * 0.25
                    overall_reference_report_score += float(np.mean(reference_report_score[request_type])) * 0.25
                input_dict["report_score"] = report_score
                input_dict["reference_report_score"] = reference_report_score
                input_dict["overall_report_score"] = overall_report_score
                input_dict["overall_reference_report_score"] = overall_reference_report_score
                input_dict["justification"] = response
                input_dict["overall_report_norm_score"] = overall_report_score / (overall_report_score + overall_reference_report_score)
                input_dict["overall_reference_report_norm_score"] = overall_reference_report_score / (overall_report_score + overall_reference_report_score)
                
            else:
                response = await self.post_request(user_question=user_question, report=report, reference_report=reference_report, race_criteria=race_criteria)
                input_dict["overall_report_score"] = response["overall_score"]
                input_dict["report_score"] = dict()
                input_dict["justification"] = response
        return input_dict
    
    

if __name__ == "__main__":
    async def main():
        file_path = "/mnt/tidalfs-bdsz01/usr/tusen/search-agent-dev/data/data1010/测试样本_2.md"
        print(file_path)
        with open(file_path, "r", encoding='utf-8') as file:
            report = file.read()
        
        service = ReportScorer(
            model_name="gemini-2.5-pro",
            max_retries=5,
            retry_delay=3,
            use_zh=True,
            use_detail=False,
            use_pairwise=False,
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
