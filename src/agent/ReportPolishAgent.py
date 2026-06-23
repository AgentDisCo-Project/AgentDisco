import json
import gin
import jinja2
import os
import sys

sys.path.append('.')

from typing import Optional, List, Union, Dict
from agent.BaseAgent import BasicAgent
from api.CriteriaGeneratorService import CriteriaGenerator
from api.ReportScorerService import ReportScorer
from api.utils.key_operator import ApiKeyCycler
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.ValidateReferenceService import ValidateReference

DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


@gin.configurable()
class ReportPolish(BasicAgent):
    def __init__(
        self,
        name: Optional[str] = "report_polish",
        description_en: Optional[str] = "A report judge and refine for report generation.",
        description_zh: Optional[str] = "报告润色器。",
        tool_bank: Optional[List[Union[str, Dict]]] = "",
        use_zh: bool = False,
        model_name: str = "",
        max_retries: int = 5,
        retry_delay: int = 3,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file: str = "ReportPolish_EN.jinja2",
        system_template_zh_file: str = "ReportPolish_ZH.jinja2",
        system_template_en_file_before_render: str = "ReportPolishBeforeRender_EN.jinja2",
        system_template_zh_file_before_render: str = "ReportPolishBeforeRender_EN.jinja2",
        need_fact: bool = False,
        need_race: bool = False,
        use_detail: bool = False,
        use_pairwise: bool = False,
        use_polish_before_render: bool = False,
    ):
        super().__init__(
            name=name,
            description_en=description_en,
            description_zh=description_zh,
            tool_bank=tool_bank,
            use_zh=use_zh
        )
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
        if use_polish_before_render:
            self.jinja_file = system_template_zh_file_before_render if self.use_zh else system_template_en_file_before_render
        else:
            self.jinja_file = system_template_zh_file if self.use_zh else system_template_en_file
        self.reference_judge = ValidateReference(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
            use_zh=use_zh,
        )
        self.criteria_generator = CriteriaGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
            use_zh=use_zh,
        )
        self.report_scorer = ReportScorer(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
            use_zh=use_zh,
        )
        self.need_fact = need_fact
        self.need_race = need_race
        self.use_detail = use_detail
        self.use_pairwise = use_pairwise
        
        
    def get_system_prompt(
        self
    ):
        template_vars = {}
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    @staticmethod
    def check_func(
        response: str
    ):
        return response
        
    
    async def post_request(
        self,
        query: str,
        report: str,
        reference_error: Dict,
        report_justification: str,
    ):
        if "qwen" in self.model.model_name or "deepseek" in self.model.model_name or "qwq" in self.model.model_name:
            system_prompt = [
                {
                    "type": "text",
                    "text": self.get_system_prompt()
                }
            ]
            
            user_prompt = []
            if self.use_zh:
                _user_prompt = f"""
# 下面是研究报告的正文：
{report}
"""
            else:
                _user_prompt = f"""
# Here is the main text of the research report:
{report}
"""
            user_prompt.append(
                {
                    "type": "text",
                    "text": _user_prompt,
                }
            )
            
            if len(reference_error) > 0:
                for error in reference_error.values():
                    error_id, error_fact, error_content = error["id"], error["fact"], error["content"]
                    if self.use_zh:
                        _user_prompt = f"""
# 下面是研究报告的引用错误需要修复：
引用ID：{error_id}
报告对应内容：{error_fact}
引用原文内容：{error_content}
"""
                    else:
                        _user_prompt = f"""
# Below are citation errors in the research report that need to be fixed:
Citation ID: {error_id}
Corresponding content in the report: {error_fact}
Original citation content: {error_content}
"""
                    user_prompt.append(
                        {
                            "type": "text",
                            "text": _user_prompt,
                        }
                    )
            
            if len(report_justification) > 0:
                if self.use_zh:
                    if not self.use_detail:
                        score, justification = report_justification[0]['overall_score'], report_justification[0]['justification']
                        _user_prompt = f"""
# 下面是研究报告的评价：
评价分数：{score}
评价内容：{justification}
"""
                    else:
                        raise NotImplementedError
                else:
                    if not self.use_detail:
                        score, justification = report_justification[0]['overall_score'], report_justification[0]['justification']
                        _user_prompt = f"""
# Below are justification of the report:
Score: {score}
Justification: {justification}
"""
                    else:
                        raise NotImplementedError
                
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt,
                    }
                )

            if self.use_zh:
                _user_prompt = f"""
# 用户问题：
{query}
"""
            else:
                _user_prompt = f"""
# User question:
{query}
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
                _user_prompt = f"""
# 下面是研究报告的正文：
{report}
"""
            else:
                _user_prompt = f"""
# Here is the main text of the research report:
{report}
"""
            user_prompt.append({"text": _user_prompt})
        
            if len(reference_error) > 0:
                for error in reference_error.values():
                    error_id, error_fact, error_content = error["id"], error["fact"], error["content"]
                    if self.use_zh:
                        _user_prompt = f"""
# 下面是研究报告的引用错误需要修复：
引用ID：{error_id}
报告对应内容：{error_fact}
引用原文内容：{error_content}
"""
                    else:
                        _user_prompt = f"""
# Below are citation errors in the research report that need to be fixed:
Citation ID: {error_id}
Corresponding content in the report: {error_fact}
Original citation content: {error_content}
"""
                    user_prompt.append({"text": _user_prompt})
            
            if len(report_justification) > 0:
                if self.use_zh:
                    if not self.use_detail:
                        _user_prompt = f"""
# 下面是研究报告的评价：
评价内容：{report_justification}
"""
                    else:
                        raise NotImplementedError
                else:
                    if not self.use_detail:
                        _user_prompt = f"""
# Below are justification of the report:
Justification: {report_justification}
"""
                    else:
                        raise NotImplementedError
                
                user_prompt.append({"text": _user_prompt})
            
            if self.use_zh:
                _user_prompt = f"""
# 用户问题：
{query}
"""
            else:
                _user_prompt = f"""
# User question:
{query}
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
        
        return response
    
    
    async def act(
        self,
        input_dict: Dict,
        input_key: str = "rendered_report",
        output_key: str = "polished_report"
    ):
        query = input_dict.get("query", "") or input_dict.get("query_text", "")
        report = input_dict[f"{input_key}"]
        reference_error = dict()
        
        if self.need_fact:
            input_dict = await self.reference_judge.act(input_dict=input_dict)
            validate_map = input_dict["validate_map"]
            for res in validate_map.values():
                if res["valid"] == "unsupported":
                    reference_error[res["id"]] = res["id"]
                    reference_error[res["fact"]] = res["fact"]
                    reference_error[res["content"]] = res["content"]
                    reference_error[res["url"]] = res["url"]
        
        report_justification = []
        if self.need_race:
            if self.use_detail:
                input_dict = await self.criteria_generator.act(input_dict=input_dict)
            input_dict = await self.report_scorer.act(input_dict=input_dict)
            report_justification = input_dict["justification"]
        polished_report = await self.post_request(
            query=query,
            report=report, 
            reference_error=reference_error, 
            report_justification=report_justification
        )
        input_dict[f"{output_key}"] = polished_report
        return input_dict
        
        
        
        
        
        