import gin
import asyncio
import jinja2
import json
import re
import sys
import asyncio
import os
sys.path.append('.')

from typing import Dict
from api.utils.key_operator import ApiKeyCycler
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.url_operator import compress_and_convert_base64, compress_url
from tool.DocParserService import WebParser
from dotenv import load_dotenv
from api.utils.string_operator import json_fix


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


@gin.configurable()
class ValidateReference:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        max_retries: int = 50,
        retry_delay: int = 3,
        use_zh: bool = False,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file_for_extractor: str = "ReferenceExtractor_EN.jinja2",
        system_template_zh_file_for_extractor: str = "ReferenceExtractor_ZH.jinja2",
        system_template_en_file_for_checker: str = "ReferenceChecker_EN.jinja2",
        system_template_zh_file_for_checker: str = "ReferenceChecker_ZH.jinja2",
    ):
        self.model_for_extractor = CustomizeChatGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
        )
        self.model_for_checker = CustomizeChatGenerator(
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
        self.jinja_file_for_extractor = system_template_zh_file_for_extractor if use_zh else system_template_en_file_for_extractor
        self.jinja_file_for_checker = system_template_zh_file_for_checker if use_zh else system_template_en_file_for_checker
        self.use_zh = use_zh
    
    
    def get_system_prompt_for_extractor(
        self
    ):
        template_vars = {}
        template = self.jinja_env.get_template(self.jinja_file_for_extractor)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    def parser_response_for_extractor(
        self,
        response: str
    ):
        response = json_fix(response)
        response = json.loads(response)
        return response
    
    
    def get_system_prompt_for_checker(
        self
    ):
        template_vars = {}
        template = self.jinja_env.get_template(self.jinja_file_for_checker)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    
    @staticmethod
    def parser_response_for_checker(
        response: str
    ):
        response = response.strip().strip('"\'').lower()
        assert response in ("supported", "unsupported", "unknown")
        return response
    
    
    
    async def extract_reference(
        self,
        report: str,
    ):
        if "qwen" in self.model_for_extractor.model_name or "deepseek" in self.model_for_extractor.model_name or "qwq" in self.model_for_extractor.model_name:
            system_prompt = [
                {
                    "type": "text",
                    "text": self.get_system_prompt_for_extractor()
                }
            ]
            
            user_prompt = []
            if self.use_zh:
                _user_prompt = f"""
下面是研究报告的正文：
{report}
"""
            else:
                _user_prompt = f"""
Here is the main text of the research report:
{report}
"""
            user_prompt.append(
                {
                    "type": "text",
                    "text": _user_prompt,
                }
            )
            
            response = await self.model_for_extractor.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.parser_response_for_extractor,
                return_cot=False,
            )

        elif "gemini" in self.model_for_extractor.model_name:
            system_prompt = [
                {
                    "text": self.get_system_prompt_for_extractor()
                }
            ]
            user_prompt = []
            if self.use_zh:
                _user_prompt = f"""
下面是研究报告的正文：
{report}
"""
            else:
                _user_prompt = f"""
Here is the main text of the research report:
{report}
"""
            user_prompt.append(
                {
                    "text": _user_prompt,
                }
            )
            response = await self.model_for_extractor.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.parser_response_for_extractor,
                return_cot=False,
            )
        else:
            raise ValueError(f"Unsupported {self.model_for_extractor.model_name}")
        
        return response
    
    
    
    async def check_reference(
        self,
        reference: str,
        statement: str,
    ):
        if "qwen" in self.model_for_checker.model_name or "deepseek" in self.model_for_checker.model_name or "qwq" in self.model_for_checker.model_name:
            system_prompt = [
                {
                    "type": "text",
                    "text": self.get_system_prompt_for_checker()
                }
            ]
            
            user_prompt = []
            
            if self.use_zh:
                _user_prompt = f"""
下面是参考资料和statements：
<reference>
{reference}
</reference>

<statement>
{statement}
</statement>
"""
            else:
                _user_prompt = f"""
Below is the reference and statements:
<reference>
{reference}
</reference>

<statement>
{statement}
</statement>
"""
            
            user_prompt.append(
                {
                    "type": "text",
                    "text": _user_prompt,
                }
            )
            response = await self.model_for_checker.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.parser_response_for_checker,
                return_cot=False,
            )
        
        elif "gemini" in self.model_for_extractor.model_name:
            system_prompt = [
                {
                    "text": self.get_system_prompt_for_checker()
                }
            ]
            
            user_prompt = []
            
            if self.use_zh:
                _user_prompt = f"""
下面是参考资料和statements：
<reference>
{reference}
</reference>

<statement>
{statement}
</statement>
"""
            else:
                _user_prompt = f"""
Below is the reference and statements:
<reference>
{reference}
</reference>

<statement>
{statement}
</statement>
"""
            
            user_prompt.append(
                {
                    "text": _user_prompt,
                }
            )
            
            response = await self.model_for_extractor.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.parser_response_for_checker,
                return_cot=False,
            )

        else:
            raise ValueError(f"Unsupported {self.model_for_checker.model_name}")
        
        return response
    
    

    async def act(
            self,
            input_dict: Dict,
            input_key: str = "report",
            report_key: str = "report"
        ):
        report = input_dict[f"{input_key}"]
        report = await self.extract_reference(report=report)
        
        webparser = WebParser()
        report_url_set = set()
        unique_reports = []
        
        # 去重URL，准备并发任务
        for _report in report:
            report_id = _report["ref_idx"]
            report_fact = _report["fact"]
            report_url = _report["url"]
            if report_url not in report_url_set:
                report_url_set.add(report_url)
                unique_reports.append({
                    "id": report_id,
                    "fact": report_fact,
                    "url": report_url
                })
        
        # 并发获取网页内容
        async def fetch_content(report_info):
            params = {"url": report_info["url"]}
            content = await webparser.call(params=params)
            return {
                "id": report_info["id"],
                "fact": report_info["fact"],
                "content": content,
                "url": report_info["url"]
            }
        
        # 并发执行所有webparser调用
        content_tasks = [fetch_content(report_info) for report_info in unique_reports]
        report_contents = await asyncio.gather(*content_tasks)
    
        # 构建report_map
        report_map = {content["id"]: content for content in report_contents}
        
        # 并发执行所有验证检查
        async def validate_report(report):
            response = await self.check_reference(
                statement=report["fact"], 
                reference=report["content"]
            )
            return {
                "id": report["id"],
                "fact": report["fact"],
                "content": report["content"],
                "url": report["url"],
                "valid": response
            }, response
        
        # 并发执行所有验证任务
        validate_tasks = [validate_report(report) for report in report_map.values()]
        validate_results_with_response = await asyncio.gather(*validate_tasks)
        
        # 分离结果
        validate_map = {}
        validate_responses = []
        for validate_result, response in validate_results_with_response:
            validate_map[validate_result["id"]] = validate_result
            validate_responses.append(response)
        
        # 计算统计信息
        num_citations, num_valid_citations = 0, 0
        for res in validate_responses:
            if res != "unknown":
                num_citations += 1
                if res == "supported":
                    num_valid_citations += 1
        
        input_dict[f"{report_key}_validate_map"] = validate_map
        input_dict[f"{report_key}_num_all_citations"] = len(validate_responses)
        input_dict[f"{report_key}_num_citations"] = num_citations
        input_dict[f"{report_key}_num_valid_citations"] = num_valid_citations
        input_dict[f"{report_key}_valid_rate"] = num_valid_citations / num_citations if num_citations > 0 else 0
        return input_dict




if __name__ == "__main__":
    async def main():
        # file_path = "/mnt/tidalfs-bdsz01/usr/tusen/search-agent-dev/dev/1024/ragengine/task_3_065a86a8_output.md"
        # file_path = "/mnt/tidalfs-bdsz01/usr/tusen/search-agent-dev/data/data1010/测试样本_1.md"
        file_path = "/mnt/tidalfs-bdsz01/usr/tusen/search-agent-dev/data/data1010/测试样本_2.md"
        print(file_path)
        with open(file_path, "r", encoding='utf-8') as file:
            report = file.read()
        
        service = ValidateReference(
            model_name="gemini-2.5-pro",
            max_retries=5,
            retry_delay=3,
            use_zh=True,
        )
        input_dict = {
            "report": report,
        }
        input_dict = await service.act(input_dict=input_dict)
        print(input_dict)
    
    asyncio.run(main())