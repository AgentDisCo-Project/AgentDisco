import gin
import jinja2
import re
import os
import time
import json
import logging
import asyncio
import sys
sys.path.append('.')

from typing import Optional, Union, Dict, List
from agent.BaseAgent import BasicAgent
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.key_operator import ApiKeyCycler
from api.utils.string_operator import markdown_fix
from collections import OrderedDict
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")


@gin.configurable()
class ReportWriter(BasicAgent):
    def __init__(
        self,
        name: Optional[str] = "report_writer",
        description_en: Optional[str] = "A hierarchical writer for report generation.",
        description_zh: Optional[str] = "分层报告生成器。",
        tool_bank: Optional[List[Union[str, Dict]]] = "",
        use_zh: bool = False,
        model_name: str = "",
        max_retries: int = 5,
        retry_delay: int = 3,
        max_concurrent: int = 50,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file: str = "ReportWriter_EN.jinja2",
        system_template_zh_file: str = "ReportWriter_ZH.jinja2",
        system_template_en_file_hierarchical: str = "HierarchicalWriter_EN.jinja2",
        system_template_zh_file_hierarchical: str = "HierarchicalWriter_ZH.jinja2",
        system_template_en_file_evidence: str = "ReportWriterEvidence_EN.jinja2",
        system_template_zh_file_evidence: str = "ReportWriterEvidence_ZH.jinja2",
        system_template_en_file_evidence_hierarchical: str = "HierarchicalWriterEvidence_EN.jinja2",
        system_template_zh_file_evidence_hierarchical: str = "HierarchicalWriterEvidence_ZH.jinja2",
        use_hierarchical_writer: bool = False,
        use_evidence_as_key: bool = False,
        use_mask_evidence: bool = False,
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
        self.use_hierarchical_writer = use_hierarchical_writer
        self.use_evidence_as_key = use_evidence_as_key
        
        if use_hierarchical_writer:
            if use_evidence_as_key:
                self.jinja_file = system_template_zh_file_evidence_hierarchical if self.use_zh else system_template_en_file_evidence_hierarchical
            else:
                self.jinja_file = system_template_zh_file_hierarchical if self.use_zh else system_template_en_file_hierarchical
        else:
            if use_evidence_as_key:
                self.jinja_file = system_template_zh_file_evidence if self.use_zh else system_template_en_file_evidence
            else:
                self.jinja_file = system_template_zh_file if self.use_zh else system_template_en_file
        
        self.max_concurrent = max_concurrent
        self.use_mask_evidence = use_mask_evidence
        self.mask_evidences = set()
        self.mask_documents = set()
    
    
    def get_system_prompt(
        self,
    ):
        template_vars = {}
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
        chunk_id: int = -1,
    ):
        report_writer_st = time.time()
        query_text = input_dict["query_text"]
        if self.use_hierarchical_writer:
            outline = input_dict[f"outline_chunk_{chunk_id}"]
        else:
            outline = input_dict["outline"]
        content, references = outline["content"], outline["references"]
        
        if "qwen3-next-80b-a3b-instruct" in self.model.model_name:
            system_prompt = self.get_system_prompt()
            user_prompt = ""
            
            if self.use_zh:
                _user_prompt = f"""
# 报告大纲
{content}
"""
            else:
                _user_prompt = f"""
# Report Outline
{content}
"""
            user_prompt += _user_prompt
  
            for reference in references:
                if len(reference) == 0:
                    continue
                idx, title, evidences = reference["id"], reference["title"], reference["evidence_map"]
                if self.use_mask_evidence and idx in self.mask_documents:
                    continue
                if self.use_zh:
                    _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
观点列表如下：
"""
                
                else:
                    _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Evidences are listed below:
"""
                
                user_prompt += _user_prompt
                
                for evidence_id, evidence in evidences.items():
                    if self.use_mask_evidence and evidence_id in self.mask_evidences:
                        continue
                    if self.use_zh:
                        _user_prompt = f"""
观点ID：{evidence_id}
观点内容：{evidence}
"""
                    else:
                        _user_prompt = f"""
Evidence ID: {evidence_id}
Evidence Content: {evidence}
"""
                    
                    user_prompt += _user_prompt
            
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
            
            user_prompt += _user_prompt
            
            if self.use_hierarchical_writer:
                previous_chunk = ""
                for _chunk_id in range(chunk_id):
                    previous_chunk += input_dict[f"writer_chunk_{_chunk_id}"]
                    previous_chunk += "\n\n"
                
                if self.use_zh:
                    _user_prompt = f"""
# 前章已经写的内容
{previous_chunk}
"""
                else:
                    _user_prompt = f"""
# Previous Written Content
{previous_chunk}
"""
                
                user_prompt += _user_prompt
            
            response = await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                return_cot=False,
            )
            
        elif "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            system_prompt = self.get_system_prompt()
            user_prompt = ""
            
            if self.use_zh:
                _user_prompt = f"""
# 报告大纲
{content}
"""
            else:
                _user_prompt = f"""
# Report Outline
{content}
"""
                user_prompt += _user_prompt
            
            for reference in references:
                if len(reference) == 0:
                    continue
                idx, title, evidences = reference["id"], reference["title"], reference["evidence_map"]
                if self.use_mask_evidence and idx in self.mask_documents:
                    continue
                if self.use_zh:
                    _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
观点列表如下：
"""
                
                else:
                    _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Evidences are listed below:
"""
                
                user_prompt += _user_prompt
                
                for evidence_id, evidence in evidences.items():
                    if self.use_mask_evidence and evidence_id in self.mask_evidences:
                        continue
                    if self.use_zh:
                        _user_prompt = f"""
观点ID：{evidence_id}
观点内容：{evidence}
"""
                    else:
                        _user_prompt = f"""
Evidence ID: {evidence_id}
Evidence Content: {evidence}
"""
                    
                    user_prompt += _user_prompt
            
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
            
            user_prompt += _user_prompt
            
            if self.use_hierarchical_writer:
                previous_chunk = ""
                for _chunk_id in range(chunk_id):
                    previous_chunk += input_dict[f"writer_chunk_{_chunk_id}"]
                    previous_chunk += "\n\n"
                
                if self.use_zh:
                    _user_prompt = f"""
# 前章已经写的内容
{previous_chunk}
"""
                else:
                    _user_prompt = f"""
# Previous Written Content
{previous_chunk}
"""
                
                user_prompt += _user_prompt
            
            response = await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                return_cot=False,
            )
        
        elif "gemini" in self.model.model_name:
            system_prompt = [{"text": self.get_system_prompt()}]
            user_prompt = []
            
            if self.use_zh:
                _user_prompt = f"""
# 报告大纲
{content}
"""
            else:
                _user_prompt = f"""
# Report Outline
{content}
"""
            user_prompt.append({"text": _user_prompt})
            
            for reference in references:
                if reference is None or len(reference) == 0:
                    continue
                idx, title, evidences = reference["id"], reference["title"], reference["evidence_map"]
                if self.use_mask_evidence and idx in self.mask_documents:
                    continue
                if self.use_zh:
                    _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
观点列表如下：
"""
                
                else:
                    _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Evidences are listed below:
"""
                
                user_prompt.append({"text": _user_prompt})
                
                for evidence_id, evidence in evidences.items():
                    if self.use_mask_evidence and evidence_id in self.mask_evidences:
                        continue
                    if self.use_zh:
                        _user_prompt = f"""
观点ID：{evidence_id}
观点内容：{evidence}
"""
                    else:
                        _user_prompt = f"""
Evidence ID: {evidence_id}
Evidence Content: {evidence}
"""
                    
                    user_prompt.append({"text": _user_prompt})
            
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
            
            if self.use_hierarchical_writer:
                previous_chunk = ""
                for _chunk_id in range(chunk_id):
                    previous_chunk += input_dict[f"writer_chunk_{_chunk_id}"]
                    previous_chunk += "\n\n"
                
                if self.use_zh:
                    _user_prompt = f"""
# 前章已经写的内容
{previous_chunk}
"""
                else:
                    _user_prompt = f"""
# Previous Written Content
{previous_chunk}
"""
                
                user_prompt.append({"text": _user_prompt})
            
            response = await self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                return_cot=False,
            )
        
        elif "gpt-oss" in self.model.model_name:
            system_prompt = self.get_system_prompt()
            user_prompt = ""
            
            if self.use_zh:
                _user_prompt = f"""
# 报告大纲
{content}
"""
            else:
                _user_prompt = f"""
# Report Outline
{content}
"""
                user_prompt += _user_prompt
            
            for reference in references:
                if len(reference) == 0:
                    continue
                idx, title, evidences = reference["id"], reference["title"], reference["evidence_map"]
                if self.use_mask_evidence and idx in self.mask_documents:
                    continue
                if self.use_zh:
                    _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
观点列表如下：
"""
                
                else:
                    _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Evidences are listed below:
"""
                
                user_prompt += _user_prompt
                
                for evidence_id, evidence in evidences.items():
                    if self.use_mask_evidence and evidence_id in self.mask_evidences:
                        continue
                    if self.use_zh:
                        _user_prompt = f"""
观点ID：{evidence_id}
观点内容：{evidence}
"""
                    else:
                        _user_prompt = f"""
Evidence ID: {evidence_id}
Evidence Content: {evidence}
"""
                    
                    user_prompt += _user_prompt
            
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
            
            user_prompt += _user_prompt
            
            if self.use_hierarchical_writer:
                previous_chunk = ""
                for _chunk_id in range(chunk_id):
                    previous_chunk += input_dict[f"writer_chunk_{_chunk_id}"]
                    previous_chunk += "\n\n"
                
                if self.use_zh:
                    _user_prompt = f"""
# 前章已经写的内容
{previous_chunk}
"""
                else:
                    _user_prompt = f"""
# Previous Written Content
{previous_chunk}
"""
                
                user_prompt += _user_prompt
            
            response = await self.model.chat_gpt_oss(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
            )
            
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        response = markdown_fix(response)
        if self.use_hierarchical_writer:
            input_dict[f"writer_chunk_{chunk_id}"] = response
            if self.use_mask_evidence:
                evidence_ids = self.parser_response_with_reference(response=response)
                for evidence_id in evidence_ids:
                    self.mask_evidences.add(evidence_id)
                for ref in outline["references"]:
                    if ref is None:
                        continue
                    evidences = ref["evidence_map"]
                    if set(evidences.keys()).issubset(self.mask_evidences):
                        self.mask_documents.add(ref["id"])
            # breakpoint()
        else:
            input_dict["report"] = response
        
        report_writer_et = time.time()
        if chunk_id >= 0:
            logging.info(f"report wirter {chunk_id} costs: {report_writer_et-report_writer_st}")
        else:
            logging.info(f"report wirter costs: {report_writer_et-report_writer_st}")

        return input_dict
    
    
    @staticmethod
    def parser_response_with_reference(
        response: str,
    ):
        if not response:
            return set()
        
        # 先提取cite标签内容
        cite_pattern = r'<cite>(.*?)</cite>'
        cite_contents = re.findall(cite_pattern, response)
        
        reference_ids = set()
        for content in cite_contents:
            # 使用字符模式匹配提取所有ID
            indices = re.findall(r'[a-zA-Z0-9_]+', content)
            for idx in indices:
                # 检查是否符合 turn_数字_数字_数字 的格式
                if re.match(r'turn_\d+_\d+_\d+', idx):
                    reference_ids.add(idx)
        
        return list(reference_ids)
    
    
    @staticmethod
    def merge_chunks_into_report(
        input_dict: Dict,
        num_chunks: int,
    ):
        report = ""
        for chunk_id in range(num_chunks):
            _report = input_dict[f"writer_chunk_{chunk_id}"]
            report += _report
            report += "\n\n"
        input_dict["report"] = report
        return input_dict
    


        
        
if __name__ == "__main__":
    async def main():
        service = ReportWriter(
            use_zh=True,
            model_name="qwen3-next-80b-a3b-instruct",
            use_hierarchical_writer=True,
            use_evidence_as_key=False,
            use_mask_evidence=True,
        )
        sample_data_path = "/mnt/tidalfs-bdsz01/usr/tusen/search-agent-dev/dev/0109/ragengine/data/sample_data.json"
        with open(sample_data_path, 'r', encoding='utf-8') as f:
            input_dict = json.load(f)
        breakpoint()
        # for chunk_id in range(input_dict['num_chunks']):
        #     input_dict = await service.act(
        #         input_dict=input_dict,
        #         chunk_id=chunk_id,
        #     )
        input_dict = await service.act(
            input_dict=input_dict,
            chunk_id=0,
        )
        breakpoint()
        print(input_dict)
    
    asyncio.run(main())
        