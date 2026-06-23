import gin
import jinja2
import re
import os
import json
import time
import asyncio
import logging
import sys
sys.path.append('.')

from functools import partial
from datetime import datetime
from typing import Optional, Union, Dict, List
from agent.BaseAgent import BasicAgent
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.DisentangledOutlineJudgeService import DisentangledOutlineJudge
from api.utils.key_operator import ApiKeyCycler
from api.utils.string_operator import json_fix
from tool.WebSearchService import WebSearch
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


@gin.configurable()
class DisentangledOutlineGenerator:
    def __init__(
        self,
        use_zh: bool = False,
        model_name: str = "",
        max_retries: int = 5,
        retry_delay: int = 3,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file: str = "DisentangledOutlineGeneratorQA_EN.jinja2",
        system_template_zh_file: str = "DisentangledOutlineGeneratorQA_ZH.jinja2",
        system_template_en_file_evidence: str = "DisentangledOutlineGeneratorEvidenceQA_EN.jinja2",
        system_template_zh_file_evidence: str = "DisentangledOutlineGeneratorEvidenceQA_ZH.jinja2",
        include_prev_outline: bool = True,
        include_prev_judge: bool = True,
        use_summary: bool = True,
        use_evidence_as_key: bool = False,
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
        if not use_evidence_as_key:
            self.jinja_file = system_template_zh_file if self.use_zh else system_template_en_file
        else:
            self.jinja_file = system_template_zh_file_evidence if self.use_zh else system_template_en_file_evidence
        self.include_prev_outline = include_prev_outline
        self.include_prev_judge = include_prev_judge
        self.use_evidence_as_key = use_evidence_as_key
        self.use_summary = use_summary
    
    
    def get_system_prompt(
        self,
        include_prev_outline: bool = False,
        include_prev_judge: bool = False,
    ):
        template_vars = {
            "include_prev_outline": include_prev_outline,
            "include_prev_judge": include_prev_judge,
            "curr_date": datetime.now().strftime("%Y年%m月%d日"),
        }
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    def check_func(
        self,
        response: str,
    ):
        return response
    
    
    @staticmethod
    def divide_outline_into_chunks(outline: str):
        def split_by_header_level(content: str, header_pattern: str):
            """按照指定的标题级别分割内容"""
            if not content.strip():
                return []
            
            lines = content.split('\n')
            chunks = []
            curr_chunk = {'content': []}  # 初始化curr_chunk
            
            for line in lines:
                # 检查是否是指定级别的标题
                if re.match(header_pattern, line.strip()):
                    # 如果当前chunk有内容，先保存
                    if curr_chunk['content']:
                        curr_chunk['content'] = '\n'.join(curr_chunk['content'])
                        chunks.append(curr_chunk)
                    
                    # 开始新的chunk
                    curr_chunk = {
                        'content': [line]  # 将标题行加入content
                    }
                else:
                    # 将内容添加到当前chunk
                    curr_chunk['content'].append(line)
            
            # 添加最后一个chunk
            if curr_chunk['content']:
                curr_chunk['content'] = '\n'.join(curr_chunk['content'])
                chunks.append(curr_chunk)
            
            return chunks
        
        def contains_level2_title(content: str):
            """检查chunk是否有二级标题（## ）"""
            lines = content.split('\n')
            for line in lines:
                if re.match(r'^##\s+', line.strip()):
                    return True
            return False
        
        def is_only_title(content: str):
            """检查chunk是否只包含标题"""
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if len(lines) == 1:
                # 如果只有一行，检查是否是标题格式（以#开头）
                return re.match(r'^#+\s+', lines[0])
            elif len(lines) == 0:
                return True  # 空内容也算只有标题
            return False
        
        if not outline or not outline.strip():
            return []
        
        # 首先按一级标题分割
        chunks = split_by_header_level(outline, r'^#\s+')
        
        # 如果只有一个chunk，按二级标题重新分割
        if len(chunks) <= 1:
            chunks = split_by_header_level(outline, r'^##\s+')
            
            # 如果还是只有一个chunk，按三级标题分割
            if len(chunks) <= 1:
                chunks = split_by_header_level(outline, r'^###\s+')
            
            if len(chunks) >= 2 and not contains_level2_title(chunks[0]['content']):
                # 将第一个chunk与第二个chunk合并
                merged_content = chunks[0]['content'] + '\n' + chunks[1]['content']
                merged_chunk = {
                    'content': merged_content
                }
                # 更新chunks列表：第一个chunk变成合并后的chunk，剩余的chunk从第三个开始
                chunks = [merged_chunk] + chunks[2:]
        
        # 检查第一个chunk是否只包含标题，如果是，与下一个chunk合并
        if len(chunks) >= 2 and is_only_title(chunks[0]['content']):
            merged_content = chunks[0]['content'] + '\n' + chunks[1]['content']
            merged_chunk = {
                'content': merged_content
            }
            chunks = [merged_chunk] + chunks[2:]
        
        for idx, chunk in enumerate(chunks):
            chunk["id"] = idx
        return chunks


    
    
    async def act(
        self,
        input_dict: Dict,
        turn_id: int,
    ):
        outline_generator_st = time.time()
        query_text = input_dict["query_text"]
        include_prev_outline = (turn_id > 0) and self.include_prev_outline
        if "qwen3-next-80b-a3b-instruct" in self.model.model_name:
            system_prompt = self.get_system_prompt(include_prev_outline=include_prev_outline)
            user_prompt = ""

            if include_prev_outline:
                prev_outline = input_dict[f"outline_turn_{turn_id-1}"]
                if self.use_zh:
                    _user_prompt = f"""
# 上一轮迭代的报告大纲
{prev_outline}
"""
                else:
                    _user_prompt = f"""
# Previous Outline
{prev_outline}
"""
                user_prompt += _user_prompt
        
            prev_judge = input_dict[f"judge_turn_{turn_id}"]["justification"]
            if self.use_zh:
                _user_prompt = f"""
# 上一轮大纲的评价
{prev_judge}
"""
            else:
                _user_prompt = f"""
# Judge for Previous Outline
{prev_judge}
"""
            
            user_prompt += _user_prompt
            
            if not self.use_evidence_as_key:
                if self.use_zh:
                    _user_prompt = f"""
# 外源搜索结果
"""
                else:
                    _user_prompt = f"""
# External Search Results
"""
                user_prompt += _user_prompt
                
                docs = input_dict[f"search_result_turn_{turn_id}"]
                for doc in docs:
                    idx, title, content, search_from = doc["id"], doc["title"], doc["content"], doc["search_from"]
                    use_summary = ("google" in search_from or self.use_summary) and ("summary" in doc)
                    
                    if use_summary:
                        content = doc["summary"]
                        if self.use_zh:
                            _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
摘要：{content}
"""
                        else:
                            _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Summary: {content}
"""
                    else:
                        if self.use_zh:
                            _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
内容：{content}
"""
                        else:
                            _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Content: {content}
"""
                    
                    user_prompt += _user_prompt
            
            else:
                if self.use_zh:
                    _user_prompt = f"""
# 外源搜索观点
"""
                else:
                    _user_prompt = f"""
# External Search Results
"""
                
                user_prompt += _user_prompt,
                
                evidences = input_dict[f"search_evidence_turn_{turn_id}"]
                for evidence in evidences:
                    idx, content, references = evidence["id"], evidence["content"], evidence["references"]
                    if self.use_zh:
                        _user_prompt = f"""
## 外源搜索观点{idx}的内容如下
观点ID：{idx}
内容：{content}
支持观点数量：{len(references)}
"""
                    else:
                        _user_prompt = f"""
## External Search Document CLaim {idx}
Evidence ID: {idx}
Content: {content}
Number of supported references: {len(references)}
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
            
            cycler = ApiKeyCycler(api_key_list=list(DIRECTLLM_API_KEY_USER.values()))
            response = await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                cycler=cycler,
                return_cot=False,
            )
            
        elif "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            system_prompt = self.get_system_prompt(include_prev_outline=include_prev_outline)
            user_prompt = ""
            
            if include_prev_outline:
                prev_outline = input_dict[f"outline_turn_{turn_id-1}"]
                if self.use_zh:
                    _user_prompt = f"""
# 上一轮迭代的报告大纲
{prev_outline}
"""
                else:
                    _user_prompt = f"""
# Previous Outline
{prev_outline}
"""
                user_prompt += _user_prompt
            
            prev_judge = input_dict[f"judge_turn_{turn_id}"]["justification"]
            if self.use_zh:
                _user_prompt = f"""
# 上一轮大纲的评价
{prev_judge}
"""
            else:
                _user_prompt = f"""
# Judge for Previous Outline
{prev_judge}
"""
            
            user_prompt += _user_prompt
            
            if not self.use_evidence_as_key:
                if self.use_zh:
                    _user_prompt = f"""
# 外源搜索结果
"""
                else:
                    _user_prompt = f"""
# External Search Results
"""
                user_prompt += _user_prompt
                
                docs = input_dict[f"search_result_turn_{turn_id}"]
                for doc in docs:
                    idx, title, content, search_from = doc["id"], doc["title"], doc["content"], doc["search_from"]
                    use_summary = ("google" in search_from or self.use_summary) and ("summary" in doc)
                    
                    if use_summary:
                        content = doc["summary"]
                        if self.use_zh:
                            _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
摘要：{content}
"""
                        else:
                            _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Summary: {content}
"""
                    else:
                        if self.use_zh:
                            _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
内容：{content}
"""
                        else:
                            _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Content: {content}
"""
                    
                    user_prompt += _user_prompt
            
            else:
                if self.use_zh:
                    _user_prompt = f"""
# 外源搜索观点
"""
                else:
                    _user_prompt = f"""
# External Search Results
"""
                
                user_prompt += _user_prompt
                
                evidences = input_dict[f"search_evidence_turn_{turn_id}"]
                for evidence in evidences:
                    idx, content, references = evidence["id"], evidence["content"], evidence["references"]
                    if self.use_zh:
                        _user_prompt = f"""
## 外源搜索观点{idx}的内容如下
观点ID：{idx}
内容：{content}
支持观点数量：{len(references)}
"""
                    else:
                        _user_prompt = f"""
## External Search Document CLaim {idx}
Evidence ID: {idx}
Content: {content}
Number of supported references: {len(references)}
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
            
            response = await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                return_cot=False,
            )
        
        
        elif "gemini" in self.model.model_name:
            system_prompt = [
                {
                    "text": self.get_system_prompt(
                        include_prev_outline=include_prev_outline,
                    )
                }
            ]
            user_prompt = []
            
            if include_prev_outline:
                prev_outline = input_dict[f"outline_turn_{turn_id-1}"]
                if self.use_zh:
                    _user_prompt = f"""
# 上一轮迭代的报告大纲
{prev_outline}
"""
                else:
                    _user_prompt = f"""
# Previous Outline
{prev_outline}
"""
                user_prompt.append(
                    {
                        "text": _user_prompt
                    }
                )
            
            prev_judge = input_dict[f"judge_turn_{turn_id}"]["justification"]
            if self.use_zh:
                _user_prompt = f"""
# 上一轮大纲的评价
{prev_judge}
"""
            else:
                _user_prompt = f"""
# Judge for Previous Outline
{prev_judge}
"""
            
            user_prompt.append({"text": _user_prompt})
                
            if not self.use_evidence_as_key:
                if self.use_zh:
                    _user_prompt = f"""
# 外源搜索结果
"""
                else:
                    _user_prompt = f"""
# External Search Results
"""
                    
                user_prompt.append({"text": _user_prompt,})
                    
                docs = input_dict[f"search_result_turn_{turn_id}"]
                for doc in docs:
                    idx, title, content, search_from = doc["id"], doc["title"], doc["content"], doc["search_from"]
                    use_summary = ("google" in search_from or self.use_summary) and ("summary" in doc)
                    
                    if use_summary:
                        content = doc["summary"]
                        if self.use_zh:
                            _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
摘要：{content}
"""
                        else:
                            _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Summary: {content}
"""
                    else:
                        if self.use_zh:
                            _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
内容：{content}
"""
                        else:
                            _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Content: {content}
"""
                        
                    user_prompt.append(
                        {
                            "text": _user_prompt,
                        }
                    )
                
            else:
                if self.use_zh:
                    _user_prompt = f"""
# 外源搜索观点
"""
                else:
                    _user_prompt = f"""
# External Search Results
"""
                    
                user_prompt.append({"text": _user_prompt})
                    
                evidences = input_dict[f"search_evidence_turn_{turn_id}"]
                for evidence in evidences:
                    idx, content, references = evidence["id"], evidence["content"], evidence["references"]
                    if self.use_zh:
                        _user_prompt = f"""
## 外源搜索观点{idx}的内容如下
观点ID：{idx}
内容：{content}
支持观点数量：{len(references)}
"""
                    else:
                        _user_prompt = f"""
## External Search Document Evidence {idx}
Evidence ID: {idx}
Content: {content}
Number of supported references: {len(references)}
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
            
            response = await self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                return_cot=False,
            )
        
        elif "gpt-oss" in self.model.model_name:
            system_prompt = self.get_system_prompt(include_prev_outline=include_prev_outline)
            user_prompt = ""
            
            if include_prev_outline:
                prev_outline = input_dict[f"outline_turn_{turn_id-1}"]
                if self.use_zh:
                    _user_prompt = f"""
# 上一轮迭代的报告大纲
{prev_outline}
"""
                else:
                    _user_prompt = f"""
# Previous Outline
{prev_outline}
"""
                user_prompt += _user_prompt
            
            prev_judge = input_dict[f"judge_turn_{turn_id}"]["justification"]
            if self.use_zh:
                _user_prompt = f"""
# 上一轮大纲的评价
{prev_judge}
"""
            else:
                _user_prompt = f"""
# Judge for Previous Outline
{prev_judge}
"""
            
            user_prompt += _user_prompt
            
            if not self.use_evidence_as_key:
                if self.use_zh:
                    _user_prompt = f"""
# 外源搜索结果
"""
                else:
                    _user_prompt = f"""
# External Search Results
"""
                user_prompt += _user_prompt
                
                docs = input_dict[f"search_result_turn_{turn_id}"]
                for doc in docs:
                    idx, title, content, search_from = doc["id"], doc["title"], doc["content"], doc["search_from"]
                    use_summary = ("google" in search_from or self.use_summary) and ("summary" in doc)
                    
                    if use_summary:
                        content = doc["summary"]
                        if isinstance(content, List):
                            content = "\n\n".join(content)
                        if self.use_zh:
                            _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
摘要：{content}
"""
                        else:
                            _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Summary: {content}
"""
                    else:
                        if self.use_zh:
                            _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
内容：{content}
"""
                        else:
                            _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Content: {content}
"""
                    
                    user_prompt += _user_prompt
            
            else:
                if self.use_zh:
                    _user_prompt = f"""
# 外源搜索观点
"""
                else:
                    _user_prompt = f"""
# External Search Results
"""
                
                user_prompt += _user_prompt
                
                evidences = input_dict[f"search_evidence_turn_{turn_id}"]
                for evidence in evidences:
                    idx, content, references = evidence["id"], evidence["content"], evidence["references"]
                    if self.use_zh:
                        _user_prompt = f"""
## 外源搜索观点{idx}的内容如下
观点ID：{idx}
内容：{content}
支持观点数量：{len(references)}
"""
                    else:
                        _user_prompt = f"""
## External Search Document CLaim {idx}
Evidence ID: {idx}
Content: {content}
Number of supported references: {len(references)}
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
            
            response = await self.model.chat_gpt_oss(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
            )
        
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        outline_generator_et = time.time()
        logging.info(f"outline generator costs: {outline_generator_et-outline_generator_st}")
        input_dict[f"outline_turn_{turn_id}"] = response
        return input_dict



if __name__ == "__main__":
    async def main():
        service = DisentangledOutlineGenerator(
            use_zh=True,
            model_name="gemini-3-pro",
            max_retries=5,
            retry_delay=3,
            include_prev_outline=True,
            include_prev_judge=True,
            use_evidence_as_key=False,
        )
        input_dict = dict()
        input_dict["query_text"] = input_dict["query"] = "湿气重怎么调理？"
        input_dict["judge_turn_0"] = {'rating': 0.0, 'justification': '评分理由：用户完全没有提供任何形式的大纲，仅提出了一个问题。本次评估的核心任务是评价一份“报告大纲”的质量，而用户提交的内容为空，这直接违反了最基本的指令。由于缺乏任何可供评估的结构、内容或思路，大纲在指令遵循、深度、平衡性、广度、支撑性和洞察力等所有判别准则上的得分均为零。这并非一个不完善的大纲，而是根本不存在大纲，因此只能给予最低分。', 'search_query': ['湿气重 表现 自测', '健脾祛湿 食疗', '祛湿茶 配方', '红豆薏米水 正确做法', '艾灸 祛湿 穴位', '懒人 祛湿 方法', '夏天 湿气重 调理', '湿热体质 怎么调理', '脾虚 湿气重 食谱', '运动 排湿气']}
        input_dict["search_query_turn_0"] = ['湿气重 表现 自测', '健脾祛湿 食疗', '祛湿茶 配方', '红豆薏米水 正确做法', '艾灸 祛湿 穴位', '懒人 祛湿 方法', '夏天 湿气重 调理', '湿热体质 怎么调理', '脾虚 湿气重 食谱', '运动 排湿气']
        input_dict["search_result_turn_0"] = [{'id': 'turn_0_0', 'search_from': 'search_note', 'content': '你是不是一个“湿气女”呢❓\n“整个人油腻腻的，头发🦱不到一天就油叽叽，脸上也是油光🈵面”；\n“上课或者上班打不起精神，整天都很疲劳🤕”；\n“每天都有刷牙😬，可是嘴里还是有异味儿”；\n......\n✨——————————✨\nㅤ\n这是如今很多女生都存在的一些问题，其实很🈶可能就是一个问题：湿气重❗在生活中，常常以“十人九湿”来形容湿气侵袭人体的普遍性，这样一看，你是不是心里舒服点了🎈。\n其湿气重跟日常饮食🍲、生活环境、生活刁惯、体质等因素🈶关，具体可以参照图2，看看自己有没有对上的。\n✨——————————✨\nㅤ\n“无湿一身轻”，赶紧试试下面这些小妙招，减➖少体内的湿气：\n、改变不好的饮食刁惯🙊；\n、坚持泡脚🦶；\n、规律作息，不熬夜😴；\n、多通风；\n#健康养生[话题]# #湿气[话题]# #湿气重[话题]# #祛湿[话题]# #去湿气[话题]# #养生[话题]# #健脾祛湿[话题]#', 'title': '女生湿气重有这些表现❗自测+改善妙招✔', 'url': 'https://www.xiaohongshu.com/explore/6685013e000000000d00d5c0', 'date': '2024-07-03 15:43:58', 'note_type': 'images', 'video': {'noteId': '6685013e000000000d00d5c0', 'url': ''}, 'images': [], 'like_count': '7273', 'collect_count': 3862, 'view_count': '554302', 'comments': [], 'confidence': -1}, {'id': 'turn_0_1', 'search_from': 'search_note', 'content': '💃女生必看！轻松赶走湿气，焕活美丽新姿态💃\n🌸亲爱的姐妹们，你是否常常感到身体沉重、困倦乏力？是否总是被痘痘、湿疹等皮肤问题困扰？如果是，那很可能是湿气在作祟！今天就来和大家分享女生去湿气的mi.诀，让你重拾活力，美丽绽放。\n🌟湿气是什么？\n湿气是一种🀄医概念，当体内湿气过重时，会影响我们的身体健康和外在美丽。湿气可能来自于不良的饮食习惯、潮湿的环境、缺乏运动等。\n💡女生湿气重的9个表现，你中招了几个？赶紧自测一下吧！\n💕姐妹们，去湿气是一个长期的过程，需要我们坚持做好饮食、运动、环境等方面的调理。只要我们养成良好的生活习惯，就一定能够赶走湿气，拥有健康美丽的身体，一起加油吧！\n#女生去湿气[话题]# #祛湿方法[话题]# #健康美丽[话题]# #养生秘籍[话题]#\n#中医养生[话题]# #养生日常[话题]# #我的健康生活[话题]# #女性健康[话题]# #女生必看[话题]# #喝出好气色[话题]#', 'title': '湿女必看｜湿气重的9个表现', 'url': 'https://www.xiaohongshu.com/explore/6710962e000000001402c9a3', 'date': '2024-10-17 12:44:30', 'note_type': 'images', 'video': {'noteId': '6710962e000000001402c9a3', 'url': ''}, 'images': [], 'like_count': '101', 'collect_count': 57, 'view_count': '6524', 'comments': [], 'confidence': -1}, {'id': 'turn_0_2', 'search_from': 'search_note', 'content': '女生湿气重这些表现!自测+改善方法湿气#湿气排出来的表现[话题]#\n整天油油腻腻的，头昏昏沉沉的，脸油得快、头油又臭……\n看似很多问题，其实都是一个问题\t:湿气重\n常见的加重湿气因素，有很多朋友身上都会有湿气。\n【减少湿气】这样做\n1.休息规律少熬夜;\n2.空调不要对着吹\n3.清淡饮食少吃冷食;\n4.穿衣要保暖，夏天也要护好肚子\n5.洗澡或淋雨后及时擦干\n6.定期泡脚非常好', 'title': '女生湿气重这些表现!自测+改善方法湿气', 'url': 'https://www.xiaohongshu.com/explore/6607725400000000140072af', 'date': '2024-03-30 10:00:52', 'note_type': 'images', 'video': {'noteId': '6607725400000000140072af', 'url': ''}, 'images': [], 'like_count': '864', 'collect_count': 331, 'view_count': '80204', 'comments': [], 'confidence': -1}, {'id': 'turn_0_4', 'search_from': 'search_note', 'content': '你是不是一个“湿气女”呢❓\n“整个人油腻腻的，头发🦱不到一天就油叽叽，脸上也是油光🈵面”；\n“上课或者上班打不起精神，整天都很疲劳🤕”；\n“每天都有刷牙😬，可是嘴里还是有异味儿”；\n......\n✨——————————✨\nㅤ\n这是如今很多女生都存在的一些问题，其实很🈶可能就是一个问题：湿气重❗在生活中，常常以“十人九湿”来形容湿气侵袭人体的普遍性，这样一看，你是不是心里舒服点了🎈。\n其湿气重跟日常饮食🍲、生活环境、生活习惯、体质等因素🈶关，具体可以参照图2，看看自己有没有对上的。\n✨——————————✨\nㅤ\n“无湿一身轻”，赶紧试试下面这些小妙招，减➖少体内的湿气：\n[一R]、改变不好的饮食习惯🙊；\n[二R]、坚持泡脚🦶；\n[三R]、规律作息，不熬夜😴；\n[四R]、勤晒被子多通风[太阳R]；\n[五R]、可以喝祛湿茶🍵。\n✨——————————✨\nㅤ\nPs：内容仅供参考，实际情况根据个人体质和健康而定~\n#健康养生[话题]# #湿气[话题]# #湿气重[话题]# #祛湿气[话题]# #祛湿[话题]# #健脾祛湿[话题]# #去湿气[话题]# #我的日常[话题]# #养生[话题]#', 'title': '女生湿气重有这些表现❗❗自测+改善妙招✔', 'url': 'https://www.xiaohongshu.com/explore/649ac8890000000007039527', 'date': '2023-06-27 19:31:21', 'note_type': 'images', 'video': {'noteId': '649ac8890000000007039527', 'url': ''}, 'images': [], 'like_count': '19977', 'collect_count': 9867, 'view_count': '1713452', 'comments': [], 'confidence': -1}, {'id': 'turn_0_6', 'search_from': 'search_note', 'content': '首先，了解“湿气”是什么？\n在中医理论中，“湿气”是一种致病因素，分为“内湿”和“外湿”。\n外湿：主要由环境引起，如长期居住在潮湿地区、淋雨、经常接触水等。\n内湿：主要由于脾胃运化功能失调所致。脾主运化，喜燥恶湿。如果饮食不节（如过食生冷、油腻、甜腻食物），或劳累过度，就会损伤脾胃，导致水湿在体内停滞，形成内湿。\n常见的湿气重表现有：\n身体沉重、乏力、容易疲倦\n头发、面部爱出油\n舌苔厚腻、舌头边缘有齿痕\n大便粘稠、不成形，容易粘马桶\n食欲不振，肚子胀胀的\n水肿、虚胖\n皮肤起疹、瘙痒\n如何简单自测？\n一个非常直观的方法是：观察你的舌头和大便。\n看舌头：对照镜子，如果发现舌头胖大，边缘有牙齿挤压的痕迹，舌苔又厚又腻，覆盖了整个舌面，大概率是湿气重。\n看大便：如果大便总是稀软不成形，粘在马桶上不易冲走，也是湿气重的典型表现。\n饮食调整：\n多吃：健脾利湿的食物，如薏米、赤小豆、山药、茯苓、冬瓜、南瓜、小米等。\n少吃/不吃：生冷寒凉（冰淇淋、冷饮）、油腻甜腻（蛋糕、油炸食品）、辛辣刺激的食物，这些都会加重湿气。\n生活方式改变：\n避免久居潮湿环境，保持居住环境干燥通风。\n不要穿未干透的衣物。\n适当运动出汗（如快走、瑜伽、跑步），是排出湿气的好方法。\n保证充足睡眠，不熬夜。\n#中医养生[话题]# #身体去湿气[话题]# #湿寒与湿热的区分[话题]# #体质[话题]# #女人养生养脾胃[话题]# #寒湿是万病之源[话题]# #健康养生[话题]#  #先健脾还是先祛湿[话题]#  #排湿气寒气[话题]# #在小红书轻养生[话题]#', 'title': '女生湿气重的10大表现，快来自查！', 'url': 'https://www.xiaohongshu.com/explore/68cbbd620000000012017f0b', 'date': '2025-09-18 16:05:54', 'note_type': 'images', 'video': {'noteId': '68cbbd620000000012017f0b', 'url': ''}, 'images': [], 'like_count': '30', 'collect_count': 16, 'view_count': '176', 'comments': [], 'confidence': -1}, {'id': 'turn_0_7', 'search_from': 'search_note', 'content': '前阵子每天睡醒还是昏沉沉的😪，上午像没睡醒一样，拼命补气血反而更严重！直到发现💩黏马桶，又发现舌头边缘一圈齿痕…才惊觉是湿气在作祟！💢现在狂吃祛湿食物，身体轻快多啦～\n\t\n🌟脾虚才是湿气的元凶！\n脾主运化却怕湿，湿气堆在体内→脾越虚→湿越排不出→恶性循环！这6个信号你中几个？👇\n\t\n1️⃣ \u200b\u200b大便黏马桶\u200b\u200b\n2️⃣ \u200b\u200b容易胀气\u200b\u200b\n3️⃣ \u200b\u200b舌苔厚腻+齿痕\u200b\u200b\n4️⃣ \u200b\u200b头昏脑涨，容易犯困\n5️⃣ \u200b\u200b浮肿水肿\u200b\u200b\n6️⃣ 皮肤油腻长痘\n\t\n✅祛湿饮食红榜\n🌾主食： 燕麦、小米、薏米（炒过更温和）、赤小豆、山药、芡实、南瓜\n🥬蔬菜： 冬瓜、白萝卜、芹菜、南瓜、马齿苋\n🍖肉类： 鲫鱼、白切鸡、卤牛肉 烹饪切记少油少盐！\n\t\n【❌忌口】\n生冷寒凉  |高糖食物| 油腻煎炸|  奶制品 | 辛辣刺激 | 酒精\n\t\n#中医养生[话题]# #食疗养生[话题]# #身体去湿气[话题]# #养好脾胃不生病[话题]#  #湿气重[话题]#  #脾虚湿气重[话题]# #大便黏马桶[话题]# #齿痕舌[话题]# #总犯困[话题]# #祛湿[话题]#', 'title': '湿气重的6大表现！黏马桶，犯困乏力……', 'url': 'https://www.xiaohongshu.com/explore/68dc4b1a00000000040122ed', 'date': '2025-10-01 05:26:50', 'note_type': 'images', 'video': {'noteId': '68dc4b1a00000000040122ed', 'url': ''}, 'images': [], 'like_count': '1096', 'collect_count': 673, 'view_count': '96775', 'comments': [], 'confidence': -1}, {'id': 'turn_0_8', 'search_from': 'search_note', 'content': '#大大方方做自己[话题]#\n\t\n🥵湿热\n💢脸油头发油，脸上爱长痘\n💢口干、口苦、口气重\n💢脾气躁，容易上火\n💢尿液偏黄\n💢女生白带多且颜色黄\n💢糖稀或便㊙️\n💢怕热，手脚爱出\n-\n😰寒湿\n💢下肢易浮肿\n💢头发，皮肤爱出油，眼睛\n💢口黏，发甜，痰多\n💢易感到乏力没精神\n💢小bian清长，粘马桶\n💢女生白带多且清稀\n💢手脚冰凉，比一般人要怕冷\n\t\n✅湿热调理方式（图4）\n✅寒湿调理方式（图5）\n\t\n#健康养生[话题]# #祛湿气[话题]# #湿热体质[话题]# #湿气重[话题]# #祛湿[话题]# #祛湿寒[话题]# #寒湿[话题]# #祛湿减肥[话题]# #养生[话题]#', 'title': '祛湿！先分清楚湿热与寒湿，自测➕改善方法👉', 'url': 'https://www.xiaohongshu.com/explore/6705fc43000000002c02c802', 'date': '2024-10-09 11:45:07', 'note_type': 'images', 'video': {'noteId': '6705fc43000000002c02c802', 'url': ''}, 'images': [], 'like_count': '459', 'collect_count': 427, 'view_count': '21718', 'comments': [], 'confidence': -1}, {'id': 'turn_0_10', 'search_from': 'search_note', 'content': '以下是一份一周祛湿食谱，从早餐、午餐到晚餐，结合了常见的祛湿食材，帮助身体排出湿气，保持清爽健康。\n✅周一\n早餐：红豆薏仁粥 + 蒸山药\n午餐：冬瓜海带排骨汤 + 糙米饭 + 清炒莴笋丝\n晚餐：玉米须茶 + 凉拌黄瓜\n-\n✅周二\n早餐：芡实莲子粥 + 蒸玉米\n午餐：鲫鱼豆腐汤 + 荞麦面 + 炒豆角\n晚餐：茯苓粥 + 凉拌木耳\n-\n✅周三\n早餐：绿豆粥 + 蒸红薯\n午餐：海带豆腐汤 + 紫米饭 + 炒西葫芦\n晚餐：荷叶茶 + 凉拌生菜\n-\n✅周四\n早餐：小米南瓜粥 + 蒸芋头\n午餐：冬瓜肉丸汤 + 燕麦饭 + 炒油麦菜\n晚餐：赤小豆薏仁茶 + 凉拌豆芽\n-\n✅周五\n早餐：黑米粥 + 蒸胡萝卜\n午餐：丝瓜蛋汤 + 藜麦饭 + 炒芦笋\n晚餐：陈皮茶 + 凉拌海带丝\n-\n✅周六\n早餐：燕麦片 + 牛奶 + 坚果\n午餐：番茄鱼片汤 + 玉米 + 炒西兰花\n晚餐：冬瓜荷叶茶 + 凉拌菠菜\n-\n✅周日\n早餐：山药红枣粥 + 蒸南瓜\n午餐：海带排骨汤 + 红薯饭 + 炒豆角茄子\n晚餐：薏仁红豆茶 + 凉拌黄瓜木耳\n-\n⭐️祛湿小贴士*\n1⃣ **忌口**：减少甜食、油腻、生冷（如冰饮、西瓜）、辛辣刺激食物。\n2⃣   **饮水**：可搭配玉米须茶、薏米水、陈皮普洱茶等代茶饮。\n3⃣. **运动**：适当出汗（如快走、瑜伽）帮助湿气排出。\n4⃣根据个人体质调整食材，湿热体质可多加绿豆、苦瓜；寒湿体质建议加生姜、陈皮温补。\n\t\n#养生食疗[话题]# #身体去湿气[话题]# #节气养生[话题]# #先健脾还是先祛湿[话题]# #吃出你的健康[话题]# #夏季养生[话题]# #顺应季节养生[话题]# #食谱[话题]# #祛湿[话题]# #祛湿食谱[话题]# #祛湿气[话题]#', 'title': '祛湿一周食谱，排出湿气清爽健康', 'url': 'https://www.xiaohongshu.com/explore/6858d007000000000b02cd3f', 'date': '2025-06-23 11:54:47', 'note_type': 'images', 'video': {'noteId': '6858d007000000000b02cd3f', 'url': ''}, 'images': [], 'like_count': '1188', 'collect_count': 1284, 'view_count': '24890', 'comments': [], 'confidence': -1}, {'id': 'turn_0_11', 'search_from': 'search_note', 'content': '湿气真的是女生变丑元凶啊！！\n测测你体内有没有湿\n▫️常年手脚冰凉、身体像灌了铅\n▫️头发后背疯狂出油\n▫️下巴反复长红肿痘\n▫️浑身黏腻像裹保鲜膜\n▫️手冷脚冷，却燥热难眠\n▫️喝凉水都胖成馒头脸\n作为10年抗湿斗士，试错无数后才明白：\n⚠️去湿茶千万别乱喝！\n❗️❗️去湿要分体质❗️\n湿气分4种类型，喝错反而伤身！\n现在把吐血整理的祛湿茶方分享给你们\n\t\n✅4种体质祛湿茶配方（附症状）\n1️⃣寒湿体质喝：\n党参红枣茶（党参+红枣+茯苓+甘草+白术）\n👉🏻适合：舌苔白腻/手脚冰凉/容易水肿/便便黏马桶/姨妈痛到直不起腰\n⚠️喝错表现：喝完更怕冷/拉肚子\n\t\n2️⃣湿热体质喝：\n去湿清火饮（茯苓+荷叶+菊花+赤小豆+甘草）\n👉🏻脸油头油/便便臭粘/爱长红肿痘\n⚠️喝错表现：口干舌燥/爆痘加剧\n\t\n3️⃣痰湿体质喝：\n消肿瘦身茶（黄芪+山楂+茯苓+陈皮）\n👉🏻喝水都胖/舌苔厚腻/小腹松软\n⚠️喝错表现：越喝越虚/喉咙卡痰\n\t\n4️⃣脾虚体质喝：\n健脾四宝粥（五指毛桃+桂圆+芡实+薏仁）\n👉🏻虚胖减不动/喝凉水都胀气/气血不足\n⚠️喝错表现：消化不畅\n\t\n✨去湿小tips‼️：\n1️⃣煮前用温水泡30分钟 2️⃣每天喝500ml足够别过量\n3️⃣经期/孕妇慎喝薏仁赤小豆\n小贴士：祛湿是持久战，坚持喝对+泡脚+晒背， 现在皮肤透亮到反光，小肚子也平了！ 湿气人一定要对症下药，别再跟风乱喝啦～姐妹们快存好这份攻略\n特殊体质或孕妇建议咨询中医师哦~\n不知道自己是什么体质的，可以问小岐哦~\n\ufeff#体质[话题]#\ufeff \ufeff#干了这杯养生茶[话题]#\ufeff \ufeff#去湿气[话题]#\ufeff \ufeff#喝出好气色[话题]#\ufeff \ufeff#祛湿[话题]#\ufeff \ufeff#健脾祛湿[话题]#\ufeff \ufeff#湿气重[话题]#\ufeff \ufeff#三伏天养生[话题]#\ufeff \ufeff#食疗养生[话题]#\ufeff \ufeff#中医养生[话题]#\ufeff', 'title': '去湿踩坑10年😭终于找对方法!!', 'url': 'https://www.xiaohongshu.com/explore/67c9588f000000002902b6e3', 'date': '2025-03-06 18:31:04', 'note_type': 'images', 'video': {'noteId': '67c9588f000000002902b6e3', 'url': ''}, 'images': [], 'like_count': '33', 'collect_count': 33, 'view_count': '2310', 'comments': [], 'confidence': -1}, {'id': 'turn_0_13', 'search_from': 'search_note', 'content': '[向右R]皮肤容易出油，暗沉，出现湿疹～\n[向右R]嘴巴会有异味，吃什么都感觉没有味道～\n[向右R]整天昏沉，没有精神，感觉很疲惫～\n[种草R]可能是身体湿气太重噢‼\n—\n打工人久坐办公室，缺少运动，经常在空调房，不通风，体内湿气也不意外！下面给大家分享10种让湿气越吃越少的食物，喜欢的收藏点赞哦！\n—\n[一R]山药：山药是中药里的上品，滋阴利湿，健脾补肾。山药可以炖汤、煮粥，也可以炒菜。\n[二R]芹菜：芹菜能很好的帮助我们对付因湿气而导致的浮肿、虚胖，也是祛.湿的佳品。\n[三R]紫苏：紫苏被称为“纯阳之草”，适合夏日食用，促进人体气血流通，赶走湿寒。\n[四R]绿豆：绿豆具有清热解毒、利尿消肿的作用，可以帮助身体排出多余的水分和湿气。\n[五R]生姜：生姜具有温中散寒、解表发汗的运用，能帮助人体排除寒湿之气，可以作为食材烹饪，也可以泡脚，祛湿效果好。\n[六R]陈皮：陈皮具有理气健脾、燥湿化痰的功效，能促进体内湿气的运化和排出，泡水煲汤皆可，泡脚也可以添加。\n[七R]花椒：花椒有排寒祛湿、杀菌消炎的运用。除了日常调味，也可泡水，泡脚加一些也对身体大有益处。\n[八R]薏米：薏米在中药里称为“薏苡仁”，它可以治湿痹，利肠胃，消水肿，健脾益胃。薏米和红豆一起煮粥，是经典的祛湿食谱。\n[九R]苦瓜：苦瓜具有清热解毒、利尿消肿的作用，对于湿气重的人来说，适量食用苦瓜有助于减少湿气。\n[一R][零R]冬瓜：冬瓜有利水、消肿、清热的作用，是果蔬里祛湿的佳品。冬瓜炒山药、冬瓜汤等都是很好的选择。\n—\n请注意，虽然这些食物有助于减少湿气，但并不能完全替代药物治疗。如果湿气过重或伴随其他症状，建议及时就医并遵循医生的建议进行治疗。同时，保持健康的饮食习惯和生活方式也是减少湿气的关键。\n#食疗[话题]# #身体去湿气[话题]# #脾胃健康从饮食开始[话题]# #湿寒与湿热的区分[话题]# #食物中的营养物质[话题]# #含糖量高的食物[话题]# #吃出好身体[话题]# #食物与健康[话题]#', 'title': '湿气越吃越少的10种食物‼️颜值飙升先去湿', 'url': 'https://www.xiaohongshu.com/explore/67f4543e000000000900d0b4', 'date': '2025-04-08 06:39:58', 'note_type': 'images', 'video': {'noteId': '67f4543e000000000900d0b4', 'url': ''}, 'images': [], 'like_count': '23', 'collect_count': 15, 'view_count': '999', 'comments': [], 'confidence': -1}, {'id': 'turn_0_14', 'search_from': 'search_note', 'content': '🌧️湿气自测！中3条以上快抄作业👇\n👅 舌苔黄腻/边缘发红\n💆 脸头油汪汪，痘痘反复长\n☀️ 晨起口干口苦，异味重\n💩 小便黄、粑粑黏马桶（湿气重灾区！）\n😤 情绪易燃易爆炸，一点就着\n🥱 身体沉甸甸，上班像渡劫…\n中招的姐妹！！湿气锁住代谢，喝凉水都胖[叹气R]！\n\t\n💡祛湿黄金公式：炒薏米+健脾CP\n薏米是“祛湿界扛把子”💪，但生薏米寒性重！必须炒焦黄才能去寒性，健脾不伤胃（寒体姐妹尤其注意！）\n\t\n💦6款刮湿饮｜一周不重样掉秤攻略\n⚠️通用做法：食材洗净→破壁机豆浆模式→喜甜加冰糖（控糖用红枣代替）\n\t\n✅祛湿美白-白白饮\n炒薏米➕山yao➕百合➕银耳➕茯苓\n✅祛湿补气-山yao 薏米小米糊\n炒薏米➕小米➕山yao\n✅祛湿暖胃-薏米南瓜小米糊\n炒薏米➕小米➕南瓜➕红枣\n✅祛湿养颜-百合银耳绿豆浆\n炒薏米➕绿豆➕百合➕银耳\n✅祛湿掉秤-绿豆薏米燕麦羹\n炒薏米➕绿豆➕燕麦➕茯苓\n✅祛湿健脾-红豆薏米豆浆\n炒薏米➕红豆➕黄豆➕红枣\n\t\n📌【小Tips】\n[一R]薏米处理：一次炒500g密封罐存好！锅烧热无油→小火翻炒10分钟→焦香微裂即可\n[二R]体质适配：寒体选（姜/红枣/黑豆）；热体选（绿豆/百合）\n[三R]喝对时间：早上9点前喝！升阳气排浊水，避开睡前（会水肿）\n✨ 懒人急救💼：薏米饮➕拍八虚（腋窝/肘窝各拍50下），祛湿速度翻倍‼️\n\t\n🤗湿气一除，小肚子自己消失！姐妹们快冲这波“自带瘦感”水光肌套餐💦✨\n\t\n#破壁机食谱[话题]# #自制神仙饮品[话题]#\n#豆浆[话题]# #养生[话题]##祛湿减肥[话题]# #早餐[话题]#\n#养生饮品[话题]##健脾养胃[话题]#\n#健康养生[话题]# #祛湿气[话题]#', 'title': '湿气不走，肥肉不走！6款薏米饮刮油又掉秤！', 'url': 'https://www.xiaohongshu.com/explore/6842fb4b0000000020029c22', 'date': '2025-06-06 22:29:31', 'note_type': 'images', 'video': {'noteId': '6842fb4b0000000020029c22', 'url': ''}, 'images': [], 'like_count': '301', 'collect_count': 431, 'view_count': '10190', 'comments': [], 'confidence': -1}, {'id': 'turn_0_15', 'search_from': 'search_note', 'content': '不同体质四神汤搭配，全家可喝，养脾胃❗\n🌷四神汤是中y经典食疗方，由药食同源的茯苓、山药、莲子和芡实组成。被誉为“健脾祛湿di一方”性质平和，男女老少皆宜，既能调里脾胃虚弱，又能画解现代人常见的“湿胖”体质。民间甚至有“四神汤一煮，脾胃湿气无”的说法。\n⭐核心成分⭐\n✅健脾养胃:山药补脾阴，茯苓健脾气，拯救虚不受补”的现代脆弱胃\n✅祛湿：茯苓利水渗湿，芡实固肾防湿气反扑，专克“喝水都胖”虚胖体质\n✅养心an神：莲子清心火，an神助M，缓/解压力大导致的失M多梦\n🌸🍀分享11种四神汤的搭配，适合不同体质。\n1️⃣四神汤+黄芪、大枣\n2️⃣四神汤+桂圆、大枣\n3️⃣四神汤+薏米、赤小豆\n4️⃣四神汤+姜片、红糖\n5️⃣四神汤+百合、酸枣仁\n6️⃣四神汤+党参、黄芪\n7️⃣四神汤+百合、麦冬\n8️⃣四神汤+苹果\n9️⃣四神汤+山楂、陈皮\n1️⃣0️⃣四神汤+玉竹、灯芯草\n1️⃣1️⃣四神汤+百合、无花果\n🌾湿气在三焦的表现\n✅湿在上焦:头晕，头沉重，口黏腻，胸闷气短，头发油，脸油。\n✅湿在中焦:气血虚，乏力，不想吃饭，胀气，腹泻，肚子大，口臭口苦。\n✅湿在下焦:四肢沉重，尿路gan染，白带多，腰膝冷痛脚臭。\n🌸有什么人不适合喝！\n1、阴虚体质:阴虚的人一般都会有内热，💩秘的情况，未免症状加重,不适合喝原方四神汤。可搭配养阴润肺的沙参或者玉竹\n2、💩秘、腹胀:💩秘腹胀的患者不建议直接使用。需另外搭配其他材料。\n3、过min：对于过min的人群要谨慎使用。\n4、四神汤也不适合在经期吃\n📚🌷古人留下的经典方剂，养脾去湿又温和。宝子们喝起来吧，不用再担心夏天的湿气入侵。\n#在小红书轻养生[话题]# #四神汤[话题]# #女人养生茶[话题]# #脾胃健康从饮食开始[话题]# #食疗[话题]# #养脾胃[话题]# #先健脾还是先祛湿[话题]# #健康养生[话题]# #养脾胃就是养命[话题]# #祛湿[话题]#', 'title': '', 'url': 'https://www.xiaohongshu.com/explore/68f3765b000000000402acf7', 'date': '2025-10-18 19:13:31', 'note_type': 'images', 'video': {'noteId': '68f3765b000000000402acf7', 'url': ''}, 'images': [], 'like_count': '61', 'collect_count': 81, 'view_count': '2910', 'comments': [], 'confidence': -1}, {'id': 'turn_0_16', 'search_from': 'search_note', 'content': '🌈想要\u202e胄啤\u202c好，其实就\u202e个一\u202c字：饿，饿一饿才会\u202e健更\u202c康。\n.\n🌈并不是说要\u202e正真\u202c的饿肚子，啥也\u202e要不\u202c吃，\u202e是而\u202c要少吃、会吃、吃的刚刚\u202e，好\u202c在啤胄\u202e够能\u202c肖化运化的能力\u202e围范\u202c内，\u202e当适\u202c的去吃。\n.\n所以我\u202e日们\u202c常要\u202e会学\u202c给啤胄“做减法”👇\n1⃣少喝水\n2⃣少吃\n3⃣消积/食\n4⃣晚\u202e少上\u202c吃\n5⃣少寒凉\n6⃣多泡脚\n7⃣多拍屁股\n8⃣十指抓地\n9⃣勤揉腹\n🔟盘腿托举\n1⃣1⃣垫脚尖\n.\n另外啤胄\u202e蒻嘘\u202c的人\u202e以可\u202c从以下几\u202e方个\u202c面来tiao养👇\n👉饮\u202e调食\u202c整\n✅选择健脾食物\u202e像，\u202c山葯能补\u202e养脾\u202c胄，可\u202e粥煮\u202c或者清炒；\u202e米薏\u202c祛湿健啤，\u202e来用\u202c煲汤很不错；还\u202e芡有\u202c实，能益/shen固/精、补\u202e止啤\u202c写，可做\u202e芡成\u202c实粥。\n.\n❎避免\u202e用食\u202c生冷油腻的食物，\u202e冰如\u202c淇淋、生鱼片和过\u202e的多\u202c油炸食品\u202e这，\u202c些食物会加/重啤\u202e负胄\u202c担。\n.\n👉生\u202e习活\u202c惯\n✅保持规律\u202e息作\u202c，早睡早\u202e，起\u202c每晚可\u202e保以\u202c持7 - 8小\u202e的时\u202c睡眠，让啤胄\u202e夜在\u202c间能很好地自\u202e修我\u202c复\n.\n✅注意腹部保暖\u202e换，\u202c季或者\u202e冷寒\u202c天气时\u202e适，\u202c当增添衣物\u202e晚，\u202c上睡觉\u202e好盖\u202c被子，防\u202e啤止\u202c胄受寒。\n.\n👉情\u202e方绪\u202c面\n✅保持良\u202e的好\u202c情绪\u202e焦，\u202c虑、生气\u202e负等\u202c面情绪会影响\u202e的脾\u202c运化功能。\n.\n👉运动方面\n像\u202e极太\u202c拳、八段锦这类较为舒缓\u202e运的\u202c动，\u202e以可\u202c促进气您循环，增/强啤胄\u202e能功\u202c。也\u202e以可\u202c选择散步\u202e每，\u202c天饭后散步\u202e小半\u202c时左右\u202e能，\u202c够帮助～消 化，改善脾\u202e状嘘\u202c况。\n.\n以\u202e可上\u202c以根据\u202e己自\u202c的情况选\u202e适择\u202c合自己的\u202e式方\u202c改善啤\u202e嘘胄\u202c蒻的情况❤\n脾胃｜脾胃虚弱｜脾胃不好｜脾胃虚寒｜脾胃不和｜消化不良｜便秘｜脾虚｜胃强脾弱｜积食健脾｜宝宝脾胃｜口臭｜脾胃调理\n#宝宝脾胃[话题]# #脾胃[话题]# #脾胃虚弱[话题]# #积食[话题]# #宝宝脾胃虚弱怎么办[话题]# #调理脾胃[话题]# #调理脾胃宝宝积食便秘口臭[话题]# @薯队长@薯队长@薯宝宝@薯宝宝@薯条小助手@薯条小助手@薯管家', 'title': '🔥简单11个小习惯，养好脾胃', 'url': 'https://www.xiaohongshu.com/explore/691eb852000000001f00d34f', 'date': '2025-11-20 20:01:15', 'note_type': 'images', 'video': {'noteId': '691eb852000000001f00d34f', 'url': ''}, 'images': [], 'like_count': '1244', 'collect_count': 938, 'view_count': '55309', 'comments': [], 'confidence': -1}, {'id': 'turn_0_17', 'search_from': 'search_note', 'content': '💡 为什么你补气血没用？\n👉🏻 直接补气血=往漏水的水池加水！\n👉🏻 肝郁→克脾→生湿→气血虚，顺序错了=白补！\n\t\n🌟 跟着中医四步走，体质逆袭！\n✅1.肝主疏泄，负责全身气机的调畅。\n✅2.脾主运化，是水湿代谢和气血生成的关键。肝气调畅后，脾胃才能正常运作。\n✅3.湿气是脾胃虚弱后的病理产物，但湿邪本身又会进一步困 脾，形成恶性循环。\n✅4.最后补气血：巩固“根本”。\n只有在前面三步完成后，补气血（如黄芪、当归、党参）才能被有效吸收利用。\nStep1️⃣ 先疏肝：把“气”捋顺**\n❌ 肝yu表现：爱生气、失M多梦、胸胁胀T、长斑长痘。\n✅ 食材：百合+桑葚+玫瑰+红枣+茯苓\nStep2️⃣ 再健脾：修复“消化发动机”\n❌ 脾xu表现：脸黄垮脸、饭后腹胀、大便黏马桶、舌有齿痕。\n✅ 食材：五指毛桃+薏苡仁+芡实+无花果+山药+龙眼肉\nStep3️⃣ 后祛湿：扫除“身体垃圾”\n❌ 湿气表现：头脸出油、虚胖水月中、口臭黏腻、浑身沉重。\n✅ 食材：黄芪+陈皮+茯苓+山楂\nStep4️⃣ 最后补气血：真正“养回来”\n❌ 气血虚表现：脱发、姨妈量少、手脚冰凉、脸黄脸垮。\n✅ 食材：黄芪+枸杞+党参+太子参+当归+桂圆+红枣+红糖\n\t\n📌 关键：前3步做完再补！否则越补越堵！\n经验总结\n1️⃣ 顺序不能乱！肝→脾→湿→气血，环环相扣。\n2️⃣ 坚持28天！身体代谢周期=1个月，别急！\n3️⃣ 搭配运动：八段锦、散步，助气机流动。\n\t\n📢 收藏🌟转发💖，和闺蜜一起打卡变美！#情绪养生[话题]# #垮脸自救[话题]# #肝郁[话题]##祛湿[话题]# #补气血[话题]# #内养外调好气色[话题]# #打工人养生[话题]# #健康养生[话题]# #脸色发黄暗沉[话题]# #变美[话题]#', 'title': '长期焦虑、失眠、长斑？先疏肝❗四部调理拯救你', 'url': 'https://www.xiaohongshu.com/explore/67e138e2000000001203cdcb', 'date': '2025-03-24 18:50:10', 'note_type': 'images', 'video': {'noteId': '67e138e2000000001203cdcb', 'url': ''}, 'images': [], 'like_count': '739', 'collect_count': 1073, 'view_count': '27202', 'comments': [], 'confidence': -1}, {'id': 'turn_0_18', 'search_from': 'search_note', 'content': '手脚冰凉、脸色蜡黄、姨妈少？\n🚫 气血不足=脾胃失调+湿气堆积+吸收不良，直接补血上火、姨妈更乱！\n💡 正确顺序：先调理后补！ （通→健→补→固）\n1️⃣ 先健脾！——脾胃好，气血才吸收\n脾虚=气血漏勺，补再多都白费！ 先把消化系统养好，后续补养才有效！\n🍵\n✔️ 陈皮山药茶（健脾助消化）🌿 陈皮+山药片+大枣，饭后喝，避免脾虚积食！\n⚠️ 忌冰奶茶、生冷水果！每天揉腹10分钟\n2️⃣ 祛湿！——湿气重，补啥都堵着\n💦 湿气重=舌苔厚、浮肿、易困！一定要先把湿气排掉，否则补品只会越补越腻！\n🥤\n✔️ 赤小豆薏米水（祛湿排水肿）🌿 炒薏米+赤小豆，煮水当茶喝，一周见效！\n🚶\u200d♀️ 每天快走30分钟微微出汗！\n3️⃣ 后补气血！——气足才能生血，血足才能脸色红润\n气是推动血液运行的动力！ 先补气，再补血，否则会“血虚不生气”！\n🍯五红汤（气血双补）🌿 红皮花生+红豆+红枣+枸杞+红糖，每周喝3次！\n💡 每天还阳卧10分钟（脚掌相对躺），促进气血循环！\n4️⃣ 最后固本！——防止气血流失，保持好状态\n补进去的气血，要让它“待得住”，不然容易掉回去！\n😡 少生气！肝郁=气血不畅\n✔️ 玫瑰佛手茶（疏肝理气）🌿 玫瑰+佛手+陈皮，调节情绪，让气血通畅！\n💤 23点前睡！熬夜=烧干灯油\n🧘\u200d♀️ 瑜伽/八段锦，激活气血循环\n✅ 气血回升自测👇\n✔️ 指甲小月牙回来了！\n✔️ 睡醒不累，脸色红闰！\n✔️ 姨妈正常，量足无血块！\n除此之外还可以做养生项目改善，像拔罐就能通过扩张皮肤毛孔，顺畅气血，促进血液循环，祛湿排毒增强体质，弥补熬夜亏空。\n🌟 姐妹们别乱补！先通后补，30天后你会发现自己气色变美了！ #湿气[话题]# #养生茶喝出好气色[话题]# #脸色暗沉发黄[话题]# #脾胃[话题]# #补气血有顺序[话题]#', 'title': '补气血顺序错了！会越补越虚', 'url': 'https://www.xiaohongshu.com/explore/6806e178000000001d020b2e', 'date': '2025-04-22 08:23:20', 'note_type': 'images', 'video': {'noteId': '6806e178000000001d020b2e', 'url': ''}, 'images': [], 'like_count': '160', 'collect_count': 152, 'view_count': '8565', 'comments': [], 'confidence': -1}, {'id': 'turn_0_20', 'search_from': 'search_note', 'content': '给大家分享 4 款针对不同体质的祛湿茶，快看看自己适合哪种👇\n🍃湿热体质：薏米赤小豆茶\n🌿食材：炒薏米 15g、赤小豆 15g、茯苓 10g（可加少量芡实）\n🍵做法：材料洗净，加水煮 20-30 分钟。\n💪功效：清热利湿、健脾消肿，适合舌苔黄腻、皮肤油腻、易长痘的湿热体质。\n⚠️注意：寒湿体质、孕妇、经期女性慎喝，脾胃虚弱者不宜过量。\n\t\n❄️寒湿体质：生姜茯苓茶\n🌿食材：生姜 3-5 片（去皮）、茯苓 10g、陈皮 5g（可选）\n🍵做法：食材放入锅中煮 15-20 分钟。\n💪功效：温阳散寒、健脾利湿，改善怕冷、手脚凉、大便黏的寒湿症状。\n⚠️注意：湿热体质、易上火者禁用。\n\t\n💧痰湿体质：陈皮茯苓茶\n🌿食材：陈皮 5g（越陈越好）、茯苓 10g、炒白术 5g（可选）\n🍵做法：材料用开水泡 20 分钟或煮 10 分钟。\n💪功效：理气健脾、燥湿化痰，缓解腹胀、食欲不振、痰多，适合痰湿体质。\n⚠️注意：阴虚火旺者慎用。\n\t\n😩湿气重且易疲劳：红豆芡实茶\n🌿食材：红豆 15g、芡实 15g、山药 10g（干品，可选）\n🍵做法：洗净浸泡后煮 30-40 分钟。\n💪功效：健脾祛湿、固肾止泻，适合湿气重、大便稀溏、易疲劳人群。\n⚠️注意：便秘人群慎用。\n喝祛湿茶同时，也要调整生活习惯，少吃生冷油腻，多运动～记得根据体质选茶，让身体更健康💕\n#祛湿茶[话题]# #湿气重[话题]# #体质养生[话题]# #身体去湿气[话题]# #湿寒与湿热的区分[话题]# #先健脾还是先祛湿[话题]# #补阳气祛寒湿[话题]# #养生茶[话题]# #食疗[话题]#', 'title': '湿气重别瞎喝！不同体质适合的祛湿茶不一样', 'url': 'https://www.xiaohongshu.com/explore/68b830ae000000001d025e48', 'date': '2025-09-03 20:12:30', 'note_type': 'images', 'video': {'noteId': '68b830ae000000001d025e48', 'url': ''}, 'images': [], 'like_count': '1517', 'collect_count': 1617, 'view_count': '98832', 'comments': [], 'confidence': -1}, {'id': 'turn_0_21', 'search_from': 'search_note', 'content': '第1️⃣种:湿寒喝一姜丝丁香茶.\n第2️⃣种:痰湿喝一山药茯苓茶.\n第3️⃣种: .上热下寒喝一桑葚双花茶 .\n第4️⃣种:湿热喝一山楂荷花茶.\n#湿气重[话题]# #湿气重的表现症状[话题]##祛湿茶[话题]##祛湿气[话题]##祛湿[话题]##除湿[话题]##湿气重如何调理[话题]# #祛湿最有效的方法[话题]# #食疗养生[话题]##上热下寒[话题]#', 'title': '湿气重！大肚腩腿粗❗️真心建议大家喝这4款茶', 'url': 'https://www.xiaohongshu.com/explore/672b1f6d000000001d03833b', 'date': '2024-11-06 15:49:01', 'note_type': 'images', 'video': {'noteId': '672b1f6d000000001d03833b', 'url': ''}, 'images': [], 'like_count': '9407', 'collect_count': 7917, 'view_count': '791021', 'comments': [], 'confidence': -1}, {'id': 'turn_0_23', 'search_from': 'search_note', 'content': '宝子们！是不是总觉得自己“湿气重”，身体像裹了块湿毛巾🧖\u200d♀️，喝了一堆祛湿茶却半点用没有？！\n听我的！先辨体质再喝茶，不然越喝越虚真的会哭死😭！整理了超全【湿气体质茶方】，赶紧对照看你是哪款👇\n🔥 湿热体质（油痘肌+口干口苦+大便粘马桶）\n家人们谁懂啊！又湿又热真的太折磨，脸上爆痘就算了，嘴巴还总发苦😩，舌头一伸全是黄腻苔！\n✨ 救命茶方：茯苓+薏米+芡实+陈皮+荷叶+菊花\n💡 喝法：抓一把煮10分钟，或者焖泡15分钟，喝起来清清爽爽，像给身体开了“除/湿+降温”双buff🧊！\n❄️ 寒湿体质（冬天手脚冰+肚子常年凉+舌苔白到发光）\n寒和湿缠在一起，就像身体里揣了块冰🧊！夏天还贪凉喝冰饮？只会雪上加霜！千万别再碰凉茶了！\n✨ 救命茶方：茯苓+生姜+甘草+党参+红枣\n💡 喝法：煮成暖暖的茶汤，喝完整个人从胃暖到脚尖，寒湿直接被“赶跑”，手脚终于不冰了✌️！\n🍂 痰湿体质（虚胖水仲+总犯困+喉咙卡痰咳不出）\n感觉身体像灌了铅⚖️，脸仲成“发面馒头”，喉咙还总黏糊糊的？这是痰湿在搞鬼👻！\n✨ 救命茶方：茯苓+黄芪+陈皮+山楂\n💡 喝法：煮水当日常茶喝，帮你把痰湿化掉，喝着喝着就觉得身体变轻盈，犯困感都少了～\n🌪️ 上热下寒（脸上爆痘+手脚冰+姨妈痛到冒冷汗）\n体质矛盾到离谱！上面上火长痘，下面冰凉痛经，简直是“冰/火两重天”😤！\n✨ 救命茶方：茯苓+陈皮+山楂+乌梅+玫瑰花+桑葚\n💡 喝法：用开水泡5分钟，酸酸甜甜的超好喝，帮你把“上热下寒”的体质调顺，姨妈期都舒服了❤️！\n⚠️ 血泪提醒：\n• 体质分不清的宝，别瞎喝！记得辨清体质～\n• 茶方别天天喝！喝1周停3天，给身体留缓冲期⏳～\n• 光喝茶不够！少吃冰饮、多走路🚶\u200d♀️，湿气才会真的跑光光！\n快收藏！下次被湿/气缠/上，直接翻出来对号喝🍵 跟油痘、水仲、疲惫说拜拜～我是影子食养，关🐷我，学习更多的食养知识。\n#湿气重[话题]# #去湿气[话题]# #养生茶[话题]# #女生养生[话题]# #体质调理[话题]# #食疗养生[话题]# #健康小知识[话题]# #养生茶[话题]##体质[话题]# #上热下寒茶[话题]#', 'title': '湿气重喝什么茶❓看体质喝对才有校❗️', 'url': 'https://www.xiaohongshu.com/explore/68ef852f000000000700d9aa', 'date': '2025-10-15 19:27:43', 'note_type': 'images', 'video': {'noteId': '68ef852f000000000700d9aa', 'url': ''}, 'images': [], 'like_count': '543', 'collect_count': 659, 'view_count': '19065', 'comments': [], 'confidence': -1}, {'id': 'turn_0_25', 'search_from': 'search_note', 'content': '✨祛湿是日常作业，有多少肥胖是痰湿引起的，来社湿减肥吧！\n\t\n🌾【材料黄金配比】\n薏米150g（祛湿界ACE选手）\n赤小豆100g（红豆的祛湿plus版！）\n茯苓30g（健脾界的低调大佬）\n芡实30g（锁住营养的固肾神器）\n⚠️重点：全程不用水洗！直接开炒！\n🔥【懒人快手步骤】\n1️⃣ 所有材料倒入无水无油的锅\n2️⃣ 小火干炒❗️\n👉薏米微微焦黄卷边\n👉赤小豆变身深酒红色\n👉空气里飘出治愈系谷物香就对了！\n3️⃣ 晾凉装密封罐 随喝随取超方便\n🍵【神仙喝法】\n抓一把洗净后丢进养生壶➕1L水\n煮开后再焖20分钟‼️\n你会得到一壶琥珀色的“液态祛湿符”\n每天打包上班喝上一壶，感觉身体在咕嘟咕嘟排水💦\n\t\n💡【知识点卡】\n✅赤小豆≠红豆！认准细长身形\n✅炒制后寒性大减 女生友好👩❤️\n✅密封罐放阴凉处 喝出三个月新鲜感\n\t\n⚠️经期/孕妇别喝❗️体寒姐妹加两片姜。\n#身体去湿气[话题]# #排湿气寒气[话题]# #食疗养生[话题]#\n\t\n#祛湿茶[话题]#  #懒人养生[话题]# #中医养生[话题]# #办公室必备[话题]# #健康饮品[话题]# #养生日常[话题]# #茯苓[话题]#', 'title': '超简单！自制超强祛湿茶！', 'url': 'https://www.xiaohongshu.com/explore/67d18618000000000900c6b9', 'date': '2025-03-12 21:03:20', 'note_type': 'images', 'video': {'noteId': '67d18618000000000900c6b9', 'url': ''}, 'images': [], 'like_count': '2828', 'collect_count': 3800, 'view_count': '142604', 'comments': [], 'confidence': -1}, {'id': 'turn_0_26', 'search_from': 'search_note', 'content': '#我的养生仪式感[话题]#\n一天不洗头就油腻腻\n拉粑粑粘马桶🚽\n每天感觉疲惫睡不饱\n肚子大，怕热又怕累\n……………………\n办公室坐久了的毛病！\n强烈安利大家喝这个祛湿茶，真的有效！\n准备：赤小豆，薏米，茯苓，芡实\n比例2：2：1：1炒熟（大概炒10-15分钟，中途不停翻遍以免炒糊了）\n养生壶煮20分钟左右，每天喝，连续一周。\n不要加其他乱七八糟的，就这四样就可以，臭宝给我去喝！#养生茶[话题]# #食疗养生[话题]# #我的养生仪式感[话题]# #养生好物[话题]# #茯苓[话题]# #养生女孩[话题]# #祛湿茶[话题]# #祛湿[话题]#', 'title': '祛湿茶给你们试了，有点东西！', 'url': 'https://www.xiaohongshu.com/explore/669f12e2000000000a0070af', 'date': '2024-07-23 10:18:10', 'note_type': 'images', 'video': {'noteId': '669f12e2000000000a0070af', 'url': ''}, 'images': [], 'like_count': '2699', 'collect_count': 2244, 'view_count': '144565', 'comments': [], 'confidence': -1}, {'id': 'turn_0_27', 'search_from': 'search_note', 'content': '每年都会给自己安排一段时间喝祛湿茶，但走过不少弯路后才明白，湿气原来分不同类型。\n\t\n今天就把这些整理出来分享给有需要的姐妹们，赶紧收藏起来吧！\n\t\n✅寒湿\n食材：白术＋红枣＋茯苓＋甘草＋党参\n\t\n✅痰湿\n食材：陈皮＋茯苓＋山楂＋黄芪\n\t\n✅湿热\n食材：薏仁＋芡实＋菊花＋荷叶＋赤小豆＋茯苓＋陈皮\n\t\n✅脾虚\n食材：怀山＋无花果＋桂圆＋芡实＋薏仁＋五指毛桃\n\t\n🧑🏻\u200d🍳食用方法\n1️⃣食材清洗干净放入保温壶，加1000毫升水煮15分钟即可\n2️⃣保温杯焖泡25分钟\n反复续杯，可以喝一整天哦~\n#养生茶[话题]##健脾胃[话题]##喝出好气色[话题]# #健脾祛湿[话题]# #祛湿气[话题]# #食疗养生[话题]# #冬季养生[话题]# #祛湿喝什么[话题]# #养生食补进行时[话题]# #养生内调茶[话题]#', 'title': '喝对茶|让你冬天不再被湿气“缠身”', 'url': 'https://www.xiaohongshu.com/explore/676a6b42000000000b00d7a9', 'date': '2024-12-24 16:05:22', 'note_type': 'images', 'video': {'noteId': '676a6b42000000000b00d7a9', 'url': ''}, 'images': [], 'like_count': '365', 'collect_count': 432, 'view_count': '21462', 'comments': [], 'confidence': -1}, {'id': 'turn_0_28', 'search_from': 'search_note', 'content': '👩很多女生可能都觉得自己湿气很重，但是自己不知道怎么祛湿，分享这4款调理茶☕️给大家！\n不同的湿气，对应调理也不同！\n👉湿寒体质喝：姜丝丁香茶\n👉湿热体质喝：山楂荷叶茶\n👉痰湿体质喝：山药茯苓茶\n👉上热下寒喝：桑葚双花茶\n\ufeff#春日养生茶[话题]#\ufeff \ufeff#健康养生[话题]#\ufeff \ufeff#我的健康生活[话题]#\ufeff \ufeff#祛湿气[话题]#\ufeff \ufeff#湿气重[话题]#\ufeff \ufeff#祛湿茶[话题]#\ufeff \ufeff#湿寒体质[话题]#\ufeff\ufeff#湿热体质[话题]#\ufeff \ufeff#健脾祛湿[话题]#\ufeff', 'title': '湿气重❗️大肚子腿粗多喝这4款祛湿茶🔥', 'url': 'https://www.xiaohongshu.com/explore/67fcb4da000000001c0165ac', 'date': '2025-04-14 18:06:07', 'note_type': 'images', 'video': {'noteId': '67fcb4da000000001c0165ac', 'url': ''}, 'images': [], 'like_count': '946', 'collect_count': 832, 'view_count': '51129', 'comments': [], 'confidence': -1}, {'id': 'turn_0_29', 'search_from': 'search_note', 'content': '薏米200g，赤小豆200g，茯苓100g，芡实100g；薏米和芡实炒了10分钟，然后放赤小豆和茯苓炒了10分钟，全程小火。煮水50分钟左右，真的很香，类似大麦茶的味道！', 'title': '自制祛湿茶', 'url': 'https://www.xiaohongshu.com/explore/67a2056a000000001902ee80', 'date': '2025-02-04 20:17:46', 'note_type': 'images', 'video': {'noteId': '67a2056a000000001902ee80', 'url': ''}, 'images': [], 'like_count': '5240', 'collect_count': 4866, 'view_count': '208018', 'comments': [], 'confidence': -1}, {'id': 'turn_0_31', 'search_from': 'search_note', 'content': '❤️\u200d🔥赤小豆+薏米真的是祛💦的王炸\n\t\n坚持喝姐妹们👭\n\t\n步骤：\n1.准备赤小豆100g 薏米100g\n（一定要买赤小豆，不是红豆\U0001fad8）\n2.将赤小豆和薏米分别在锅里炒2分钟左右\n（薏米性微寒，一定要炒熟，薏米颜色有点变黄，赤小豆有点变深红色）\n3.炒好后装起来，每次取四分之一的量，加入1L水，煮水之前把炒熟的薏米和赤小豆洗一下，只洗要煮的那些，其他等喝之前再洗\n4.养生壶20分钟或者用锅小火煮15分钟\n\t\n姐妹们行动起来\n祛💦气后整个人都很轻松\n体重真的会降❗️❗️#喝出好气色[话题]# #身体去湿气[话题]# #红豆薏米水[话题]# #湿气重喝什么[话题]# #薏米水[话题]# #赤小豆薏米茶[话题]# #我的养生日常[话题]#', 'title': '红豆薏米水\U0001fad8去湿就喝它❗️', 'url': 'https://www.xiaohongshu.com/explore/67a45148000000002802b3ef', 'date': '2025-02-06 14:06:00', 'note_type': 'images', 'video': {'noteId': '67a45148000000002802b3ef', 'url': ''}, 'images': [], 'like_count': '6833', 'collect_count': 4638, 'view_count': '462964', 'comments': [], 'confidence': -1}, {'id': 'turn_0_32', 'search_from': 'search_note', 'content': '🌟先说重点：祛湿神水，你真的喝对了吗？\n⚠️很多人为了祛湿气喝红豆薏米水\n肚子却越喝越大\n身体越喝越疲惫\n舌苔越喝越明显\n不仅没有起到减肥效果，反而越喝越肿\n❌其实是你的打开方式不对\n\t\n👇下面就给大家分享一下我喝红豆薏米水的实践心得     🍸一起正确祛湿，轻盈入夏🥗\n\t\n✨选材关键\n✅红豆要用赤小豆（细长形，药效更强），不是圆红豆（补血为主）\n✅薏米建议选炒薏米（药店或自制：生薏米干锅小火炒至微黄），减轻寒性\n✨经典配比\n赤小豆60g + 炒薏米30g（1人份）\n体质偏寒加3片生姜/5颗红枣\n易上火加10g茯苓\n✨正确煮法\n浸泡2小时后煮1小时（不建议用保温杯焖泡！有效成分难析出）\n只喝汤水不吃渣（豆类淀粉易生湿）\n📢被夸大的功效\n🙅不能减肥！（短期排水≠减脂）\n🙅不能祛痘！（湿热痘有效，激素痘无效）\n🙅不能天天喝！（过度利水反伤津液）\n✨体质筛选：\n✅适合：舌胖有齿痕、便溏、头面出油多\n❌不适合：手脚冰凉、舌红无苔、易口干者\n✨最佳饮用时机：\n上午9-11点（脾经运行时）喝200ml温饮，下午喝易尿频影响睡眠\n✨我的使用感悟\n作为易水肿体质，这些年试过各种祛湿方，最终回归到最朴实的红豆薏米水：\n\t\n早晨脸肿、眼皮沉时喝1杯（利水效果明显）\n舌苔白腻+大便粘马桶时连喝3天（个人体验：黏腻感减轻）\n暑天吹空调后头重如裹时喝（比冰美式管用）\n连续喝一周不如「喝2天停1天」效果好\n配合拍打脾经（小腿内侧）效果加成\n雨季搭配藿香正气水泡脚有奇效\n\t\n⚠️切记：它不是网红水，而是中医食疗方。尊重体质差异，才能用得巧妙。\n体质自测表/不同加减方详解👉评论区留言获取\n#中医养生[话题]# #养生茶[话题]# #喝出好气色好喝不贵[话题]# #红豆薏米水[话题]##祛湿气[话题]#', 'title': '\U0001fad8红豆薏米水，喝对比喝多更重要❗️', 'url': 'https://www.xiaohongshu.com/explore/67f741510000000009016dff', 'date': '2025-04-10 11:56:01', 'note_type': 'images', 'video': {'noteId': '67f741510000000009016dff', 'url': ''}, 'images': [], 'like_count': '4078', 'collect_count': 3875, 'view_count': '243545', 'comments': [], 'confidence': -1}, {'id': 'turn_0_33', 'search_from': 'search_note', 'content': '俗话说：“热在三伏，冷在三九，冬病夏治。”\n.\n养生就要趁着三伏天这个好“天时”，借助一年中阳气zui旺盛的季节来帮扶身体，养阳驱寒、治已病、防未病，事半功倍，请务必抓住三伏天黄金养生的40天。\n.\n\U0001fad8自制红豆薏米水步骤\nㅤ\n1.准备赤小豆100g、薏米100g（比例1:1就可以，一定要买赤小豆，不是红豆哈）\n2.把赤小豆和薏米分别放在锅里炒2-3分钟左右（薏米性微寒，一定要炒熟）\n3.炒好后密封罐装起来就可以分次食用了。\n4.养生壶煮30分钟。\nㅤ\n💡多种搭配方式\nㅤ\n1.炒薏米30g+赤小豆30g+陈皮5g\n2.炒薏米30g+赤小豆30g+大枣5个\n3.炒薏米30g+赤小豆30g+生姜3片\nㅤ\n这几种搭配非常方便，适合办公室懒人养生，喝完整个人都很轻松，体重真的会降～\n\t\n#真正煮出来的自在水[话题]# #养生内调茶[话题]# #在小红书轻养生[话题]# #养生日常[话题]# #三伏天养生[话题]# #身体去湿气[话题]# #祛湿气[话题]# #红豆薏米茶[话题]# #减肥[话题]#\nㅤ', 'title': '红豆薏米水，这个喝法绝了！（自制步骤全解析）', 'url': 'https://www.xiaohongshu.com/explore/6859efce000000002201c6a4', 'date': '2025-06-24 08:22:39', 'note_type': 'images', 'video': {'noteId': '6859efce000000002201c6a4', 'url': ''}, 'images': [], 'like_count': '832', 'collect_count': 723, 'view_count': '71335', 'comments': [], 'confidence': -1}, {'id': 'turn_0_34', 'search_from': 'search_note', 'content': '🔥 祛湿水喝错=白喝+伤身？难怪你越喝越肿！ 🔥\n💢 你是不是也这样：\n👉 跟风喝红豆薏米水祛湿...\n👉 结果：小肚子鼓鼓💢 | 身体沉沉💢 | 舌苔厚厚💢\n❌ 不仅没瘦，反而像泡发的馒头？\n💡 真相是：90%的人都喝错了！祛湿不成反伤身！\n👇 亲测有效的【正确打开方式】大公开，抄作业就能轻盈一夏！\n\t\n🚫 先避雷！祛湿水不是“万能神水”！\n不能减肥！(排的是水，不是脂肪！)\n不能治痘！(湿热痘有用，其他类型无效)\n不能天天灌！(过度喝反伤津液，越喝越干！)\n\t\n✅ 喝对才有效！核心要点Get！\n1️⃣  【选材是灵魂！】\n❌ 红豆？NO！→ ✅ 必须是【赤小豆】！ (细长身形，食养作用更好)\n❌ 生薏米？NO！→ ✅ 一定选【炒薏米】！(减寒性！药店有或自己小火炒黄)\n🧊 体寒星人？→ 加3片生姜/5颗红枣\n🔥 易上火？→ 加10g茯苓\n2️⃣【黄金配比】\n赤小豆 60g + 炒薏米 30g (1人份量，精准才有效！)\n3️⃣【正确煮法=效果翻倍】\n⏳ 浸泡2小时 + 煮足1小时！(拒绝保温杯焖泡！无效！)\n\t\n🕒 【最佳饮用时机】— 事半功倍！\n⏰ 黄金时间：上午9-11点！(脾经当令，吸收最好) ☕ 喝200ml温热的！(拒绝冰镇！)\n⚠️ 下午晚上别喝！(易跑厕所，影响睡眠！)\n\t\n❓ 你适合喝吗？快速自测！\n✅ 适合你喝：\n舌头胖有齿痕👅\n便便溏软不成形💩\n头发脸爱出油💧\n❌ 先别喝：\n手脚常年冰凉❄️\n舌头红、光溜溜无苔🔴\n容易口干口渴💦 (这类朋友先养阴！)\n\t\n🌟 我的【真实体验报告】🌟\n(易水肿体质亲测！)\n☀️ 早晨脸肿、眼皮重：喝1杯，排水超明显！💦\n💩 舌苔白+便便粘马桶：连喝3天，浑身清爽不粘腻！\n🌬️ 空调房待久头重脚轻：来一杯，比冰咖啡管用10倍！👍\n⚡ 关键技巧：喝2天停1天，效果 > 天天喝！配合拍打小腿内侧(脾经)，效果加成！\n\t\n📢 重要提醒：\n这碗水是老祖宗的【食养智慧】，不是网红快消品！\n⚠️ 效果因人而异！了解自己体质是关键！\n#中医养生[话题]# #养生茶[话题]# #芡实赤豆薏米茶[话题]# #身体去湿气[话题]#  #健康养生[话题]# #红豆薏仁水[话题]# #祛湿气[话题]# #喝出好气色[话题]# #祛湿茶[话题]# #养生内调茶[话题]# #我的养生仪式感[话题]#\n#记录吧就现在[话题]#', 'title': '红豆薏米水，喝对比喝多更重要❗️', 'url': 'https://www.xiaohongshu.com/explore/683ae32b000000002102c7b2', 'date': '2025-05-31 19:08:27', 'note_type': 'images', 'video': {'noteId': '683ae32b000000002102c7b2', 'url': ''}, 'images': [], 'like_count': '239', 'collect_count': 198, 'view_count': '25552', 'comments': [], 'confidence': -1}, {'id': 'turn_0_35', 'search_from': 'search_note', 'content': '最近开始爱喝红豆薏米水的味道\n查了下功效 很适合自己 那肯定要搞起来！\n材料：赤小豆+薏米+水\n工序：豆子炒熟—分装—清洗、浸泡—开煮\n最后开喝～～～\n#自在水红豆薏米水[话题]# #红豆薏米水[话题]##赤小豆薏仁[话题]# #喝出好气色[话题]# #想记录下此刻[话题]#', 'title': '不是5.5元的红豆薏米水喝不起 是自制更香', 'url': 'https://www.xiaohongshu.com/explore/65f930da0000000012037036', 'date': '2024-03-19 14:29:46', 'note_type': 'images', 'video': {'noteId': '65f930da0000000012037036', 'url': ''}, 'images': [], 'like_count': '2159', 'collect_count': 1561, 'view_count': '144509', 'comments': [], 'confidence': -1}, {'id': 'turn_0_36', 'search_from': 'search_note', 'content': '🌿赤小豆（红豆）利水消肿、薏米健脾祛湿，二者搭配山药、茯苓、玉米须等食材，可 “健脾祛湿、利水消肿”。肚子大，腿粗，脸肿，皮肤沉暗的人，要健脾袪湿。\n\t\n🌸养生依据：《本草纲目》记载赤小豆（红豆） “甘，酸，平，能利水消肿，解.毒 排 脓”，薏米 “甘，淡，凉，能利水渗湿，健脾止泻”，现代研究证实二者含丰富膳食纤维与活性成分，可强化祛湿效果，助力体质改 善。\n\t\n🍀 10 款红豆薏米水慢慢调理，收藏起来喝出健康好状态。\n✅1. 经典红豆薏米水（基础祛湿）\n✅2. 红豆薏米茯苓水（深层祛湿）\n✅3. 红豆薏米玉米须水（消肿祛湿）\n✅4. 红豆薏米冬瓜水（利水祛湿）\n✅5. 红豆薏米山药水（健脾养胃）\n✅6. 红豆薏米芡实水（健脾固肠）\n✅7. 红豆薏米红枣水（健 脾 补 血）\n✅8. 红豆薏米荷叶水（祛湿减脂）\n✅9. 红豆薏米荞麦水（饱腹减脂）\n✅10. 红豆薏米绿豆水（清热减脂）\n\t\n⚠️ 小提醒：\n1.\t薏米提前炒香可降低寒性，避免脾胃不适，红豆需提前浸泡 30 分钟（易煮烂），煮制时间 20-25 分钟为宜，不喜甜或需控 糖人建议不加糖；\n\t\n2.\t脾胃虚寒者（易腹泻）建议搭配生姜、红枣中和寒性。\n\t\n☘️你喜欢哪种搭配：欢迎分享你的体验！\n☘️我是‘楹进’，非常相信中式古法养生，以及食疗的力量，持续分享简单、实用的养生小知识，以及普通人能上手的养生食谱，关🐷我，一起用家常食材养出好身体！”\n\t\n#小红书三餐四季[话题]# #中式古法养生[话题]# #健康养生[话题]# #养生小知识[话题]# #养生[话题]# #红豆薏米[话题]# #食疗养生[话题]# #养生[话题]##养生花茶[话题]# #红豆薏米水[话题]#  #养生日常 #喝出健康喝出好气色！', 'title': '红豆薏米煮水这样喝掉秤很快，10种搭配❗', 'url': 'https://www.xiaohongshu.com/explore/6922af6d000000000d00dc06', 'date': '2025-11-23 14:53:33', 'note_type': 'images', 'video': {'noteId': '6922af6d000000000d00dc06', 'url': ''}, 'images': [], 'like_count': '461', 'collect_count': 446, 'view_count': '20305', 'comments': [], 'confidence': -1}, {'id': 'turn_0_37', 'search_from': 'search_note', 'content': '浑身提不上力气，小肚子大大，\n头发又爱出油，身体浮肿，\nロ干ロ苦口臭，手脚冰凉，\n减肥困难，痰多黏腻，大便粘马桶。\n舌苔厚腻，舌头肥大两边有齿痕，\n有以上的症状就是湿气重的表现！\n\t\n《黄帝内经》中说“湿气通于脾”\n健脾祛湿，疏旰泻🔥是养生调理的重点。\n\t\n湿气很狡猾，\n一旦进入身体就会直入脾胃\n导致机能下降，\n而脾胃又正好是解毒利水的位置\n这又很容易导致体内进一步生湿😭\n\t\n分享一款简单有效的祛湿茶\n食材:\n1️⃣赤小豆2️⃣薏米\n3️⃣茯苓 4️⃣干芡实\n这几个搭配在一起可以\n很好的去除我们体内湿气。\n\t\n做法:\n1️⃣薏米偏寒，要先炒一下。\n2️⃣食材洗净放入养生壶煮30分钟✅\n#小红书的三餐四季[话题]# #祛湿[话题]# #减肥[话题]#\n#赤小豆薏米芡实茶[话题]# #霍思燕[话题]#\n#减肥经验分享[话题]# #消肿[话题]# #湿气[话题]#', 'title': '红豆薏米水｜霍思燕同款的瘦身祛湿水！！', 'url': 'https://www.xiaohongshu.com/explore/67726802000000000900f152', 'date': '2024-12-30 17:29:38', 'note_type': 'images', 'video': {'noteId': '67726802000000000900f152', 'url': ''}, 'images': [], 'like_count': '390', 'collect_count': 322, 'view_count': '22694', 'comments': [], 'confidence': -1}, {'id': 'turn_0_38', 'search_from': 'search_note', 'content': '红豆薏米汤越喝湿气越重？原来我都喝错了\n之前我喝红豆薏米，就是直接煮。学了中医后才发现，原来我一直喝错了，难怪一点变化都没有。\n之前我每天坚持喝，却发现：\n👉🏻 舌苔越来越厚，齿痕更明显\n👉🏻 明明没吃多少，小肚子却越来越鼓\n👉🏻 喝完反而更容易累，手脚沉重\n现在才知道，原来我完全没弄明白。\n《黄帝内经》说：“湿胜则濡泄。”意思是湿气重的人，脾胃运化能力差，盲目祛湿反而会加重负担。\n1️⃣生薏米寒凉，越喝脾越虚\n很多人直接用生薏米煮水，却不知道生薏米性寒，长期喝会不利脾阳。脾主运化，脾虚了，湿气反而排不出去。\n✅ 正确做法：\n薏米一定要炒过，炒至微黄，减少寒性\n搭配食材：山药、茯苓、芡实\n2️⃣只祛湿不健pi，湿气永远除不尽\n湿气的根源在脾虚，光喝红豆薏米，就像只擦地板不关水龙头，水（湿气）还是会一直冒出来。\n✅ 正确搭配：\n四神汤（山药+莲子+芡实+茯苓）健pi为主，祛湿为辅\n五指毛桃+白术，补气祛湿，适合容易疲劳的人\n3️⃣体质不对，越喝越糟糕\n阴虚、阳虚、气血不足的人，喝红豆薏米反而会加重问题：\n阴虚火旺（口干、睡眠不好、易上火）→ 越喝越燥\n阳虚怕冷（手脚冰凉、容易腹泻）→ 寒上加寒\n✅ 体质调理方案：\n阴虚：换成银耳百合汤\n阳虚：喝生姜红枣茶\n气血虚：用黄芪+当归煮水\n真正有效的祛湿茶配方\n如果你真的想祛湿，试试这个升级版：\n🍵 炒薏米20g + 赤小豆15g + 山药10g + 陈皮3g\n薏米炒过，赤小豆（不是红豆！）提前浸泡\n小火煮30分钟，只喝水不吃渣\n每周喝3次，不要天天喝\n坚持1个月的变化：\n✔️ 舌苔变薄，齿痕减轻\n✔️ 晨起不浮肿，身体轻盈\n✔️ 消化变好，小肚子慢慢消下去\n#祛湿[话题]# #健脾[话题]# #中医养生[话题]# #红豆薏米[话题]# #湿气重表现[话题]# #身体去湿气[话题]# #夏季养生[话题]# #女人养生养脾胃[话题]# #喝出健康喝出好气色[话题]# #先健脾还是先祛湿[话题]# #生活化养生[话题]# #杂粮粥[话题]# #健康养生[话题]# #喝出好气色[话题]# #五谷杂粮[话题]# #养生粥[话题]# #养生补气血[话题]# #寒湿是万病之源[话题]#', 'title': '红豆薏米汤越喝湿气越重？原来我都喝错了🙅', 'url': 'https://www.xiaohongshu.com/explore/68332c140000000022007ec2', 'date': '2025-05-25 22:41:24', 'note_type': 'images', 'video': {'noteId': '68332c140000000022007ec2', 'url': ''}, 'images': [], 'like_count': '500', 'collect_count': 444, 'view_count': '51256', 'comments': [], 'confidence': -1}, {'id': 'turn_0_39', 'search_from': 'search_note', 'content': '三伏天女性艾灸去湿💦，找对穴位是关键‼\n三伏天，万物阳气蒸腾，体内阳气空虚，此时用艾灸驱寒升阳，以调节脏腑机能，使阴阳达到平衡，冬病夏治，美丽由内而外散发。\n艾灸一般前七天，每天灸一次，之后可一周灸3-5次。症状有所缓解后可减至隔日灸甚至一周1次，🉑视情况而定。\n肚脐(神阙穴)：每天艾灸5~10分钟\n关元穴：艾灸5~10分钟，隔-天一次\n三阴交：艾灸5~10分钟，隔-天一次\n足三里：艾灸5~10分钟，隔-天一次\n#艾灸[话题]###祛湿[话题]##新手爸妈[话题]# #小儿[话题]# #小孩[话题]# #三伏天养生[话题]# #三伏天[话题]#', 'title': '三伏天女性艾灸去湿💦，找对穴位是关键‼ 三', 'url': 'https://www.xiaohongshu.com/explore/66ac475a0000000005020750', 'date': '2024-08-02 10:41:30', 'note_type': 'images', 'video': {'noteId': '66ac475a0000000005020750', 'url': ''}, 'images': [], 'like_count': '5235', 'collect_count': 4695, 'view_count': '254768', 'comments': [], 'confidence': -1}, {'id': 'turn_0_40', 'search_from': 'search_note', 'content': '一.为什么说艾灸能帮助减肥？\n1️⃣ 调月卑胃=控食欲\n👉 适合：管不住嘴、爱吃重口味、肚子胀的姐妹\n2️⃣ 活气血=提代xie\n👉 适合：手脚冰凉、没力气、代谢慢的懒人\n3️⃣ 降食欲=少暴食\n👉 适合：一饿就失控、暴饮暴食的吃货\n4️⃣ 排寒湿=去虚胖\n👉 适合：肚子松垮、阴雨天更累的湿胖党\n\t\n二.关键帮助减肥的穴位\n1️⃣ 中脘穴\n📍肚脐上5指（4寸）\n💡给月卑胃“加油”，消化变好，肚子脂肪难堆积！胃zhang、消化慢的姐妹锁死～\n2️⃣ 天枢穴\n📍肚脐左右各3指（2寸），两侧都有\n💡当肠道“清道夫”，通便秘、排垃圾，小肚腩直接瘪下去！\n3️⃣ 关元穴\n📍肚脐下4指（3寸）\n💡虚寒体质姐妹必灸！补阳气、调代谢，把小腹凉凉和脂肪一起赶跑～\n4️⃣ 足三里\n📍膝盖下4指（3寸）\n💡 给月卑胃“开外挂”！消化变好，腿部水肿直接退～ 腿粗、总胀肚的快灸！\n5️⃣ 丰隆穴\n📍小腿外侧，膝盖到外脚踝连线中点\n💡 虚胖天敌！祛湿化痰超[强]，浑身肿肿、肉松软的姐妹锁死，把虚胖体质扳回来～\n6️⃣ 带脉穴\n📍侧腰，和肚脐平行的位置（左右各1个）\n💡 瘦腰腹！“游泳圈”粉碎机，灸完腰直接细一圈，腰腹肉肉堆堆必灸🔥\n\t\n艾灸操作建议\n- 每次挑 3 - 5 个穴位，每个穴位灸个 10 - 15 分钟就行，隔一天来一次 。\n- 顺序，由上到下。\n- 工具用艾条悬灸，注意艾条年份，3-5年的\n\t\n三.注意事项\n（一）适合人群\n- 适合体质虚寒、阳气不足、伴有湿气（寒湿）的人\n- 结合饮食和运动，并长期坚持，安全操作\n（二）禁忌人群\n- 容易上火,红痒过min的，孕妇，别灸 。\n- 发着高烧、急性yan症犯的时候，别灸 。\n\t\n四.其他提醒\n-  刚吃饱别着急灸，等1小时左右。\n- 下雨🌧️天也先别灸，空气湿气大。\n- 灸完记得喝口温水！ 4 小时内别洗澡、别猛吹冷风。\n- 这事儿急不得，坚持灸上 1 个来月，慢慢就显出来啦\n\t\n五.饮食建议\n- 生冷、油乎乎的少吃！多吃山药、红豆、茯苓这些能帮“干活”、排排湿气的食物 。\n- 晚上吃饭别超量，睡前 3 小时就别往嘴里塞东西啦。\n\t\n总结：想通过艾灸掉秤，得有耐心长期坚持！日常把健康小习惯（规律作息、合理饮食这些 ）跟上，双管齐下～要是坚持一阵体重没啥变化，找靠谱专业人士根据具体情况给调调方案。\n#穴位养生[话题]# #中医养生[话题]# #艾灸[话题]# #减肥[话题]# #艾灸养生[话题]# #养生日常[话题]# #一起来养生[话题]# #排湿气寒气[话题]# #艾灸艾条艾柱[话题]# #补阳气祛寒湿[话题]#', 'title': '怎么艾灸可以瘦下来', 'url': 'https://www.xiaohongshu.com/explore/68a42469000000001c005cec', 'date': '2025-08-19 17:02:03', 'note_type': 'images', 'video': {'noteId': '68a42469000000001c005cec', 'url': ''}, 'images': [], 'like_count': '952', 'collect_count': 1057, 'view_count': '23120', 'comments': [], 'confidence': -1}, {'id': 'turn_0_41', 'search_from': 'search_note', 'content': '三伏天，万物阳气蒸腾，体内阳气空虚，此时用艾灸驱寒升阳，以调节脏腑机能，使阴阳达到平衡，冬病夏治，美丽由内而外散发。\n艾灸一般前七天，每天灸一次，之后可一周灸3-5次。症状有所缓解后可减至隔日灸甚至一周1次，🉑视情况而定。\n\t\n🌿三伏灸的5大好处\n✔️驱寒祛湿：改善手脚冰凉、关节冷痛、宫寒痛经等问题。\n✔️增强免疫力：减少感冒、鼻炎、哮喘等呼吸道疾病发作。\n✔️调理脾胃：改善消化不良、胃寒腹泻、食欲不振。\n✔️冬病夏治：对慢性支气管炎、风湿性关节炎等冬季易发病有显著缓解作用。\n✔️抗疲劳、抗衰老：提升阳气，改善亚健康状态。\n⚠️注意事项\n✅\xa0最佳艾灸时间：上午9-11点（阳气最旺），避免晚上9点后施灸（易影响睡眠）。\n✅\xa0禁忌人群：孕妇、高热、皮肤破损、阴虚火旺者慎灸。\n✅\xa0灸后护理：2小时内不碰冷水、不吹空调，多喝温水。\n⚠️如出现头晕、口干等上火症状，可减少灸量或加灸涌泉穴引火下行。\n坚持三伏灸30天，可显著改善体质，尤其适合寒湿重、易疲劳、慢性病患者！\xa0🌿🔥\n#灸出好气色[话题]# #穴位养生[话题]# #灸出好气色[话题]# #三伏灸[话题]# #三伏天[话题]# #三伏天养生[话题]# #三伏天去湿[话题]# #三伏天灸[话题]# #三伏天祛湿[话题]# #三伏天减肥[话题]#', 'title': '三伏天女性艾灸去湿💦，找对穴位是关键‼', 'url': 'https://www.xiaohongshu.com/explore/687e102e00000000120333ea', 'date': '2025-07-21 18:02:22', 'note_type': 'images', 'video': {'noteId': '687e102e00000000120333ea', 'url': ''}, 'images': [], 'like_count': '1220', 'collect_count': 1766, 'view_count': '30307', 'comments': [], 'confidence': -1}, {'id': 'turn_0_45', 'search_from': 'search_note', 'content': '夏天穿好看的衣服总被赘肉绊住？别担心，中医艾灸来帮忙！针对不同部位的赘肉，几组艾灸穴位搭配，让你想瘦哪儿就瘦哪儿。试试三伏天艾灸，真的不一样！\n\t\n瘦肚子（针对脾胃弱、痰湿堆积）\n中脘穴：上腹部前正中线，脐上 4 寸（胃的募穴），帮消化不囤脂\n关元穴：下腹部前正中线，脐下 3 寸，促代谢燃腹脂\n天枢穴：腹部平脐，前正中线旁开 2 寸（大肠募穴），调肠道让小腹平\n💡 每穴灸 15-20 分钟，每周 3-4 次，肚子悄悄变小～\n\t\n细大腿（改.善湿气重、气血不畅）\n足三里：小腿外侧，犊鼻下 3 寸，调脾胃减腿围\n三阴交：小腿内侧，内踝尖上 3 寸，调气血消水仲\n丰隆穴：小腿外侧，外踝尖上 8 寸，祛湿让腿变细\n💡 每穴灸 10-15 分钟，每周 2-3 次，穿短裤更自信～\n\t\n甩拜拜肉（解决气血阻滞）\n曲池穴：肘横纹外侧端（屈肘找），促代谢甩肉\n内关穴：前臂掌侧，腕横纹上 2 寸，促循环瘦臂\n肩髃穴：肩部三角肌上（臂外展时凹陷处），通经络减赘肉\n💡 每穴灸 10-15 分钟，每周 3-4 次，无袖装放心穿～\n\t\n小贴士：温度别太高防烫.伤，灸后别马上洗澡、吃生冷。\n三伏天是老天爷赏的减.肥窗口期, 按穴位精.准 灸，易瘦体质直接焊死在身上💥 （亲测有.效果！来评论区交出你们的三伏天艾灸战绩！）\n\t\n#穴位养生[话题]# #快速瘦赘肉[话题]##艾灸瘦全身[话题]##三伏天瘦逆袭[话题]##夏天要瘦[话题]##三伏天养生[话题]# #日常养生[话题]##养生小妙招[话题]# #夏季养生[话题]# #养生小知识[话题]#', 'title': '三伏天艾灸黄金期🔥 精准灸出易瘦体质', 'url': 'https://www.xiaohongshu.com/explore/6882eae9000000002203e8c3', 'date': '2025-07-25 10:24:41', 'note_type': 'images', 'video': {'noteId': '6882eae9000000002203e8c3', 'url': ''}, 'images': [], 'like_count': '4104', 'collect_count': 4128, 'view_count': '106659', 'comments': [], 'confidence': -1}, {'id': 'turn_0_47', 'search_from': 'search_note', 'content': '为什么说艾灸能帮助减肥？\n1️⃣ 调月卑胃=控食欲\n👉 适合：管不住嘴、爱吃重口味、肚子胀的姐妹\n2️⃣ 活气🩸=提代xie\n👉 适合：手脚冰凉、没力气、代谢慢的懒人\n3️⃣ 降食欲=少暴食\n👉 适合：一饿就失控、暴饮暴食的吃货\n4️⃣ 排寒湿=去虚胖\n👉 适合：肚子松垮、阴雨天更累的湿胖党\n\t\n二.关键帮助减肥的穴位\n1️⃣ 中脘穴\n📍肚脐上5指（4寸）\n💡给月卑胃“加油”，消化变好，肚子脂肪难堆积！胃zhang、消化慢的姐妹锁死～\n2️⃣ 天枢穴\n📍肚脐左右各3指（2寸），两侧都有\n💡当肠道“清道夫”，通便秘、排垃圾，小肚腩直接瘪下去！\n3️⃣ 关元穴\n📍肚脐下4指（3寸）\n💡虚寒体质姐妹必灸！补阳气、调代谢，把小腹凉凉和脂肪一起赶跑～\n4️⃣ 足三里\n📍膝盖下4指（3寸）\n💡 给月卑胃“开外挂”！消化变好，腿部水肿直接退～ 腿粗、总胀肚的快灸！\n\t\n艾灸操作建议\n🌟每次挑 3 - 5 个穴位，每个穴位灸个 10 - 15 分钟就行，隔一天来一次 。\n🌟顺序，由上到下。\n🌟工具用艾条悬灸，注意艾条年份，3-5年的\n\t\n三.注意事项\n（一）适合人群\n- 适合体质虚寒、阳气不足、伴有湿气（寒湿）的人\n- 结合饮食和运动，并长期坚持，安全操作\n（二）禁忌人群\n- 容易上火,红痒过min的，孕妇，别灸 。\n- 发着高烧、急性yan症犯的时候，别灸 。\n\t\n四.其他提醒\n-  刚吃饱别着急灸，等1小时左右。\n- 下雨🌧️天也先别灸，空气湿气大。\n- 灸完记得喝口温水！ 4 小时内别洗澡、别猛吹冷风。\n- 这事儿急不得，坚持灸上 1 个来月，慢慢就显出来。\n\t\n#艾灸[话题]#  #补阳气祛寒湿[话题]# #穴位养生[话题]# #中医养生[话题]# #灸出好气色[话题]# #疏通经络对人体的好处[话题]# #艾灸艾条艾柱[话题]# #知艾者福善灸者寿[话题]# #养生[话题]# #健康[话题]# #艾灸养生[话题]# #一起来养生[话题]#', 'title': '艾灸养成易瘦体质，选对穴位掉秤快🔥', 'url': 'https://www.xiaohongshu.com/explore/68c2a63f000000001c036e81', 'date': '2025-09-11 18:36:47', 'note_type': 'images', 'video': {'noteId': '68c2a63f000000001c036e81', 'url': ''}, 'images': [], 'like_count': '2564', 'collect_count': 2627, 'view_count': '61546', 'comments': [], 'confidence': -1}, {'id': 'turn_0_53', 'search_from': 'search_note', 'content': '分享倪海夏老师排出陈年积寒，积痰、积瘀三个穴位，大家收藏好～\n\t\n✅三大穴位：\n丰隆穴-化痰\n血海穴-淤血\n阴陵泉 -去湿\n\t\n✅艾灸注意事项：\n①每个穴位灸十五分钟，不红不烫是基础，主要身体要感觉舒服～\n②不要玩手机，聚精会神感受艾灸和身体的对话～\n③艾灸过程，如果感觉口干舌燥，想喝水，要么停灸要么补阴津～#艾灸[话题]##艾灸穴位[话题]##倪海夏[话题]##咳嗽[话题]##湿气[话题]##艾灸养生[话题]#', 'title': '听劝❗️灸透这三大穴位排出陈常年积寒积淤', 'url': 'https://www.xiaohongshu.com/explore/67ebe766000000000b01dae7', 'date': '2025-04-01 21:17:26', 'note_type': 'images', 'video': {'noteId': '67ebe766000000000b01dae7', 'url': ''}, 'images': [], 'like_count': '963', 'collect_count': 1444, 'view_count': '32882', 'comments': [], 'confidence': -1}, {'id': 'turn_0_59', 'search_from': 'search_note', 'content': '宝子们，是不是总觉得脾胃不好，吃啥都不香，身体还总出状况？其实调理脾胃可是有门道的，得按顺序来，先疏肝，再健脾，接着祛湿，最后补气血，一步步来，脾胃才能真正好起来💪今天就来给大家详细讲讲～\n🌿先疏肝\n肝属木，脾胃属土，在中医理论里，木克土。肝气不舒，就容易影响脾胃的正常功能。想想看，如果你最近压力大，心情郁闷，是不是就容易没胃口，还腹胀？这就是肝气郁结在捣乱。\n要疏肝，平时可以多吃点疏肝理气的食物，像佛手、陈皮、玫瑰花。泡上一杯玫瑰花茶，既能疏肝理气，又能养颜，一举两得。另外，晚上早点睡，11 点前进入深度睡眠，肝才能好好排毒、藏血，这也是疏肝的关键哦～\n💪再健脾\n肝气顺畅了，就可以集中精力健脾啦。脾胃是后天之本，气血生化之源。健脾可以多吃些黄色的食物，像小米、南瓜、红薯，它们都入脾经，能很好地滋养脾胃。还可以试试艾灸足三里这个穴位，每天艾灸 15 - 20 分钟，能激发脾胃的阳气，增强脾胃功能。\n🍃接着祛湿\n脾胃虚弱，水湿运化就会失常，湿气就容易在体内堆积。湿气重，人会感觉浑身乏力、头重脚轻，脸上还爱出油。\n祛湿可以多吃薏米、赤小豆、芡实这些利水渗湿的食材。煮个薏米赤小豆粥，坚持喝一段时间，能明显感觉到湿气在减少。平时也别总呆在潮湿的环境里，多运动，让湿气随着汗液排出体外。\n🩸最后补气血\n前面的工作做好了，就可以补气血啦。因为脾胃功能正常了，湿气也祛了，补进去的气血才能被身体好好吸收。\n可以吃点红枣、桂圆、阿胶来补气血。不过要注意，补气血不能太猛，得循序渐进。像红枣桂圆粥，就是个不错的选择，既能补气血，又不会给脾胃造成太大负担。\n宝子们，调理脾胃是个长期的过程，按照这个顺序慢慢来，相信大家都能拥有好脾胃，好气色🥰记得点赞收藏，分享给身边同样需要的小伙伴～#山参养生局[话题]#\n#脾胃调理[话题]# #疏肝健脾[话题]# #祛湿补气血[话题]# #养生小知识[话题]#', 'title': '脾虚的来‼️顺序搞不对，养脾胃真的会白费', 'url': 'https://www.xiaohongshu.com/explore/6905b08400000000030391d5', 'date': '2025-11-01 15:02:28', 'note_type': 'images', 'video': {'noteId': '6905b08400000000030391d5', 'url': ''}, 'images': [], 'like_count': '609', 'collect_count': 1040, 'view_count': '14531', 'comments': [], 'confidence': -1}, {'id': 'turn_0_61', 'search_from': 'search_note', 'content': '🌞三伏天是一年中最适合艾灸的时节🌿 借助天时，找对穴位，效果直接翻倍！\n📖今天整理了不同体质对应的艾灸穴位，快码住！\n一、脾胃虚寒党🥣\n一吃凉的就胃疼、肚子胀，还总感觉脾胃 “动力不足”？这几个穴救大命！\n✨中脘穴：肚脐上 4 寸（大概四指并拢宽度），帮你理顺饮食运化，吃嘛嘛香🍚\n✨关元穴：肚脐下 3 寸，给脾胃补足能量，虚寒感退退退！\n✨足三里：外膝眼往下四横指，常灸能强脾胃、少生病，脾胃弱必冲！\n二、上热下寒人🔥❄\n明明上火（口干、咽喉痛），脚却像踩冰窖？这是气血 “上下不通”！\n🌟脾俞穴：脊柱旁开 2 指，和肚脐水平位置。平衡上下寒热，火气往下走、阳气往上补～\n🌟命门穴：后背正中央，肚脐对应的脊柱凹陷处。给下半身供暖，怕冷腿酸全改善！\n🌟涌泉穴：脚底前 1/3 凹陷（脚趾蜷起更明显），引火下行的 “大功臣”，上火 + 脚凉同时解决，超爽！\n三、肝郁血瘀族😠\n爱生气、痛经、姨妈颜色深？肝郁血瘀找上门！灸这组穴，把 “闷气”“瘀血” 全赶跑～\n🔥大敦穴：脚大拇趾甲根旁 2mm，轻轻一灸，郁结的闷气瞬间疏开，心情都变顺✨\n🔥太冲穴：足背第 1、2 跖骨间凹陷，调肝理气一把手，痛经、烦躁全缓解～\n🔥期门穴：乳头直下第 6 肋间隙，灸完气血通畅，脸色都变红润！\n四、湿气重星人🌧\n身体发沉、大便黏马桶、总觉得累？湿气在 “搞鬼”！这组穴帮你把湿气 “打包带走”🚚\n💧丰隆穴：小腿前外侧，外踝尖上 5 指宽。脾胃 “除湿机”，调理好脾胃，湿气源头就断了～\n💧解溪穴：足背与小腿交界横纹中央，“痰湿粉碎机”，把身体里的湿浊一点点排干净～\n💧阴陵泉：小腿内侧胫骨凹陷处，“脾湿克星”，从根源上健脾祛湿，身体轻松到飞起！\n五、手脚冰凉怪🧊\n❄神阙穴：肚脐中央，“肾阳大粮仓”，艾灸直接给全身补阳气，从内暖到外～\n❄大椎穴：后颈第 7 颈椎凹陷处（低头时最明显的骨突下），“经络疏通器”，打通气血通道，手脚瞬间回温🔥\n❄太溪穴：内踝尖与跟腱间凹陷，“气血加速器”，让气血顺畅跑到手脚，冰凉感退散！\n💡小 tips：艾灸后别碰冷水、吹冷风，喝杯温水促进代谢～\n#三伏天养生[话题]# #艾灸攻略[话题]# #中医调理[话题]# #体质养生[话题]# #养生[话题]# #健康养生[话题]# #穴位养生[话题]# #湿寒与湿热的区分[话题]# #补阳气祛寒湿[话题]# #灸出好气色[话题]#', 'title': '三伏天艾灸🔥 按体质选穴才有效！附穴位指南', 'url': 'https://www.xiaohongshu.com/explore/68552dd3000000001203fa1d', 'date': '2025-06-20 17:45:55', 'note_type': 'images', 'video': {'noteId': '68552dd3000000001203fa1d', 'url': ''}, 'images': [], 'like_count': '3025', 'collect_count': 3688, 'view_count': '78628', 'comments': [], 'confidence': -1}, {'id': 'turn_0_63', 'search_from': 'search_note', 'content': '春天湿气重，可以用艾灸调理脾胃，要正确取穴详细取穴位置：\n内伤脾胃，百病由生，脾胃是我们后天之本，气血生化之源，脾胃不好就会产生很多问题: 虚胖(小肚子突出)、皮肤头发易出油、脸色暗淡发黄、 便秘....总之脾胃不好真的会让人显得不清爽、老态， 养好脾胃真的会让人方方面面变好温补莫过于灸法，想要养好脾胃，可以试试【艾】， 推荐以下穴位:\n1.中脘穴:肚脐正上4寸；胃之募穴，能加强肠胃蠕动、具有健脾和胃、补中益气、降逆利水之功用2.神阙穴:就是肚脐；培元固本、回阳救脱、和胃理肠；\n3.关元穴:肚脐下3寸(约4横指)温固阳气，通经活络\n4.足三里穴:小腿前外侧，当犊鼻下3寸；是“足阳明胃经”的主要穴位之一，它具有调理脾胃、补中益气、通经活络、疏风化湿的功能；\n5.阴陵泉穴:小腿内侧，胫骨内侧髁下缘与胫骨内侧缘之间的凹陷中；可调理聚湿生痰和脾虚及胃引起的症状\n每个穴位艾灸15分钟左右，每周3-4次即可。\n#穴位养生[话题]# #调理[话题]# #经络[话题]# #食疗养生[话题]# #养生日常[话题]# #体质[话题]# #中医养生[话题]# #艾灸艾条艾柱[话题]# #中医[话题]# #美容养生[话题]#', 'title': '有宝子问，脾胃不好如何调理，推荐艾灸', 'url': 'https://www.xiaohongshu.com/explore/6611651f000000001a014c95', 'date': '2024-04-06 23:07:11', 'note_type': 'images', 'video': {'noteId': '6611651f000000001a014c95', 'url': ''}, 'images': [], 'like_count': '6173', 'collect_count': 6591, 'view_count': '276470', 'comments': [], 'confidence': -1}, {'id': 'turn_0_65', 'search_from': 'search_note', 'content': '艾灸时需要注意火候和时间，避免皮肤烫伤，通常每个穴位艾灸15到20分钟。灸完之后要注意避免吹风更不能马上洗澡。\n【个人养生经验分享，不涉及医疗建议，身体不适请及时就医，审核大大请勿误判谢谢！】\n\ufeff#养生[话题]#\ufeff \ufeff#健康养生[话题]#\ufeff \ufeff#中医养生[话题]#\ufeff \ufeff#艾灸[话题]#\ufeff \ufeff#艾灸养生[话题]#\ufeff \ufeff#脾虚[话题]#\ufeff \ufeff#湿气[话题]#\ufeff \ufeff#祛湿[话题]#\ufeff \ufeff#养生小知识[话题]#\ufeff \ufeff#湿气重[话题]#\ufeff', 'title': '脾虚湿气重，艾灸这六大穴！', 'url': 'https://www.xiaohongshu.com/explore/67f90932000000000d015983', 'date': '2025-04-11 20:21:06', 'note_type': 'video', 'video': {'noteId': '67f90932000000000d015983', 'url': ''}, 'images': [], 'like_count': '3727', 'collect_count': 4168, 'view_count': '84521', 'comments': [], 'confidence': -1}, {'id': 'turn_0_69', 'search_from': 'search_note', 'content': '不要盲目跟风！\n都说去湿对人好，但是属于哪种湿气，对应怎么去，也是有很大学问的。湿气分三种，不同情况方法大不同！\n看看你是哪一种👇\n一、三种湿的常见表现\n1️⃣ 寒湿：手脚冰凉、肚子胀、大便稀溏、舌苔白厚。\n2️⃣ 痰湿：脸油头油、痰多黏腻、肚子松软胖、舌边有齿痕。\n3️⃣ 湿热：口苦口味、长d、小便黄、舌红苔黄腻。\n二、不同湿气的艾灸搭配\n1️⃣ 寒湿体质\n▪️ 穴位：中脘（健）、足三里（补）、关元（温）、阴陵泉（祛）。\n▪️ 艾灸方法：每天选2-3个穴，每穴灸15分钟，腹部穴位可隔姜灸。\n2️⃣ 痰湿体质\n▪️ 穴位：丰隆（化）、中脘（和）、阴陵泉（利）、带脉（消）。\n▪️ 艾灸方法：早晚各灸1次，重点揉按丰隆穴后再灸，更佳。\n3️⃣ 湿热体质\n▪️ 穴位：曲池（清）、合谷（排）、阴陵泉（祛）、三阴交（调）。\n▪️ 艾灸方法：下午3-5点灸，避开上午阳气过盛时段，每穴10分钟。\n三、饮食调理重点\n• 寒湿：多吃生姜、红枣、山药粥，忌冷饮。\n• 痰湿：喝陈皮茯苓茶，少吃甜食和油炸食品。\n• 湿热：煮绿豆薏米汤，戒奶茶和辛辣。\n四、三种湿都有的调理法\n▪️ 穴位搭配：中脘+足三里+阴陵泉+丰隆，每天选2个穴轮灸。\n▪️ 饮食原则：温补不加冰，清淡少油盐，晨起喝温水促代。\n我的经验：艾灸后喝温水，1小时内别洗澡；阴虚火旺慎灸！坚持1个月，湿气悄悄跑光光～（具体也因人而异，可以根据体质调整）\n#艾灸[话题]# #艾灸养生[话题]# #我的养生仪式感[话题]# #湿寒与湿热的区分[话题]# #艾灸艾条艾柱[话题]# #灸出好气色[话题]# #先健脾还是先祛湿[话题]# #寒湿是万病之源[话题]#', 'title': '艾灸去湿，先认识你是哪种湿', 'url': 'https://www.xiaohongshu.com/explore/685e4cca0000000017032c87', 'date': '2025-06-27 15:48:26', 'note_type': 'images', 'video': {'noteId': '685e4cca0000000017032c87', 'url': ''}, 'images': [], 'like_count': '53', 'collect_count': 36, 'view_count': '1079', 'comments': [], 'confidence': -1}, {'id': 'turn_0_77', 'search_from': 'search_note', 'content': '被闺蜜按头安利的艾灸贴，这才是养生的正确打开方式吧～\n三阴交对女性很友好！搭配艾草精油热敷 upup！\n撕开即热的设计超贴心，4-6小时恒温热敷♨️\n淡淡的艾草味道不熏人，在办公室使用也不影响同事🤫\n巴掌大一片携带方便，上班通勤、逛街散步都能贴，养生原来这么简单不费劲儿！\n小小一片热敷贴，由内而外养自己一遍！养生女孩快冲！\n\t\n#记录吧就现在[话题]# #上热下寒引火下行[话题]# #穴位养生[话题]# #养生人必备[话题]# #上热下寒表现[话题]##三阴交[话题]# #足三里[话题]# #保暖实用家居好物[话题]# #精油热敷贴[话题]# #穴位养生[话题]# #灸出好气色[话题]# #当代年轻人养生[话题]# #女性健康[话题]# #1年1度购物狂欢[话题]##我的保暖秘诀[话题]#  #艾灸贴有用吗[话题]##艾草贴[话题]# #秋冬保暖[话题]# #小红书市集秋上新[话题]#', 'title': '三阴交热敷一个月，真的不是智商税💪', 'url': 'https://www.xiaohongshu.com/explore/69134adf0000000005038927', 'date': '2025-11-11 22:40:32', 'note_type': 'images', 'video': {'noteId': '69134adf0000000005038927', 'url': ''}, 'images': [], 'like_count': '2251', 'collect_count': 1201, 'view_count': '189238', 'comments': [], 'confidence': -1}, {'id': 'turn_0_80', 'search_from': 'search_note', 'content': '想在这黄金30天排走湿哒哒？那这8个黄金穴位，必贴！（附详细找穴方法和贴敷作用）\n\t\n#夏季养生[话题]# #三伏天养生[话题]# #三伏贴[话题]# #穴位养生[话题]# #中医健康养生知识[话题]# #十二经脉循行[话题]# #养生小知识[话题]# #艾灸[话题]# #在小红书轻养生[话题]# #三伏天[话题]# #艾灸贴[话题]# #艾灸艾条艾柱[话题]#', 'title': '贴错=白贴！这8个【黄金穴位】请收藏！', 'url': 'https://www.xiaohongshu.com/explore/688b1a720000000023035685', 'date': '2025-07-31 15:25:38', 'note_type': 'images', 'video': {'noteId': '688b1a720000000023035685', 'url': ''}, 'images': [], 'like_count': '566', 'collect_count': 491, 'view_count': '18991', 'comments': [], 'confidence': -1}, {'id': 'turn_0_82', 'search_from': 'search_note', 'content': '上次的隔姜灸阴陵泉不好操作，分享个简单版的，牙签容易烧断，建议用金属的签字，电扇要背着吹，就能吸走艾烟，艾柱是观二心兔年那个\n\t\n\ufeff#艾灸[话题]#\ufeff \ufeff#祛湿[话题]#\ufeff \ufeff#养生[话题]#\ufeff', 'title': '一个排湿贼牛的', 'url': 'https://www.xiaohongshu.com/explore/68416294000000002202729f', 'date': '2025-06-05 17:25:40', 'note_type': 'video', 'video': {'noteId': '68416294000000002202729f', 'url': ''}, 'images': [], 'like_count': '22637', 'collect_count': 12858, 'view_count': '590278', 'comments': [], 'confidence': -1}, {'id': 'turn_0_83', 'search_from': 'search_note', 'content': '为什么说艾灸能帮助减肥？\n1️⃣ 调月卑胃=控食欲\n👉 适合：管不住嘴、爱吃重口味、肚子胀的姐妹\n2️⃣ 活气🩸=提代xie\n👉 适合：手脚冰凉、没力气、代谢慢的懒人\n3️⃣ 降食欲=少暴食\n👉 适合：一饿就失控、暴饮暴食的吃货\n4️⃣ 排寒湿=去虚胖\n👉 适合：肚子松垮、阴雨天更累的湿胖党\n\t\n二.关键帮助减肥的穴位\n1️⃣ 中脘穴\n📍肚脐上5指（4寸）\n💡给月卑胃“加油”，消化变好，肚子脂肪难堆积！胃zhang、消化慢的姐妹锁死～\n2️⃣ 天枢穴\n📍肚脐左右各3指（2寸），两侧都有\n💡当肠道“清道夫”，通便秘、排垃圾，小肚腩直接瘪下去！\n3️⃣ 关元穴\n📍肚脐下4指（3寸）\n💡虚寒体质姐妹必灸！补阳气、调代谢，把小腹凉凉和脂肪一起赶跑～\n4️⃣ 足三里\n📍膝盖下4指（3寸）\n💡 给月卑胃“开外挂”！消化变好，腿部水肿直接退～ 腿粗、总胀肚的快灸！\n\t\n艾灸操作建议\n🌟每次挑 3 - 5 个穴位，每个穴位灸个 10 - 15 分钟就行，隔一天来一次 。\n🌟顺序，由上到下。\n🌟工具用艾条悬灸，注意艾条年份，3-5年的\n\t\n三.注意事项\n（一）适合人群\n- 适合体质虚寒、阳气不足、伴有湿气（寒湿）的人\n- 结合饮食和运动，并长期坚持，安全操作\n（二）禁忌人群\n- 容易上火,红痒过min的，孕妇，别灸 。\n- 发着高烧、急性yan症犯的时候，别灸 。\n\t\n四.其他提醒\n-  刚吃饱别着急灸，等1小时左右。\n- 下雨🌧️天也先别灸，空气湿气大。\n- 灸完记得喝口温水！ 4 小时内别洗澡、别猛吹冷风。\n- 这事儿急不得，坚持灸上 1 个来月，慢慢就显出来。#福纳健康[话题]# #艾灸[话题]#', 'title': '艾灸养成易瘦体质，选对穴位掉秤快🔥', 'url': 'https://www.xiaohongshu.com/explore/691fc39e000000001b026efe', 'date': '2025-11-21 09:42:54', 'note_type': 'images', 'video': {'noteId': '691fc39e000000001b026efe', 'url': ''}, 'images': [], 'like_count': '193', 'collect_count': 246, 'view_count': '3585', 'comments': [], 'confidence': -1}, {'id': 'turn_0_87', 'search_from': 'search_note', 'content': '中焦不通太折磨！坚持 3 个月艾灸，我终于摆脱上热下冷的困境\n朋友们，上热下冷的体质真的太难受了😫 稍微吃点温热的食物就容易上火，可腿和脚却始终冰冰凉，一碰到冷风就扛不住。其实，这多半是 “中焦不通” 在作祟！而且往往还会跟着湿气重，导致身体容易发胖。好在我找到了艾灸治理的办法，坚持 3 个月后成功打通中焦，现在就把经验分享给大家👇\n采用“灸三停一”循环模式，即连续艾灸3天，休息1天后重复，每个位置每次艾灸15分钟\n✨首先:中脘，神阙，关元(治理中焦脾胃与下焦元气)\n✨然后:足三里，丰隆，三阴交(侧重健脾祛湿、调和气血）\n✨最后:太溪，涌泉(专注滋养肾水、引火下行)\n需要注意艾灸前、后需饮用适量温水，及时补充身体水分，避免津液流失，注意保暖，避免吹风、接触冷水，防止寒气入侵\n我当时就是按照这三组，每天交替艾灸，坚持了大概三个月，效果真的特别明显！现在腰以下的部位暖和多了，再也不怕冷，舌苔也变得红润健康，整个人都充满精神💪\n如果你也是上热下冷的体质，不妨试试这个艾灸方法，说不定也能轻松摆脱中焦不通的烦恼！记得点赞收藏，要是有疑问，欢迎在评论区留言和我交流～#艾灸[话题]# #养生[话题]# #艾灸调理[话题]# #上热下寒引火下行[话题]# #中医养生[话题]# #穴位养生[话题]# #疏通经络对人体的好处[话题]# #上热下寒体质[话题]# #五脏六腑的功能[话题]# #经络疏通[话题]#', 'title': '打通中焦、真的瘦了', 'url': 'https://www.xiaohongshu.com/explore/68d349eb000000000e0320e4', 'date': '2025-09-24 18:03:51', 'note_type': 'images', 'video': {'noteId': '68d349eb000000000e0320e4', 'url': ''}, 'images': [], 'like_count': '1112', 'collect_count': 1409, 'view_count': '33157', 'comments': [], 'confidence': -1}, {'id': 'turn_0_89', 'search_from': 'search_note', 'content': '分享倪海厦老师排出陈年痰湿淤三个穴位，大家收藏好~\n[彩虹R]三大穴位:\n阴陵泉-祛湿\n丰隆穴-化痰\n血海穴-淤血\n[彩虹R]艾灸注意事项:\n[一R]一般穴位悬灸不要烫，温和舒适是基础。\n[二R]不要玩手机，聚精会神享受艾灸。\n[三R]艾灸过程，如果口干舌燥，补充滋阴汤水！\n#艾灸[话题]##穴位[话题]##倪海厦[话题]##湿气[话题]##祛湿[话题]##养生[话题]##健康[话题]#', 'title': '听劝！灸透这三大穴位排出陈年痰湿淤', 'url': 'https://www.xiaohongshu.com/explore/6810e9e7000000002001daa5', 'date': '2025-04-29 23:01:59', 'note_type': 'images', 'video': {'noteId': '6810e9e7000000002001daa5', 'url': ''}, 'images': [], 'like_count': '645', 'collect_count': 875, 'view_count': '21194', 'comments': [], 'confidence': -1}, {'id': 'turn_0_90', 'search_from': 'search_note', 'content': '咱们身体要是有上热下寒的情况，就像交通堵啦！开pi胃是疏通 “中间路”，让营养能好好运；降心火给 “上头” 降降温；补肾水给 “下边” 供供暖，三路一起使劲，寒热平衡了，身体就得劲咯～\n📅 艾灸安排：\n✅ 第一天：中脘、神阙、关元，唤醒pi胃能量\n✅ 第二天：足三里、丰隆、三阴交，祛湿又健pi\n✅ 第三天：太溪、涌泉，补足肾水，温养下肢\n💡 操作要点：\n每个穴位灸10-15分钟，灸3天停1-2天，循环调理~灸前灸后记得喝温水，别让寒气找上门❄️\n#穴位养生[话题]# #艾灸[话题]# #中医养生[话题]# #排湿气寒气[话题]# #灸出好气色[话题]# #养生日常[话题]# #一起来养生[话题]# #三伏灸🔥[话题]# #变美的尽头是养生[话题]# #养脾胃[话题]#', 'title': '中焦不通月卑胃差这样艾灸', 'url': 'https://www.xiaohongshu.com/explore/6890769400000000230330eb', 'date': '2025-08-04 17:00:04', 'note_type': 'images', 'video': {'noteId': '6890769400000000230330eb', 'url': ''}, 'images': [], 'like_count': '1047', 'collect_count': 1251, 'view_count': '26435', 'comments': [], 'confidence': -1}, {'id': 'turn_0_94', 'search_from': 'search_note', 'content': '夏天想美美的，身上肉肉却各种“绊马索”，一整个大emo住😫，挖到艾灸这宝藏法子，针对不同部位肉肉，选对穴位灸一灸，想哪里变窄就变哪，三伏天buff叠满，变窄体质拿捏住💪\n\t\n🔥窄肚子（pi胃弱、痰湿堆堆党看过来）\n肉肉爱囤肚子？中脘、关元、天枢穴安排上！\n✅中脘穴：上腹部前正中线，脐上4寸（胃的募穴），帮消化，不让脂肪“赖”在肚子\n✅关元穴：下腹部前正中线，脐下3寸 ，给代谢加buff，“烧”掉腹部脂肪\n✅天枢穴：腹部平脐，前正中线旁开2寸（大肠募穴），调顺肠道，小腹偷偷变平坦\n💡操作：每穴灸15 - 20分钟，每周3 - 4次 ，坚持灸，肚子缩水超惊喜~\n🔥细大腿（湿气重、气血不畅别慌）\n大腿肉肉松垮、水肿？足三里、三阴交、丰隆穴来救场！\n✅足三里：小腿外侧，犊鼻下3寸 ，调脾胃，给大腿“变窄”助攻\n✅三阴交：小腿内侧，内踝尖上3寸 ，理顺气血，水肿退散，大腿线条up\n✅丰隆穴：小腿外侧，外踝尖上8寸 ，猛猛祛湿，大腿慢慢变纤细\n💡操作：每穴灸10 - 15分钟，每周2 - 3次 ，穿短裤自信炸街不是梦✨\n🔥甩拜拜肉（气血阻滞、手臂肉肉顽固）\n穿无袖暴露“拜拜肉”？曲池、内关、肩髃穴安排！\n✅曲池穴：肘横纹外侧端（屈肘找凹陷），加速代谢，肉肉甩光光\n✅内关穴：前臂掌侧，腕横纹上2寸 ，循环拉满，手臂悄悄变紧致\n✅肩髃穴：肩部三角肌上（臂外展时的凹陷），通开经络，赘肉退！退！退！\n💡操作：每穴灸10 - 15分钟，每周3 - 4次 ，无袖美衣随便穿~\n🌿小贴士\n艾灸温度别太高，防烫到自己；灸完别立刻洗澡、吃生冷，让身体好好“吸收”能量✔\n三伏天可是老天爷赏的“掉秤窗口期”，按穴位精准灸，变窄体质直接焊在身上💥 （亲测有用！评论区交出你们的三伏天艾灸战绩！）\n#穴位养生[话题]# #中医养生[话题]# #艾灸[话题]# #灸出好气色[话题]# #减肥[话题]# #排湿气寒气[话题]# #养生日常[话题]# #一起来养生[话题]# #变美的尽头是养生[话题]# #三伏天灸[话题]#', 'title': '三伏天黄金期艾灸🔥 精准拿捏变窄体质', 'url': 'https://www.xiaohongshu.com/explore/6891d5920000000023039d17', 'date': '2025-08-05 17:57:38', 'note_type': 'images', 'video': {'noteId': '6891d5920000000023039d17', 'url': ''}, 'images': [], 'like_count': '1152', 'collect_count': 1258, 'view_count': '28420', 'comments': [], 'confidence': -1}]
        input_dict["judge_turn_1"] = {'rating': 7.0, 'justification': '该大纲结构清晰，逻辑性强，尤其在第三部分提出的“先疏肝，再健脾，后祛湿”的调理次序，体现了深刻的洞察力。大纲在深度方面表现出色，不仅对“湿气”进行了多维度的分类（湿热、寒湿、痰湿），还针对不同类型提供了具体的饮食和艾灸方案，包括食材、食谱、穴位等，具有很强的可操作性。同时，大纲规划了详细的证据支撑（<cite>标签），遵循了指令要求。\n\n然而，大纲存在两个主要缺陷，导致其无法获得更高分数：\n1.  **平衡性与广度严重不足**：大纲完全局限于中医理论框架，完全忽略了现代医学的视角。对于“身体沉重、疲劳、水肿、消化不良”等症状，现代医学有其自身的诊断体系（如甲状腺功能减退、慢性疲劳综合征、肠易激综合征等）。一份高质量的“综合报告”应至少包含与现代医学概念的对比分析或鉴别诊断部分，以体现其平衡性和广度。完全单方面的论述使其客观性大打折扣。\n2.  **广度仍有欠缺**：即便在在中医体系内，大纲也遗漏了一些常见的、与祛湿高度相关的外部疗法，如拔罐、刮痧等，这些疗法在祛湿方面的应用非常广泛。此外，“运动”部分的建议过于笼统，未能像饮食和艾灸部分那样，根据不同体质类型提供更具针对性的运动强度和方式建议，削弱了该部分的深度和实用性。', 'search_query': ['祛湿 拔罐 穴位', '刮痧 祛湿 经络', '脾虚 运动选择 禁忌', '湿热体质 运动推荐', '中医 湿气 现代医学 解读', '肠易激综合征 饮食调理', '甲状腺功能减退 浮肿', '慢性疲劳综合征 自我管理', '四神汤 做法 功效', '祛湿食谱 一周不重样', '疏肝解郁 按摩穴位', '艾灸祛湿 禁忌人群']}
        input_dict["search_query_turn_1"] = ['祛湿 拔罐 穴位', '刮痧 祛湿 经络', '脾虚 运动选择 禁忌', '湿热体质 运动推荐', '中医 湿气 现代医学 解读', '肠易激综合征 饮食调理', '甲状腺功能减退 浮肿', '慢性疲劳综合征 自我管理', '四神汤 做法 功效', '祛湿食谱 一周不重样', '疏肝解郁 按摩穴位', '艾灸祛湿 禁忌人群']
        input_dict['search_result_turn_1'] = [{'id': 'turn_1_0', 'search_from': 'search_note', 'content': '阴陵泉\n位置：小腿内侧，胫骨内\n侧髁后下方凹陷处\n作用：拔罐阴陵泉穴，可\n化湿通阳\n方法：选择大小适宜的火罐，在阴陵泉穴处吸\n拔，留罐10分钟左右\n曲池穴\n位置：屈肘成直角，当肘\n弯横纹尽头处\n作用：刮痧曲池穴，可清热活络\n方法：用面刮法刮拭曲池\n穴，以出痧为度，至痧退\n后再刮第二次\n三阴交\n位置：在小腿内侧，\n当足\n内踝尖上3寸，胫骨内侧\n缘后方\n作用：具有健脾,和胃,化\n湿的功效\n方法：用拇指指腹揉按\n穴位5分钟\n丰隆穴\n位置：外踝尖上8寸，胫骨前肌外缘\n作用：拔罐丰隆穴，可化\n痰袪湿\n方法：选择大小适宜的火罐，在丰隆穴处吸拔，留\n罐10分钟左右\n胃俞六\n位置：位于背部，当第12\n胸椎棘突下，旁开1.5寸\n作用：按摩胃俞穴，可健脾助运，缓解湿热引起的\n不适\n方法：用拇指指腹按揉\n胃俞穴3~5分钟，以皮肤产生酸、麻、帐感觉为佳。\n脾俞穴\n位置：位于背部，，第11\n胸椎棘突下，旁开1.5寸。\n作用：按摩脾俞穴，可健脾利湿\n方法：用拇指指腹按揉脾\n俞穴3~5分钟，以皮肤产生酸、麻、胀感觉为佳。#节气养生[话题]# #中医健康养生知识[话题]# #中医养生[话题]# #健康养生[话题]#', 'title': '湿气重 牢记这6个穴位', 'url': 'https://www.xiaohongshu.com/explore/6667ed28000000000e032645', 'date': '2024-06-11 14:22:32', 'note_type': 'images', 'video': {'noteId': '6667ed28000000000e032645', 'url': ''}, 'images': [], 'like_count': '1815', 'collect_count': 1766, 'view_count': '82065', 'comments': [], 'confidence': -1}, {'id': 'turn_1_1', 'search_from': 'search_note', 'content': '🌱你也经常被shi气“附体”吗？脸上、头发总是出油，早上睡不醒，状态差，今天给大家分享去shi“开关”！👇🏻\n\t\n1️⃣曲池\n📍屈肘成直角，肘横纹外侧端凹陷处，摸到凸起的肱骨外上髁，与内侧尺泽穴连线的中点就是它！\n\t\n2️⃣脾俞\n📍正坐低头，肩胛骨下角平第7胸椎，向下数4个椎体至第11胸椎，旁开1.5寸（约两横指），竖脊肌高点处轻按即得～\n\t\n3️⃣胃俞\n📍第12胸椎棘突下凹陷处，旁开1.5寸。肩胛骨下角平第7胸椎，垂直向下推5个椎体。\n\t\n4️⃣丰隆\n📍外踝尖上8寸（约小腿一半长度），胫骨前缘外侧两横指，条口穴旁开一横指处，酸/月长/感明显～\n\t\n5️⃣阴陵泉\n📍小腿内侧，胫骨内侧下缘凹陷处，比目鱼肌起点上方，大隐静脉旁轻按即现！\n\t\n6️⃣三阴交\n📍足内踝尖上3寸（四指并拢宽度），胫骨后缘凹陷处，三条阴经交汇的“黄金点”！\n\t\n#湿气重[话题]# #如何祛湿[话题]# #祛湿[话题]# #中医科普[话题]# #健康养生[话题]# #穴位养生[话题]# #身体去湿气[话题]# #中医养生[话题]# #排湿气寒气[话题]# #穴位图[话题]#', 'title': '🌿湿气重，记住这6个“开关”｜码住❗️', 'url': 'https://www.xiaohongshu.com/explore/68a878da000000001d006567', 'date': '2025-08-23 09:02:24', 'note_type': 'images', 'video': {'noteId': '68a878da000000001d006567', 'url': ''}, 'images': [], 'like_count': '51', 'collect_count': 39, 'view_count': '2527', 'comments': [], 'confidence': -1}, {'id': 'turn_1_3', 'search_from': 'search_note', 'content': '去湿气的8个小方法\n[一R]适当拔罐\n[二R]三伏天晒背\n[三R]五指毛桃茯苓茶\n[四R]运动\n[五R]艾灸\n[六R]泡脚\n[七R]饮食清淡\n[八R]按穴位\n我自己是比较懒的，给姐妹们分享简单又能坚持的方法大家都可以用起来\n图（9）是湿气自测，看看你的湿气在第几级，可以下方告诉我，给你调整方案，一起做个无湿美少女✌🏻\n冠🐷盼盼，我会持续给大家分享祛湿，脾胃，气🩸，体zhi调理干货，一起养出健康美\n#湿气[话题]# #湿气重[话题]###湿气重喝什么[话题]##湿气排出来的表现[话题]##女人健康养生[话题]##中医健康养生知识[话题]##舌诊[话题]# #手诊[话题]#', 'title': '去除湿气的8个快方法‼️有效好用✊', 'url': 'https://www.xiaohongshu.com/explore/684556fa000000000303c7a5', 'date': '2025-06-08 17:25:14', 'note_type': 'images', 'video': {'noteId': '684556fa000000000303c7a5', 'url': ''}, 'images': [], 'like_count': '652', 'collect_count': 309, 'view_count': '72932', 'comments': [], 'confidence': -1}, {'id': 'turn_1_9', 'search_from': 'search_note', 'content': '💡先看这份体质对照表！自测你是哪种？\n（建议先收藏⭐️反复看）\n▫️【冰雪公主】手脚冰凉/姨妈痛/拉肚子👉艾灸\n▫️【油腻girl】头脸出油/口臭/变㊙️👉刮痧\n▫️【堵车达人】肌肉酸/水肿/皮肤痒👉拔罐\n▫️【又冷又油】上热下寒/反复上火👉先刮后灸\n⚠️血泪教训：\n❌拔罐越紫≠毒素越多（其实是毛细🩸管破裂）\n❌刮痧越痛≠效果越好（可能刮伤经络）\n❌艾灸越烫≠祛湿越强（低温常灸才有效）\n💯正确👍方式：\n1️⃣ 拔罐后8小时别洗澡\n2️⃣ 刮痧后喝温蜂蜜水（排du速度×2）\n3️⃣ 艾灸后搓脚心100下（引火下行是重点）\n🌟年度王炸组合：\n☀️夏季：刮痧（每周1次）+金银花茶\n❄️冬季：艾灸（隔天1次）+肉桂煮苹果\n🌧梅雨季：拔罐（半月1次）+赤小豆薏米水\n🔥常见问题🙋QA：\nQ：能天天艾灸吗？\nA：NO！阴虚人会灸出口腔溃疡！\nQ：刮痧出痧少=没湿气？\nA：可能是气血不足刮不动！\nQ：拔罐能减肥吗？\nA：暂时消水肿可以，管不住嘴照样胖！\n#中医科普[话题]# #刮痧[话题]# #穴位养生[话题]# #灸出好气色[话题]# #排湿气寒气[话题]# #拔罐[话题]# #艾灸[话题]# #中医养生[话题]# #治疗[话题]#', 'title': '🔥拔罐/刮痧/艾灸都是祛湿，到底怎么选？', 'url': 'https://www.xiaohongshu.com/explore/68b158d5000000001d0117bb', 'date': '2025-08-29 15:37:57', 'note_type': 'images', 'video': {'noteId': '68b158d5000000001d0117bb', 'url': ''}, 'images': [], 'like_count': '178', 'collect_count': 98, 'view_count': '28343', 'comments': [], 'confidence': -1}, {'id': 'turn_1_11', 'search_from': 'search_note', 'content': '#艾灸祛湿[话题]#\n【湿气有两种】\n1.一种是外界寒湿侵袭人体导致的\n2.另一种是脾气运化不调导致的内湿。\n对于两种湿气，艾灸对外湿有很好的去湿气效果。\n【艾灸去湿】\n-艾灸主要的作用就是温、通、补，温热的效应就可以起到去寒气的作用，可以温补调和脾胃。\n-脾胃功能、运化能力强，身体里的湿就不会停聚和聚集，新陳代谢正常后，湿气也会慢慢去滁，因此艾灸可以去身体里的寒湿。\n【5个祛湿穴位】\n[一R]中脘穴\n缓解胃部不适的症状，有助于祛湿\n\t\n[二R]关元穴\n养生保健穴位，有助于调里气血、补（shen）固精，帮助祛湿。\n\t\n[三R]足三里穴\n祛湿较好的穴位，坚持艾灸段时间，可明显看到祛湿效果。\n[四R]丰隆穴 胃经、脾经水湿浊气汇集之穴，长按消痰浊。\n\t\n[五R]解溪穴\t全身祛痰湿的穴位，解除下肢。\n\t\n#穴位养生[话题]# #体质[话题]# #夏季养生[话题]# #我的养生仪式感[话题]# #艾灸养生[话题]# #艾灸祛湿排寒通经络[话题]# #艾灸祛湿减肥[话题]# #穴位[话题]# #艾灸贴减肥[话题]#', 'title': '多年湿气相伴❓灸这人体祛湿5大穴位！', 'url': 'https://www.xiaohongshu.com/explore/66c8aa4f000000001d01574a', 'date': '2024-08-23 23:27:11', 'note_type': 'images', 'video': {'noteId': '66c8aa4f000000001d01574a', 'url': ''}, 'images': [], 'like_count': '1579', 'collect_count': 1448, 'view_count': '85580', 'comments': [], 'confidence': -1}, {'id': 'turn_1_14', 'search_from': 'search_note', 'content': '夏季阳气☀️旺盛，人体的气血也充盈，是拔火罐的好时机，不少人一到夏天就会忍不住的拔火罐。但罐印其实隐藏着很多求救信号，大家可以自测自检：\n淡粉/轻度潮红\n艳红鲜红\n红黑暗红\n灰白\n紫+斑块\n紫黑暗紫\n皮纹/微痒\n水泡\n·\n不同的罐印反映不同的健康状况\n这些印记如何解读❓\n·\n✨ 无明显罐斑/轻度潮红且消失快：无病或病情轻微\n·\n✨ 鲜红而艳：阴虚，气血两虚或阴虚火旺\n小方案：\n气血两虚，食养红枣粥；阴虚火旺，按揉然谷穴2~3分钟/次\n·\n✨ 红而暗：血脂高，血黏稠度高，且有热邪\n小方案：\n决明绛栀茶-解腻茶（配方源于苏州市卫生局《🥼院制剂手册》记载的「降脂合剂」）\n穴位按揉天枢穴和阴陵泉穴位，2-3分钟/次\n·\n✨ 灰白，触之不温：多为虚寒和湿邪\n小方案：\n虚寒，参桂姜枣茶-暖阳茶（配方源自明代孙志宏《简明医彀》记载的「保元汤」）\n湿邪，参苓薏仁茶-无湿茶（配方源自宋代官修《太平惠民和剂局方》记载的「参苓白术散」）\n穴位按揉曲池穴、阴陵泉、丰隆穴，可以散寒祛湿，2-3分钟/次\n·\n✨ 紫黑而暗：供血不足，行经不畅，有气滞血淤之象\n小方案：\n拉筋拍打可以疏通经络，道路通畅，气滞血瘀的情况就会改善。先拍打通用部位，然后拍打四肢，后背脊柱，蕞好全身地毯式轰炸，可用通痹锤做敲打，因为配方独特，用的是明代医著《症因脉治》卷三收载的“家秘羌活汤”方，疏通经络与服💊香并举更好\n.\n✨ 发紫，有斑块：寒凝血瘀症\n小方案：\n寒凝血瘀，参桂姜枣茶-暖阳茶（配方源自明代孙志宏《简明医彀》记载的「保元汤」）\n穴位配合曲池、合谷、血海、三阴交、太冲，都是活X的穴位，2-3分钟/次\n·\n✨ 有皮纹或微痒：风邪或湿症\n风邪，穴位可按揉风府、风池、风市穴，都是去风的穴位，2-3分钟/次\n湿邪，参苓薏仁茶-无湿茶（配方源自宋代「参苓白术散」）\n.\n✨ 罐体内壁有水气:该部位湿气\n湿邪，参苓薏仁茶-无湿茶（配方源自宋代「参苓白术散」）\n穴位可以选择阴陵泉、水分穴、足三里穴，都是祛湿的穴位，2-3分钟/次\n·\n✨ 水泡：体内湿气重，如水泡内有血水，是湿热重的反映\n湿气重，参苓薏仁茶-无湿茶（配方源自宋代「参苓白术散」）\n如果是湿热重，参苓薏仁茶-无湿茶+决明绛栀茶-解腻茶一起喝\n.\n以上就是罐印所暗示的身体的秘密，和对症小方案，记得要自查哦~\n不清楚的同学可以留言➕找我们\xa0@正执新汉方\n#拔火罐[话题]#\xa0\xa0#正执新汉方养生茶[话题]#\xa0\xa0#正执新汉方[话题]#\xa0\xa0#参苓薏仁茶[话题]#\xa0\xa0#刮痧[话题]#', 'title': '拔🔥罐后的罐印都暗示了什么？', 'url': 'https://www.xiaohongshu.com/explore/66b435cd000000001e0197b2', 'date': '2024-08-09 09:00:48', 'note_type': 'images', 'video': {'noteId': '66b435cd000000001e0197b2', 'url': ''}, 'images': [], 'like_count': '1003', 'collect_count': 520, 'view_count': '93821', 'comments': [], 'confidence': -1}, {'id': 'turn_1_19', 'search_from': 'search_note', 'content': '如果你总感觉浑身没劲、吃点就胀、脸色蜡黄……别急着怪自己“体质差”，这是脾胃在对你呐喊：“你的活法，该撕碎重来了！”中医说“脾为后天之本”，但它不是枷锁，而是你觉醒的开关。\n\t\n《黄帝内经》讲“思伤脾”，那些让你腹胀、脸黄、便秘的元凶，其实是焦虑、内耗和麻木的生活。\n\t\n一、中医说：脾虚的人，骨子里都憋着一团火\n\t\n• 脾主运化：它像身体的“物流中心”，把食物变成气血能量\n• 脾虚的本质：长期吃外卖、熬夜刷手机、憋着情绪不发泄……这些都在让脾胃“超负荷加班”，直到它彻底宕机。\n“脾虚不是病，是身体在逼你换活法”—— 你吃进去的每一口外卖，熬的每一次夜，都是对脾胃的慢性谋杀。\n\t\n▌你的“累”是求救信号\n• 气血不足：脾虚导致气血化生无力，人就像断电的手机，明明躺着也累到虚脱。\n• 湿气淤堵：脾虚运化不了水湿，身体像泡发的海绵，浮肿、舌苔厚腻、大便黏马桶。\n中医暴言：“脾虚的人，不是缺营养，是缺一场造反！”\n\t\n▌吃：像农民一样吃饭\n• 早餐：焦香的小米南瓜粥（小米炒到微焦，暖脾力翻倍），中医说焦香入脾，是唤醒脾胃的“冲锋号”。\n• 零食：烤红薯、炒瓜子，用大地馈赠的温热，把寒湿逼出体外。\n• 禁忌：戒冰饮、戒重口味，你的脾胃不是垃圾桶。\n\t\n▌动：像孩子一样撒野\n• 晨起踩草地：赤脚踩泥土，刺激脚底脾经，湿气从涌泉穴溜走。\n• 八段锦“单举”：两臂一上一下对拉，把淤堵的脾气吼出来（亲测打嗝超爽！）。\n\t\n▌情绪：把委屈吼给风听\n• 揉腹泄洪法：睡前顺时针揉腹100圈，把胀气、怨气揉成一声响亮的嗝。\n• 发疯式疏肝：去山顶大喊、写日记骂人、跳广场舞……“脾虚的人，情绪越疯，脾胃越嗨！”\n\t\n3年前的我：\n❌ 每天靠冰美式续命，加班到凌晨\n❌ 情绪内耗，焦虑到胃痉挛\n❌ 脸色蜡黄，被同事问“是不是贫血”\n3年后的我：\n✅ 早起喝姜枣茶，晒太阳到后背发烫\n✅ 周末去农场摘菜，吃得像个“土妞”\n✅ 不爽就怼，哭完大笑，脾胃反而越作越强\n\t\n[飞吻R]我的一些养脾清单（亲测有效🔥）：\n• 早餐：四神汤（茯苓+山药+莲子+芡实）煮到黏稠，撒一把烤馒头丁\n• 穴位：足三里狠狠艾灸，痛到流泪也要按\n• 精神食粮：每天对镜子吼“去他的虚弱！”\n\t\n坚持3个月，你会收获：红润脸蛋、放肆干饭自由，和一个健康的身体。\n#养生[话题]# #脾胃[话题]# #养脾胃[话题]# #100天运动打卡挑战[话题]#', 'title': '中医说：脾虚的人，骨子里都憋着一团火', 'url': 'https://www.xiaohongshu.com/explore/6826fab0000000001101fdc9', 'date': '2025-05-16 16:43:28', 'note_type': 'images', 'video': {'noteId': '6826fab0000000001101fdc9', 'url': ''}, 'images': [], 'like_count': '1732', 'collect_count': 1662, 'view_count': '30128', 'comments': [], 'confidence': -1}, {'id': 'turn_1_23', 'search_from': 'search_note', 'content': '脾胃与湿气关系密切\n每天坚持6点，脾越来越强壮，湿越来越少\n--\n👉1.\t养成运动的习惯\n坚持运动，每天30分钟，每周运动次数不低于5次，可以选择慢跑、瑜伽、打羽毛球或者气排球等\n👉2.\t坚持吃早餐\n九点钟之前吃早餐，无论多忙，也要腾出吃早餐的时间。可以吃馒头、玉米、小米粥、或者用山药+芡实+茯苓+莲子，打成豆浆，当做早餐来喝\n👉3.\t三餐饮食有规律\n不仅要坚持吃早餐，其他的两餐也要吃。不暴饮暴食，每餐吃七八分饱即可，少吃或者不吃生冷寒凉的食物。如螃蟹、西瓜、奶茶等\n👉4.\t坚持泡脚\n每天晚上九点钟左右泡个热水脚，泡20分钟左右，泡完脚之后尽快把脚擦干，保暖好足部，不要去吹风受凉\n👉5.\t坚持吃养脾祛湿的食物\n坚持吃养脾祛湿的食物，如薏米、山药、白扁豆、红豆、茯苓、芡实、小米等\n👉6.\t按时休息不熬夜\n调整作息，按时休息不熬夜，每天晚上十点半放下手机，为进入睡眠做准备\n--\n注：内容仅供参考\n\t\n#健脾祛湿[话题]# #湿气[话题]# #脾虚湿气重[话题]#', 'title': '每天坚持6点，脾越来越强壮，湿越来越少', 'url': 'https://www.xiaohongshu.com/explore/64103d080000000027012bef', 'date': '2023-03-14 17:23:20', 'note_type': 'images', 'video': {'noteId': '64103d080000000027012bef', 'url': ''}, 'images': [], 'like_count': '316', 'collect_count': 323, 'view_count': '15702', 'comments': [], 'confidence': -1}, {'id': 'turn_1_28', 'search_from': 'search_note', 'content': '八尾帮大家总结了湿热体质的养生攻略！还是非常详细，大家可以码住，结合的教材文献有《黄帝内经》《中Y基础学》《中医基础理论》《中Y药膳学》《公共营养师-基础知识》等内容。\n#00:18[时刻]# 大纲\n#00:32[时刻]# 湿热的定义\n#00:45[时刻]# 黄帝内经中的定义\n#01:34[时刻]# 具体表现：面部\n#02:10[时刻]# 嘴舌😛\n#02:45[时刻]# 二便\n#03:01[时刻]# 体感\n#03:23[时刻]# 零件\n#03:58[时刻]# 湿\n#04:04[时刻]# 外湿\n#04:21[时刻]# 内湿\n#04:53[时刻]# 实热\n#05:10[时刻]# 虚热\n#05:32[时刻]# 营养素缺乏\n#05:50[时刻]# 膳食纤维\n#06:00[时刻]# B族维生素\n#06:17[时刻]# 钾镁\n#06:41[时刻]# 抗🐑化\n#07:23[时刻]# 调理原则\n#07:29[时刻]# 早上：健脾胃清🔥\n#07:53[时刻]# 吃什么\n#08:46[时刻]# 其他：运动、茶水、站桩\n#10:12[时刻]# 中午：（清🔥、湿）\n#10:24[时刻]# 吃什么\n#11:29[时刻]# 其他：VB、午休、防晒➕保湿\n#13:12[时刻]# 下午茶：疏🈷️干➕清🔥\n#14:20[时刻]# 晚上：健P➕去💦\n#14:27[时刻]# 吃什么：五行散\n#16:10[时刻]# 放松：音乐➕泡脚\n#16:33[时刻]# 泡脚：湿之汤\n#16:54[时刻]# 睡觉：22点之前\n#17:20[时刻]# 刮痧：膀胱经\n#17:39[时刻]# 穴位\n#18:06[时刻]# 食养\n#19:18[时刻]# 四季养生\n#湿热[话题]# #湿热质[话题]# #湿热体质[话题]# #湿热下注[话题]# #湿热调理[话题]# #脾胃[话题]# #跟着书籍学养生[话题]# #健康养生[话题]# #养生[话题]# #读懂身体语言[话题]#', 'title': '湿热体质全攻略❗️一张纸讲透湿热体质养生法', 'url': 'https://www.xiaohongshu.com/explore/68ee4b4e0000000005002c76', 'date': '2025-10-15 07:02:00', 'note_type': 'video', 'video': {'noteId': '68ee4b4e0000000005002c76', 'url': ''}, 'images': [], 'like_count': '3295', 'collect_count': 2462, 'view_count': '36417', 'comments': [], 'confidence': -1}, {'id': 'turn_1_34', 'search_from': 'search_note', 'content': '姐妹们谁能懂啊？！纯是大震惊！以前总以为自己是“易月半体质”，被濕气缠了好多年，直到調理后才发现，原来瘦下来这么轻松——两个月怒減27斤，从臃肿“泡发体”变身走路带风的瘦子，连皮肤都亮到发光\n以前，我就是个实打实的“痰濕受害者”。身材虚月半得像🎈，肉松、软趴趴的，尤其是肚子，像是带个游泳圈，穿啥都显臃月中，每次照镜子都失望叹气，头发得一天一洗，皮肤还油得发光，整天昏昏沉沉，爬三层楼梯都喘得不行，浑身无力，连情绪都跟着低落\n现在，我实现了华丽转身！肚子平坦了，皮肤变得白里透红的，整个人轻松又自信，精神饱满，连走路都带风\n👇下面就把我亲测有效的祛濕瘦身法分享给大家，简单易操作，新手也能直接冲！\n✅ 蒸苹果当早餐：\n早晨空腹来一个蒸苹果太绝了！软糯香甜还能帮肠道“大扫除”，既能健睥养胃，又能cu进排便，帮身体清垃圾，坚持一周感觉消化改善\n✅主食换馒头超养人：\n五谷是养s根本，把精米白饭换成馒头，松软好消化，睥胃越养越给力。气皿足了之后，脸上的黄气慢慢退了，脸上透出健康好气色\n✅ 陈皮养生茶助内调：\n每天煮陈皮茶，加茯苓、玫瑰就是疏月干健气茶，配雪梨、红枣就是清甜潤燥汤，喝着清爽不腻，健睥养胃、除濕化痰，还能悄悄抑制食欲，不知不觉就少吃半碗饭！掉秤效果明显\n✅ 饭后站半小时：\n吃完就坐或躺下真的容易堆小腹噢，现在我雷打不动站半小时，刷手机、听音乐，简单又有效，悄悄摘下“游泳圈”\n✅ 睡前泡脚逼出濕气：\n水温控制在42℃左右，泡15分钟至后背微微出汗，不仅能祛濕散寒，还能助眠，现在沾床不一会儿就睡着了，再也没有睡不着的烦恼\n✅ 揉肚子+推腹消大肚：\n早、晚5分钟，先顺时针揉21圈促消化，再逆时针揉21圈养睥胃，最后沿肚脐向下轻推21次排浊，坚持下来促进肠胃蠕动，肠胃消化功能渐强，肚子也逐渐变小～\n✅ 轻度运动通經络：\n不用跑马拉松，每天坚持练15分钟八段锦，晚饭后散步半小时（千万别吃完就坐下或躺下），微微出点汗有助于疏通經络，让湿气随汗排出，身体代谢逐渐改善\n✅心态调理也很重要\n压力大、纠结、焦虑的时候，就去公园散散步，约几好友唱唱歌，和闺蜜逛逛街听听舒缓的音乐，心情放松了，整个人也舒畅了，瘦身效果直接翻倍\n\t\n#祛湿[话题]# #调理脾胃#脾胃好了比啥都强[话题]# #养脾胃[话题]# #在小红书轻养生[话题]# #痰湿体质养生#脾胃好气血足[话题]##脾胃调理[话题]#', 'title': '我终于从痰湿体质变成健康体质❗️', 'url': 'https://www.xiaohongshu.com/explore/692beb10000000001e021223', 'date': '2025-11-30 14:58:24', 'note_type': 'images', 'video': {'noteId': '692beb10000000001e021223', 'url': ''}, 'images': [], 'like_count': '1259', 'collect_count': 1711, 'view_count': '30836', 'comments': [], 'confidence': -1}, {'id': 'turn_1_35', 'search_from': 'search_note', 'content': '早春古法健身祛湿操🔥真的太猛了[赞R]\n每天坚持做这🔟个动作，一个动作1️⃣分钟，每天2-3遍，排出身体湿气，疏肝解郁🌟补充阳气☀养出好气色❤快一起运动起来吧😊\n#古法健身[话题]# #坚持锻炼身体[话题]# #古法锻炼[话题]# #古法养生[话题]# #养生运动[话题]# #养气色[话题]# #祛湿排寒[话题]# #祛湿健脾[话题]# #小动作大作用[话题]# #锻炼[话题]#', 'title': '古法养生操｜🔟个动作祛湿排汗💦太牛了', 'url': 'https://www.xiaohongshu.com/explore/67c705e700000000290387eb', 'date': '2025-03-04 21:53:43', 'note_type': 'video', 'video': {'noteId': '67c705e700000000290387eb', 'url': ''}, 'images': [], 'like_count': '28084', 'collect_count': 39726, 'view_count': '448319', 'comments': [], 'confidence': -1}, {'id': 'turn_1_38', 'search_from': 'search_note', 'content': '救命！湿气重真的会让人变丑变懒😭\n我之前脸油得像反光板、头发刚洗就塌、\n小肚子鼓成球、整天困得像没睡醒…\n直到学了🀄️医才知道，原来都是湿气在搞鬼！\n之后咬牙坚持自律养生一个月，\n整个人从里到外焕然一新✨\n🌿首先明白湿气到底从哪来？\n湿气被称为“万恶之邪”\n✅外湿：长期待在潮湿环境（比如南方梅雨季）、淋雨没及时擦干、空调房温度过低…湿气就会从毛孔偷偷钻进身体！\n✅内湿：全是自己“作”出来的！熬夜、久坐不动、爱吃生冷（奶茶/冰淇淋）、重油重辣…脾胃功能变弱，无正常常运化水湿，湿气就越积越多！\n✅ 体质因素：气虚、阳虚人更容易被湿气缠上\n叁叁是如何逆袭清爽体质的？跟着做真的超有效！\n1.饮食自律，从“吃”掉湿气开始\n🀄️医讲究“💊食同源”，选对食材事半功倍\n✨祛湿黄金粥：赤小豆+薏米+芡实，煮成浓稠粥每天喝！薏米健脾祛湿，赤小豆利水消zhong，芡实固肾补脾，湿气重的姐妹闭眼喝～\n✨健脾神器：山要、茯苓、冬瓜都是天然祛湿食材！可以煮山要茯苓小米粥、冬瓜虾仁汤，好吃又养生～\n✨忌口清单：生冷食物（冰淇淋、生鱼片）、油炸食品、甜品，再馋也要忍住！\n2.作息自律\n✨早睡早起：\n✨蕞重要的是要学会午休，中午小憩20分钟，养心又养脾，下午工作更有精神了！\n3.运动自律\n✨每日必做：八段锦\n✨按摩腹部\n✨上班没时间运动的姐妹们，抽空可以靠墙站！每天15分钟，不仅能祛湿，还能改进体态～\n4.习惯自律\n细节决定成败！这些习惯坚持下来效果惊人👇\n✨ 泡脚\n✨ 穴位按摩：每天花5分钟按揉足三里（调理脾胃）、阴陵泉（利水祛湿）、丰隆穴（化痰祛湿），酸胀感越强果越好！\n✨ 远离空调房\n5. 保持好心情：\n\t\n祛湿避坑指南‼️\n1️⃣ 别过度节食：祛湿≠瘦身！盲目少吃会shang脾胃，反而加重湿气！\n2️⃣ 别三天打鱼：自律贵在坚持！偶尔放纵没关系，但别打乱整体节奏～\n坚持一个月后，你会发现脸不油了、头发蓬松了、小肚子也平了！整个人轻盈又有活力💃#身体去湿气[话题]# #寒湿是万病之源[话题]# #先健脾还是先祛湿[话题]# #排湿气寒气[话题]# #当代年轻人养生日常[话题]#', 'title': '一个月和湿气说拜拜👋', 'url': 'https://www.xiaohongshu.com/explore/682915870000000022025a74', 'date': '2025-05-18 07:02:31', 'note_type': 'images', 'video': {'noteId': '682915870000000022025a74', 'url': ''}, 'images': [], 'like_count': '9007', 'collect_count': 7041, 'view_count': '241168', 'comments': [], 'confidence': -1}, {'id': 'turn_1_41', 'search_from': 'search_note', 'content': '#00:02[时刻]# 本视频大纲\n#00:22[时刻]# 定义\n#00:34[时刻]# 5大表现\n#02:06[时刻]# 如何产生的？\n#04:16[时刻]# 2幅灵魂画作帮助理解\n#05:32[时刻]# 如何缓解和改善呢？\n#不懂就问有问必答[话题]# #湿气自我打分表[投票]#', 'title': '中医研究生｜“湿气”究竟是个什么玩意儿❓', 'url': 'https://www.xiaohongshu.com/explore/657c296100000000090237ed', 'date': '2023-12-15 18:24:33', 'note_type': 'video', 'video': {'noteId': '657c296100000000090237ed', 'url': ''}, 'images': [], 'like_count': '5252', 'collect_count': 3626, 'view_count': '158766', 'comments': [], 'confidence': -1}, {'id': 'turn_1_42', 'search_from': 'search_note', 'content': '🔍 第一步：辩证5大“湿气体质”\n❶ 寒湿体质—— “冰箱型” 🧊特征：舌苔白腻、怕冷、手脚凉、爱拉肚子、分泌物清稀\n❷ 湿热体质—— “蒸笼型” 🔥特征：舌苔黄腻、脸油长痘、口苦口干、大便粘马桶、异味重\n❸ 痰湿体质—— “淤泥型” 🛶特征：舌苔厚腻、肥胖浮肿、胸闷痰多、总觉得昏沉嗜睡\n❹ 暑湿体质—— “桑拿型”（季节性） ☀️特征：夏季特有，头昏沉、身热不畅、胸闷没胃口、口渴不想喝\n❺ 风湿体质—— “天气预报型” 💨特征：关节酸痛游走、遇阴雨天加重、皮肤易长湿疹\n🛠 第二步：掌握8大“祛湿段位” （针对不同湿气，组合出招）\n1️⃣ 芳香化湿—— 对付闷在头面的湿\n2️⃣ 苦温燥湿—— 对付寒湿困住脾胃\n3️⃣ 清热利湿—— 对付湿热交织\n4️⃣ 淡渗利湿—— 通用排水法\n5️⃣ 健脾化湿—— 治本大法\n6️⃣ 祛风胜湿—— 对付风湿在表\n7️⃣ 温阳化湿—— 对付寒湿深入\n8️⃣ 化痰祛湿—— 对付顽固痰湿\n📝 核心搭配公式（快记笔记！）\n✔️ 寒湿 = 苦温燥湿 + 温阳化湿 + 健脾化湿\n✔️ 湿热 = 清热利湿 + 淡渗利湿 + 芳香化湿\n✔️ 痰湿 = 化痰祛湿 + 健脾化湿 + 理气（陈皮）\n✔️ 风湿 = 祛风胜湿 + 根据寒热搭配\n✔️ 暑湿 = 芳香化湿 + 清热利湿 + 淡渗利湿\n💡 下期预告\n“湿气定位指南——三焦&脏腑篇！湿气堵在：\n👉 上焦（心肺）：痘痘、胸闷、头晕\n👉 中焦（脾胃）：腹胀、口臭、消化不良\n👉 下焦（肾膀胱）：水肿、带下、腰酸\n#养生[话题]# #健康养生[话题]# #中医养生[话题]#  #身体去湿气[话题]# #养生小知识[话题]# #在小红书轻养生[话题]# #湿气[话题]# #祛湿[话题]# #养生日常[话题]# #中医科普[话题]#', 'title': '祛湿宝典·上篇｜5类湿气+8大解法！💦', 'url': 'https://www.xiaohongshu.com/explore/692d5080000000001e03935c', 'date': '2025-12-01 18:01:46', 'note_type': 'images', 'video': {'noteId': '692d5080000000001e03935c', 'url': ''}, 'images': [], 'like_count': '424', 'collect_count': 591, 'view_count': '5668', 'comments': [], 'confidence': -1}, {'id': 'turn_1_304', 'search_from': 'search_note', 'content': '之前稍不注意就腹痛腹泻，整天提心吊胆，试过不少法子都没效果，整个人愁得没胃口🍵直到我找到了适合自己的调理方式，没想到，就这么坚持下来，肠胃安稳了，整个人都轻松很多。今天把我的经验分享给大家，赶紧码住～😷\n中醫讲，肠易激的反复与脾胃虚弱、湿浊内停密切相关，脾胃运化差，就像肠道失了“调节器”，功能紊乱反复出现，导致腹胀腹痛、排便异常，所以健脾、祛湿、调肠道，才是改善肠易激的关键！ 👇来看看，这些常见问题你有吗？\n🌫一吃生冷油腻就腹痛腹泻\n🌫肚子胀气频繁，屁多还尴尬\n🌫排便不规律，时而拉稀时而便秘\n🌫吃一点就饱，消化慢还没食欲\n✅要是有，别犹豫，下面方法赶紧码住！\n苐❶阶段：温肠止泻🔥肠道敏感、腹泻频繁时，关键要温和调理🍵生姜配炒米，生姜温中止呕、驱散寒邪，炒米健脾养胃、固肠止泻材料：生姜3片，炒米15克做法：放入锅中，加适量清水煮开，焖5分钟后温饮，每天1杯，坚持1-2周\n苐❷阶段：理气消胀🔥腹泻好转但仍有腹胀时，说明气滞湿阻还没改善。🍵这时用陈皮搭配炒麦芽、茯苓，陈皮理气健脾、燥湿化痰，炒麦芽助消化、疏肝气，茯苓健脾祛湿，三者搭配，既能理顺肠道气机，又能改善消化材料：陈皮5克，炒麦芽10克，茯苓8克做法：加适量清水，大火煮开后转小火煮10分钟，放温后饮用，持续2-4周\n苐❸阶段：健脾固肠🔥症状稳定后，不能掉以轻心！🍵用山药配莲子、芡实补脾胃、固肠道。山药健脾养胃，莲子补脾止泻，芡实固肾涩肠，搭配起来增强脾胃功能，让肠道更“结实”材料：山药10克（干品），莲子6克（去芯），芡实6克做法：放入锅中煮20分钟，每天1次，可长期坚持\n⚠除了喝调理茶，日常养护也至关重要：\n✔少吃生冷、辛辣、油炸食物，多吃山药、南瓜、小米等温和好消化的食材\n✔每天适当散步，饭后慢走10分钟，能促进肠道蠕动，帮助消化\n✔吃饭细嚼慢咽，避免狼吞虎咽，减轻肠道负担\n✔保持情绪稳定，焦虑紧张容易刺激肠道，试着通过听音乐、深呼吸缓解压力 ✔坚持用肠胃贴，它能通过穴位压力，改善肠道环境，帮助稳定肠易激，减少不适\n#告别肠易激综合征[话题]##肠道健康[话题]# #肠胃调理[话题]##肠胃贴[话题]##健脾养胃[话题]##肠胃[话题]#', 'title': '我调整了饮食，三年的肠易激居然改善了！', 'url': 'https://www.xiaohongshu.com/explore/68a85789000000001b03c839', 'date': '2025-08-22 19:42:01', 'note_type': 'images', 'video': {'noteId': '68a85789000000001b03c839', 'url': ''}, 'images': [], 'like_count': '6', 'collect_count': 13, 'view_count': '463', 'comments': [], 'confidence': -1}, {'id': 'turn_1_353', 'search_from': 'search_note', 'content': '很多人一听到“甲减”就紧张，但其实从中医角度看，它不过是身体3个关键部位“坏了”，\n就像机器零件出故障一样，只要我们及时调理，就能慢慢恢复！😊\n\t\n🔍 甲减，其实就是身体这几处坏了：\n在中医理论中，甲减多与“肾、脾、肝”这三个脏腑功能失调有关。\n它们就像身体的发动机，一旦出问题，\n就会导致气血不足、代谢减慢，出现疲劳、怕冷、体重增加等症状。\n\t\n下面我来详细说说：\n\t\n[一R]肾阳不足（肾“坏了”）：\n肾是先天之本，主水液代谢和温煦全身。\n如果肾阳虚弱，就像冬天没暖气，人会特别怕冷、水肿、腰酸，还容易脱发。\n中医常说“肾主骨生髓”，肾不好了，整个身体的动力就弱了，甲减的疲劳感就是这么来的～❄️\n\t\n[二R]脾气虚弱（脾“坏了”）：\n脾是后天之本，负责消化吸收和运化水湿。\n脾虚了，营养跟不上，人就会乏力、食欲差、大便溏稀，体重还莫名其妙增加。\n这就像家里的“厨房”坏了，做不出好饭菜，身体自然没能量！🍚\n\t\n[三R]肝气郁结（肝“坏了”）：\n肝主疏泄，调节情绪和气机。\n长期压力大、情绪不好，会导致肝气不畅，影响气血运行，加重甲减的症状，比如情绪低落、月经不调。\n中医说“肝郁则脾虚”，肝和脾是好朋友，一个出问题，另一个也受影响哦～😔\n\t\n🌿 中医调理小贴士（真实分享）：\n别担心，甲减不是绝症！中医强调“整体调理”，我从临床经验总结了几点建议，帮你慢慢修复这些“坏掉”的地方：\n\t\n[一R]饮食调养：\n多吃温补食物，比如生姜、山药、核桃（补肾阳）；小米、红枣（健脾）；玫瑰花茶（疏肝）。\n少吃生冷油腻，避免伤脾。\n\t\n[二R]穴位按摩：\n每天按揉足三里（健脾）、关元穴（补肾）、太冲穴（疏肝），\n每个穴位3-5分钟，能提振阳气、舒缓情绪。\n\t\n[三R]生活方式：\n保持适度运动（如散步、太极），避免熬夜，晚上11点前睡觉有助于养肾。\n心情不好时，多和朋友聊天或练习深呼吸，肝气顺了，身体自然轻松～\n\t\n希望这篇分享对你有帮助！如果喜欢，记得点赞❤️收藏⭐️\n#甲减[话题]# #甲状腺功能减退[话题]# #健康科普[话题]# #中医调理[话题]# #甲减问题解决[话题]# #甲减怎么调理[话题]# #中医治疗甲减[话题]# #甲状腺[话题]# #北京甲状腺医生冯丽娜[话题]#', 'title': '甲减，其实就是身体这3处“坏了”', 'url': 'https://www.xiaohongshu.com/explore/68f33ffd0000000003023d49', 'date': '2025-10-19 06:21:00', 'note_type': 'images', 'video': {'noteId': '68f33ffd0000000003023d49', 'url': ''}, 'images': [], 'like_count': '131', 'collect_count': 123, 'view_count': '14199', 'comments': [], 'confidence': -1}, {'id': 'turn_1_357', 'search_from': 'search_note', 'content': '作为一名中医，每天都会遇到很多甲减患者。很多人查出桥本时，医生只说"观察"，等甲减了就开优甲乐。但真相是：甲减不是病，而是身体发出的求救信号！\n\t\n🔍 甲减的真相：身体这4处"坏了"\n\t\n1⃣ 肝郁气滞\n很多患者来看诊时，我都会先问："最近是不是压力很大？"肝主疏泄，长期焦虑、生气会导致肝气郁结。肝郁克脾，影响气血生化，甲状腺功能就会紊乱。\n👉 典型表现：胸闷、月经不调、易怒\n👉 调理重点：疏肝解郁\n\t\n2⃣ 脾胃虚寒\n"医生，我一吃冷的就难受"——这是脾胃虚寒型甲减的典型主诉。脾阳不足，运化无力，气血生化不足，甲状腺得不到滋养。\n👉 典型表现：怕冷、消化不良、浮肿\n👉 调理重点：温补脾阳\n\t\n3⃣ 湿热蕴结\n长期外卖、熬夜、压力大，体内湿热堆积。湿热阻碍气血运行，影响甲状腺功能。\n👉 典型表现：口苦、头面油腻、大便黏腻\n👉 调理重点：清热化湿\n\t\n4⃣ 肾阳不足\n"医生，我整天没精神"——肾阳是生命之火，肾阳不足会影响全身机能，包括甲状腺。\n👉 典型表现：腰膝酸软、性欲减退\n👉 调理重点：温补肾阳\n\t\n🌿 甲减自救指南\n饮食调理\n✔ 早餐：山药小米粥（健脾）\n✔ 午餐：当归生姜羊肉汤（温阳）\n✔ 晚餐：清炒时蔬（少油少盐）\n🚫 忌口：冷饮、甜食、油炸食品\n\t\n生活调理\n🌞 早晨晒背15分钟（补阳气）\n🧘\u200d♀ 每天八段锦10分钟（疏通经络）\n😴 晚上11点前入睡（养肝血）\n\t\n穴位按摩\n💆 太冲穴（疏肝）\n💆 足三里（健脾）\n💆 关元穴（补肾）\n\t\n💡 医生想说\n甲减不是终点，而是身体在提醒你：该调整生活方式了！中医调理讲究"三分治，七分养"，找准病因，对症调理，甲减是可以改善的。\n如果你也有：\n❌ 整天没精神\n❌ 特别怕冷\n❌ 体重莫名增加\n可能是甲状腺在求救！\n\t\n#白大褂医生已在线[话题]# #甲减[话题]# #甲减怎么办[话题]# #甲减症状[话题]# #甲减患者[话题]# #甲减减重[话题]# #身体发出的信号[话题]# #调理[话题]# #人到中年身体健康最重要[话题]# #中医就是这么神奇[话题]# #甲减治疗[话题]# #甲减发胖[话题]# #没有什么比健康来的重要[话题]#', 'title': '甲减，其实就是身体四处坏了', 'url': 'https://www.xiaohongshu.com/explore/67f37f190000000009016028', 'date': '2025-04-07 15:30:33', 'note_type': 'images', 'video': {'noteId': '67f37f190000000009016028', 'url': ''}, 'images': [], 'like_count': '2134', 'collect_count': 1988, 'view_count': '158596', 'comments': [], 'confidence': -1}, {'id': 'turn_1_382', 'search_from': 'search_note', 'content': '#四神汤[话题]# #四神汤的做法[话题]# #调理脾胃[话题]# #煲汤[话题]# #入伏养生要喝汤[话题]# #精通每一味好汤[话题]# #秋冬一碗汤[话题]# #养生煲汤[话题]# #四神汤怎么煮[话题]# #炖汤[话题]# #脾虚湿气重吃什么[话题]# #四神汤挑选攻略[话题]# #养生食补进行时[话题]# #广东靓汤[话题]# #秋冬滋润汤水[话题]#', 'title': '健脾养胃祛湿 | 四神汤的煲汤经典搭配做法', 'url': 'https://www.xiaohongshu.com/explore/69016a620000000007001f3a', 'date': '2025-10-29 09:14:10', 'note_type': 'images', 'video': {'noteId': '69016a620000000007001f3a', 'url': ''}, 'images': [], 'like_count': '2', 'collect_count': 1, 'view_count': '142', 'comments': [], 'confidence': -1}, {'id': 'turn_1_383', 'search_from': 'search_note', 'content': '#四神汤[话题]# #四神汤的做法[话题]# #四神汤配方[话题]# #四神汤挑选攻略[话题]# #四神汤健脾祛湿[话题]# #四神汤的功效和禁忌[话题]# #怀山药[话题]# #茯苓[话题]# #四神汤一周喝几次[话题]# #苹果四神汤[话题]#', 'title': '四神汤 | 适合这些体质·三种吃法·挑选技巧', 'url': 'https://www.xiaohongshu.com/explore/691eb96f000000001f00da41', 'date': '2025-11-20 14:47:11', 'note_type': 'images', 'video': {'noteId': '691eb96f000000001f00da41', 'url': ''}, 'images': [], 'like_count': '1', 'collect_count': 1, 'view_count': '101', 'comments': [], 'confidence': -1}, {'id': 'turn_1_385', 'search_from': 'search_note', 'content': '#四神汤[话题]# #四神汤的做法[话题]# #调理脾胃[话题]# #脾胃虚弱[话题]# #四神汤怎么煮[话题]# #四神汤挑选攻略[话题]# #脾虚湿气重吃什么[话题]# #养脾胃[话题]# #四神汤健脾祛湿[话题]#', 'title': '四神汤适合哪些人？哪些人不能喝？', 'url': 'https://www.xiaohongshu.com/explore/6915c8a80000000007021bc2', 'date': '2025-11-13 20:01:44', 'note_type': 'images', 'video': {'noteId': '6915c8a80000000007021bc2', 'url': ''}, 'images': [], 'like_count': '2', 'collect_count': 1, 'view_count': '141', 'comments': [], 'confidence': -1}, {'id': 'turn_1_387', 'search_from': 'search_note', 'content': 'hello朋友们，\n\t\n最近炎热的夏天，湿气也越来越重，\n\t\n今天与我妈妈分享【经典四神汤】，还有我妈妈的一些建议，如何用辅助药材搭配更适合自己的身体！\n\t\n简单好喝，小朋友也喜欢的味道，日常多喝一些，浑身更加轻盈舒畅\n\t\n希望大家喜欢！\n\t\n#抗炎饮食[话题]# #中医药膳[话题]# #中药[话题]# #祛湿[话题]#', 'title': '湿气重？与北京中医大学的妈妈做【四神汤】', 'url': 'https://www.xiaohongshu.com/explore/683379ce000000000f030fc0', 'date': '2025-05-26 07:01:08', 'note_type': 'video', 'video': {'noteId': '683379ce000000000f030fc0', 'url': ''}, 'images': [], 'like_count': '6779', 'collect_count': 7352, 'view_count': '149083', 'comments': [], 'confidence': -1}, {'id': 'turn_1_389', 'search_from': 'search_note', 'content': '😫最近熬夜加班，总感觉肚子胀胀的，翻出家里的食材煮了碗四神汤，喝完整个人都舒服了！\n没有复杂的步骤，就是食材本身的清甜，养生真的不用费钱费时间～\n🧾♥️ 食材配比（一人份）\n•山药：15g（建议选铁棍山药，粉质足更养人）\n•芡实：10g（提前泡15分钟，煮出来更软糯）\n•茯苓：10g（干茯苓块）\n•莲子：10g（去芯！）\n✅ 多人份直接按比例翻倍，可加瘦肉/排骨/红枣，口感更丰富～\n‼️核心功效\n四神汤是经典健脾祛湿方，山药健脾养胃，芡实固肾涩精，茯苓利水渗湿，莲子养心安神，搭配起来温和不寒凉，尤其适合脾虚湿气重、容易疲劳的人。\n✨脾胃是后天之本，养好它比涂多少护肤品都管用！\n♥️🍳 懒人做法\n1.\t所有食材洗净，莲子去芯，芡实、茯苓提前泡15分钟\n2.\t食材放入砂锅/电饭煲，加500ml清水\n3.\t砂锅小火煮40分钟，电饭煲选“煲汤”模式1小时\n4.\t出锅前可加少许冰糖调味，或者直接喝原味更养生～\n🔴体质偏寒的姐妹可以加2片生姜，体热的姐妹别加冰糖，清爽更解腻～\n\t\n🔍 四神汤食材选购攻略\n1. 山药（首选铁棍山药干）\n✅ 优质特征：颜色是米白色或淡黄色，不是雪白雪白的；表面有细微纹路，摸起来有点粗糙。\n❌ 避坑点：通体雪白发亮的，大概率是硫磺熏过的，闻着有刺鼻味，吃了反而伤脾胃。\n2. 芡实（选新鲜干芡实）\n✅ 优质特征：颗粒饱满圆润，颜色是淡棕黄色，外壳薄且易剥；掰开后内部是白色粉性，没有霉点。\n❌ 避坑点：颜色发黑、发暗，捏起来发软的，是受潮发霉的；颗粒干瘪、大小不一的，口感差还没营养。\n3. 莲子（去芯白莲）\n✅ 优质特征：颜色是自然的乳白色，不是惨白；表面有淡淡的纹路，煮后软糯清香；去芯干净，没有苦味。\n❌ 避坑点：硫磺熏过的莲子会有刺鼻味，煮后汤水发黄；带芯的莲子苦味重，脾胃弱的人喝了容易不舒服。\n4. 茯苓（选粉质细腻的白茯苓）\n✅ 优质特征：切面是纯白色或淡粉色，质地坚实，用手捏不会碎成粉末；闻着有淡淡的土腥味，是正常的。\n❌ 避坑点：颜色发灰、发暗，一捏就碎的，是陈货或劣质品；有酸味、霉味的直接pass！#四神汤[话题]# #养脾胃[话题]# #养孩子就是养脾[话题]# #养好脾胃不生病[话题]# #女人养生养脾胃[话题]# #脾胃健康从饮食开始[话题]# #养脾胃就是养命[话题]# #先健脾还是先祛湿[话题]#', 'title': '‼️一碗四神汤|养好小肚腩和玻璃胃✨', 'url': 'https://www.xiaohongshu.com/explore/693ffa2e000000001e033319', 'date': '2025-12-15 20:08:14', 'note_type': 'images', 'video': {'noteId': '693ffa2e000000001e033319', 'url': ''}, 'images': [], 'like_count': '3', 'collect_count': 2, 'view_count': '44', 'comments': [], 'confidence': -1}, {'id': 'turn_1_390', 'search_from': 'search_note', 'content': '四神汤作为千古祛湿名方，健脾益阴，厚实肠胃、化湿利水\n立冬时节喝四神汤，是一个非常应季且有益的养生选择。它带来的好处主要体现在“健脾祛湿”和“为冬补打好基础”这两个核心上。\n每周喝2～3次，有意想不到的惊喜哟！\n👶特别孩子，吃饭不积极多观察一下，四神汤加颗苹果就可以日常养护脾胃蛮好的，消食解腻助消化，疏通身体了才能吃补进去👍适合全家食疗！\n✅苹果+四神汤功效\n1、四神汤健脾化湿，内调去黄，长期喝可以改善肤色暗沉。\n2、苹果煮水则可以调理脾胃，润肠通便\n3、芡实有收敛作用，不太适合便秘人群喝，但是苹果水又具有润肠通便效果，所以四神汤+苹果水搭配起来正好能中和效果，简直王炸了❗\n传统的四神汤食材都放这啦！淮山药、茯苓、芡实、莲子，可以按1：1：1：1的比例进行搭配，每种食材来上10克，再切一个苹果\n✅做法\n1个苹果+10g茯苓+10g芡实+10g莲子+10g淮山药，加入1000ml水，养生壶煮30min到1小时就可以哈。\n😋软软糯糯的食材也可以吃掉呀\n清甜爽口敲好喝！！四神汤也可以加排骨猪肚炖汤喝，都很棒！老少皆宜哦\n脾胃不适，容易腹泻便溏，长期熬夜导致肤色暗沉，舌苔厚腻身体湿气重的宝子都可以坚持喝起来哈。\n⚠️不太适合人群\n1、感冒咳嗽，糖尿病患者\n2、便秘、腹胀等人群，\n3、怀孕人群\n⚠️注意：一周煮上个2-3次就可以啦！养脾胃这件事慢慢来~\n❌该配方仅供调理脾胃\n如果身体真的很不舒服还是要线下就医哟\n#四神汤[话题]# #养好脾胃不生病[话题]# #苹果四神汤[话题]##健脾养胃[话题]##立冬的一碗汤[话题]##苹果煮水好处多[话题]##养脾胃就是养命[话题]# #儿童成长汤[话题]# #宝宝食疗方[话题]# #先健脾还是先祛湿[话题]# @薯管家 @薯店长 @生活薯 @知识薯 @薯队长 @薯条小助手', 'title': '抓住立冬好时间多喝🍎四神汤脾胃太舒服了❗', 'url': 'https://www.xiaohongshu.com/explore/690de5380000000007022b09', 'date': '2025-11-07 20:25:28', 'note_type': 'images', 'video': {'noteId': '690de5380000000007022b09', 'url': ''}, 'images': [], 'like_count': '12', 'collect_count': 11, 'view_count': '383', 'comments': [], 'confidence': -1}, {'id': 'turn_1_394', 'search_from': 'search_note', 'content': '👋 姐妹们，提到“抗炎饮食”是不是觉得那是都要吃冷冰冰的沙拉、喝果蔬汁？No No No！🙅\u200d♀️\n\t\n其实，咱们老祖宗留下的饮食智慧，才是最适合中国宝宝体质的“抗炎大法”！\n不挨饿、不吃草，也能把身体调理得通透白净！✨\n把“抗炎”吃进肚子里，中式抗炎主打一个“温补”和“顺时而食”，不用买昂贵的进口超级食物，菜市场就能搞定！\n\t\n🥬 01 .炫一口“天然灭火器”\n别总盯着羽衣甘蓝啦！咱们菜市场里的菠菜、油菜、西兰花，哪个不是抗氧化的小能手？\n这些深色蔬菜，能帮身体清理“垃圾”，把体内的“火气”降下来。建议每天蔬菜摄入量占到餐盘的一半哦！🥗\n\t\n🌽 主食要搞“粗细搭配”\n千万别为了瘦就不吃主食！长期不吃碳水，心情会变差，压力激素飙升，反而加重炎症！\n怎么吃：把精米白面换掉一半，加入糙米、小米、紫薯、玉米。这些“慢碳”不仅饱腹感强，还能稳住血糖，让情绪更平稳。🍠\n\t\n🥣 东方“食补”是王炸\n这就是中式抗炎的灵魂——药食同源！\n山药 & 茯苓：脾胃虚弱的姐妹锁死它，温和养护。\n薏仁 & 赤小豆：感觉身体沉重、肿肿的时候，煮水喝特别绝。\n莲子 & 百合：心烦睡不着？用它们煮粥，清心安神\n\t\n🧘\u200d♀️ 02. 抗炎是一种生活方式\n除了吃，把身体的“节能模式”打开也很重要！\n💤 睡好觉 = 免费医美\n熬夜 = 主动发炎！🔥 真的别再一边熬夜一边涂贵妇眼霜了。\nAction：试着把入睡时间前移到23:00。中医讲“卧则血归于肝”，高质量的睡眠是身体自我修复的黄金期，坚持一周，你会发现脸色真的会变亮！\n\t\n🌳 练一练“慢运动”\n剧烈运动产生过多皮质醇，反而可能导致炎症。\n推荐：咱们的八段锦、太极，或者简单的公园散步、瑜伽。微微出汗、身体舒展的感觉刚刚好，既提升代谢又不耗气血。\n\t\n🍵 给情绪“排排毒”\n压力大、爱生气，身体真的会“记仇”哦！\n心态：心平气和才是最大的抗炎。每天给自己留10分钟“放空时间”，喝杯茶、读几页书，把焦虑的情绪代谢掉。\n\t\n💡 碎碎念：\n抗炎不是突击战，而是细水长流的养护。\n从今天开始，好好吃饭，乖乖睡觉，在这个快节奏的生活里，把自己的状态养得稳稳的！💪\n\t\n#中式抗炎[话题]# #抗炎饮食[话题]# #养生日常[话题]# #健康饮食[话题]# #食疗养生[话题]# #适合中国胃的减肥餐[话题]# #我的健康生活[话题]# #内调[话题]# #变美思路[话题]#', 'title': '🔥适合中国胃的「中式抗炎」一周食谱', 'url': 'https://www.xiaohongshu.com/explore/6937bc1b000000001e03125c', 'date': '2025-12-09 17:00:52', 'note_type': 'images', 'video': {'noteId': '6937bc1b000000001e03125c', 'url': ''}, 'images': [], 'like_count': '600', 'collect_count': 664, 'view_count': '11414', 'comments': [], 'confidence': -1}, {'id': 'turn_1_412', 'search_from': 'search_note', 'content': '👉 寒湿体质：艾灸暖身，驱寒祛湿！\n👉 湿热体质：刮痧清热，排毒解暑！\n👉 气滞血瘀：拔罐疏通，活血化瘀！\n💡\xa0划重点：\n✅ 拔罐后别吹风，注意保暖！\n✅刮痧后多喝水，帮助代谢！\n✅ 艾灸后别碰冷水，避免寒气入侵！\n#艾灸[话题]##拔罐[话题]##刮痧[话题]##我的养生仪式感[话题]##去湿气[话题]#', 'title': '一分钟看懂👉艾灸与刮痧、拔罐的区别', 'url': 'https://www.xiaohongshu.com/explore/67cdb7dd000000002901a747', 'date': '2025-03-09 23:46:37', 'note_type': 'images', 'video': {'noteId': '67cdb7dd000000002901a747', 'url': ''}, 'images': [], 'like_count': '132', 'collect_count': 46, 'view_count': '33171', 'comments': [], 'confidence': -1}, {'id': 'turn_1_413', 'search_from': 'search_note', 'content': '姐妹们谁懂啊，之前的我舌苔又厚又腻，像发霉的豆腐乳；早起眼皮肿成泡泡眼，小腿一按一个坑；马桶永远冲不干净（湿7人都懂！）；头发刚洗就贴头皮，脸油得能煎鸡蛋……\n\t\n不过现在舌苔变薄透出淡粉底色了哈哈哈，蕞爽的是💩黄金香蕉便！马桶釉面闪闪亮✨\n\t\n闺蜜说我像开了去湿滤镜！其实，我主要是靠膝盖下这个“排水开关”——阴陵泉\n\t\n一路走来，我可有太多要说的了：\n\t\n1. 📍 小白找穴法：\u2028👉 蜷腿坐，顺着小腿内侧骨头往上摸\u2028👉 快到膝盖时摸到一个软乎乎的坑（别按太狠会酸到跺脚！）\n\t\n2. 🔥 懒人操作：\u20281️⃣ 艾熏法：手持艾条离皮肤3根手指远，温热不烫就行⚡️（新手5分钟/天）\u20282️⃣ 按压法：追剧时用保温杯底顶住穴位，边看边滚超省力！\n\t\n3.⚠小贴士：\u2028❌灸后3小时别碰冷💦\u2028❌防烫伤：艾条永远保持奶茶吸管距离（3cm以上）⚡️\u2028❌禁忌人群：孕妇别碰！皮肤破口停用！饭后半小时再操作⚡ ️\n\t\n4.💡 增效组合：\u2028• 搭配煮苹果水喝，排💦速度翻倍\u2028• 下雨天灸完套袜子保暖\n\t\n用老中医的话来说就是，膝盖下这个坑=身体排💦总闸！通了之后全身像装了隐形抽湿机~\n\t\n当然了，效果因人而异｜纯自用分享非专业建议～，也欢迎姐妹们交流分享自己的经验哦～\n#三伏养生局[话题]##艾灸[话题]##艾灸养生[话题]##艾灸艾条[话题]##艾柱[话题]##艾灸工具[话题]##灸出好气色[话题]##阴陵泉[话题]##祛湿[话题]##浪漫生活的记录者[话题]#', 'title': '啊啊啊我发现了一个排湿气贼牛的穴……', 'url': 'https://www.xiaohongshu.com/explore/685f64fe00000000120163d0', 'date': '2025-06-28 11:43:58', 'note_type': 'images', 'video': {'noteId': '685f64fe00000000120163d0', 'url': ''}, 'images': [], 'like_count': '116', 'collect_count': 95, 'view_count': '11356', 'comments': [], 'confidence': -1}, {'id': 'turn_1_421', 'search_from': 'search_note', 'content': '核心功效与作用\n1. 温经散寒，驱寒祛湿\n艾灸盒通过燃烧艾条产生的热力渗透肌肤，能够有效驱散体内寒湿之气，特别适合寒湿体质人群。对于因寒凉导致的关节疼痛、胃寒腹痛、女性痛经等问题有显著缓解作用，尤其适合手脚冰凉、畏寒怕冷的人群。\n2. 活血通络，缓解疼痛\n通过温热刺激穴位，艾灸盒能够促进局部血液循环，改善气血瘀滞，对于肩颈僵硬、腰肌劳损、运动后肌肉酸痛等症状有良好的调理效果。艾灸的热力可温通经络，行气活血，对慢性疼痛有显著的缓解作用。\n3. 调节气血，增强免疫力\n艾草的药性成分随热量进入人体，温和刺激免疫系统，帮助改善疲劳、失眠、易感冒等亚健康状态。长期使用可增强体质，提高身体的抗病能力，对于体虚乏力、免疫力低下的人群特别有益。\n4. 美容养颜，改善气色\n艾灸盒能够促进新陈代谢，减少皱纹和色斑，使皮肤更加光滑细腻。通过促进全身血液循环，带动面部气血流动，能够慢慢改善暗沉、蜡黄，让肌肤透着健康光泽。\n适用人群与场景\n适用人群\n• 亚健康人群：体虚乏力、睡眠质量差、易疲劳者\n• 慢性病患者：风湿性关节炎、慢性胃炎、月经不调等需要长期调理者\n• 寒湿体质人群：手脚冰凉、畏寒怕冷者\n• 日常保健需求者：预防感冒、缓解工作疲劳、提升精力\n适用部位\n• 腹部：调理脾胃、改善消化功能\n• 腰部：缓解腰肌劳损、肾虚腰痛\n• 背部：疏通督脉、缓解肩颈僵硬\n• 关节部位：改善关节炎、风湿痛\n与传统艾灸的区别\n操作便利性\n艾灸盒相比传统手持艾灸更加便捷安全，解放双手，适合家庭使用。\n温度控制\n艾灸盒通过设计控制温度，能均匀传导热量，减少烫伤风险。\n覆盖范围\n艾灸盒可同时作用于多个穴位或大面积区域，适合需要大面积施灸的情况。而传统手持艾灸更适合精准定位特定穴位。\n使用注意事项\n禁忌人群\n• 孕妇：腹部及腰骶部禁用\n• 皮肤破损者：伤口及感染处不宜使用\n• 高热患者：体温过高时不宜使用\n• 凝血功能障碍者：需谨慎使用\n使用要点\n• 时间控制：单次艾灸时间建议15-30分钟\n• 温度调节：从低温开始逐渐适应，避免烫伤\n• 艾条选择：选择优质陈艾条，燃烧稳定、烟雾小\n• 艾灸后护理：注意保暖，避免吹风受凉，2小时内不宜洗澡 #疏通经络对人体的好处[话题]# #排湿气寒气[话题]# #艾灸盒[话题]# #艾灸[话题]# #艾灸养生[话题]# #艾灸盒[话题]#', 'title': '', 'url': 'https://www.xiaohongshu.com/explore/691af79a000000000503ab28', 'date': '2025-11-17 18:23:22', 'note_type': 'images', 'video': {'noteId': '691af79a000000000503ab28', 'url': ''}, 'images': [], 'like_count': '0', 'collect_count': 0, 'view_count': '20', 'comments': [], 'confidence': -1}, {'id': 'turn_1_26', 'search_from': 'search_note', 'content': '1.蹲马步\n每次做2分钟左右。\n\t\n2.仰卧起坐\n每天起床和入睡前做20次。\n\t\n3.揉腹\n用掌火四指摩腹五分钟左右（顺时针消食通便，逆时针止泻）。\n\t\n4.单手举臂\n两臂一松一紧的上下对拉，可以按摩到脾胃。\n\t\n5.脚趾抓地\n脚趾抓紧的时候要停留2-3秒，然后再放松，连续做60-90次。\n\t\n6.叩击牙齿\n放松身心，口唇轻闭，上下牙齿有节奏的互相轻轻叩击。\n\t\n7.漱津液\n首先叩牙齿36次，然后咬紧牙齿，用舌头在口腔中搅动，以唾液满口后，再分次咽下。\n\t\n8.打太极拳\n促进身体中的新陈代谢或者脾胃的运转功能，改善脾胃虚弱。\n\t\n9.骑自行车\n骑行过程中，可以促进消化功能，有助于改善脾胃功能。\n\t\n\ufeff#中医[话题]#\ufeff \ufeff#脾胃[话题]#\ufeff \ufeff#脾胃不好[话题]#\ufeff \ufeff#调理脾胃[话题]#\ufeff', 'title': '脾虚最怕这些运动，赶紧做起来赶走脾胃虚弱', 'url': 'https://www.xiaohongshu.com/explore/67da2460000000000703785e', 'date': '2025-03-19 14:00:40', 'note_type': 'images', 'video': {'noteId': '67da2460000000000703785e', 'url': ''}, 'images': [], 'like_count': '122', 'collect_count': 115, 'view_count': '3874', 'comments': [], 'confidence': -1}, {'id': 'turn_1_114', 'search_from': 'search_note', 'content': '1.\xa0肠易激的宝子，日常饮食别贪多，吃到七分饱即可，给肠道和脾胃减少负担\n2.\xa0少吃高刺激食物，像辣椒、花椒、芥末等\n3.\xa0选温和的运动方式，慢走或八段锦很合适\n4.\xa0每晚用40℃左右热水泡脚，每次15-20分钟，泡到微微出汗，暖腹又舒眠\n5.\xa0坚持使用肠胃贴，通过穴位压力稳住肠易激，把它贴于肚脐周围，帮助舒缓腹痛胀气，改善腹泻便秘等问题\n6.\xa0养成晨起排便习惯，避免便秘时腹压增大，加重肠道蠕动紊乱\n7.\xa0别久坐不动，每小时起身活动5分钟，做做简单拉伸\n8.\xa0上午9-10点喝300ml温温水，此时身体代谢活跃\n9.\xa0坚持23点前入睡，每天睡够7-8小时，让肠道得到充分休息\n10.\xa0少喝含咖啡因的饮品，咖啡、浓茶尽量不碰\n11.\xa0常按天枢穴（肚脐旁开2寸）和足三里（外膝下3寸），每次按1-2分钟\n12.\xa0少吃油炸、油煎食物，比如炸鸡、油条，减少油脂摄入，减轻消化负担\n13.\xa0奶制品适量吃，优先选无乳糖奶或发酵型酸奶\n14.\xa0别盲目节食或做高强度运动，过度消耗适得其反\n15.\xa0少吃寒凉食物，像冰饮、冰水果、螃蟹等，避免加重负担\n16.\xa0戒掉含糖饮料，可乐、奶茶全停掉，多喝温水或陈皮茶\n17.\xa0少吃不新鲜或变质食物，比如隔夜菜、发霉水果\n18.\xa0少吃腌制食物，咸菜、酱肉等过咸食物别碰\n19.\xa0调整饮食要避开过敏或吃了不适的食物，及时忌口\n20.\xa0保证规律作息，千万别熬夜，熬夜易加重不适\n建议常做这5个动作：\n1.\xa0腹部绕环\n平躺在床上，双手放在肚脐处，顺时针缓慢绕环按摩腹部，重复10-15圈，每天做1组\n2.\xa0腹式呼吸\n平躺在床上，双手放在腹部，吸气时肚子鼓起，呼气时肚子收紧，每次做5-10分钟，缓解小腹坠胀感\n3.\xa0靠墙站立\n后背贴墙，双脚分开与肩同宽，双手自然下垂，保持10-15分钟，改善体态的同时，减轻腹部压力\n4.\xa0屈膝卷腹\n平躺在床上，双腿屈膝，双手抱头，缓慢抬起上半身再放下，幅度以不感到腹部不适为宜，每次5-10分钟\n5.\xa0搓腹暖肠\n双手搓热后，放在肚脐两侧，上下搓揉至腹部发热，每次3-5分钟\n#肠易激 #肠道调理#肠胃#腹泻#胀气#照顾好自己的胃', 'title': '', 'url': 'https://www.xiaohongshu.com/explore/68c10481000000001d021d39', 'date': '2025-09-10 12:54:25', 'note_type': 'images', 'video': {'noteId': '68c10481000000001d021d39', 'url': ''}, 'images': [], 'like_count': '229', 'collect_count': 179, 'view_count': '21209', 'comments': [], 'confidence': -1}, {'id': 'turn_1_120', 'search_from': 'search_note', 'content': '1.❌忌口\n孩子肠易激简单来说就是脾胃出了问题，肝火旺盛，所以咱们首先忌口，但是摄入面比较广这些注意忌口不吃：（大麦，小麦，黑麦，以及上述加工的食品，馒头，包子，面条，面筋，油条，面包，蛋糕，饼干等）\n可以替换为：糙米，荞麦，无麸质燕麦，高梁米，薏米，山药，黑米，亚麻籽等\n2.❌乳制品\n不喝（牛奶，羊奶，酸奶，奶酪等）\n可以替换为：椰子粉，骆驼奶，椰浆等\n3.❌含大豆蛋白\n（大豆，豆豉，豆瓣酱，酱油，调味料）\n4.❌少喝\n不喝：咖啡，奶茶，酒水，饮料，浓茶等可以替换为：菊花茶，蒲公英茶和玫瑰茶\n5.❌蔬菜\n不吃：西兰花，花菜，羽衣甘蓝，孢子甘蓝，芥菜，如果特别喜欢吃一定要焯熟后才可以可以多吃：油麦菜，空心菜，生菜，芹菜，冬瓜，芥蓝，黄瓜，菠菜类，\n6.❌油\n不吃：花生油，动物油和调和油\n可以吃：椰子油，菜籽油，橄榄油等\n7.❌坚果\n不吃：口味重的坚果，炒焦的坚果，霉变的坚果，花生也不建议吃\n可以吃：南瓜子，腰果，夏威夷果，核桃\n8.❌豆类\n不吃：红豆，黄豆，黑豆，蚕豆，芸豆等\n可以吃：鹰嘴豆，豌豆，缸豆，绿豆等\n9.❌水果\n少吃：榴莲，菠萝蜜，龙眼，荔枝，西瓜，芒果，苹果，山楂等\n可以多吃：（以莓果类为主） 蓝莓，树莓，蔓越莓，草莓，巴西莓，樱桃，橙子，火龙果，等\n10.❌含碘食物\n不吃：碘盐，海带，紫菜，海苔，海鲜等\n补充蛋白质可以吃：淡水鱼，牛肉，瘦猪肉，鸡胸肉，鸭肉，兔肉等\n坚持5个动作\n1、转手腕、脚腕\n每天晨起前/三餐前后转转手腕、脚腕，转6~12圈四肢末端乃“气之根”，气血从这里向身体中心汇聚，手腕、脚腕是重要的“渡口”，越通畅越好\n2、拍八虚\n腋窝→肘窝→腹股沟→腘窝每个位置拍81下。八虚分别指身体8个“折叠”的点，是气血通行的要道，也是内脏 废浊向外排时容易堵塞的点\n3、揉腹\n晨起或睡前，顺时针揉腹180下。推揉时会有气泡或推到硬块疙瘩，早上气力充沛反应会更强烈些。坚持多推/揉一会儿，推到打ge、放P、排便是蕞好的\n4、按摩肩颈\n睡前按一按肩/颈，或用热毛巾敷5~6分钟。肩颈这块经常暴露在外面，容易受风/寒/湿侵袭，僵硬难以伸展，影响头和身体的气血通畅\n5、晨起梳头\n晨起梳头300下。头为诸暘之会，梳齿划过头皮，ci激脉络里的气/血流动。很多人梳食物前几天会感觉头发越来越油，其实是在排浊，多梳一阵子就不油了\n除了忌口加适量运动外，咱还得坚持内调，坚持补充益生菌，调节肠道菌群，维护肠胃健康~~~\n#肠易激[话题]##肠胃健康[话题]# #肠易激怎么办[话题]#', 'title': '肠易激上岸啦！', 'url': 'https://www.xiaohongshu.com/explore/66cc524c000000001d038693', 'date': '2024-08-26 18:00:44', 'note_type': 'images', 'video': {'noteId': '66cc524c000000001d038693', 'url': ''}, 'images': [], 'like_count': '2605', 'collect_count': 2373, 'view_count': '89061', 'comments': [], 'confidence': -1}, {'id': 'turn_1_166', 'search_from': 'search_note', 'content': '慢性肠胃炎 脾胃虚弱，建议6个中成方\n.\n1、脾胃虚弱——长期腹泻，消化功能弱，进食过多则泻，腹胀隐痛，反反复复\n中成药：六君子丸、补脾益肠丸、人参健脾丸、参苓白术参考选一种。\n适合长期调理方：泰渭散（针对胃粘膜受损的修复）\n.\n2、伤食腹泻——因饮食不当引起的腹泻，腹痛肠鸣，泻后痛减，泻未消化食物\n中成药：保和丸、加味保和丸参考选一种。\n调理：神曲、山楂、茯苓、半夏、陈皮、连翘、莱菔子加减。\n.\n3、脾肾阳虚——好发中老年人，黎明之前，腹痛肠鸣即泻，泻后舒适，腹部怕凉喜温\n中成药：四神丸、固本益肠片、温脾止泻丸参考选一种。\n调理：四神丸（补骨脂、五味子、肉豆蔻、吴茱萸）加减。\n.\n4、寒湿腹泻——大便清稀，像水一样，腹痛肠鸣，大便酸臭\n中成药：泰渭散、金免幽、藿香正气丸、附子理中丸参考选一种。\n方药：藿香正气散（藿香、白芷、紫苏、茯苓、半夏曲、白术、陈皮、厚朴、苦桔梗、大腹皮、甘草）加减。\n.\n5、湿热腹泻——常发于夏季暑湿天气，泻后不爽，肛门灼热，大便色黄褐味臭\n中成药：葛根芩连丸、香连丸、复方黄连素片、肠康片参考选一种。\n方药：葛根芩连汤（葛根、黄芩、黄连、甘草）加减。\n.\n6、肝气乘脾——生气紧张时腹痛腹泻，泻时急迫，泻后痛减，类似肠易激综合征\n中成药：柴芍六君丸、柴胡舒肝丸+肠胃益生菌金免幽\n方药：痛泻药方（陈皮、白术、白芍、防风）加减。\n.\n小贴士：\n以上均为提供参考，请在指导下用。\n#脾胃[话题]# #养胃[话题]# #修复胃黏膜[话题]# #胃炎胃胀气胃病养胃[话题]# #胃肠道健康[话题]#', 'title': '', 'url': 'https://www.xiaohongshu.com/explore/65b65bc700000000100396c8', 'date': '2024-01-28 21:51:03', 'note_type': 'images', 'video': {'noteId': '65b65bc700000000100396c8', 'url': ''}, 'images': [], 'like_count': '246', 'collect_count': 375, 'view_count': '19766', 'comments': [], 'confidence': -1}, {'id': 'turn_1_384', 'search_from': 'search_note', 'content': '#中医[话题]# #综艺推荐[话题]# #药食同源[话题]#\n\t\n📺：#初入职场的我们中医季[话题]#（2025）\n\t\nEp6 如何守“胃”健康（下）\n\t\n嘉宾：#李承铉[话题]# ，#罗昉芊[话题]# ，#伯远[话题]# ，#高卿尘[话题]#\n\t\n中医界大咖：#王金贵[话题]#，#游柏稳[话题]#\n\t\n[种草R]下午考核是让学员们从【药食同源】出发，给嘉宾制定专属健康食谱\n\t\n“民以食为天”➡️“药食同源”，古人的智慧太强大，现代人的创新好厉害[哭惹R]\n\t\n——李妍婧给承铉哥的药膳：1️⃣红枣山药馒头（中筋面粉➕酵母，掺入碾碎红枣与蒸软山药，揉面！切小馒放入蒸箱！）<连山东小齐都说好，咱妍姐这揉面技术太厉害了[色色R]>\n2️⃣中药奶茶（陈皮➕红茶先煮，后放入牛奶）➡️针对痰湿症状，晚上要是睡不着半杯就好～\n\t\n——赵乐怡给feezy的药膳：小排汤🥣（白扁豆先炒，后和陈皮、决明子【明目清脑】、白术、茯苓、山楂煮水💦，捞出药材➡️ 放入大麦米和藜麦&排骨、山药煮熟）\n\t\n——王煜给伯远的药膳：1️⃣百合、桑叶（润肺）➕薄荷、陈皮（疏肝）➕山药、茯苓（健脾）➕小排（老师建议要和赵姐一样捞出药材口感更好！）<晚饭小排也不要太多噢噢>\n2️⃣ 四神汤：由茯苓、山药、莲子、芡实四味药食同源的食材组成【健脾养胃、祛湿利水、增强体质】➕绿茶、薄荷叶、陈皮、决明子<薄荷不要煮太久否则会苦苦\U0001f979的>\n\t\n——白松妍给伯远的药膳：1️⃣ 山药➕红枣➕醪糟【补益脾气，温养脾胃】➕玫瑰花🌹【行气解郁、活血止痛、疏肝和胃】\n2️⃣南瓜鸡蛋羹（南瓜🎃去心➕四神汤汁混鸡蛋液➡️上锅蒸）；3️⃣蒙古咸奶茶（好想喝喝看啊啊啊啊）\n\t\n——王涵给小九的药膳：1️⃣ 南瓜药膳（南瓜🎃去心，放入百合➕玉竹➕山药➕大枣中药，上锅蒸）；2️⃣山楂薄荷饮（山楂➕薄荷➕陈皮 【疏散风热、利咽、透疹、疏肝行气】）\n\t\n——阿依给小九及大家的甜品药膳：1️⃣ 温中健胃饼（研磨茯苓➕生姜➕白术➕山药【健脾养胃、温中散寒】；适量低筋面粉➕药材粉末➕少许盐；面粉中加入姜汁；揉面塑形进烤箱）2️⃣ 酸奶冷瓜汤（新疆酸奶➕鹰嘴豆➕黄瓜丁➕薄荷叶）\n\t\n❗️中医小知识：\n[种草R]以酸味与甘味食药材配伍增强滋阴养血、益气健脾、生津补液效果\n[种草R]益生菌只有🈶 积食//经常打嗝//腹胀//大便有残渣物现象才吃，不用一定补！！\n\t\n\U0001f979最后大家一起去参拜张仲景祠好有感觉，还有传承感[哭惹R]', 'title': '👀中医知识之如何守“胃”健康（下）', 'url': 'https://www.xiaohongshu.com/explore/68f89690000000000703987e', 'date': '2025-10-22 16:32:16', 'note_type': 'images', 'video': {'noteId': '68f89690000000000703987e', 'url': ''}, 'images': [], 'like_count': '37', 'collect_count': 32, 'view_count': '1410', 'comments': [], 'confidence': -1}, {'id': 'turn_1_18', 'search_from': 'search_note', 'content': '脾虚人丑是真的！脾主运化，是气血生化之源。脾虚，气血生化的速度就会减弱，从而导致面色蜡黄没气色，松弛垮脸，人也没精神。说真的，脾虚和脾好，一眼就能看出来。\n\t\n很多人一听到脾虚，立马就想要通过吃什么来补脾胃，我们要清楚脾虚不是短时间形成的，而是长时间的耗损造成的。所以想要养好脾胃，调理顺序很重要，有步骤有顺序的调理，才能让脾胃越来越好～今天和大家分享我养脾胃这些年一直在做的：\n\t\n✅正确的调养顺序：\n一、疏肝（图4）\n二、健脾（图5）\n三、祛湿（图6）\n四、补气养血（图7）\n\t\n疏肝是调理脾胃的前提，健脾胃是核心，去湿气是辅，补气血是目的。\n总之，养好脾胃是一个需要长期坚持的过程，我们慢慢的把这些融入到我们的生活中，慢慢调理，就能切实的感受到变化，脾胃养好了，气血就充盈，也会面若桃花，这种养出来的好气色比再精致的妆都要美得多！希望大家都能脾好人美！\n\t\n#90后养生日常[话题]# #存钱不如存健康[话题]# #我的养生仪式感[话题]# #养脾胃[话题]# #脾虚人丑[话题]# #变瘦变美变好看[话题]# #变美逆袭[话题]# #内调外养好气色[话题]# #养生[话题]# #年轻人养生方式[话题]#', 'title': '养生4年心得｜原来养脾胃是有顺序的❗️', 'url': 'https://www.xiaohongshu.com/explore/671b79600000000016021398', 'date': '2024-10-25 18:56:33', 'note_type': 'images', 'video': {'noteId': '671b79600000000016021398', 'url': ''}, 'images': [], 'like_count': '25914', 'collect_count': 26509, 'view_count': '470861', 'comments': [], 'confidence': -1}, {'id': 'turn_1_25', 'search_from': 'search_note', 'content': '#重新养自己一遍[话题]# #隔离班味计划[话题]#\n🍐阿梨来啦，前面分享过的\n【脾胃不好的快手早餐吃哪些】，有姐妹偷偷已经执行起来并且真的有用哦，还没有看的姐妹去翻一翻。\n[太阳R]那么，我们除了保持良好的饮食习惯，日常生活方式也是改善脾胃健康的重要因素哦。\n[太阳R]今天分享5种方式适合脾胃的锻炼方式↓\n①散步：低强度、适合各年龄段\n[举手R]---随时随地可以进行哦\n②慢跑：有氧运动，提高身体的代谢率\n[举手R]---周末空闲时间多就可以执行起来哦\n③瑜伽：猫牛式、树式、下犬式等，促进消化系统的健、康\n[举手R]---我知道有些人不喜欢第二种方式，那么这种安安静静的瑜伽也是不错的哦\n④太极拳或者八段锦：促进脾胃的运化功能，改善气血的流通\n[举手R]---八段锦真的是特别火啊，而且动作做对了真的能感受到传统健身的力量哦\n⑤腹部按摩：促进肠胃蠕动，改善消化系统的jian康\n[举手R]---当噗噗不顺利的时候多按摩腹部（我有出了一个穴位跟练，又瘦肚子又能改善便秘），宿便排出去才不会增加肠胃压力哦\n.\n[太阳R]好啦，这几种方式有跑着的，有站着的，有坐着的，有躺着的，总有一种适合你哦。\n[害羞R]不要找借口偷懒啦，赶紧选择一种坚持起来，不管以后有多少目标计划，身体健康是首要任务[扯脸H]\n.\n🍐我是阿梨，每天分享养生小知识\n.\n#脾胃虚弱[话题]# #脾胃调理[话题]# #脾胃健康[话题]# #调理脾胃[话题]# #养脾胃[话题]# #食疗养生[话题]# #每天坚持[话题]# #中医养生[话题]# #我的健康生活[话题]# #我的养生仪式感[话题]#', 'title': '🔥脾胃不好，一周锻炼计划', 'url': 'https://www.xiaohongshu.com/explore/6652d5b2000000001303ed9a', 'date': '2024-05-26 14:24:50', 'note_type': 'images', 'video': {'noteId': '6652d5b2000000001303ed9a', 'url': ''}, 'images': [], 'like_count': '532', 'collect_count': 352, 'view_count': '45488', 'comments': [], 'confidence': -1}, {'id': 'turn_1_31', 'search_from': 'search_note', 'content': '湿热下注指湿热邪气侵犯下部脏腑，治疗以“清热利湿”为核心，结合中药、针灸和生活方式调整。根据病位和体质加减药物，推荐清热利湿食物，避免辛辣刺激，保持干爽和规律运动，防止复发。Ai生成，仅供参考！', 'title': '湿热下注治疗思路', 'url': 'https://www.xiaohongshu.com/explore/68d9b8d20000000012020852', 'date': '2025-09-29 06:38:10', 'note_type': 'images', 'video': {'noteId': '68d9b8d20000000012020852', 'url': ''}, 'images': [], 'like_count': '688', 'collect_count': 721, 'view_count': '24570', 'comments': [], 'confidence': -1}, {'id': 'turn_1_46', 'search_from': 'search_note', 'content': '当初接触到中医的湿毒这个概念时，不理解到底是什么，只是认为它又是一个不接地气的理论模型，但在这些年深入学习微生物组学后我才发现，原来这就是现代西医的SIBO（小肠细菌过度生长）。\n\t\n中医的湿毒内盛、食积郁滞是什么？\n饮食停滞不化，阻滞中焦，影响升降，导致腹胀、嗳气、反酸、食欲差、大便不畅。\n\t\n并且湿邪长期滞留体内，与热相合形成毒。\n导致身体沉重、皮肤问题、口黏、舌苔厚腻、粪便黏臭。\n\t\n后果：食欲不振、体倦乏力、脾胃虚弱等等\n\t\n而现代医学的SIBO是什么？\n小肠内细菌数量异常增多，或菌群类型异常。其核心病理是小肠自清洁功能（MMC）失灵，导致细菌从大肠逆流或在小肠内繁殖。\n症状主要表现为慢性腹胀、腹痛、腹泻或便秘（取决于产气类型），并导致营养吸收不良和全身性炎症，是许多肠道甚至全身问题的根源。\n\t\n[向右R]说人话就是：本应该呆在大肠的细菌却跑到了小肠里大量繁殖，不仅和我们抢夺营养物质，还利用这些产生大量毒素，毒害我们的身体。\n\t\n并进而导致我们出现：\n精神：焦虑、抑郁和多动症\n肠胃：肠漏、腹胀和拉肚子\n身体：慢性疲劳、气血虚弱、全身炎症，甚至非酒精型脂肪肝。\n\t\n我们咋一看，会发现，这两者原理和表现不是几乎一模一样吗？\n\t\n湿毒就是对应西医所说的细菌发酵与代谢毒素累积。\n细菌发酵产生的氢气、硫化氢、甲烷、组胺、尸胺、腐胺这类胺类会刺激神经、引发炎症、损伤黏膜。\n\t\n长期积累的后果：容易过敏、焦虑、皮肤炎症、脑雾、心某管问题，甚至严重损伤肝脑功能。\n\t\n而当我们仔细分析用于改善湿毒内盛、食积郁滞的中药其实就可以发现其中的奥妙。\n\t\n[一R]黄连： 苦寒燥湿，清热解毒的要药，针对湿毒内盛、肠道湿热（相当于炎症和毒素）。\n\t\n天然广谱抗菌剂： 黄连主要有效成分是盐酸小檗碱 (Berberine)。小檗碱是一种强大的天然抗生素，能有效抑制多种肠道有害菌（包括许多产 H2S 的细菌、致病性大肠杆菌、艰难梭菌等），防止毒素生产累计，这就对应了中医的解毒。\n\t\n[二R]神曲： 属于消食药，善于消谷积、化积滞，针对食积郁滞，帮助脾胃运化。（有点类似开菲尔酸奶）\n消化酶和益生菌来源： 神曲由面粉等和多种药材发酵而成，天然含有淀粉酶、脂肪酶等多种消化酶，以及酵母菌和乳酸菌等发酵产物。\n\t\n促进消化： 补充这些酶直接帮助分解未消化的食物 (食积)，减轻小肠负担，减少细菌的食物来源。\n\t\n下篇笔记讲讲SIBO的三大类型和鉴别方法。\n#sibo[话题]# #中医[话题]#', 'title': '中医sibo三把火：黄连、神曲、茯苓', 'url': 'https://www.xiaohongshu.com/explore/691b2dd3000000000700bed9', 'date': '2025-11-17 22:14:43', 'note_type': 'images', 'video': {'noteId': '691b2dd3000000000700bed9', 'url': ''}, 'images': [], 'like_count': '127', 'collect_count': 125, 'view_count': '3053', 'comments': [], 'confidence': -1}, {'id': 'turn_1_101', 'search_from': 'search_note', 'content': '以下是一些针对胃肠功能紊乱的饮食调理方法：\n饮食原则\n•  少食多餐：建议每天吃5~6顿小餐，而不是3顿大餐，这样可以减轻胃肠负担，让胃肠有足够的时间消化食物。\n•  定时定量：保持规律的饮食时间，每餐的量也尽量保持一致，避免过饥或过饱。\n•  饮食清淡：选择易消化、无刺激性的食物，避免油腻、辛辣、生冷等刺激性食物。\n•  细嚼慢咽：充分咀嚼食物，有助于唾液分泌，减轻胃肠负担。\n适合的食物\n•  高蛋白食物：如瘦肉、鱼虾、蛋清、脱脂牛奶等，有助于修复胃肠道黏膜。\n•  高纤维素食物：如带叶子的蔬菜、水果等，可促进胃肠蠕动，预防便秘。\n•  易消化食物：如小米粥、南瓜粥、山药粥等，对胃肠刺激小，容易被吸收。\n•  新鲜蔬菜水果：如胡萝卜、苹果、菠菜、油麦菜等，能补充维生素和矿物质。\n避免的食物\n•  辛辣刺激性食物：如辣椒、芥茉、胡椒等，会刺激胃黏膜。\n•  油腻食物：如肥肉、奶油、油炸食品等，难以消化，会增加胃肠负担。\n•  产气食物：如牛奶、巧克力、洋葱、韭菜等，容易引起腹胀。\n•  刺激性饮料：如咖啡、酒精、碳酸饮料等，会刺激胃酸分泌。\n食谱推荐\n•  莲枣山药粥：大枣、山药、莲子、小米一起煮成的粥，对改善脾胃虚弱、腹泻、反酸、胀气有很好的效果。\n•  佛手蛋：用茉莉花、佛手、鸡蛋煮成，有疏肝理气、醒脾固肠的作用。\n•  扁豆粳米粥：粳米、扁豆、梅花、白糖煮成的粥，健脾去湿，适合胃肠功能紊乱患者食用。\n特殊情况的饮食调整\n•  功能性便秘：多喝水，增加膳食纤维的摄入，如芹菜、韭菜、香蕉等，适当食用产气食物和油脂食物，有助于刺激肠道蠕动。\n•  肠易激综合征：采用低发漫饮食，增加可溶性膳食纤维，避免生冷、辛辣刺激性食物。\n其他注意事项\n•  保持良好的心态：情绪紧张、压力大等不良心理状态会影响胃肠功能，应尽量保持心情舒畅。\n•  适量运动：如散步、慢跑等有氧运动，有助于促进胃肠蠕动，改善胃肠功能。\n•  注意保暖：避免突然接触冷空气或大量吃冷饮，以免导致胃肠蠕动加快，出现疼痛和腹泻。\n以上建议仅供参考，具体的饮食调理方案应根据个人的具体情况制定，如有需要可在医生或营养师的指导下进行。', 'title': '胃肠功能紊乱如何调理饮食', 'url': 'https://www.xiaohongshu.com/explore/67dd938c000000000603ca97', 'date': '2025-03-22 00:27:56', 'note_type': 'images', 'video': {'noteId': '67dd938c000000000603ca97', 'url': ''}, 'images': [], 'like_count': '202', 'collect_count': 148, 'view_count': '16263', 'comments': [], 'confidence': -1}, {'id': 'turn_1_122', 'search_from': 'search_note', 'content': '肠易激综合征（IBS）病位在肠，\n与肝、脾、肾密切相关，\n核心病机为“脏腑功能失调、气机阻滞”，\n常见证型分为肝郁脾虚型“脾胃虚寒型”“湿热下注型”“脾肾阳虚型”。\n\t\n情绪波动致肝气犯脾，易引发腹痛腹泻；\n饮食生冷致脾胃虚寒，可能伴随畏寒、便稀；\n辛辣油腻致湿热困肠，常表现为腹痛、便黏；\n...\n长期腹泻耗伤肾阳，多出现五更泄泻、腰膝酸软，常见于压力人群、饮食不规律者、慢性肠功能紊乱患者。\n\t\n⚠️:\n内容仅供个人笔记📝分享，\n不构成任何医疗🏥指导。\n如有不适及时就医👩🏻\u200d⚕️噢～\n\t\n#学了中医才知道[话题]# #中药学[话题]# #消化与吸收[话题]# #肠易激[话题]# #肠易激腹胀[话题]# #腹胀腹泻[话题]# #肠道功能紊乱[话题]# #消化功能[话题]# #肠易激综合征[话题]# #胃肠[话题]#', 'title': '📝‖腹痛腹泻常用的4类28种中成药', 'url': 'https://www.xiaohongshu.com/explore/690489b20000000003037384', 'date': '2025-11-04 07:00:20', 'note_type': 'images', 'video': {'noteId': '690489b20000000003037384', 'url': ''}, 'images': [], 'like_count': '13', 'collect_count': 15, 'view_count': '342', 'comments': [], 'confidence': -1}, {'id': 'turn_1_134', 'search_from': 'search_note', 'content': '你是不是经常遇到这种情况？ 💩 便便软烂、粘马桶、冲不干净 💩 稍微吃点凉的/油腻的就拉肚子 💩 明明没吃坏东西，却总是腹胀、咕噜叫！\n\t\n别慌！今天教你快速判断是胃还是肠的问题，并给出调理方案！\n\t\n🔍 1分钟自测：胃or肠？\n👉 胃的问题（消化不良型） ✔️ 饭后容易胀气、打嗝 ✔️ 反酸、烧心 ✔️ 大便不成形，但无明显腹痛\n原因：胃动力差、胃酸不足 → 食物消化不完全 → 肠道负担重 → 便便稀烂\n\t\n👉 肠的问题（敏感or炎症型） ✔️ 一紧张/受凉就拉肚子 ✔️ 腹痛后马上想上厕所 ✔️ 便便带黏液（警惕肠炎！）\n原因：肠道菌群失衡、肠易激综合征（IBS）→ 肠道蠕动过快 → 水分吸收不足\n\t\n✨ 对症调理方案\n✅ 胃弱型（消化不良） ▪️ 饮食：少吃难消化食物（糯米、油炸）、细嚼慢咽 ▪️ 救急：饭后喝山楂陈皮水（促消化）\n▪️ 长期养胃：早餐吃烤馒头片、小米粥\n\t\n✅ 肠敏感型（易腹泻） ▪️ 饮食：避开生冷、乳制品、辛辣 ▪️ 救急：蒸苹果+山药粉（收敛止泻）\n▪️ 长期调理：补充益生菌、睡前揉腹\n\t\n⚠️ 什么情况要看医生？\n❗ 持续2周以上 + 体重下降 ❗ 便便有血、黑便 ❗ 夜间腹痛/腹泻（可能不是简单功能性问题）\n\t\n💬 互动时间： 你是胃弱型还是肠敏感型？\n👇 评论区分享你的经历，一起交流调理心得！\n\t\n#大便不成型[话题]# #便便软烂[话题]# #粘马桶[话题]##活谓素[话题]# #我在小红书轻养生[话题]# #腹泻[话题]# #大便不成型是胃还是肠的[话题]# #腹胀[话题]# #保护胃黏膜[话题]# #恶心呕吐[话题]# @健康薯', 'title': '大便不成型是胃还是肠的问题?', 'url': 'https://www.xiaohongshu.com/explore/68a28db1000000001d00f47d', 'date': '2025-08-18 11:18:51', 'note_type': 'images', 'video': {'noteId': '68a28db1000000001d00f47d', 'url': ''}, 'images': [], 'like_count': '267', 'collect_count': 227, 'view_count': '36311', 'comments': [], 'confidence': -1}, {'id': 'turn_1_141', 'search_from': 'search_note', 'content': '一、原因解析：为何便便“软趴趴”？\n1. 消化不良：脾胃虚弱或暴饮暴食，导致食物未充分分解，形成糊状便。\n2. 肠道疾病：慢性肠炎、溃疡性结肠炎等炎症，或肠道肿瘤、息肉等器质性病变，破坏肠道吸收功能。\n3. 菌群失衡：肠道有害菌过度繁殖，影响水分吸收，常见于滥用抗生素或长期饮食不规律。\n4. 精神压力：焦虑、紧张等情绪通过“脑-肠轴”扰乱肠道蠕动，引发肠易激综合征。\n5. 饮食不当：高脂肪、高蛋白食物过量（如油炸食品、坚果）或乳糖/麸质不耐受，导致脂肪吸收不良。\n---\n二、科学改善：从根源调节肠道功能\n❶ 饮食调整\n- 多吃膳食纤维：燕麦、红薯、绿叶菜等促进肠道蠕动，帮助成形。\n- 补充益生菌：酸奶、泡菜或益生菌制剂（如整肠生）平衡菌群。\n- 忌口清单：减少辛辣、酒精、高脂食物，避免刺激肠道黏膜。\n❷ 生活习惯优化\n- 定时排便：晨起或餐后2小时内如厕，利用“胃结肠反射”规律排便。\n- 适度运动：每天快走30分钟或练习八段锦“调理脾胃”招式，增强肠道动力。\n- 缓解压力：冥想、瑜伽降低焦虑，避免情绪性腹泻。\n❸ 中医调理\n- 穴位按摩：按揉天枢穴（肚脐旁2寸）、支沟穴（手腕上3寸）疏通肠道气机。\n- 中药方剂：脾虚者可用参苓白术散，湿热者选葛根芩连汤，需遵医嘱。\n❹ 药物干预\n- 急性期：细菌感染用头孢类抗生素，寄生虫感染用甲硝唑。\n- 慢性调理：肠炎患者可短期使用蒙脱石散吸附毒素，搭配益生菌修复黏膜。\n❺ 警惕重症信号\n若伴随以下症状，需立即就医排查肠癌等疾病：\n- 便血、黑便或白色陶土样便\n- 体重骤降、持续腹痛\n- 家族有肠道肿瘤史\n---\n三、冷知识：便便健康自测\n- 理想形态：香蕉状、表面光滑（布里斯托大便分类4型）。\n- 颜色预警：绿色（消化过快）、红色（肛肠出血）、黑色（上消化道出血）需警惕。\n---\n🌱 肠道是第二大脑，调理需耐心！转发提醒：别让“软便”拖成健康隐患～\n数据来源：\n- 中国医药信息查询平台\n- 中医调理与临床案例\n- 消化科专家解析\n*✨ 科学养肠，从每一餐开始！\n#拉粑粑自由[话题]# #有趣的知识[话题]# #促进肠道健康[话题]# #肠胃[话题]# #快乐从肠计议[话题]# #有趣的知识又增长[话题]# #肠道菌群[话题]##大便调理[话题]##消化不良[话题]# #大便问题[话题]# #健康科普[话题]# #大便[话题]# #拉肚子[话题]# #拉肚子软便[话题]# #拉肚子怎么办[话题]# #拉肚子怎么缓解[话题]# #大便困难[话题]#', 'title': '大便不成形全攻略｜5招调理肠道健康💩', 'url': 'https://www.xiaohongshu.com/explore/67ef53e7000000001c029f31', 'date': '2025-04-04 11:37:11', 'note_type': 'images', 'video': {'noteId': '67ef53e7000000001c029f31', 'url': ''}, 'images': [], 'like_count': '68', 'collect_count': 53, 'view_count': '23060', 'comments': [], 'confidence': -1}, {'id': 'turn_1_188', 'search_from': 'search_note', 'content': '一吃凉的、辣的就拉肚子？每天跑厕所N次，人都拉虚了… 分享亲测有效的调理办法，从吃、养、护全方面改善，跟“玻璃胃”说拜拜～\n\t\n🔍 先搞懂：你为啥总拉肚子？\n\t\n常见的2个原因，对号入座：\n▪️ 肠易激综合征：压力大、吃不对就拉，大便带黏液但没血，拉完肚子还隐隐痛\n▪️ 肠道菌群乱了：抗生素吃多了、熬夜多，好菌变少，肠道“扛不住”刺激\n\t\n💡 小提醒：如果拉血、发烧、体重掉，一定要先去查肠镜！排除器质性问题～\n\t\n🍚 第1步：吃对3类食物，给肠道“减负”\n\t\n▪️ 养肠主食：小米粥（熬出米油最好）、蒸山药、南瓜、软烂的白米饭（别吃杂粮饭）\n▪️ 温和蛋白质：蒸蛋羹、水煮蛋、嫩豆腐、去皮的清蒸鱼（别吃油炸肉、肥肉）\n▪️ 护肠蔬菜：胡萝卜（煮软）、土豆、冬瓜（去皮去籽，别吃韭菜、芹菜、菠菜）\n\t\n❌ 3类“雷区”千万别碰：\n\t\n• 生冷：冰饮、刺身、生腌（直接刺激肠道）\n• 辛辣：火锅、辣条、酒精（加重肠道充血）\n• 产气：牛奶、豆类、洋葱（会胀气+拉肚子）\n\t\n🌿 第2步：2个习惯帮肠道“重建秩序”\n\t\n▪️ 少食多餐，细嚼慢咽：\n每天吃5-6小餐，每顿7分饱（别撑），每口嚼20下再咽——减轻肠道消化负担，尤其别饿到胃疼才吃\n\t\n▪️ 固定排便+腹部保暖：\n每天早上起床后去厕所（养成生物钟），排便别玩手机（5分钟内搞定）；平时穿个小肚兜，别让肚子受凉（受凉超容易拉）\n\t\n🛡️ 第3步：给肠道“补点菌”，增强抵抗力\n\t\n▪️ 吃天然发酵食物：无糖酸奶（选配料表干净的，含活性菌）、纳豆、泡菜（少量，别太咸）\n▪️ 选对益生菌：认准“双歧杆菌、乳酸菌”，饭后吃（空腹吃会被胃酸杀死），坚持1-2个月才有效\n\t\n💡 别依赖益生菌！重点还是靠饮食调理，益生菌只是辅助～\n\t\n🧘 第4步：情绪+作息，比吃药更管用\n\t\n▪️ 压力大真的会拉！每天花10分钟深呼吸：吸气4秒→屏息4秒→呼气6秒，帮肠道放松\n▪️ 熬夜会毁肠道！尽量11点前睡，睡够7-8小时——肠道修复主要在夜间，睡不好恢复慢\n\t\n▪️ 轻度运动：每天散步20分钟，或做“腹式呼吸”（吸气时肚子鼓，呼气时瘪），促进肠道蠕动\n\t\n肠胃弱是“三分治七分养”，别指望一下子好透～ 按这4步坚持1-2个月，慢慢减少拉肚子频率，肠胃会越来越稳！\n\t\n\ufeff#拉肚子[话题]#\ufeff\ufeff#肠胃调理[话题]#\ufeff\ufeff#肠易激综合征[话题]#\ufeff\ufeff#无痛胃肠镜[话题]#\ufeff \ufeff#肠胃[话题]#\ufeff \ufeff#胃肠镜[话题]#\ufeff \ufeff#便秘[话题]#\ufeff \ufeff#腹痛[话题]#\ufeff\ufeff#腹泻[话题]#\ufeff\ufeff#腹泻调理[话题]#\ufeff', 'title': '肠胃弱总拉肚子？4步调理法有效不反复！', 'url': 'https://www.xiaohongshu.com/explore/689d5d80000000001c03f0e6', 'date': '2025-08-14 18:37:02', 'note_type': 'images', 'video': {'noteId': '689d5d80000000001c03f0e6', 'url': ''}, 'images': [], 'like_count': '7', 'collect_count': 6, 'view_count': '1491', 'comments': [], 'confidence': -1}, {'id': 'turn_1_214', 'search_from': 'search_note', 'content': '李女士长期遭受肠易激综合征的困扰，主要表现为腹痛、腹胀、腹泻和便秘交替出现。这些症状导致她情绪不稳，严重影响了工作和生活质量。她曾尝试多种西药治疗，但效果不佳，症状反复发作。\n在一次朋友的推荐下，李女士来到了我这里就诊。开始的时候对她进行了详细的病史询问，通过中医四诊合参的方法，诊断为脾胃虚弱、肝郁气滞导致的肠易激综合征。\n首先，通过中药汤剂调理脾胃，疏肝理气，以改善肠道功能和调节情绪。同时，结合艾灸疗法，选取相应的穴位进行刺激，以调和气血、疏通经络。此外，还指导李女士进行饮食调整，避免食用刺激性食物，增加膳食纤维的摄入。\n肠易激综合征患者在后续饮食中需要注意以下几点：\n饮食清淡：应减少食用辛辣、油腻、生冷等刺激性食物，如辣椒、油炸食品、冷饮等，以免加重肠道负担。同时，适当摄入新鲜蔬菜和水果，如香蕉、猕猴桃、西红柿、黄瓜等，以提供足够的膳食纤维和维生素。\n\t\n避免饮食过饱：过度饮食可能导致腹胀、胃烧灼感、恶心、呕吐等不适症状，因此患者应避免暴饮暴食，尽量做到少食多餐。\n\t\n减少产气食物的摄取：如豆浆、奶制品、洋葱、大豆、扁豆等，这些食物在肠道内易产生气体，加重腹胀等不适症状。\n\t\n注意食物不耐受：部分患者可能对某些食物如乳糖、麸质等存在不耐受情况，应尽量避免食用这些食物。\n\t\n均衡饮食：保持饮食的均衡和多样性，确保摄入足够的蛋白质、碳水化合物、脂肪、维生素和矿物质。\n\t\n饮食规律：尽量保持规律的进餐时间，避免长时间饥饿或暴饮暴食，有助于维持肠道运动的平衡。\n\t\n多喝水：适量饮水有助于软化粪便，促进肠道蠕动，预防便秘。\n\t\n此外，肠易激综合征患者还应注意劳逸结合，积极锻炼身体，避免精神刺激，缓解紧张情绪。同时，定期随访复查，根据医生建议调整治疗方案和饮食计划。\xa0#肠易激综合征[话题]#\xa0\xa0#中医脾胃蔡铁如[话题]#', 'title': '肠易激综合征看这一篇就够了', 'url': 'https://www.xiaohongshu.com/explore/65fff81d000000001203c2ae', 'date': '2024-03-24 19:02:40', 'note_type': 'images', 'video': {'noteId': '65fff81d000000001203c2ae', 'url': ''}, 'images': [], 'like_count': '139', 'collect_count': 64, 'view_count': '9920', 'comments': [], 'confidence': -1}, {'id': 'turn_1_222', 'search_from': 'search_note', 'content': '[向右R]肠易激综合征 IBS是一种常见的功能性胃肠道疾病，其主要症状包括腹.痛、腹胀、腹泻或便.秘等。调理肠易激需要综合饮食调整、生活方式改善以及调理肠道菌裙\n-\n‼️先判断自己是否存在肠易激👇\n[向右R]符合罗马IV标准腹.痛>3月+排便相关症状等等)\n-\n如果是，那该如何调理肠易激呢？\n[种草R]实行低FODMAP饮食：\nFODMAP(可发酵短链碳水化合物)在肠道中易被细.菌发酵产气，引发腹胀、腹.痛。\n\t\n[种草R]生活方式调整：\n1、每周5次运动，每次时长30分钟的有氧运动，改善肠道蠕动和压力反应\n2、每日补充1.5-3升水，避免咖啡.因/酒精饮料\n3、戒烟戒酒\n\t\n[种草R]补充益生菌：\n1️⃣菌qun失衡矫正:IBS患者都存在菌qun失调，益生菌竞争性抑制,减少有害菌，重建菌qun平衡:\n2️⃣肠道屏障修复:益生菌刺激闭合蛋白和闭锁小带蛋白表达，减少肠漏。\n3️⃣情绪调节:益生菌调节肠道5-羟色胺(5-HT)合成，调节情绪且缓解肠道动力紊乱。\n-\n肠易激的调理不是一蹴而就的事情～就像给肠道 “重启系统”，需要时间和耐心⏳。别因为短期看不到效果焦虑，坚持调整饮食、生活习惯，配合科学补充益生菌，肠道一定会慢慢恢复活力✨！一起养出健康好肠胃～\n\ufeff#健康科普[话题]#\ufeff \ufeff#肠易激[话题]#\ufeff \ufeff#肠易激怎么办[话题]#\ufeff \ufeff#肠易激惹综合征[话题]#\ufeff \ufeff#肠易激怎么调理[话题]#\ufeff \ufeff#肠易激调理[话题]#\ufeff \ufeff#肠易激食谱[话题]#\ufeff \ufeff#肠道健康[话题]#\ufeff \ufeff#肠易激日常[话题]#\ufeff \ufeff#健康养生[话题]#\ufeff', 'title': '肠易激该如何调理？如何转变为正常肠？', 'url': 'https://www.xiaohongshu.com/explore/68073865000000001200de1d', 'date': '2025-04-22 17:26:03', 'note_type': 'images', 'video': {'noteId': '68073865000000001200de1d', 'url': ''}, 'images': [], 'like_count': '354', 'collect_count': 205, 'view_count': '28523', 'comments': [], 'confidence': -1}, {'id': 'turn_1_223', 'search_from': 'search_note', 'content': '⭕随着生活水平提高及生活节奏加快，受胃肠功能紊乱困扰的患者日渐增多，尤其是年青人非常多。很多患者深受其扰，痛苦不堪。\n胃肠功能紊乱，临床上称之为功能性胃肠病，即经过各项检查后，并未发现胃肠道器质性病变，而出现胃肠道相关不适症状，主要包括肠易激综合征、功能性消化不良、以及功能性便秘。\n⭕1、功能性消化不良\n多表现为餐后饱胀不适、吃一点就饱、胃口差、上腹烧灼、疼痛等。\n⭕2、功能性便秘\n多表现为排便次数减少、排使困难、大便干硬、排便不尽、肛门梗阻等，需要使用泻药或开塞露通便。\n⭕3、肠易激综合征\n多表现为腹泻，每日可3-5次，多在晨起或餐后，或者腹泻便秘交替，也会出现腹痛，多在排气或排便后级解，但是无血使，部分人也会出现便秘，附有黏液等。\n⭕除了这些症状外，还会伴有不同程度的心理、精神问题，如焦虑、抑郁、失眠、头痛等，比较影响日常生活质量，但不会发生癌变\n对于该病的治疗要有耐心，遵医嘱用药，胃肠功能紊乱在自我调节上主要应该注意以下几点：\n⭕注意保暖\n由于机体内植物神经功能异常，应变能力差，所以要尽量避免突然接触冷空气，大量吃冷饮和凉拌菜，导致胃肠蠕动加快，出现疼痛和腹泻。\n⭕调整食谱\n应该尽量少吃使胃肠胀气、排气增多的食物，如牛奶、豆类、减少小麦、土豆和玉米等淀粉样食物的摄入量，因为这些食物只能被人体吸收小部分，大部分要靠大肠内部的细菌代谢分解，从而使产气增多。炒菜烹调时可以适当加入姜、蒜、胡椒等调味品，可以使排气减少。\n⭕调理情绪、生活规律\n稳定情绪，劳逸适度，这一点非常重要，因为情绪紧张可以直接加快胃肠道的蠕动导致腹泻。\n⭕加强锻炼\n可以每天早晚在空腹时按摩腹部。左手在上，从右向左顺时针转，右手在下，以肚脐眼为中心从左向右逆时针转着按摩；每次50次。这样的按摩可以改善胃肠道的运动和血液循环。\n⭕定时用餐、细嚼慢咽\n吃饭宜定时定量，少食多餐，餐与餐之间，不间隔太短或太久的时间才进食。每一顿都不宜吃得过饱，尤其是下午和晚上。\n⭕避免刺激，少吃油炸食物\n不吸烟，因为吸烟使胃部血管收缩，影响胃壁细胞的血液供应，使胃黏膜抵抗力降低而诱发胃病。应少饮酒，少吃辣椒，胡椒等辛辣食物。生冷和刺激性强的食物对消化道黏膜具有较强的刺激作用。少吃油炸食，因为这类食物不容易消化，会加重消化道负担。\n⭕温馨提示:如身体感到不适，请及时线下就医！\n#肠胃[话题]# #知识科普[话题]# #杜芳祥[话题]# #肠胃功能紊乱[话题]#', 'title': '肠胃功能紊乱，应该如何调理？', 'url': 'https://www.xiaohongshu.com/explore/663880cc000000001e01c491', 'date': '2024-05-06 15:03:40', 'note_type': 'images', 'video': {'noteId': '663880cc000000001e01c491', 'url': ''}, 'images': [], 'like_count': '128', 'collect_count': 80, 'view_count': '14589', 'comments': [], 'confidence': -1}, {'id': 'turn_1_237', 'search_from': 'search_note', 'content': '神曲捣碎，加入高丽参，放入铁锅进行小火翻炒。炒制几分钟后冷却装入容器。日常泡水饮用，会有不错效果。\n作为10年的胃胀患者，胃动力不足，这10年各大医院都看过，都是辛酸泪。\n除了日常饮食要很注意之外，吃东西切记尽量不要为了减脂瘦身就去大量服用粗粮，会给脾胃增加负担。豆浆、牛奶、洋葱等十字花科的蔬菜都尽量避开，会引发胀气。糯米制品少吃。\n☀️补充一个对我来说很重要的体会就是，避免吃寒凉的食物。例如像我每次吃1️⃣生冷的日料，不管是生鱼片还是生硬的寿司米，都会消化不良，胀气发作。2️⃣生蚝也是大寒之物，体寒的人少吃，虽然它有丰富蛋白质和💪阳效果😂。3️⃣螃蟹我一般也会少吃一点，喜欢的话我会用麻油生姜煎制后食用，麻油生姜热补的性质中和螃蟹的寒性即可食用。所以并不是不能吃，而是少量吃，脾胃养好了都能吃，烹饪手法改变后也能吃。4️⃣改天再出一贴饮食篇吧，想分享的太多了。毕竟我真的是一路亲身排雷过来的，久病成医。\n\t\n易导致胃胀气的蔬菜:\n1.十字花科蔬菜\n·西兰花、花椰菜、卷心菜、羽衣甘蓝·原因:含棉子糖和硫化物，肠道细菌分解时易产气。\n2.豆类\n·黄豆、黑豆、扁豆、鹰嘴豆(虽属豆科，常被当作蔬菜食用)\n·原因:含低聚糖，需肠道细菌分解，产生气体。\n3.洋葱与大蒜\n·原因:含果聚糖(高FODMAP)，部分人难以消化。\n4.根茎类蔬菜\n·红薯、白萝卜、甜菜根\n·原因:高纤维或甘露醇(多元醇)，可能增加胀气。\n5.其他高FODMAP蔬菜\n·芦笋、\t芹菜(茎部)。\n6.生吃蔬菜\n·生胡萝卜、生西兰花等\n·原因:生食纤维更硬，可能加重消化负担。\n替代建议:\n·低FODMAP蔬菜:黄瓜、菠菜、生菜、西葫芦、胡萝卜(熟)、南瓜、青椒。·烹饪技巧:煮熟或蒸软蔬菜，破坏纤维结构，更易消化。\n·分餐策略:少量多次摄入，避免一次性过量食用高纤维食物。\n注意事项:\n避免碳酸饮料、咀嚼过快吞咽空气等习惯，结合适度运动促进肠道蠕动。\n有个好的脾胃，就成功了一半了。[红色心形R]\n画重点[红色心形R]：情绪稳定很重要，胃是情绪的反射器。像我原先有肠易激综合征，一旦情绪不稳定的时候，胃就会开始发作难受，产生剧烈的躯体反应。所以珍爱生命，情绪稳定。远离任何让你情绪不稳定的脏东西。\n#养胃打卡[话题]# #肠易激综合征[话题]##胃胀气[话题]#', 'title': '多年的胃胀气患者', 'url': 'https://www.xiaohongshu.com/explore/67ca8586000000000900cd96', 'date': '2025-03-07 13:35:02', 'note_type': 'images', 'video': {'noteId': '67ca8586000000000900cd96', 'url': ''}, 'images': [], 'like_count': '499', 'collect_count': 651, 'view_count': '12190', 'comments': [], 'confidence': -1}, {'id': 'turn_1_260', 'search_from': 'search_note', 'content': '【⚠️胃肠功能紊乱别忽视！这些症状可能是身体在求救】\n最近总感觉肚子胀胀的，吃完东西就嗳气反酸？\n熬夜加班后突然腹痛腹泻，却查不出病因？\n明明没吃错东西，却频繁便秘腹泻交替……\n这些可能是胃肠功能紊乱在作怪！\n🚨 胃肠功能紊乱的6大典型症状\n1️⃣ 腹痛腹胀：\n· 餐后上腹隐痛或胀满感，按压时加重\n· 尤其在吃油腻、辛辣食物后更明显（比如火锅次日必拉肚子！）\n2️⃣ 排便异常：\n· 腹泻：大便稀溏，每天3次以上（喝冰咖啡必窜稀的举手！）\n· 便秘：排便困难，每周少于3次（像羊屎蛋一样干结）\n· 腹泻便秘交替：今天拉明天堵，毫无规律\n3️⃣ 消化不良：\n· 早饱感：吃半碗饭就撑得难受\n· 嗳气反酸：喉咙冒酸水，打嗝有腐臭味\n· 食欲不振：看到美食却没胃口\n4️⃣ 恶心呕吐：\n· 晨起刷牙干呕（误以为慢性咽炎）\n· 进食后呕吐，吐完胃更难受\n5️⃣ 肠道“闹情绪”：\n· 焦虑时腹痛加重，排便后缓解（肠易激综合征典型表现！）\n6️⃣ 全身乏力：\n· 长期营养吸收不良导致精神萎靡\n· 明明睡够8小时，仍感觉疲惫不堪\n🔍 自测：你是胃肠功能紊乱高危人群吗？\n✅ 饮食不规律：长期外卖、暴饮暴食、节食\n✅ 压力山大：熬夜加班、情绪焦虑、抑郁\n✅ 肠道敏感：乳糖不耐受、吃生冷就腹泻\n✅ 药物影响：长期服用抗生素、泻药\n✅ 家族史：直系亲属有肠易激综合征\n💡 三步缓解胃肠功能紊乱\n1️⃣ 饮食调理是基础\n· 规律进食：每天固定三餐时间，细嚼慢咽（每口咀嚼15次！）\n· 避开雷区：\n❌ 辛辣刺激（火锅、辣条）\n❌ 高油高糖（炸鸡、奶茶）\n❌ 产气食物（碳酸饮料、豆类）\n· 补充膳食纤维：燕麦、西兰花（促进肠道蠕动）\n· 益生菌辅助：酸奶、发酵食品（调节肠道菌群）\n2️⃣ 生活习惯大改造\n· 腹部保暖：空调房必备护腰，经期别穿露脐装\n· 适度运动：饭后散步10分钟，每周3次瑜伽\n· 睡眠管理：23点前入睡，避免熬夜（胃肠修复黄金期！）\n3️⃣ 心理调节是关键\n· 压力释放：冥想、写日记、听白噪音\n· 情绪管理：焦虑时深呼吸（4-7-8呼吸法：吸气4秒→屏息7秒→呼气8秒）\n· 认知行为疗法：严重焦虑者可寻求心理咨询\n⚠️ 出现这些信号必须立刻就医！\n▫️ 便血或黑便（警惕消化道出血）\n▫️ 体重骤降（1个月内下降5%以上）\n▫️ 持续呕吐无法进食\n▫️ 腹部包块或剧烈疼痛\n▫️ 发热或贫血（可能合并感染或器质性病变）\n#胃肠功能紊乱[话题]##北京怡德医院[话题]# #北京怡德医院体检中心[话题]#', 'title': '', 'url': 'https://www.xiaohongshu.com/explore/67fe0f5d000000001b0263d0', 'date': '2025-04-15 15:48:45', 'note_type': 'images', 'video': {'noteId': '67fe0f5d000000001b0263d0', 'url': ''}, 'images': [], 'like_count': '522', 'collect_count': 220, 'view_count': '84230', 'comments': [], 'confidence': -1}, {'id': 'turn_1_326', 'search_from': 'search_note', 'content': '吃坏肚子腹胀腹泻，或压力大时便秘反酸，总问“肠胃啥时候能好”？其实恢复时间看诱因和调理方式，分 3 种情况讲清，帮你少走弯路～\n\t\n🔍 肠胃紊乱是“失衡信号”\n\t\n多因饮食不当、情绪压力、菌群失衡导致，是功能性问题（非器官病变）。恢复关键在及时去诱因、给肠胃修复时间。\n\t\n[彩虹R] 分 3 种情况看恢复时间！\n\t\n[一R] 短期紊乱（1-2 周）：饮食 / 感染致，好得快\n• 诱因：变质食物、冰饮辣食刺激、急性肠胃炎恢复期；\n• 症状：腹泻腹胀、轻微腹痛，无黏液脓血；\n• 恢复：1-2 周；\n• 调理：吃小米粥、蒸蛋羹（避油腻粗纤维），腹泻后喝淡盐水补电解质。\n\t\n[二R] 中期紊乱（1-3 个月）：习惯 / 情绪致，需规律调\n• 诱因：熬夜、三餐不规律、工作压力大；\n• 症状：便秘腹泻交替、餐后反酸胀气，休息好症状减轻；\n• 恢复：1-3 个月；\n• 调理：固定三餐（早 7-8、午 12-13、晚 18-19 点），饭后慢走、睡前顺时针揉腹 50 圈。\n\t\n[三R] 长期紊乱（3 个月 +）：慢性诱因致，需针对性治\n• 诱因：肠易激、慢性菌群失衡、长期焦虑抑郁；\n• 症状：吃特定食物腹泻、长期便秘（每周＜3 次）；\n• 恢复：3-6 个月甚至更久；\n• 调理：先查大便常规、肠镜，肠易用药物调蠕动，菌群失衡补高活性益生菌。\n\t\n❌ 别踩 3 个拖慢恢复的坑！\n\t\n1. 症状轻就放纵：腹泻刚停吃火锅冰饮，致紊乱反复；\n2. 靠吃药掩盖：便秘用泻药、腹泻用止泻药，伤肠胃功能；\n3. 过度焦虑：加重肠脑轴紊乱，形成恶性循环。\n\t\n✅ 3 个加速恢复技巧！\n\t\n1. 记肠胃日记：找诱因（如喝牛奶腹胀就避开）；\n2. 每天喝 1500-2000ml 温水，促肠道蠕动；\n3. 每坐 1 小时起身走 5 分钟，改善肠道循环。\n\t\n恢复需耐心！短期紊乱 1-2 周好，长期紊乱明确诱因 + 科学干预也能控制。先从固定三餐、少熬夜开始，给肠胃一点时间～\n\t\n#肠胃功能紊乱[话题]##重庆胃镜[话题]##重庆无痛胃肠镜[话题]##肠胃调理[话题]##肠道健康[话题]##重庆胃肠镜[话题]##重庆胃肠镜医院[话题]##消化问题[话题]#', 'title': '肠胃功能紊乱，正常需要多久才好？', 'url': 'https://www.xiaohongshu.com/explore/690aba41000000000700013f', 'date': '2025-11-05 18:39:01', 'note_type': 'images', 'video': {'noteId': '690aba41000000000700013f', 'url': ''}, 'images': [], 'like_count': '29', 'collect_count': 13, 'view_count': '4089', 'comments': [], 'confidence': -1}, {'id': 'turn_1_386', 'search_from': 'search_note', 'content': '#四神汤[话题]# #四神汤的做法[话题]# #四神汤配方[话题]# #四神汤功效和禁忌[话题]# #四神汤挑选攻略[话题]# #四神汤一周喝几次[话题]# #四神汤怎么选[话题]# #四神汤怎么煮[话题]# #四神汤搭配[话题]# #养脾胃[话题]#', 'title': '四神汤的挑选技巧：适合体质·吃法·挑选标准', 'url': 'https://www.xiaohongshu.com/explore/693fb87f000000001e036ff9', 'date': '2025-12-15 15:27:59', 'note_type': 'images', 'video': {'noteId': '693fb87f000000001e036ff9', 'url': ''}, 'images': [], 'like_count': '1', 'collect_count': 2, 'view_count': '31', 'comments': [], 'confidence': -1}, {'id': 'turn_1_391', 'search_from': 'search_note', 'content': '药食同源目录下经典之方：四神汤。\n甘淡渗利为阳，适合脾胃比较虚弱的人食用。\n#四神汤[话题]# #脾胃健康从饮食开始[话题]# #中药养生[话题]# #养好脾胃不生病[话题]# #中医养生[话题]# #女人养生养脾胃[话题]# #补阳气祛寒湿[话题]# #养脾胃就是养命[话题]#', 'title': '药食同源又是经方：四神汤', 'url': 'https://www.xiaohongshu.com/explore/67c9990f000000002900bbd9', 'date': '2025-03-06 20:46:07', 'note_type': 'images', 'video': {'noteId': '67c9990f000000002900bbd9', 'url': ''}, 'images': [], 'like_count': '7912', 'collect_count': 8671, 'view_count': '419171', 'comments': [], 'confidence': -1}, {'id': 'turn_1_399', 'search_from': 'search_note', 'content': '元旦前一个月瘦瘦计划开始！👋今天分享央妈推荐的5+2冬断食食谱！\n-\n🍱5天正常日🎉5天里你需要摄入热量略低于你的基础DX，这样既能维持日常所需的营养，又能开启燃🔥脂模式。吃饱的同时还—秤！\n-\n🍱2天是断食日🎉在这2天里，你的热量摄入会减少到大约500-600kcal左右，这两天体重下降很明显喔～\n-很多宝子留言说为什么瘦下来后小肚子还是凸凸的，早上起床脸跟发面馒头一样，整个人感觉沉沉的，我也有过这样的经历，网上一搜才知道原来是因为💧气太重了，南方地区的天气大多都是潮湿为主，加上饮食不注意，湿气重是很难避免的我们广东人吃维特健灵祛湿轻比较多，排水就像喝水一样简单～\n-\n老广人应该对这个品牌不陌生，香港30年的老品牌了，这个是我妈妈辈都吃的，不是贴牌货 从采摘草本-生产-包装都有自己的工厂，还是GMP认证的，老广人口碑代代相传的～\n-\n✅每天早晚各1粒，我都吃空好几瓶了，慢慢的身体的正向反馈真的很多！排完‘废水‘后整个人都有种轻盈感！在冬天对减减真的很友好，感觉我的体重称每天都在给我惊喜！久坐小肚肚囤积的姐妹可以去搜搜看！\n#央妈减肥食谱[话题]##跟着央妈减肥[话题]##祛湿轻[话题]##冬断食[话题]##维特健灵祛湿轻[话题]##轻断食[话题]##食谱分享[话题]##科学饮食[话题]#', 'title': '央妈5+2轻断食食谱一周跟练！', 'url': 'https://www.xiaohongshu.com/explore/69327178000000001e00a950', 'date': '2025-12-06 20:21:08', 'note_type': 'images', 'video': {'noteId': '69327178000000001e00a950', 'url': ''}, 'images': [], 'like_count': '1036', 'collect_count': 1190, 'view_count': '14557', 'comments': [], 'confidence': -1}]
        input_dict["outline_turn_0"] = '# 湿气重综合调理报告\n\n### **一、 深入认知“湿气”：根源、成因与分类**\n\n1.  **湿气的概念与来源**\n    *   **概念定义**：“湿气”是中医理论中的一种致病因素，当体内水液运化失常，就容易形成湿气<cite>turn_0_1, turn_0_6</cite>。\n    *   **内外来源**：湿气分为“外湿”和“内湿”。外湿多由外部环境造成，如气候潮湿、久居湿地、淋雨等；内湿则主要源于脏腑功能失调，特别是脾胃运化功能减弱<cite>turn_0_6</cite>。\n\n2.  **湿气形成的关键：脾虚是元凶**\n    *   中医认为“脾主运化”，负责运化水谷精微和水液，但脾“喜燥恶湿”<cite>turn_0_6</cite>。若因饮食不节、劳累过度等因素损伤脾胃，会导致水湿无法正常代谢，在体内停滞，形成“内湿”<cite>turn_0_6, turn_0_7</cite>。\n    *   因此，健脾是祛湿的根本，否则湿气会反复产生，形成恶性循环<cite>turn_0_7, turn_0_17, turn_0_18, turn_0_38, turn_0_59</cite>。\n\n3.  **湿气的常见分类与辨析**\n    *   **湿热体质**：体内湿气与热邪结合。表现为面部和头发油腻、易长痘、口干口苦、口气重、脾气急躁、小便色黄、大便黏腻或便秘<cite>turn_0_8, turn_0_14, turn_0_69</cite>。\n    *   **寒湿体质**：体内湿气与寒邪结合。表现为怕冷、手脚冰凉、下肢易浮肿、大便稀溏、女性白带清稀量多<cite>turn_0_8, turn_0_11, turn_0_69</cite>。\n    *   **痰湿体质**：湿气凝聚成痰。表现为体型肥胖（尤其是腹部）、喉咙总有痰、身体困重、舌苔厚腻<cite>turn_0_11, turn_0_23, turn_0_69</cite>。\n\n### **二、 湿气自测：识别身体发出的“潮湿”信号**\n\n1.  **常见全身表现**\n    *   **精神状态**：感觉身体沉重、四肢无力、精神不振、容易疲劳犯困<cite>turn_0_0, turn_0_4, turn_0_6, turn_0_7, turn_0_11, turn_0_37</cite>。\n    *   **皮肤与毛发**：皮肤和头发爱出油，面部油光满面，容易长湿疹、长痘<cite>turn_0_0, turn_0_2, turn_0_4, turn_0_6, turn_0_11, turn_0_13</cite>。\n    *   **消化系统**：食欲不振、腹部胀气、口中黏腻或有异味<cite>turn_0_0, turn_0_4, turn_0_6, turn_0_7, turn_0_13</cite>。\n    *   **身材外形**：出现水肿、虚胖，体重增加<cite>turn_0_6, turn_0_7, turn_0_11</cite>。\n\n2.  **核心自测方法：观舌苔与察大便**\n    *   **观察舌头**：健康的舌头应是淡红色、薄白苔。湿气重的人通常舌体胖大，舌头边缘有牙齿压迫的痕迹（齿痕），舌苔白厚或黄腻<cite>turn_0_6, turn_0_7, turn_0_14, turn_0_37</cite>。\n    *   **观察大便**：大便质地稀软、不成形，容易粘在马桶上，不易冲干净，是湿气重的典型表现<cite>turn_0_6, turn_0_7, turn_0_14, turn_0_37</cite>。\n\n### **三、 科学调理：系统性祛湿策略**\n\n1.  **调理总纲：分步调理，事半功倍**\n    *   中医调理讲究顺序，对于复杂体质，单纯祛湿或进补可能效果不佳。一个有效的调理顺序是：**先疏肝，再健脾，后祛湿，最后补气血**<cite>turn_0_17, turn_0_18, turn_0_59</cite>。\n    *   **疏肝**：肝气郁结会影响脾胃功能（肝木克脾土），导致消化不良。可通过情绪调节、充足睡眠或饮用玫瑰花茶等方式疏肝理气<cite>turn_0_17, turn_0_59</cite>。\n    *   **健脾**：脾胃功能恢复，才能有效运化水湿，从根源上杜绝湿气生成<cite>turn_0_17, turn_0_18, turn_0_38</cite>。\n\n2.  **饮食调理：吃对食物，从内“干燥”**\n    *   **推荐食材（“红榜”）**：多吃健脾利湿的食物，如薏米、赤小豆、山药、茯苓、芡实、冬瓜、南瓜、小米等<cite>turn_0_6, turn_0_7, turn_0_13, turn_0_16</cite>。\n    *   **禁忌食物（“黑榜”）**：避免或减少生冷寒凉（如冷饮、冰淇淋）、油腻甜腻（如油炸食品、蛋糕）、辛辣刺激的食物，这些会加重脾胃负担，助长湿气<cite>turn_0_6, turn_0_7, turn_0_10, turn_0_16</cite>。\n    *   **辩证食养：祛湿茶饮与汤方**\n        *   **经典祛湿方**：赤小豆薏米水是广为人知的祛湿方。需注意：应选用细长形的**赤小豆**而非圆形红豆；薏米性寒，需**炒制**至微黄以减其寒性<cite>turn_0_14, turn_0_31, turn_0_32, turn_0_34, turn_0_38</cite>。\n        *   **分型调理茶饮**：\n            *   **湿热体质**：可用薏米、赤小豆、茯苓、荷叶、菊花等泡水，以清热利湿<cite>turn_0_11, turn_0_20, turn_0_23, turn_0_27</cite>。\n            *   **寒湿体质**：可用生姜、茯苓、党参、红枣、陈皮等煮水，以温阳散寒祛湿<cite>turn_0_11, turn_0_20, turn_0_23, turn_0_27</cite>。\n            *   **痰湿体质**：可用陈皮、茯苓、山楂、黄芪等泡水，以理气健脾、燥湿化痰<cite>turn_0_11, turn_0_20, turn_0_23, turn_0_27</cite>。\n        *   **健脾祛湿第一方**：“四神汤”（由茯苓、山药、莲子、芡实组成）性质平和，适合多数人用于健脾祛湿<cite>turn_0_15, turn_0_38</cite>。\n        *   **祛湿食谱参考**：可参考制定一周的祛湿食谱，如早餐安排红豆薏仁粥、山药，午餐搭配冬瓜汤等<cite>turn_0_10, turn_0_14</cite>。\n\n3.  **生活方式干预：养成“干爽”好习惯**\n    *   **规律作息**：保证充足睡眠，避免熬夜，有助于脏腑自我修复<cite>turn_0_0, turn_0_2, turn_0_4, turn_0_6</cite>。\n    *   **适度运动**：进行快走、瑜伽、跑步等运动，使身体微微出汗，是排出湿气的有效途径<cite>turn_0_1, turn_0_6, turn_0_10</cite>。\n    *   **改善环境**：保持居住环境的干燥通风，衣物要晒干再穿，洗澡或淋雨后及时擦干身体<cite>turn_0_2, turn_0_4, turn_0_6</cite>。\n    *   **坚持泡脚**：用温水或加入生姜、花椒、陈皮等泡脚，可以促进循环，帮助祛除寒湿<cite>turn_0_0, turn_0_2, turn_0_4, turn_0_13, turn_0_16</cite>。\n\n4.  **外部疗法：艾灸温阳化湿**\n    *   **原理与功效**：艾灸通过温热刺激穴位，能够温通经络、补足阳气、活血化瘀，从而有效驱散体内的寒湿之气<cite>turn_0_40, turn_0_41, turn_0_47, turn_0_83</cite>。\n    *   **核心祛湿穴位**：\n        *   **健脾胃穴位**：中脘穴、足三里<cite>turn_0_40, turn_0_63</cite>。\n        *   **祛湿要穴**：阴陵泉（祛湿）、丰隆穴（化痰湿）<cite>turn_0_53, turn_0_61, turn_0_63, turn_0_89</cite>。\n        *   **补阳固本穴位**：关元穴、神阙穴（肚脐）<cite>turn_0_39, turn_0_40, turn_0_63</cite>。\n    *   **分型选穴与操作**：\n        *   **寒湿体质**：可重点灸关元、中脘、足三里等穴位<cite>turn_0_69</cite>。\n        *   **湿热体质**：可选择曲池、合谷、阴陵泉等穴位<cite>turn_0_69</cite>。\n        *   **操作要点**：每穴艾灸10-20分钟，每周进行3-5次。灸后注意保暖，避免吹风和接触冷水，并多喝温水<cite>turn_0_39, turn_0_40, turn_0_41, turn_0_65, turn_0_83</cite>。\n\n### **四、 总结与建议**\n\n1.  **长期坚持与个体化方案**：祛湿是一个需要长期坚持生活方式调理的慢性过程<cite>turn_0_1</cite>。同时，每个人的体质不同，需要根据自身具体情况（如寒热、虚实）选择最适合自己的方法，切忌盲目跟风<cite>turn_0_11, turn_0_32, turn_0_38</cite>。\n2.  **专业咨询建议**：如果湿气症状严重，或伴有其他疾病，亦或自身无法准确判断体质，建议及时咨询专业的中医师，在医生指导下进行针对性治疗<cite>turn_0_11, turn_0_13</cite>。'

        # input_dict = await service.act(
        #     input_dict=input_dict,
        #     turn_id=1,
        # )
        # print(input_dict)
        outline = '# **湿气重全面调理报告**\n\n# 一、 导言：认识与辨识体内湿气\n本部分旨在介绍“湿气”的基本概念，并提供一套系统的自我辨识方法，帮助您初步判断自身是否存在湿气过重的问题。\n\n## 1.1 什么是湿气\n湿气，在中医理论中，被视为一种“阴邪”，易损伤人体阳气，阻滞气机<cite>turn_0_3</cite>。当体内水液代谢失衡，湿邪便会滞留体内，引发一系列不适症状<cite>turn_0_235</cite>。现代生活中，不良的饮食习惯、缺乏运动、环境潮湿等因素均可能导致湿气过重<cite>turn_0_99, turn_0_275</cite>。\n\n## 1.2 湿气重的典型表现\n您可以通过观察以下几方面的身体信号，来综合判断体内湿气状况：\n\n### 1.2.1 舌象观察：最直观的“晴雨表”\n- **舌苔**：舌苔白厚、厚腻，甚至刮不干净，是湿气重的典型特征<cite>turn_0_0, turn_0_5, turn_0_12, turn_0_27, turn_0_39</cite>。若舌苔发黄且厚腻，则可能为湿热表现<cite>turn_0_5, turn_0_22, turn_0_27</cite>。\n- **舌体**：舌体胖大，边缘有明显的牙齿印记（齿痕舌），多因脾虚湿盛，水湿停滞所致<cite>turn_0_0, turn_0_5, turn_0_7, turn_0_11, turn_0_22, turn_0_28</cite>。\n\n### 1.2.2 身体感受：沉重与疲乏\n- **精神状态**：早晨起床困难，感觉昏昏沉沉，精神不振，即使睡眠充足也感觉疲劳<cite>turn_0_0, turn_0_7, turn_0_21, turn_0_28</cite>。\n- **肢体感觉**：感觉身体、四肢沉重，像“背了沙袋”或“裹了湿衣服”一样，懒得活动<cite>turn_0_6, turn_0_7, turn_0_28, turn_0_116, turn_0_235</cite>。\n\n### 1.2.3 排泄状况：黏滞不爽\n- **大便**：大便形态软烂、不成形，或黏腻，容易粘在马桶上，不易冲干净，总有排不尽的感觉<cite>turn_0_6, turn_0_7, turn_0_8, turn_0_28, turn_0_116</cite>。\n- **小便**：小便可能出现短黄、浑浊等情况，尤其在湿热体质中更为常见<cite>turn_0_16, turn_0_44, turn_0_174</cite>。\n\n### 1.2.4 皮肤与外貌：油腻与浮肿\n- **面部与头发**：面部容易出油，头发刚洗完不久就变得油腻<cite>turn_0_7, turn_0_16, turn_0_28, turn_0_44</cite>。\n- **皮肤问题**：容易长痘痘、粉刺或湿疹，皮肤摸起来有些黏腻<cite>turn_0_7, turn_0_16, turn_0_44</cite>。\n- **体型与浮肿**：出现虚胖，尤其是腹部赘肉松软、小肚子突出<cite>turn_0_8, turn_0_11, turn_0_21</cite>。早晨起床时可能出现脸部、眼睑浮肿<cite>turn_0_7, turn_0_37</cite>。\n\n### 1.2.5 其他相关症状\n- **消化系统**：食欲不振，吃一点就觉得胃胀<cite>turn_0_7, turn_0_11, turn_0_223</cite>。\n- **口腔问题**：口中发黏，或伴有口苦、口臭<cite>turn_0_9, turn_0_12, turn_0_21, turn_0_116</cite>。\n- **关节**：在阴雨天或潮湿环境下，关节可能感到酸痛、沉重<cite>turn_0_7, turn_0_9, turn_0_116</cite>。\n\n# 二、 湿气的成因与分类\n了解湿气的来源和具体类型，是精准调理的第一步。中医认为，湿气的形成有内、外两大因素，并根据其性质和兼夹的病邪，分为不同类型。\n\n## 2.1 湿气的主要成因\n### 2.1.1 内因：脾胃虚弱是根本\n- **脾主运化**：中医认为“脾主运化”，负责将食物和水液转化为能量并输送全身<cite>turn_0_21</cite>。若脾胃功能虚弱，运化水湿的能力下降，多余的水分就会在体内积聚成湿<cite>turn_0_6, turn_0_12, turn_0_35, turn_0_255</cite>。\n- **肝郁克脾**：长期压力大、情绪不畅导致肝气郁结，也会影响脾胃的运化功能，间接导致湿气内生<cite>turn_0_189, turn_0_20</cite>。\n\n### 2.1.2 外因：生活与环境的渗透\n- **饮食不节**：过食生冷（如冷饮、沙拉）、油腻、甜腻、辛辣的食物，会直接损伤脾胃，加重湿气<cite>turn_0_0, turn_0_35, turn_0_100, turn_0_192</cite>。“湿从口入”是重要原因之一<cite>turn_0_0</cite>。\n- **环境潮湿**：长期生活在潮湿的环境（如南方回南天<cite>turn_0_239, turn_0_242, turn_0_255</cite>）、淋雨后未及时擦干、衣物不干爽等，外界的湿邪会侵入人体<cite>turn_0_35, turn_0_256, turn_0_275</cite>。\n- **不良习惯**：久坐不动导致气血循环不畅<cite>turn_0_11, turn_0_75</cite>、长期吹空调使汗液无法正常排出<cite>turn_0_35, turn_0_75</cite>、熬夜损耗阳气等，都会削弱身体排湿能力<cite>turn_0_24, turn_0_40, turn_0_100</cite>。\n\n## 2.2 湿气的常见分类\n湿气常与其他病邪结合，形成不同类型的湿证，调理方法各有侧重<cite>turn_0_17, turn_0_177</cite>。\n- **寒湿体质**：如同身体里的“冰箱”，表现为怕冷、手脚冰凉、舌苔白腻、大便稀溏<cite>turn_0_9, turn_0_44, turn_0_120</cite>。\n- **湿热体质**：如同身体里的“蒸笼”，表现为面部油光、爱长痘、口干口苦、小便黄、大便黏臭、舌苔黄腻<cite>turn_0_9, turn_0_44, turn_0_120, turn_0_174</cite>。\n- **痰湿体质**：如同身体里的“淤泥”，常表现为体型肥胖、浮肿、胸闷痰多、嗜睡、身体困重、舌苔厚腻<cite>turn_0_6, turn_0_9, turn_0_44</cite>。\n- **上热下寒**：身体同时存在两种矛盾症状，上半身易上火（如口腔溃疡、长痘），下半身却怕冷（如手脚冰凉、腹泻），舌象常表现为舌尖红、舌根白腻<cite>turn_0_18, turn_0_32, turn_0_43</cite>。\n- **其他类型**：还包括季节性的暑湿（夏季特有）和影响关节的风湿等<cite>turn_0_9</cite>。\n\n# 三、 核心调理策略：以内养为本，多维祛湿\n调理湿气需遵循系统性原则，核心在于“健脾胃，断湿源”，并结合生活方式的全面调整<cite>turn_0_138, turn_0_122</cite>。这是一个需要持之以恒的长期过程<cite>turn_0_15, turn_0_124</cite>。\n\n## 3.1 饮食调理：吃对食物，从源头祛湿\n饮食是调理湿气的重中之重，应遵循“清淡、温和、易消化”的原则。\n\n### 3.1.1 饮食禁忌：避开“生湿”食物\n- **忌生冷寒凉**：戒除冷饮、冰淇淋、冰镇水果、凉拌菜等，以免损伤脾胃阳气<cite>turn_0_0, turn_0_1, turn_0_35, turn_0_105</cite>。\n- **忌油腻甜腻**：减少炸鸡、烧烤、蛋糕、奶茶等高油高糖食物，这些会加重脾胃负担，滋生痰湿<cite>turn_0_0, turn_0_35, turn_0_100, turn_0_160</cite>。\n- **忌滋补过度**：在湿气未清时，盲目进补（如阿胶、鹿角胶）会“闭门留寇”，越补越堵<cite>turn_0_51, turn_0_160, turn_0_181</cite>。\n\n### 3.1.2 推荐食材：善用“祛湿”帮手\n- **健脾祛湿类**：山药、茯苓、芡实、莲子、白扁豆是健脾的经典食材，常被用于“四神汤”等食疗方<cite>turn_0_31, turn_0_45, turn_0_48, turn_0_168</cite>。\n- **利水渗湿类**：赤小豆、炒薏米、冬瓜（带皮效果更佳）、玉米须，有助于将体内多余水分排出<cite>turn_0_31, turn_0_136, turn_0_151, turn_0_196</cite>。\n- **行气化湿类**：陈皮、生姜、砂仁等，能理顺气机，帮助化开中焦的湿滞<cite>turn_0_31, turn_0_33, turn_0_122</cite>。\n\n### 3.1.3 食疗茶饮方\n- **针对寒湿**：可饮用姜丝丁香茶、姜枣茶，以温中散寒<cite>turn_0_3, turn_0_136, turn_0_165</cite>。\n- **针对湿热**：可饮用荷叶山楂茶、绿豆薏米汤、菊花金银花茶等，以清热利湿<cite>turn_0_3, turn_0_159, turn_0_165, turn_0_180</cite>。\n- **针对痰湿**：可饮用山药茯苓茶、陈皮茯苓茶，以健脾化痰<cite>turn_0_3, turn_0_165, turn_0_180</cite>。\n\n## 3.2 运动与作息：动以升阳，静以养血\n规律的运动和作息是激发身体自身排湿能力的关键。\n\n### 3.2.1 坚持适度运动\n- **运动原则**：选择温和的有氧运动，以“微微出汗”为宜，避免大汗淋漓损伤阳气<cite>turn_0_0, turn_0_132, turn_0_162, turn_0_210</cite>。出汗是排出湿气的重要途径<cite>turn_0_1, turn_0_201</cite>。\n- **运动推荐**：快走、散步、慢跑、瑜伽、游泳、跳绳等都是不错的选择<cite>turn_0_100, turn_0_132, turn_0_223, turn_0_210</cite>。爬山能很好地调和气血、祛湿排毒<cite>turn_0_205, turn_0_217</cite>。\n- **传统功法**：八段锦、五禽戏、太极拳等传统养生功法，动作缓和，能有效调理脾胃、疏通经络<cite>turn_0_0, turn_0_221, turn_0_231</cite>。\n\n### 3.2.2 保证规律作息\n- **拒绝熬夜**：尽量在晚上11点前入睡，保证充足睡眠。长期熬夜会耗损阳气和精气，加重体内湿气<cite>turn_0_1, turn_0_24, turn_0_100, turn_0_132</cite>。\n- **避免久坐**：每坐一小时起身活动，伸展四肢，促进气血流通，防止湿气在下半身积聚<cite>turn_0_11, turn_0_256</cite>。\n\n## 3.3 生活习惯调整：细节防湿，内外兼修\n- **环境干燥**：保持居住环境的通风与干燥，尤其在回南天或梅雨季，可使用除湿机或空调的除湿功能<cite>turn_0_99, turn_0_240, turn_0_252, turn_0_268</cite>。衣物被褥要彻底晾干，勤晒太阳<cite>turn_0_150, turn_0_256</cite>。\n- **注意保保暖**：避免身体直接吹空调、风扇，洗澡、洗头后要及时擦干、吹干<cite>turn_0_1, turn_0_256</cite>。\n- **情绪舒畅**：保持心情愉悦，避免长期焦虑、抑郁等不良情绪，因情绪会影响肝脾功能，进而影响水湿代谢<cite>turn_0_35, turn_0_189</cite>。\n\n# 四、 外部辅助疗法：多途径加速湿气排出\n除了内调，中医外治法也是祛除湿气的有效手段，可以作为综合调理的补充。\n\n## 4.1 艾灸疗法：温阳散寒，化湿通络\n- **原理与优势**：艾灸通过温热刺激穴位，能温补阳气、驱寒散湿，尤其适合寒湿和脾虚湿盛的体质<cite>turn_0_101, turn_0_105, turn_0_132</cite>。\n- **常用祛湿穴位**：\n    - **健脾胃要穴**：足三里、中脘、脾俞、太白<cite>turn_0_101, turn_0_102, turn_0_111, turn_0_129, turn_0_146</cite>。\n    - **利水排湿要穴**：阴陵泉、丰隆、承山<cite>turn_0_36, turn_0_101, turn_0_109, turn_0_111, turn_0_112, turn_0_121</cite>。\n    - **温补元气要穴**：关元、气海、神阙<cite>turn_0_101, turn_0_114, turn_0_139, turn_0_141</cite>。\n- **注意事项**：艾灸后应避风寒，2小时内不宜洗澡<cite>turn_0_144</cite>。易上火体质可加灸涌泉穴引火下行<cite>turn_0_123, turn_0_144</cite>。体质不明者建议在专业人士指导下进行<cite>turn_0_102</cite>。\n\n## 4.2 拔罐疗法：疏通经络，排出瘀湿\n- **原理与作用**：拔罐利用负压吸附于体表，造成局部充血或瘀血，以达到疏通经络、行气活血、祛风散寒、排出湿气的目的<cite>turn_0_60, turn_0_61, turn_0_63, turn_0_81, turn_0_91</cite>。\n- **罐印解读**：拔罐后留下的印记颜色可在一定程度上反映身体状况。如印记紫黑可能提示供血不足、有血瘀；出现水泡则表明体内湿气较重<cite>turn_0_61, turn_0_65, turn_0_97</cite>。但罐印颜色深浅也与拔罐力度、留罐时间有关，不应作为唯一评判标准<cite>turn_0_62</cite>。\n- **注意事项与禁忌**：\n    - **非人人适宜**：体质虚弱、气血不足者、孕妇、皮肤病患者及有出血倾向者不宜拔罐<cite>turn_0_65, turn_0_78</cite>。拔罐属于“泄法”，过度使用会耗伤正气<cite>turn_0_60, turn_0_78, turn_0_93</cite>。\n    - **操作规范**：不宜在同一部位频繁拔罐，留罐时间一般为5-15分钟<cite>turn_0_62, turn_0_68, turn_0_78</cite>。拔罐后需注意保暖，8小时内不宜洗澡<cite>turn_0_56, turn_0_60</cite>。\n\n## 4.3 刮痧与按摩\n- **刮痧**：作用于肌表，通过刮拭经络达到活血化瘀、疏通经络的效果，尤其适合湿热体质<cite>turn_0_56, turn_0_64, turn_0_86</cite>。\n- **穴位按摩**：日常可按揉足三里、阴陵泉、丰隆等祛湿穴位，每个穴位3-5分钟，是一种简便易行的保健方法<cite>turn_0_75, turn_0_100, turn_0_111, turn_0_275</cite>。\n\n## 4.4 泡脚疗法\n- **作用**：睡前用热水泡脚，可以促进血液循环，疏通经络，通过发汗帮助排出寒湿<cite>turn_0_1, turn_0_105, turn_0_110</cite>。\n- **方法**：水温约40℃，浸泡15-20分钟至微微出汗即可。可在水中加入生姜、艾叶、花椒等增强效果<cite>turn_0_122, turn_0_132, turn_0_219</cite>。\n\n# 五、 调理效果评估与总结\n调理湿气是一个持续改善的过程，通过观察身体的积极变化，可以评估调理效果，并坚定持续调理的信心。\n\n## 5.1 湿气排出的积极信号\n当调理方法奏效时，身体会发出一系列积极信号，表明湿气正在减少<cite>turn_0_15</cite>：\n- **舌苔变薄**：舌苔由厚腻变得薄白、干净<cite>turn_0_15, turn_0_48</cite>。\n- **精神改善**：精力变得充沛，不再昏沉嗜睡<cite>turn_0_15</cite>。\n- **大便成形**：大便不再黏腻，变得通畅易排出<cite>turn_0_15</cite>。\n- **身体轻盈**：四肢沉重感消失，浮肿减轻，体重可能下降<cite>turn_0_15, turn_0_2</cite>。\n- **皮肤清爽**：面部和头发的出油情况改善，痘痘减少<cite>turn_0_15</cite>。\n- **口气清新**：口中黏腻感和异味消失<cite>turn_0_15, turn_0_24</cite>。\n- **睡眠改善**：入睡更容易，睡眠质量提高<cite>turn_0_15</cite>。\n\n## 5.2 结语：构建长期健康的“无湿”体质\n调理湿气并非一蹴而就，而是一场关乎生活方式的修行<cite>turn_0_124</cite>。其核心在于深刻理解“辨证施治”的重要性，即根据自身寒、热、虚、实的不同体质类型，选择最适合的调理方案<cite>turn_0_177, turn_0_189</cite>。所有祛湿方法都应以“健脾”为根本，因为脾胃是运化水湿的枢纽，只有脾胃功能强大，才能从源头上杜绝湿气的生成<cite>turn_0_42, turn_0_122, turn_0_138</cite>。通过长期坚持健康的饮食、规律的运动、平和的心态和良好的作息，结合必要的中医外治法，您将能逐步构建一个清爽、轻盈、充满活力的健康体质。'
        
        input_dict = await service.act(
            input_dict=input_dict,
            turn_id=1,
        )
        breakpoint()
        # outline = input_dict["outline_turn_0"]
        chunks = service.divide_outline_into_chunks(outline)
        breakpoint()
    
    asyncio.run(main())
        
        
        
        
        
        
        
