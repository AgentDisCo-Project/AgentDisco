import gin
import jinja2
import re
import os
import json
import asyncio
import logging
import sys

from api.NoteFilterService import NoteFilter

sys.path.append('.')

from functools import partial
from datetime import datetime
from typing import Optional, Union, Dict, List
from agent.BaseAgent import BasicAgent
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.key_operator import ApiKeyCycler
from api.utils.string_operator import json_fix
from tool.WebSearchService import WebSearch
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


@gin.configurable()
class OutlineGenerator(BasicAgent):
    def __init__(
        self,
        name: Optional[str] = "outline_generator",
        description_en: Optional[str] = "An outline generator.",
        description_zh: Optional[str] = "报告大纲生成器。",
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
        system_template_en_file: str = "OutlineGenerator_EN.jinja2",
        system_template_zh_file: str = "OutlineGenerator_ZH.jinja2",
        system_template_en_file_combine: str = "OutlineGeneratorCombine_EN.jinja2",
        system_template_zh_file_combine: str = "OutlineGeneratorCombine_ZH.jinja2",
        system_template_en_file_evidence: str = "OutlineGeneratorEvidence_EN.jinja2",
        system_template_zh_file_evidence: str = "OutlineGeneratorEvidence_ZH.jinja2",
        system_template_en_file_evidence_combine: str = "OutlineGeneratorEvidenceCombine_EN.jinja2",
        system_template_zh_file_evidence_combine: str = "OutlineGeneratorEvidenceCombine_ZH.jinja2",
        include_prev_query: bool = False,
        include_summary: bool = False,
        num_searches: int = 10,
        llm_filter_model_name: str = "",
        search_engine: str = "google",
        top_k: int = 50,
        return_score: bool = True,
        use_flash_filter: bool = False,
        use_evidence_as_key: bool = False,
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
        
        if search_engine == "combine":
            if not use_evidence_as_key:
                self.jinja_file = system_template_zh_file_combine if self.use_zh else system_template_en_file_combine
            else:
                self.jinja_file = system_template_zh_file_evidence_combine if self.use_zh else system_template_en_file_evidence_combine
        else:
            if not use_evidence_as_key:
                self.jinja_file = system_template_zh_file if self.use_zh else system_template_en_file
            else:
                self.jinja_file = system_template_zh_file_evidence if self.use_zh else system_template_en_file_evidence
        
        self.max_concurrent = max_concurrent
        self.include_prev_query = include_prev_query
        
        self.searcher = WebSearch(
            use_zh=use_zh,
            num_searches=num_searches,
            search_engine=search_engine,
        )
        
        self.filter = NoteFilter(
            model_name=llm_filter_model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            include_search_query=True,
            search_engine=search_engine,
            top_k=top_k,
            return_score=return_score,
            use_flash_filter=use_flash_filter,
        )
        self.include_summary = include_summary
        self.search_engine = search_engine
        self.use_evidence_as_key = use_evidence_as_key
    
    
    def get_system_prompt(
        self,
        include_prev_outline: bool = False,
        include_prev_query: bool = False,
        include_search: bool = False,
    ):
        if self.search_engine == "combine":
            template_vars = {
                "include_prev_outline": include_prev_outline,
                "include_prev_query": include_prev_query,
                "include_search": include_search,
                "curr_date": datetime.now().strftime("%Y年%m月%d日"),
            }
        else:
            template_vars = {
                "include_prev_outline": include_prev_outline,
                "include_prev_query": include_prev_query,
                "include_search": include_search,
                "curr_date": datetime.now().strftime("%Y年%m月%d日"),
                "search_engine": self.search_engine,
            }
        
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
        
        if "outline" not in response or "is_finish" not in response:
            raise ValueError()
        if self.search_engine == "combine":
            if "xiaohongshu_search_query" not in response or "google_search_query" not in response:
                raise ValueError()
        else:
            if "search_query" not in response:
                raise ValueError()
        if not isinstance(response["outline"], str) or not isinstance(response["search_query"], List) or not isinstance(response["is_finish"], bool):
            raise ValueError() 
        
        if "# " not in response["outline"]:
            raise ValueError() 
        
        return response
    
    
    def check_func(
        self,
        response: str,
    ):
        return self.parser_response(response)
    
    
    @staticmethod
    def count_reference(
        response: str
    ):
        pattern = r'<cite>(.*?)</cite>'
        return len(re.findall(pattern, response))


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
        
        if not outline or not outline.strip():
            return []
        
        # 首先按一级标题分割
        chunks = split_by_header_level(outline, r'^#\s+')
        
        # 如果只有一个chunk，按二级标题重新分割
        if len(chunks) <= 1:
            chunks = split_by_header_level(outline, r'^##\s+')

            if len(chunks) >= 2 and not contains_level2_title(chunks[0]['content']):
                # 将第一个chunk与第二个chunk合并
                merged_content = chunks[0]['content'] + '\n' + chunks[1]['content']
                merged_chunk = {
                    'content': merged_content
                }
                # 更新chunks列表：第一个chunk变成合并后的chunk，剩余的chunk从第三个开始
                chunks = [merged_chunk] + chunks[2:]
        
        for idx, chunk in enumerate(chunks):
            chunk["id"] = idx
        return chunks
    
    
    async def act(
        self,
        input_dict: Dict,
        turn_id: int,
    ):
        query_text = input_dict["query_text"]
        
        include_prev_outline = turn_id > 0
        include_prev_query = turn_id > 0 and self.include_prev_query
        include_search = turn_id > 0 and f"search_result_turn_{turn_id-1}" in input_dict
        
        if "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            system_prompt = [
                {
                    "type": "text",
                    "text": self.get_system_prompt(
                        include_prev_outline=include_prev_outline,
                        include_prev_query=include_prev_query,
                        include_search=include_search
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
                        "type": "text",
                        "text": _user_prompt
                    }
                )   
            
            if include_prev_query:
                prev_queries = input_dict[f"query_{turn_id-1}"]
                if self.use_zh:
                    _user_prompt = f"""
# 上一轮迭代的搜索词
{prev_queries}
"""
                
                else:
                    _user_prompt = f"""
# Previous Search Query Terms
{prev_queries}
"""
                
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt
                    }
                )
            
            
            if include_search:
                if not self.use_evidence_as_key:
                    if self.use_zh:
                        _user_prompt = f"""
# 外源搜索结果
"""
                    else:
                        _user_prompt = f"""
# External Search Results
"""
            
                    user_prompt.append(
                        {
                            "type": "text",
                            "text": _user_prompt,
                        }
                    )
                
                    docs = input_dict[f"search_result_turn_{turn_id-1}"]
                    for doc in docs:
                        idx, title, content, search_from = doc["id"], doc["title"], doc["content"], doc["search_from"]
                        include_summary = ("google" in search_from or self.include_summary) and ("summary" in doc)
                        content = content if not include_summary else doc["summary"]
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
                                "type": "text",
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
                    
                    user_prompt.append(
                        {
                            "type": "text",
                            "text": _user_prompt,
                        }
                    )
                    
                    evidences = input_dict[f"search_evidence_turn_{turn_id-1}"]
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
                        user_prompt.append(
                            {
                                "type": "text",
                                "text": _user_prompt,
                            }
                        )
            
            else:
                if self.use_zh:
                    _user_prompt = f"""
# 无外源搜索结果
"""
                else:
                    _user_prompt = f"""
# External Search Results are Empty
"""
                
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt,
                    }
                )
            
            
            
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
                    "text": self.get_system_prompt(
                        include_prev_outline=include_prev_outline,
                        include_prev_query=include_prev_query,
                        include_search=include_search
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
                user_prompt.append({"text": _user_prompt})
            
            if include_prev_query:
                prev_queries = input_dict[f"query_{turn_id-1}"]
                if self.use_zh:
                    _user_prompt = f"""
# 上一轮迭代的搜索词
{prev_queries}
"""
                
                else:
                    _user_prompt = f"""
# Previous Search Query Terms
{prev_queries}
"""
                
                user_prompt.append({"text": _user_prompt})
            
            
            if include_search:
                if not self.use_evidence_as_key:
                    if self.use_zh:
                        _user_prompt = f"""
# 外源搜索结果
"""
                    else:
                        _user_prompt = f"""
# External Search Results
"""
            
                    user_prompt.append({"text": _user_prompt})
                
                    docs = input_dict[f"search_result_turn_{turn_id-1}"]
                    for doc in docs:
                        idx, title, content, search_from = doc["id"], doc["title"], doc["content"], doc["search_from"]
                        include_summary = ("google" in search_from or self.include_summary) and ("summary" in doc)
                        content = content if not include_summary else doc["summary"]
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
                    
                        user_prompt.append({"text": _user_prompt})
                
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
                    
                    evidences = input_dict[f"search_evidence_turn_{turn_id-1}"]
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
                
            else:
                if self.use_zh:
                    _user_prompt = f"""
# 无外源搜索结果
"""
                else:
                    _user_prompt = f"""
# External Search Results are Empty
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
                check_func=self.parser_response,
                return_cot=False,
            )
        
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        if not response["is_finish"]:
            if self.search_engine == "combine":
                response["xiaohongshu_search_query"].append(query_text)
                response["google_search_query"].append(query_text)
            else:
                response["search_query"].append(query_text)
        input_dict[f"outline_turn_{turn_id}"] = response["outline"]
        input_dict[f"is_finish_turn_{turn_id}"] = response["is_finish"]
        if self.search_engine == "combine":
            input_dict[f"xiaohongshu_search_query_turn_{turn_id}"] = response["xiaohongshu_search_query"]
            input_dict[f"google_search_query_turn_{turn_id}"] = response["google_search_query"]
        else:
            input_dict[f"search_query_turn_{turn_id}"] = response["search_query"]
        
        if len(response.get("search_query", [])) > 0 or len(response.get("xiaohongshu_search_query", [])) > 0 and len(response.get("google_search_query", [])) > 0:
            search_docs = await self.searcher.call({
                "query_text": input_dict["query_text"],
                "search_query": response.get("search_query", []),
                "turn_id": f"turn_{turn_id}",
                "xiaohongshu_search_query": response.get("xiaohongshu_search_query", []),
                "google_search_query": response.get("google_search_query", [])
            })
            input_dict[f"search_result_turn_{turn_id}"] = search_docs
        
        # cnt_references = self.count_reference(response["outline"])
        # logging.info(f"count number of references: {cnt_references}")
        return input_dict, turn_id
    

if __name__ == "__main__":
    async def main():
        service = OutlineGenerator(
            model_name="gemini-2.5-pro",
            use_zh=True,
        )
        input_dict = dict()
        input_dict["query_text"] = input_dict["query"] = "湿气重怎么调理？"
        outline = '# 湿气重调理全方位指南\n\n## 1. 认识湿气：成因、分类与自我诊断\n- **1.1 湿气的来源**: 湿气分为内湿和外湿。内湿主要与脾胃运化功能失调有关，不良饮食习惯如过食生冷、油腻、甜食会加重内湿<cite>turn_0_137, turn_0_59</cite>。外湿则与气候和生活环境潮湿相关，如淋雨、久居湿地、常处空调房等<cite>turn_0_72, turn_0_137</cite>。长期熬夜、久坐不动等不良生活习惯也会耗损阳气，加重体内湿气<cite>turn_0_59, turn_0_72</cite>。\n- **1.2 湿气重的典型症状**: 身体湿气重时会发出多种信号，简易自测方法包括观察：\n    - **舌头**: 舌苔厚腻，发白或发黄，舌体胖大有齿痕<cite>turn_0_8</cite>。\n    - **皮肤与头发**: 面部和头发易出油，皮肤长痘或湿疹<cite>turn_0_8, turn_0_96</cite>。\n    - **身体感觉**: 精神不振，浑身乏力困重，感觉身体像被湿布包裹<cite>turn_0_8</cite>。\n    - **消化与排泄**: 大便黏腻，粘在马桶上不易冲净，腹部胀满<cite>turn_0_8, turn_0_94</cite>。\n- **1.3 湿气的常见分类**: 根据《黄帝内经》理论，湿气可结合体质分为不同类型，常见的有<cite>turn_0_96</cite>：\n    - **湿热体质**: 表现为面部油光、易长痘、口干口苦、大便黏滞<cite>turn_0_96, turn_0_94</cite>。\n    - **湿寒体质**: 表现为手脚冰凉、怕冷、容易水肿、经期不适<cite>turn_0_96, turn_0_94</cite>。\n    - **痰湿体质**: 表现为体型肥胖、喉咙有痰、舌苔厚腻、易疲劳<cite>turn_0_96, turn_0_94</cite>。\n    - **上热下寒**: 表现为上半身易上火，但下半身却怕冷<cite>turn_0_96, turn_0_84</cite>。\n\n## 2. 饮食调理：对症“吃”掉湿气\n- **2.1 宜食清单**: 日常应多吃健脾祛湿、利水消肿的食物<cite>turn_0_31</cite>。\n    - **主食**: 推荐小米、燕麦、红豆、薏米仁、玉米、山药、芡实等<cite>turn_0_31, turn_0_59</cite>。\n    - **蔬菜**: 推荐冬瓜、胡萝卜、黄瓜、芹菜、苦瓜等<cite>turn_0_31, turn_0_72</cite>。\n    - **水果**: 可适量选择木瓜、苹果、橙子等<cite>turn_0_31, turn_0_72</cite>。\n- **2.2 忌食清单**: 避免食用会加重脾胃负担、助长湿气的食物，如生冷饮品、冰咖啡、西瓜、油腻（炸鸡、麻辣烫）、甜食（蛋糕、奶油、奶茶）及辛辣食物<cite>turn_0_137, turn_0_8, turn_0_59</cite>。\n- **2.3 对症祛湿食谱与茶饮**: 针对不同湿气类型，选择不同的调理茶饮效果更佳。\n    - **湿热体质**: 可饮用山楂荷叶茶、菊花薏仁茶等以清热化湿<cite>turn_0_96, turn_0_84, turn_0_94</cite>。\n    - **湿寒体质**: 推荐饮用姜丝丁香茶、生姜红枣桂圆茶等以温中散寒<cite>turn_0_84, turn_0_94</cite>。\n    - **痰湿体质**: 适合饮用山药茯苓茶、山楂黄芪茶以健脾祛痰<cite>turn_0_84, turn_0_94</cite>。\n    - **实用参考**: 可借鉴“24小时排湿作息表”，将祛湿饮食融入一日三餐，如早餐姜枣茶，午餐冬瓜排骨汤，晚餐南瓜小米粥<cite>turn_0_8</cite>。\n\n## 3. 生活方式干预：养成祛湿好习惯\n- **3.1 坚持适度运动**: 运动能促进新陈代谢和血液循环，通过出汗帮助排出体内湿气<cite>turn_0_137, turn_0_59</cite>。推荐快走、慢跑、瑜伽、游泳、跳操等运动<cite>turn_0_137, turn_0_8</cite>。\n- **3.2 优化起居习惯**:\n    - **充足睡眠不熬夜**: 长期熬夜会耗损阳气，是加重湿气的元凶之一<cite>turn_0_59, turn_0_64, turn_0_72</cite>。\n    - **保持环境干爽**: 避免淋雨和长时间处于潮湿环境，室内要勤通风，勤晒被褥，洗头后要及时吹干头发<cite>turn_0_137, turn_0_8, turn_0_133</cite>。\n    - **坚持热水泡脚**: 泡脚能疏通经络、驱寒除湿。可在热水中加入艾草、花椒，水位没过脚踝，泡至身体微微发汗效果更佳<cite>turn_0_59, turn_0_8</cite>。\n\n## 4. 中医外治法辅助：多途径加速湿气排出\n- **4.1 穴位经络保健**: \n    - **穴位按摩**: 脾主运化水湿，可经常按揉健脾胃的关键穴位，如足三里穴<cite>turn_0_59, turn_0_64</cite>。\n    - **拍打经络**: 每日拍打腋窝、肘窝等“八虚”部位，有助于疏通经络，排出湿毒<cite>turn_0_8</cite>。\n- **4.2 善用传统疗法**: 在专业人士指导下，可借助艾灸、拔罐等方法辅助祛湿<cite>turn_0_137, turn_0_64</cite>。艾灸是补充阳气、排湿散寒的有效手段，可使用艾灸贴敷于肚脐、脚底等部位<cite>turn_0_59, turn_0_8</cite>。'
        chunks = service.divide_outline_into_chunks(outline=outline)
        # results = await service.act(
        #     input_dict=input_dict,
        #     turn_id=0,
        # )
        # print(results)
    
    asyncio.run(main())