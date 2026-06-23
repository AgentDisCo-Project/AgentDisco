import gin
import jinja2
import re
import os
import json
import time
import logging
import asyncio
import sys
sys.path.append('.')

from functools import partial
from datetime import datetime
from typing import Optional, Union, Dict, List, Tuple
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.NoteFilterService import NoteFilter
from api.utils.key_operator import ApiKeyCycler
from api.utils.string_operator import json_fix
from tool.WebSearchService import WebSearch
from dotenv import load_dotenv

load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)


@gin.configurable()
class DisentangledOutlineJudgeBlueprintAPI:
    def __init__(
        self,
        model_name: str = "",
        max_retries: int = 5,
        retry_delay: int = 3,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file: str = "DisentangledOutlineJudgeBlueprintQA_EN.jinja2",
        system_template_zh_file: str = "DisentangledOutlineJudgeBlueprintQA_ZH.jinja2",
        system_template_en_file_combine: str = "DisentangledOutlineJudgeBlueprintCombineQA_EN.jinja2",
        system_template_zh_file_combine: str = "DisentangledOutlineJudgeBlueprintCombineQA_ZH.jinja2",
        system_template_en_file_style: str = "DisentangledOutlineJudgeBlueprintStyleQA_EN.jinja2",
        system_template_zh_file_style: str = "DisentangledOutlineJudgeBlueprintStyleQA_ZH.jinja2",
        include_prev_query: bool = False,
        num_searches: int = 10,
        llm_filter_model_name: str = "",
        search_engine: str = "google",
        top_k: int = 50,
        return_score: bool = True,
        use_flash_filter: bool = False,
        use_evidence_as_key: bool = False,
        use_zh: bool = False,
        add_origin_query: bool = True,
        need_filter: bool = False,
        outline_judge_threshold: float = 8.5,
        max_outline_generator_turns: int = 3,
        min_outline_generator_turns: int = 1,
        use_response_style: bool = True,
        # 新增优化参数
        rating_volatility_threshold: float = 0.3,
        rating_improvement_threshold: float = 0.2,
        enable_rating_calibration: bool = True,
        max_blueprints_len: int = 10,
        min_blueprints_len: int = 5,  # 最小时候要点数量
        max_query_len: int = 30,
        min_query_len: int = 10,  # 新增：最少搜索词数量
        min_query_per_blueprint: int = 2,  # 每个要点最少搜索词
        enable_query_quality_check: bool = True,  # 启用搜索词质量检查
        first_round_rating_boost: float = 0.5,  # 首轮评分提升（避免过低）
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
        if search_engine == "combine":
            self.jinja_file = system_template_zh_file_combine if self.use_zh else system_template_en_file_combine
        else:
            if use_response_style:
                self.jinja_file = system_template_zh_file_style if self.use_zh else system_template_en_file_style
            else:
                self.jinja_file = system_template_zh_file if self.use_zh else system_template_en_file
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
        self.search_engine = search_engine
        self.use_evidence_as_key = use_evidence_as_key
        self.add_origin_query = add_origin_query
        self.num_searches = num_searches
        self.need_filter = need_filter
        self.outline_judge_threshold = outline_judge_threshold
        self.max_outline_generator_turns = max_outline_generator_turns
        self.min_outline_generator_turns = min_outline_generator_turns
        self.use_response_style = use_response_style
        
        # 新增参数
        self.rating_volatility_threshold = rating_volatility_threshold
        self.rating_improvement_threshold = rating_improvement_threshold
        self.enable_rating_calibration = enable_rating_calibration
        self.max_blueprints_len = max_blueprints_len
        self.min_blueprints_len = min_blueprints_len
        self.max_query_len = max_query_len
        self.min_query_len = min_query_len
        self.min_query_per_blueprint = min_query_per_blueprint
        self.enable_query_quality_check = enable_query_quality_check
        self.first_round_rating_boost = first_round_rating_boost

    def is_stop_outline_generator(
        self,
        input_dict: Dict,
        turn_id: int,
    ) -> bool:
        """
        智能停止条件判断
        """
        # 必须满足最小轮次
        if turn_id < self.min_outline_generator_turns:
            return False
        
        # 超过最大轮次
        if turn_id >= self.max_outline_generator_turns:
            logging.info(f"达到最大轮次 {self.max_outline_generator_turns}，停止迭代")
            return True
        
        # 检查当前轮次是否有 judge 结果
        curr_judge_key = f"judge_turn_{turn_id}"
        if curr_judge_key not in input_dict:
            return False
        
        curr_rating = input_dict[curr_judge_key].get("rating", 0)
        
        # 条件1：评分超过阈值
        if curr_rating >= self.outline_judge_threshold:
            logging.info(f"当前评分 {curr_rating} 超过阈值 {self.outline_judge_threshold}，停止迭代")
            return True
        
        # 条件2：连续评分下降检测
        if turn_id >= 2:
            prev_1_rating = input_dict.get(f"judge_turn_{turn_id-1}", {}).get("rating", 0)
            prev_2_rating = input_dict.get(f"judge_turn_{turn_id-2}", {}).get("rating", 0)
            
            if prev_2_rating > prev_1_rating > curr_rating:
                logging.info(f"检测到连续评分下降: {prev_2_rating} -> {prev_1_rating} -> {curr_rating}，停止迭代")
                return True
        
        # 条件3：评分波动检测
        if turn_id >= 1:
            ratings = [input_dict.get(f"judge_turn_{i}", {}).get("rating", 0) for i in range(turn_id + 1)]
            if len(ratings) >= 2:
                volatility = self._calculate_rating_volatility(ratings)
                if volatility > self.rating_volatility_threshold:
                    logging.info(f"评分波动过大 ({volatility:.2f})，停止迭代")
                    return True
        
        # 条件4：改进幅度检测
        if turn_id >= 1:
            prev_rating = input_dict.get(f"judge_turn_{turn_id-1}", {}).get("rating", 0)
            improvement = curr_rating - prev_rating
            
            if improvement < self.rating_improvement_threshold and curr_rating < self.outline_judge_threshold:
                logging.info(f"改进幅度过小 ({improvement:.2f})，停止迭代")
                return True
        
        return False
    
    def _calculate_rating_volatility(self, ratings: List[float]) -> float:
        """计算评分波动率（标准差/平均值）"""
        if len(ratings) < 2:
            return 0.0
        
        mean = sum(ratings) / len(ratings)
        if mean == 0:
            return 0.0
        
        variance = sum((r - mean) ** 2 for r in ratings) / len(ratings)
        std_dev = variance ** 0.5
        
        return std_dev / mean

    def _calibrate_rating(self, rating: float, justification: str, turn_id: int) -> float:
        """
        评分校准：检测并修正过于宽松或严苛的评分
        """
        if not self.enable_rating_calibration:
            return rating
        
        original_rating = rating
        justification_lower = justification.lower()
        
        # 首轮评分校准：避免首轮评分过低
        if turn_id == 0 and rating < 4.0:
            # 检查是否提到了明显的优点
            positive_keywords = ["结构完整", "覆盖", "清晰", "合理", "基础"]
            has_positive = any(kw in justification_lower for kw in positive_keywords)
            
            if has_positive:
                # 给予首轮一定的评分提升
                rating = min(rating + self.first_round_rating_boost, 5.0)
                logging.info(f"首轮评分校准: {original_rating:.1f} -> {rating:.1f} (发现优点，给予提升)")
        
        # 关键词映射校准
        calibration_keywords = {
            "完美": 9.5, "卓越": 9.5, "无可挑剔": 9.5,
            "优秀": 8.5, "出色": 8.5, "高质量": 8.5,
            "良好": 7.5, "不错": 7.5,
            "及格": 6.0, "一般": 6.0, "基础": 5.5,
            "不足": 5.0, "欠缺": 5.0,
            "较差": 4.0, "严重": 3.0, "很差": 3.0,
        }
        
        detected_scores = []
        for keyword, expected_score in calibration_keywords.items():
            if keyword in justification_lower:
                detected_scores.append(expected_score)
        
        if detected_scores:
            avg_expected = sum(detected_scores) / len(detected_scores)
            
            # 如果评分与预期差距过大，进行校准
            if abs(rating - avg_expected) > 1.5:
                calibrated = (rating + avg_expected) / 2
                logging.info(f"评分校准: {rating:.1f} -> {calibrated:.1f} (基于评价内容)")
                return round(calibrated, 1)
        
        return round(rating, 1)

    def _validate_and_fix_blueprints(self, blueprints: List[Dict], query_text: str) -> List[Dict]:
        """
        验证并修复 blueprints
        
        检查项：
        1. 每个 blueprint 必须有 content
        2. 每个 blueprint 必须有 search_query
        3. search_query 数量必须 >= min_query_per_blueprint
        4. 总 search_query 数量必须 >= min_query_len
        """
        if not isinstance(blueprints, list):
            logging.warning(f"blueprints 不是列表，而是 {type(blueprints)}")
            blueprints = []
        
        fixed_blueprints = []
        all_search_queries = []
        
        for idx, blueprint in enumerate(blueprints):
            if not isinstance(blueprint, dict):
                continue
            
            # 确保有 content
            content = blueprint.get("content", "").strip()
            if not content:
                content = f"补充分析维度 {idx + 1}"
            
            # 获取 search_query
            if self.search_engine == "combine":
                search_query = blueprint.get("xhs_search_query", []) + blueprint.get("google_search_query", [])
            else:
                search_query = blueprint.get("search_query", [])
            
            # 清理搜索词
            if isinstance(search_query, list):
                search_query = [sq.strip() for sq in search_query if isinstance(sq, str) and sq.strip()]
            else:
                search_query = []
            
            # 智能分割包含逗号的搜索词
            search_query = self._smart_split_query(search_query)
            
            # 去重
            search_query = list(dict.fromkeys(search_query))
            
            # 如果搜索词数量不足，生成补充搜索词
            if len(search_query) < self.min_query_per_blueprint:
                additional_queries = self._generate_additional_queries(content, query_text, self.min_query_per_blueprint - len(search_query))
                search_query.extend(additional_queries)
            
            # 更新 blueprint
            fixed_blueprint = {
                "content": content,
                "search_query": search_query
            }
            
            if self.search_engine == "combine":
                fixed_blueprint["xhs_search_query"] = search_query[:len(search_query)//2 + 1]
                fixed_blueprint["google_search_query"] = search_query[len(search_query)//2:]
            
            fixed_blueprints.append(fixed_blueprint)
            all_search_queries.extend(search_query)
        
        # 检查总搜索词数量
        if len(all_search_queries) < self.min_query_len:
            logging.warning(f"总搜索词数量 {len(all_search_queries)} 少于最小要求 {self.min_query_len}，生成补充搜索词")
            additional = self._generate_additional_queries(query_text, query_text, self.min_query_len - len(all_search_queries))
            
            # 将补充搜索词分配到各个 blueprint
            for i, query in enumerate(additional):
                if fixed_blueprints:
                    fixed_blueprints[i % len(fixed_blueprints)]["search_query"].append(query)
        
        return fixed_blueprints

    def _generate_additional_queries(self, content: str, query_text: str, num_queries: int) -> List[str]:
        """生成补充搜索词"""
        additional = []
        
        # 从内容中提取关键词
        keywords = re.findall(r'[\u4e00-\u9fa5]{2,6}|[a-zA-Z]+', content)
        keywords = list(dict.fromkeys(keywords))[:5]  # 去重，取前5个
        
        # 生成组合搜索词
        for i in range(num_queries):
            if keywords:
                keyword = keywords[i % len(keywords)]
                if self.search_engine == "google":
                    additional.append(f"{keyword} {query_text[:20]}")
                else:
                    additional.append(f"{keyword} 分析")
            else:
                additional.append(f"补充搜索 {i + 1}")
        
        return additional

    def _check_query_quality(self, search_queries: List[str]) -> Tuple[bool, str]:
        """
        检查搜索词质量
        
        返回: (是否通过, 问题描述)
        """
        if not self.enable_query_quality_check:
            return True, ""
        
        issues = []
        
        # 检查数量
        if len(search_queries) < self.min_query_len:
            issues.append(f"搜索词数量不足: {len(search_queries)} < {self.min_query_len}")
        
        # 检查质量：避免过于宽泛的搜索词
        vague_terms = ["什么", "怎么", "如何", "介绍", "分析", " overview", " introduction"]
        vague_count = sum(1 for sq in search_queries if any(vt in sq.lower() for vt in vague_terms))
        
        if vague_count > len(search_queries) * 0.5:
            issues.append(f"宽泛搜索词过多: {vague_count}/{len(search_queries)}")
        
        # 检查重复
        unique_queries = set(search_queries)
        if len(unique_queries) < len(search_queries) * 0.8:
            issues.append("搜索词重复率过高")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, ""

    def get_system_prompt(self, turn_id: int = 0) -> str:
        """生成系统 Prompt"""
        template_vars = {
            "curr_date": datetime.now().strftime("%Y年%m月%d日"),
            "search_engine": self.search_engine,
            "max_blueprints_len": self.max_blueprints_len,
            "min_blueprints_len": self.min_blueprints_len,
            "max_query_len": self.max_query_len,
            "min_query_len": self.min_query_len,
            "turn_id": turn_id,
        }
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt

    def check_func(self, response: str) -> Dict:
        """响应校验函数"""
        return self.parser_response(response=response, turn_id=0)

    def parser_response(self, response: str, turn_id: int = 0) -> Dict:
        """
        解析并校验模型响应
        """
        response = json_fix(response)
        
        try:
            response = json.loads(response)
        except json.JSONDecodeError as e:
            logging.error(f"JSON 解析失败: {e}")
            raise ValueError(f"Invalid JSON format: {e}")
        
        # 校验响应结构
        if not isinstance(response, Dict):
            raise ValueError("Response must be a JSON object")
        
        # 必需字段检查
        required_fields = ["rating", "justification", "blueprints"]
        for field in required_fields:
            if field not in response:
                raise ValueError(f"Missing required field: {field}")
        
        # rating 字段校验
        rating = response["rating"]
        if not isinstance(rating, (float, int)):
            raise ValueError(f"Rating must be a number, got {type(rating)}")
        if rating < 0 or rating > 10:
            raise ValueError(f"Rating must be between 0 and 10, got {rating}")
        
        # justification 字段校验
        justification = response["justification"]
        if not isinstance(justification, str):
            raise ValueError(f"Justification must be a string, got {type(justification)}")
        if len(justification) < 10:
            raise ValueError(f"Justification too short ({len(justification)} chars), minimum 10")
        
        # blueprints 字段校验和修复
        blueprints = response.get("blueprints", [])
        blueprints = self._validate_and_fix_blueprints(blueprints, "")
        response["blueprints"] = blueprints
        
        # 评分校准
        response["rating"] = self._calibrate_rating(response["rating"], response["justification"], turn_id)
        
        return response

    def _smart_split_query(self, search_query: List[str]) -> List[str]:
        """智能分割包含逗号的搜索词"""
        result = []
        for item in search_query:
            if isinstance(item, str) and (',' in item or '，' in item):
                sub_items = re.split('[,，]', item)
                result.extend([sub_item.strip() for sub_item in sub_items if sub_item.strip()])
            else:
                result.append(item.strip() if isinstance(item, str) else item)
        return result

    def _build_user_prompt(
        self,
        query_text: str,
        outline: str,
        blueprints: List,
        turn_id: int,
        input_dict: Dict,
    ) -> Union[str, List]:
        """构建用户 Prompt"""
        
        include_prev_query = (turn_id > 0) and self.include_prev_query
        
        prompt_parts = []
        
        # 添加迭代轮次信息
        if self.use_zh:
            prompt_parts.append(f"# 当前迭代轮次\n第 {turn_id} 轮\n")
        else:
            prompt_parts.append(f"# Current Iteration\nRound {turn_id}\n")
        
        # 历史搜索词
        if include_prev_query:
            prev_queries = []
            for t in range(turn_id):
                prev_queries.extend(input_dict.get(f"search_query_turn_{t}", []))
            
            if self.use_zh:
                prompt_parts.append(f"# 历史的搜索词\n{prev_queries}\n")
            else:
                prompt_parts.append(f"# Previous Search Query Terms\n{prev_queries}\n")
        
        # 用户提问
        if self.use_zh:
            prompt_parts.append(f"# 用户提问\n{query_text}\n")
        else:
            prompt_parts.append(f"# User Question\n{query_text}\n")
        
        # 回复风格
        if self.use_response_style:
            response_style = input_dict.get('response_style', '')
            if response_style:
                if self.use_zh:
                    prompt_parts.append(f"# 回复风格（供参考）\n{response_style}\n")
                else:
                    prompt_parts.append(f"# Response Style (for reference)\n{response_style}\n")
        
        # 报告大纲
        if outline:
            if self.use_zh:
                prompt_parts.append(f"# 报告大纲\n{outline}\n")
            else:
                prompt_parts.append(f"# Report Outline\n{outline}\n")
        else:
            if self.use_zh:
                prompt_parts.append("# 报告大纲\n（首轮生成，暂无大纲内容）\n")
            else:
                prompt_parts.append("# Report Outline\n(First round, no outline yet)\n")
        
        # 大纲要点列表
        if blueprints:
            if self.use_zh:
                prompt_parts.append(f"# 大纲要点列表\n{blueprints}\n")
            else:
                prompt_parts.append(f"# Report Outline Blueprints\n{blueprints}\n")
        
        # 引用率信息
        if turn_id >= 1:
            unique_cnt = input_dict.get(f"outline_turn_{turn_id-1}_unique_cnt", "")
            search_result = input_dict.get(f"search_result_turn_{turn_id-1}", [])
            search_cnt = len(search_result) if isinstance(search_result, list) else 0
            
            if search_cnt > 0 and unique_cnt:
                try:
                    ratio = int(unique_cnt) / int(search_cnt)
                    if self.use_zh:
                        prompt_parts.append(f"# 大纲引用文档个数/搜索引擎返回文档个数\n{unique_cnt}/{search_cnt}={ratio:.2f}\n")
                    else:
                        prompt_parts.append(f"# Number of outline referenced documents / Search engine returned documents\n{unique_cnt}/{search_cnt}={ratio:.2f}\n")
                except (ValueError, ZeroDivisionError):
                    pass
        
        if self.use_zh:
            return "\n".join(prompt_parts)
        else:
            return "\n".join(prompt_parts)

    async def _call_llm(self, system_prompt: Union[str, List], user_prompt: Union[str, List], turn_id: int) -> Dict:
        """调用 LLM 模型"""
        
        if "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            return await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=lambda r: self.parser_response(r, turn_id),
                return_cot=False,
            )
        elif "gemini" in self.model.model_name:
            return await self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=lambda r: self.parser_response(r, turn_id),
                return_cot=False,
            )
        elif "gpt-oss" in self.model.model_name:
            return await self.model.chat_gpt_oss(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=lambda r: self.parser_response(r, turn_id),
            )
        else:
            raise ValueError(f"Unsupported model: {self.model.model_name}")

    async def act(
        self,
        input_dict: Dict,
        turn_id: int,
    ) -> Tuple[bool, Dict]:
        """
        执行大纲评判和搜索词生成
        """
        outline_judge_st = time.time()
        query_text = input_dict.get("query_text", "") or input_dict.get("query", "")
        
        if not query_text:
            logging.error("Missing query_text in input_dict")
            raise ValueError("query_text is required")
        
        # 获取大纲和要点
        outline = input_dict.get(f"outline_turn_{turn_id-1}", "")
        blueprints = input_dict.get(f"blueprints_turn_{turn_id-1}", [])
        
        # 构建 Prompt
        system_prompt = self.get_system_prompt(turn_id)
        user_prompt = self._build_user_prompt(
            query_text=query_text,
            outline=outline,
            blueprints=blueprints,
            turn_id=turn_id,
            input_dict=input_dict,
        )
        
        # Gemini 需要特殊格式
        if "gemini" in self.model.model_name:
            system_prompt = [{"text": system_prompt}]
            user_prompt = [{"text": user_prompt}]
        
        # 调用 LLM
        try:
            response = await self._call_llm(system_prompt, user_prompt, turn_id)
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            raise
        
        outline_judge_et = time.time()
        
        # 记录评分信息
        rating = response.get('rating', 0)
        unique_cnt = input_dict.get(f"outline_turn_{turn_id-1}_unique_cnt", "")
        search_result = input_dict.get(f"search_result_turn_{turn_id-1}", [])
        search_cnt = len(search_result) if isinstance(search_result, list) else 0
        
        logging.info(f"Outline judge rating: {rating:.1f} (turn {turn_id})")
        logging.info(f"Outline referenced docs / search returned docs: {unique_cnt}/{search_cnt}")
        logging.info(f"Outline judge costs: {outline_judge_et - outline_judge_st:.2f}s")
        logging.info(f"Search engine: {self.search_engine}")
        
        # 保存评判结果
        input_dict[f"judge_turn_{turn_id}"] = response
        
        # 提取搜索词
        search_query, xhs_search_query, google_search_query = self._extract_search_queries(
            response, query_text
        )
        
        # 检查搜索词质量
        quality_passed, quality_issue = self._check_query_quality(search_query)
        if not quality_passed:
            logging.warning(f"搜索词质量检查未通过: {quality_issue}")
        
        logging.info(f"Number of search queries generated: {len(search_query)}")
        
        input_dict[f"search_query_turn_{turn_id}"] = search_query
        input_dict[f"xhs_search_query_turn_{turn_id}"] = xhs_search_query
        input_dict[f"google_search_query_turn_{turn_id}"] = google_search_query
        
        # 提取 blueprints
        blueprints_content = [bp.get("content", "") for bp in response.get("blueprints", [])]
        input_dict[f"blueprint_turn_{turn_id}"] = blueprints_content
        input_dict[f"blueprints_turn_{turn_id}"] = response.get("blueprints", [])
        
        # 检查是否停止
        should_stop = self.is_stop_outline_generator(input_dict=input_dict, turn_id=turn_id)
        
        # 如果需要继续，执行搜索
        if not should_stop and response.get("blueprints"):
            await self._execute_search(input_dict, turn_id, search_query, xhs_search_query, google_search_query)
        
        return should_stop, input_dict

    def _extract_search_queries(
        self,
        response: Dict,
        query_text: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """从响应中提取搜索词"""
        
        search_query = []
        xhs_search_query, google_search_query = [], []
        
        # 添加原始查询
        if self.search_engine == "combine":
            if self.add_origin_query:
                xhs_search_query.append(query_text)
                google_search_query.append(query_text)
        else:
            if self.add_origin_query or self.search_engine == "sandbox":
                search_query.append(query_text)
        
        # 提取 blueprints 中的搜索词
        for blueprint in response.get("blueprints", []):
            if self.search_engine == "combine":
                xhs_sq = blueprint.get('xhs_search_query', [])
                google_sq = blueprint.get('google_search_query', [])
                
                xhs_sq = self._smart_split_query(xhs_sq) if isinstance(xhs_sq, list) else []
                google_sq = self._smart_split_query(google_sq) if isinstance(google_sq, list) else []
                
                xhs_search_query.extend(xhs_sq)
                google_search_query.extend(google_sq)
            else:
                sq = blueprint.get('search_query', [])
                sq = self._smart_split_query(sq) if isinstance(sq, list) else []
                search_query.extend(sq)
        
        # 去重
        search_query = list(dict.fromkeys(search_query))
        xhs_search_query = list(dict.fromkeys(xhs_search_query))
        google_search_query = list(dict.fromkeys(google_search_query))
        
        # 过滤空字符串
        search_query = [sq for sq in search_query if sq.strip()]
        xhs_search_query = [sq for sq in xhs_search_query if sq.strip()]
        google_search_query = [sq for sq in google_search_query if sq.strip()]
        
        return search_query, xhs_search_query, google_search_query

    async def _execute_search(
        self,
        input_dict: Dict,
        turn_id: int,
        search_query: List[str],
        xhs_search_query: List[str],
        google_search_query: List[str]
    ):
        """执行搜索并保存结果"""
        
        query_search_st = time.time()
        
        async def search_single(q: str) -> Dict:
            """单次搜索"""
            try:
                search_docs = await self.searcher.call({
                    "query_text": input_dict.get("query_text", ""),
                    "search_query": [q],
                    "turn_id": f"turn_{turn_id}",
                    "xiaohongshu_search_query": xhs_search_query,
                    "google_search_query": google_search_query
                })
                return {"query": q, "docs": search_docs}
            except Exception as e:
                logging.error(f"Search failed for query '{q}': {e}")
                return {"query": q, "docs": [], "error": str(e)}
        
        # 并发执行搜索
        search_docs_per_q = await asyncio.gather(
            *[search_single(q) for q in search_query if q.strip()]
        )
        
        search_docs_per_q = list(search_docs_per_q)
        input_dict[f"search_result_turn_{turn_id}"] = search_docs_per_q
        
        query_search_et = time.time()
        logging.info(f"Query search costs: {query_search_et - query_search_st:.2f}s")
        
        # 执行过滤（如果启用）
        if self.need_filter:
            filter_st = time.time()
            try:
                input_dict = await self.filter.act(input_dict=input_dict, turn_id=turn_id)
            except Exception as e:
                logging.error(f"Filter failed: {e}")
            filter_et = time.time()
            logging.info(f"Filter out search results costs: {filter_et - filter_st:.2f}s")


if __name__ == "__main__":
    async def main():
        service = DisentangledOutlineJudgeBlueprintAPI(
            model_name="gemini-2.5-flash",
            llm_filter_model_name="gemini-2.5-flash",
            num_searches=10,
            use_zh=True,
            search_engine="google",
            top_k=50,
            return_score=True,
            use_flash_filter=False,
            use_evidence_as_key=False,
            add_origin_query=False,
            need_filter=False,
            max_outline_generator_turns=3,
            outline_judge_threshold=8.5,
            min_query_len=10,
            min_query_per_blueprint=2,
        )
        
        input_dict = {
            "query_text": "收集整理目前中国9阶层实际收入和财务状况，特别研究得出中国的中产有哪些特点，实际中产人数，财力等等",
            "query": "收集整理目前中国9阶层实际收入和财务状况，特别研究得出中国的中产有哪些特点，实际中产人数，财力等等",
            "blueprint_turn_0": [],
            "response_style": "专业研究报告风格"
        }
        
        is_finish, result = await service.act(
            input_dict=input_dict,
            turn_id=0,
        )
        
        print(f"Finished: {is_finish}")
        print(f"Rating: {result.get('judge_turn_0', {}).get('rating', 0)}")
    
    asyncio.run(main())
