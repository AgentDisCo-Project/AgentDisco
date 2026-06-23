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
class ConsultScoreGenerator:
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
        system_template_en_file: str = "PairwiseConsultScoreGenerator_EN.jinja2",
        system_template_zh_file: str = "PairwiseConsultScoreGenerator_EN.jinja2",
        num_samples: int = 1,
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
        self.max_concurrent = max_concurrent
        self.jinja_file = system_template_zh_file if use_zh else system_template_en_file
        self.num_samples = num_samples
        
    
    def get_system_prompt(
        self
    ):
        template_vars = {}  # 如果需要传递变量可以在这里添加
        
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    def check_func(
        self,
        response: str,
    ):
        response = json_fix(response)
        if "instruction_following" not in response or "comprehensiveness" not in response:
            raise ValueError()
        if "completeness" not in response or "writing_quality" not in response:
            raise ValueError()
        response = json.loads(response)
        return response
    

    async def post_request(
        self,
        user_question: str,
        report: str,
        reference_report: str,
    ):
        if "gpt" in self.model.model_name:
            system_prompt = self.get_system_prompt()
            user_prompt = f"""
<prompt>
{user_question}
</prompt>

<report_a>
{report}
</report_a>

<report_b>
{reference_report}
</report_b>
"""
            response = await self.model.chat_gpt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
            )
        
        elif "gemini" in self.model.model_name:
            system_prompt = [
                {
                    "text": self.get_system_prompt()
                }
            ]
            user_prompt = []
            _user_prompt = f"""
<prompt>
{user_question}
</prompt>

<report_a>
{report}
</report_a>

<report_b>
{reference_report}
</report_b>
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
    
    
    async def post_request_all_trials(
        self,
        user_question: str,
        report: str,
        reference_report: str,
    ):
        orign_task = [
            self.post_request(user_question=user_question, report=report, reference_report=reference_report)
            for _ in range(self.num_samples)
        ]
        flip_task = [
            self.post_request(user_question=user_question, report=reference_report, reference_report=report)
            for _ in range(self.num_samples)
        ]
        
        res = await asyncio.gather(
            *orign_task, *flip_task,
            return_exceptions=True,
        )
        origin_res, flip_res = [], []
        for i, r in enumerate(res):
            if isinstance(r, Exception):
                print(f"Trial {i} failed: {r}")
                continue
            if i < self.num_samples:
                origin_res.append(r)
            else:
                flip_res.append(r)
        return origin_res, flip_res
    
    
    def parse_response(
        self,
        origin_res: List,
        flip_res: List,
        dimension: str,
    ):
        all_votes  = []
        all_scores = []
        
        for resp in origin_res:
            dim = resp.get(dimension, {})
            preferred = dim.get("preferred", "a")
            gap_score = dim.get("gap_score", 0)
            score = gap_score if preferred == "a" else -gap_score
            all_votes.append(preferred)
            all_scores.append(score)
            
        for resp in flip_res:
            dim = resp.get(dimension, {})
            preferred = dim.get("preferred", "a")
            gap_score = dim.get("gap_score", 0)
            flipped_preferred = "b" if preferred == "a" else "a" # 翻转：A/B 互换后，原来的 candidate 变成了 report_a
            score = gap_score if flipped_preferred == "a" else -gap_score
            all_votes.append(flipped_preferred)
            all_scores.append(score)
        
        num_wins   = sum(1 for v in all_votes if v == "a")
        num_losses = sum(1 for v in all_votes if v == "b")
        
        if num_wins > num_losses:
            grade = "win"
        elif num_wins < num_losses:
            grade = "lose"
        else:
            grade = "tie"
        
        avg_score_b = sum(all_scores) / len(all_scores) if all_scores else 0
        consensus_score = avg_score_b + 5 # ── 平均分（0-10，5=tie）
        
        return {
            "grade":      grade,
            "is_win":     grade == "win",
            "is_tie":     grade == "tie",
            "is_lose":    grade == "lose",
            "score":      consensus_score,
            "votes":      all_votes,          # 原始投票记录（可用于 debug）
            "all_scores": all_scores,
            "raw": {
                "original": origin_res,
                "flipped":  flip_res,
            },
        }
    
        
    
    async def act(
        self,
        input_dict: Dict,
        report_key: str = "report"
    ):
        user_question = input_dict["query_text"]
        report = input_dict[f"{report_key}"]
        reference_report = input_dict.get("reference_report", "")
        
        origin_res, flip_res = await self.post_request_all_trials(user_question=user_question, report=report, reference_report=reference_report)
        res = dict()
        for dimension in ("instruction_following", "comprehensiveness", "completeness", "writing_quality"):
            res[dimension] = self.parse_response(
                origin_res=origin_res,
                flip_res=flip_res,
                dimension=dimension,
            )
        
        instruction_following_res = res["instruction_following"]
        res["instruction_following_res"] = {
            "win_rate": 1. if instruction_following_res["is_win"] else 0.,
            "tie_rate": 1. if instruction_following_res["is_tie"] else 0.,
            "lose_rate": 1. if instruction_following_res["is_lose"] else 0.,
            "avg_score": instruction_following_res["score"]
        }
        
        completeness_res = res["completeness"]
        res["completeness_res"] = {
            "win_rate": 1. if completeness_res["is_win"] else 0.,
            "tie_rate": 1. if completeness_res["is_tie"] else 0.,
            "lose_rate": 1. if completeness_res["is_lose"] else 0.,
            "avg_score": completeness_res["score"],
        }
        
        comprehensiveness_res = res["comprehensiveness"]
        res["comprehensiveness_res"] = {
            "win_rate": 1. if comprehensiveness_res["is_win"] else 0.,
            "tie_rate": 1. if comprehensiveness_res["is_tie"] else 0.,
            "lose_rate": 1. if comprehensiveness_res["is_lose"] else 0.,
            "avg_score": comprehensiveness_res["score"],
        }
        
        writing_quality_res = res["writing_quality"]
        res["writing_quality_res"] = {
            "win_rate": 1. if writing_quality_res["is_win"] else 0.,
            "tie_rate": 1. if writing_quality_res["is_tie"] else 0.,
            "lose_rate": 1. if writing_quality_res["is_lose"] else 0.,
            "avg_score": writing_quality_res["score"],
        }
        
        dim_res = [res[d] for d in ("instruction_following", "comprehensiveness", "completeness", "writing_quality")]
        win_count  = sum(1 for d in dim_res if d["is_win"])
        tie_count  = sum(1 for d in dim_res if d["is_tie"])
        lose_count = sum(1 for d in dim_res if d["is_lose"])
        res["overall"] = {
            "win_rate": win_count / len(dim_res),
            "tie_rate": tie_count / len(dim_res),
            "lose_rate": lose_count / len(dim_res),
            "avg_score": sum(d["score"] for d in dim_res) / len(dim_res),
            "net_win_rate": (
                win_count / (win_count + lose_count)
                if (win_count + lose_count) > 0 else 0.0
            ),
        }

        input_dict["eval_overall_res"] = res["overall"]
        input_dict["eval_instruction_following_res"] = res["instruction_following_res"]
        input_dict["eval_comprehensiveness_res"] = res["comprehensiveness_res"]
        input_dict["eval_completeness_res"] = res["completeness_res"]
        input_dict["eval_writing_quality_res"] = res["writing_quality_res"]
        return input_dict



if __name__ == "__main__":
    async def main():
        service = ConsultScoreGenerator(
            model_name="gpt-4.1",
            use_zh=False,
        )
        input_dict = dict()
        input_dict["query_text"] = "ABC"
        input_dict["rendered_report"] = "abc"
        input_dict["reference_report"] = "abc"
        input_dict = await service.act(
            input_dict=input_dict,
            report_key="rendered_report"
        )
        breakpoint()
        print(input_dict)
    
    asyncio.run(main())
        
