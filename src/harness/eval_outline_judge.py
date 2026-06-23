import argparse
import asyncio
import logging
import multiprocessing as mp
import os
import json
import time
import gin
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)                  
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'api'))               # 解决 api/ 内部的扁平导入
sys.path.insert(0, os.path.join(root_dir, 'agent'))             # 解决 agent/ 内部的扁平导入

from typing import Dict
from datetime import datetime
from tqdm import tqdm
from agent.QueryMinerAgent import QueryMiner
from agent.MemoryBankAgent import MemoryBankManager
from agent.ReportWriterAgent import ReportWriter
# from agent.ReportWriterEvidenceAgent import ReportWriter
from agent.ReportPolishAgent import ReportPolish
from api.ReferenceRenderService import ReferenceRender
from api.IntentPlannerService import IntentPlanner
from harness.api.DisentangledOutlineJudgeBlueprintAPIService import DisentangledOutlineJudgeBlueprintAPI
from harness.api.JudgeSearchQueryDiversityAPIService import JudgeSearchQueryDiversityAPI
from harness.api.SummaryQAGeneratorAPIService import SummaryQAGeneratorAPI
from run_workflow import set_logger, load_data_from_cache, cache_data


@gin.configurable()
async def run_agent(
    job_name: str,
    input_path: str,
    cache_dir: str,
    log_dir: str,
    output_path: str = None,
    use_zh: bool = True,
    use_debug: bool = True,
    input_dict: Dict = None,
):
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = job_name + "_" + cur_time if not use_debug else job_name
    set_logger(log_dir=log_dir, job_name=job_name)
    os.makedirs(cache_dir, exist_ok=True)
    
    input_dict = load_data_from_cache(cache_path=input_path) if input_dict is None else input_dict
    logging.info(f"successfully load input dict from: {input_path}")
    
    logging.info(f"start with job name: {job_name}")
    input_query = input_dict.get("query_text", "") or input_dict.get("query", "")
    logging.info(f"processing query {input_query}")

    cache_path = os.path.join(cache_dir, f"{job_name}_cache.json")
    if os.path.exists(cache_path) and False:
        input_dict = load_data_from_cache(cache_path=cache_path)
    else:
        query_miner_st = time.time()
        query_miner = QueryMiner(
            use_zh=use_zh,
            disable_video=True,
            disable_images=True,
            disable_multi_images=True,
            disable_comment=True,
            use_input_query=True,
            include_summary=False,
        )
        input_dict = await query_miner.act(
            input_dict=input_dict,
            response_key="query_text"
        )
        query_miner_et = time.time()
        logging.info(f"query miner costs: {query_miner_et-query_miner_st}")

        intent_planner_st = time.time()
        intent_planner = IntentPlanner(
            use_zh=use_zh,
        )
        input_dict = await intent_planner.act(
            input_dict=input_dict,
        )
        intent_planner_et = time.time()
        logging.info(f"intent planner costs: {intent_planner_et-intent_planner_st}")
        
        outline_judge = DisentangledOutlineJudgeBlueprintAPI(
            use_zh=use_zh,
            use_evidence_as_key=False,
            need_filter=False,
            outline_judge_threshold=9,
            max_outline_generator_turns=2,
            min_outline_generator_turns=1,
            use_response_style=True,
        )

        outline_judge_st = time.time()
        is_finish, input_dict = await outline_judge.act(
            input_dict=input_dict,
            turn_id=0,
        )
        outline_judge_et = time.time()
        logging.info(f"finish outline judge costs: {outline_judge_et-outline_judge_st}")

        judge_and_evidence_generator = SummaryQAGeneratorAPI(use_zh=use_zh)
        generate_judge_and_evidence_st = time.time()
        input_dict = await judge_and_evidence_generator.act(input_dict=input_dict, turn_id=0)
        generate_judge_and_evidence_et = time.time()
        logging.info(f"generate summary costs: {generate_judge_and_evidence_et-generate_judge_and_evidence_st}")

        cache_data(input_dict=input_dict, cache_path=cache_path)
    
    judge_query_generator = JudgeSearchQueryDiversityAPI(use_zh=use_zh)
    generate_judge_query_st = time.time()
    score, input_dict = await judge_query_generator.act(
        input_dict=input_dict,
        turn_id=0,
    )
    generate_judge_query_et = time.time()
    logging.info(f"judge score for current setting is {score}")
    return input_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gin-config-file",
        type=str,
        required=False,
        default="./config/auto_eval.gin",
    )
    args = parser.parse_args()
    
    gin.parse_config_file(args.gin_config_file)
    asyncio.run(run_agent())
 





