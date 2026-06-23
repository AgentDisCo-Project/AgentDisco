import argparse
import asyncio
import logging
import multiprocessing as mp
import os
import json
import time
import gin
import sys
sys.path.append('./api')
sys.path.append('./agent')

from typing import Dict
from datetime import datetime
from tqdm import tqdm
from agent.QueryMinerAgent import QueryMiner
from agent.MemoryBankAgent import MemoryBankManager
# from agent.ReportWriterAgent import ReportWriter
from agent.ReportWriterEvidenceAgent import ReportWriter
from agent.ReportPolishAgent import ReportPolish
from api.DisentangledOutlineGeneratorService import DisentangledOutlineGenerator
from api.DisentangledOutlineJudgeService import DisentangledOutlineJudge
from api.ReferenceRenderService import ReferenceRender
from run_workflow import set_logger, load_data_from_cache, cache_data



@gin.configurable()
async def run_batch_agent(
    job_name: str,
    input_path: str,
    log_dir: str,
    cache_dir: str,
    output_path: str = None,
    max_concurrent: int = 10,
    require_cache_data: bool = False,
    load_from_excel: bool = False,
):
    input_list = load_data_from_cache(cache_path=input_path, return_list=True, load_from_excel=load_from_excel)
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(input_list), desc="Processing workflows")
    
    async def run_agent_with_semaphore(input_dict, index):
        async with semaphore:
            try:
                result = await run_agent(
                    input_dict=input_dict,
                    job_name=f"{job_name}_{index}",
                    require_cache_data=require_cache_data,
                    log_dir=f"{log_dir}/{job_name}_{index}",
                    cache_dir=f"{cache_dir}/{job_name}_{index}"
                )
                pbar.update(1)
                return index, result, None
            except Exception as e:
                pbar.update(1)
                return index, None, e
    
    tasks = [
        run_agent_with_semaphore(input_dict=input_dict, index=i)
        for i, input_dict in enumerate(input_list)
    ]
    task_results = await asyncio.gather(*tasks, return_exceptions=True)
    pbar.close()
    
    results = [None] * len(input_list)
    errors = [None] * len(input_list)
    
    for task_result in task_results:
        if isinstance(task_result, Exception):
            continue
        index, result, error = task_result
        results[index] = result
        errors[index] = error
    
    merged_list = []
    for i, input_dict in enumerate(input_list):
        if errors[i] is not None:
            merged_dict = {
                **input_dict,
                'error': str(errors[i]),
                'status': 'error',
                'index': i
            }
        else:
            merged_dict = {
                **input_dict,
                'result': results[i],
                'status': 'success',
                'index': i
            }
        merged_list.append(merged_dict)
    
    output_path = output_path or f"{input_path.rsplit('.', 1)[0]}_out.jsonl"
    cache_data(input_dict=merged_list, cache_path=output_path, use_list=True)
    return merged_list



@gin.configurable()
async def run_agent(
    job_name: str,
    input_path: str,
    cache_dir: str,
    log_dir: str,
    skip_query_miner: str,
    skip_outline_generator: bool,
    skip_report_writer: bool,
    skip_report_polish: bool,
    disable_query_miner: bool,
    disable_images: bool,
    disable_multi_images: bool,
    disable_comment: bool,
    disable_video: bool,
    disable_outline_generator: bool,
    disable_report_writer: bool,
    disable_report_polish: bool,
    output_path: str = None,
    input_dict: Dict = None,
    require_cache_data: bool = True,
    max_outline_generator_turns: int = 10,
    min_outline_generator_turns: int = 2,
    outline_judge_threshold: int = 8,
    use_zh: bool = False,
    use_input_query: bool = False,
    include_summary: bool = False,
    use_hierarchical_writer: bool = False,
    use_debug: bool = False,
    use_polish_before_render: bool = False,
    use_evidence_as_key: bool = False,
):
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = job_name + "_" + cur_time if not use_debug else job_name
    set_logger(log_dir=log_dir, job_name=job_name)
    if require_cache_data:
        os.makedirs(cache_dir, exist_ok=True)
    
    input_dict = load_data_from_cache(cache_path=input_path) if input_dict is None else input_dict
    logging.info(f"successfully load input dict from: {input_path}")
    
    logging.info(f"start with job name: {job_name}")
    input_query = input_dict.get("query_text", "") or input_dict.get("query", "")
    logging.info(f"processing query {input_query}")
    
    if disable_report_polish:
        use_polish_before_render = False
        save_key = "rendered_report"
    else:
        save_key = "polished_rendered_report"
    
    # query miner
    if not disable_query_miner:
        cache_path = os.path.join(cache_dir, f"{job_name}_query_miner.json")
        if skip_query_miner and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load query miner results from cache: {cache_path}")
        else:
            query_miner_st = time.time()
            query_miner = QueryMiner(
                use_zh=use_zh,
                disable_video=disable_video,
                disable_images=disable_images,
                disable_multi_images=disable_multi_images,
                disable_comment=disable_comment,
                use_input_query=use_input_query,
                include_summary=include_summary,
            )
            input_dict = await query_miner.act(
                input_dict=input_dict,
                response_key="query_text"
            )
            query_miner_et = time.time()
            logging.info(f"query miner costs: {query_miner_et-query_miner_st}")
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"finish cache query miner: {cache_path}")
    
    # outline generator
    if not disable_outline_generator:
        cache_path = os.path.join(cache_dir, f"{job_name}_outline_generator.json")
        if skip_outline_generator and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            num_turns = input_dict["num_turns"]
            num_chunks = input_dict["num_chunks"]
            logging.info(f"load outline generator results from cache: {cache_path}")
        else:
            outline_judge = DisentangledOutlineJudge(
                use_zh=use_zh,
                use_evidence_as_key=False,
                need_filter=False,
                outline_judge_threshold=outline_judge_threshold
            )
            outline_generator = DisentangledOutlineGenerator(
                use_zh=use_zh,
                use_evidence_as_key=False,
            )
            memory_bank = MemoryBankManager(
                use_zh=use_zh,
                use_evidence_as_key=False,
                use_hierarchical_writer=use_hierarchical_writer
            )
            
            turn_id = 0
            while turn_id < max_outline_generator_turns:
                outline_judge_st = time.time()
                input_dict = await outline_judge.act(
                    input_dict=input_dict,
                    turn_id=turn_id,
                )
                outline_judge_et = time.time()
                logging.info(f"finish outline judge turn {turn_id} costs: {outline_judge_et-outline_judge_st}")
                
                is_finish = input_dict[f"judge_turn_{turn_id}"]["rating"] >= outline_judge_threshold
                input_dict[f"is_finish_turn_{turn_id}"] = is_finish
                logging.info(f"finsh outline generator at {turn_id}: {is_finish}, judge score is: {input_dict[f'judge_turn_{turn_id}']['rating']}")
                if (is_finish and (turn_id >= min_outline_generator_turns)) or (turn_id > max_outline_generator_turns):
                    if require_cache_data:
                        cache_data(input_dict=input_dict, cache_path=cache_path)
                        logging.info(f"finish cache memory bank: {cache_path}")
                    if is_finish:
                        input_dict["outline"] = input_dict[f"outline_turn_{turn_id-1}"]
                        input_dict["judge"] = input_dict[f"judge_turn_{turn_id}"]
                        input_dict["search_result"] = input_dict[f"search_result_turn_{turn_id-1}"]
                        input_dict["search_result_map"] = input_dict[f"search_result_map_turn_{turn_id-1}"]
                    else:
                        max_judge_score_turn = max(
                            range(turn_id),
                            key=lambda _turn_id: input_dict[f"judge_turn_{_turn_id}"]["rating"]
                        )
                        input_dict["outline"] = input_dict[f"outline_turn_{max_judge_score_turn-1}"]
                        input_dict["judge"] = input_dict[f"judge_turn_{max_judge_score_turn}"]
                        input_dict["search_result"] = input_dict[f"search_result_turn_{max_judge_score_turn-1}"]
                        input_dict["search_result_map"] = input_dict[f"search_result_map_turn_{max_judge_score_turn-1}"]
                    break
                
                memory_bank_st = time.time()
                input_dict = await memory_bank.act(
                    input_dict=input_dict,
                    turn_id=turn_id,
                )
                memory_bank_et = time.time()
                logging.info(f"memory bank costs: {memory_bank_et-memory_bank_st}")
                
                outline_generator_st = time.time()
                input_dict = await outline_generator.act(
                    input_dict=input_dict,
                    turn_id=turn_id,
                )
                outline_generator_et = time.time()
                logging.info(f"finish outline generator turn {turn_id} costs: {outline_generator_et-outline_generator_st}")

                turn_id += 1

            num_turns = turn_id
            if use_hierarchical_writer:
                chunks = outline_generator.divide_outline_into_chunks(input_dict[f"outline"])
                num_chunks = len(chunks)
                for chunk_id, chunk in enumerate(chunks):
                    input_dict[f"outline_chunk_{chunk_id}"] = chunk
                input_dict["num_chunks"] = num_chunks
            else:
                input_dict["num_chunks"] = num_chunks = 1.
            input_dict["num_turns"] = num_turns
            
            input_dict = memory_bank.get_docs_with_reference(
                input_dict=input_dict,
                num_chunks=num_chunks,
            )
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"finish cache outline generator: {cache_path}")
    else:
        num_turns = num_chunks = 1
    

    # report writer
    if not disable_report_writer:
        cache_path = os.path.join(cache_dir, f"{job_name}_report_writer.json")
        if skip_report_writer and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load report writer results from cache: {cache_path}")
            num_turns = input_dict["num_turns"]
            num_chunks = input_dict["num_chunks"]
        else:
            report_writer = ReportWriter(
                use_zh=use_zh, 
                use_hierarchical_writer=use_hierarchical_writer,
                use_evidence_as_key=use_evidence_as_key
            )
            if use_hierarchical_writer:
                for chunk_id in range(num_chunks):
                    report_writer_st = time.time()
                    input_dict = await report_writer.act(
                        input_dict=input_dict,
                        chunk_id=chunk_id,
                    )
                    report_writer_et = time.time()
                    logging.info(f"finish hierarchical writer chunk {chunk_id} costs: {report_writer_et-report_writer_st}")
                
                input_dict = report_writer.merge_chunks_into_report(
                    input_dict=input_dict,
                    num_chunks=num_chunks,
                )
            
            else:
                report_writer_st = time.time()
                input_dict = await report_writer.act(
                    input_dict=input_dict,
                )
                report_writer_et = time.time()
                logging.info(f"finish report writer costs: {report_writer_et-report_writer_st}")
            
            if not use_polish_before_render:
                reference_render = ReferenceRender()
                reference_render_st = time.time()
                input_dict = reference_render.act(
                    input_dict=input_dict,
                    input_key="report",
                    output_key="rendered_report",
                )
                reference_render_et = time.time()
                logging.info(f"finish reference render costs: {reference_render_et-reference_render_st}")
            
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"finish cache report writer: {cache_path}")
    
    if not disable_report_polish:
        cache_path = os.path.join(cache_dir, f"{job_name}_report_writer.json")
        if skip_report_polish and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load report polish results from cache: {cache_path}")
        else:
            report_polish = ReportPolish(use_zh=use_zh, use_polish_before_render=use_polish_before_render)
            report_polish_st = time.time()
            input_key = "rendered_report" if use_polish_before_render else "report"
            output_key = "polished_rendered_report" if use_polish_before_render else "polished_report"
            input_dict = await report_polish.act(
                input_dict=input_dict,
                input_key=input_key,
                output_key=output_key
            )
            report_polish_et = time.time()
            logging.info(f"finish report polish costs: {report_polish_et-report_polish_st}")
            
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
        
        if not use_polish_before_render:
            reference_render = ReferenceRender()
            reference_render_st = time.time()
            input_dict = reference_render.act(
                input_dict=input_dict,
                input_key="polished_report",
                output_key="polished_rendered_report",
            )
            reference_render_et = time.time()
            logging.info(f"finish reference render costs: {reference_render_et-reference_render_st}")
    
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + f"_{job_name}_output" + '.md'
    
    logging.info(f"start output data at {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(input_dict[f"{save_key}"])
    
    output_path = None
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + f"_{job_name}_output" + '.json'
    
    logging.info(f"start output data at {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(input_dict["search_result_map"], f, ensure_ascii=False, indent=2)
    
    logging.info(f"finish output data at {output_path}")
    
    if require_cache_data:
        cache_data(input_dict=input_dict, cache_path=output_path)
        logging.info(f"finish output data at {output_path}")
    breakpoint()
    return input_dict





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gin-config-file",
        type=str,
        required=False,
        default="./config/video_agent_demo.gin",
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        required=False,
        default="",
    )
    args = parser.parse_args()
    
    gin.parse_config_file(args.gin_config_file)
    mp.set_start_method("spawn", force=True)
    if args.run_mode == "batch":
        asyncio.run(run_batch_agent())
    else:
        asyncio.run(run_agent())