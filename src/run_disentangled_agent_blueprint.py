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
from agent.ReportWriterAgent import ReportWriter
# from agent.ReportWriterEvidenceAgent import ReportWriter
from agent.ReportPolishAgent import ReportPolish
from api.DisentangledOutlineGeneratorBlueprintService import DisentangledOutlineGeneratorBlueprint
from api.DisentangledOutlineJudgeBlueprintService import DisentangledOutlineJudgeBlueprint
from api.ReferenceRenderService import ReferenceRender
from api.IntentPlannerService import IntentPlanner
from api.NoteSelectorService import NoteSelector
from api.ImageManagerService import RednoteImageManager
from api.ImageSelectorService import ImageSelector
from api.HTMLRenderService import HTMLRender
from api.PosterSlideContentGeneratorService import PosterSlideContentGenerator
from api.SlideImageRenderService import SlideImageRender
from api.SlideRenderService import SlideRender
from api.SlideVideoRenderService import SlideVideoRender
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


def find_last_completed_turn(cache_dir: str, job_name: str, max_turns: int) -> int:
    """找到最后完成的轮次"""
    last_completed_turn = -1
    for turn_id in range(max_turns):
        turn_cache_path = os.path.join(cache_dir, f"{job_name}_outline_generator_turn_{turn_id}.json")
        if os.path.exists(turn_cache_path):
            last_completed_turn = turn_id
        else:
            break
    return last_completed_turn


def load_turn_data(cache_dir: str, job_name: str, turn_id: int) -> Dict:
    """加载指定轮次的数据"""
    turn_cache_path = os.path.join(cache_dir, f"{job_name}_outline_generator_turn_{turn_id}.json")
    return load_data_from_cache(cache_path=turn_cache_path)


def save_turn_data(input_dict: Dict, cache_dir: str, job_name: str, turn_id: int):
    """保存指定轮次的数据"""
    turn_cache_path = os.path.join(cache_dir, f"{job_name}_outline_generator_turn_{turn_id}.json")
    cache_data(input_dict=input_dict, cache_path=turn_cache_path)
    return turn_cache_path


def get_max_judge_turn_id(input_dict: Dict):
    max_judge_turn_id = -1
    max_judge_value = -1
    turn_id = 0
    while f"judge_turn_{turn_id}" in input_dict:
        if input_dict[f"judge_turn_{turn_id}"]["rating"] > max_judge_value:
            max_judge_value = input_dict[f"judge_turn_{turn_id}"]["rating"]
            max_judge_turn_id = turn_id
        turn_id += 1
    return input_dict[f"outline_turn_{max_judge_turn_id-1}"]



@gin.configurable()
async def run_agent(
    job_name: str,
    input_path: str,
    cache_dir: str,
    download_dir: str,
    log_dir: str,
    skip_query_miner: bool,
    skip_intent_planner: bool,
    skip_outline_generator: bool,
    skip_report_writer: bool,
    skip_report_polish: bool,
    skip_note_selector: bool,
    skip_image_selector: bool,
    skip_html_render: bool,
    skip_slide_content_planner: bool,
    skip_slide_image_render: bool,
    skip_slide_render: bool,
    skip_slide_video_render: bool,
    disable_query_miner: bool,
    disable_intent_planner: bool,
    disable_images: bool,
    disable_multi_images: bool,
    disable_comment: bool,
    disable_video: bool,
    disable_outline_generator: bool,
    disable_report_writer: bool,
    disable_report_polish: bool,
    disable_note_selector: bool,
    disable_image_selector: bool,
    disable_html_render: bool,
    disable_slide_content_planner: bool,
    disable_slide_image_render: bool,
    disable_slide_render: bool,
    disable_slide_video_render: bool,
    render_with_image: bool = True,
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
    force_restart_outline_generator: bool = False,
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
    
    # breakpoint()
    # intent planner
    if not disable_intent_planner:
        cache_path = os.path.join(cache_dir, f"{job_name}_intent_planner.json")
        if skip_intent_planner and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load intent planner results from cache: {cache_path}")
        else:
            intent_planner_st = time.time()
            intent_planner = IntentPlanner(
                use_zh=use_zh,
            )
            input_dict = await intent_planner.act(
                input_dict=input_dict,
            )
            intent_planner_et = time.time()
            logging.info(f"intent planner costs: {intent_planner_et-intent_planner_st}")
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"finish cache intent planner: {cache_path}")
        
    # use_hierarchical_writer = use_hierarchical_writer and input_dict["intent"] in ("事实查询", "状态进展", "新闻资讯", "探索深入", "资源定位")

    # breakpoint()
    # outline generator
    if not disable_outline_generator:
        final_cache_path = os.path.join(cache_dir, f"{job_name}_outline_generator_final.json")
        
        # 检查是否跳过整个outline generator
        if skip_outline_generator and os.path.exists(final_cache_path):
            input_dict = load_data_from_cache(cache_path=final_cache_path)
            num_turns = input_dict["num_turns"]
            num_chunks = input_dict["num_chunks"]
            logging.info(f"load outline generator final results from cache: {final_cache_path}")
        else:
            # 找到最后完成的轮次（除非强制重新开始）
            last_completed_turn = -1
            if not force_restart_outline_generator:
                last_completed_turn = find_last_completed_turn(cache_dir, job_name, max_outline_generator_turns)
                logging.info(f"last completed turn: {last_completed_turn}")
            
            # 如果有已完成的轮次，加载最后一轮的数据
            if last_completed_turn >= 0:
                input_dict = load_turn_data(cache_dir, job_name, last_completed_turn)
                logging.info(f"loaded data from turn {last_completed_turn}")
            
            # 初始化组件
            outline_judge = DisentangledOutlineJudgeBlueprint(
                use_zh=use_zh,
                use_evidence_as_key=False,
                need_filter=False,
                outline_judge_threshold=outline_judge_threshold,
                max_outline_generator_turns=max_outline_generator_turns,
                min_outline_generator_turns=min_outline_generator_turns,
            )
            outline_generator = DisentangledOutlineGeneratorBlueprint(
                use_zh=use_zh,
                use_evidence_as_key=False,
            )
            memory_bank = MemoryBankManager(
                use_zh=use_zh,
                use_evidence_as_key=False,
                use_hierarchical_writer=use_hierarchical_writer
            )
            
            # 从下一轮开始执行
            start_turn = last_completed_turn + 1
            turn_id = start_turn
            
            logging.info(f"starting outline generator from turn {start_turn}")
            
            while turn_id < max_outline_generator_turns:
                logging.info(f"starting turn {turn_id}")
                
                turn_cache_path = os.path.join(cache_dir, f"{job_name}_outline_generator_turn_{turn_id}.json")
                if os.path.exists(turn_cache_path) and not force_restart_outline_generator:
                    input_dict = load_data_from_cache(cache_path=turn_cache_path)
                    logging.info(f"turn {turn_id} already exists, loaded from cache: {turn_cache_path}")
                    turn_id += 1
                    continue
                if f"is_finish_turn_{turn_id-1}" in input_dict and input_dict[f"is_finish_turn_{turn_id-1}"]:
                    break
                
                # 执行outline judge
                outline_judge_st = time.time()
                is_finish, input_dict = await outline_judge.act(
                    input_dict=input_dict,
                    turn_id=turn_id,
                )
                outline_judge_et = time.time()
                logging.info(f"finish outline judge turn {turn_id} costs: {outline_judge_et-outline_judge_st}")
                
                input_dict[f"is_finish_turn_{turn_id}"] = is_finish
                logging.info(f"turn {turn_id} is_finish: {is_finish}")
                
                # 保存当前轮次的judge结果
                if require_cache_data:
                    turn_cache_path = save_turn_data(input_dict, cache_dir, job_name, turn_id)
                    logging.info(f"saved turn {turn_id} judge result: {turn_cache_path}")
                
                # 如果决定停止，或者已到最大轮次，结束循环
                if is_finish:
                    logging.info(f"outline generator finished at turn {turn_id}")
                    break
                
                # 执行memory bank
                memory_bank_st = time.time()
                input_dict = await memory_bank.act(
                    input_dict=input_dict,
                    turn_id=turn_id,
                )
                memory_bank_et = time.time()
                logging.info(f"memory bank turn {turn_id} costs: {memory_bank_et-memory_bank_st}")
                
                # 执行outline generator
                outline_generator_st = time.time()
                input_dict = await outline_generator.act(
                    input_dict=input_dict,
                    turn_id=turn_id,
                )
                outline_generator_et = time.time()
                logging.info(f"finish outline generator turn {turn_id} costs: {outline_generator_et-outline_generator_st}")
                
                # 保存完整的轮次结果
                if require_cache_data:
                    turn_cache_path = save_turn_data(input_dict, cache_dir, job_name, turn_id)
                    logging.info(f"saved complete turn {turn_id} result: {turn_cache_path}")
                
                turn_id += 1
            
            num_turns = turn_id + 1  # 包含最后执行的轮次
            
            # 选择最佳结果
            if turn_id >= 0:
                # 查找最高评分的轮次
                best_turn = -1
                best_score = -1
                for i in range(num_turns):
                    if f"judge_turn_{i}" in input_dict:
                        score = input_dict[f"judge_turn_{i}"]["rating"]
                        if score > best_score:
                            best_score = score
                            best_turn = i
                        logging.info(f"turn {i} score: {score}")
                
                if best_turn >= 0:
                    # 使用最佳轮次的outline（outline是在前一轮或同一轮生成的）
                    # 如果best_turn是judge轮次，outline应该来自前一轮的生成
                    outline_turn = max(0, best_turn - 1) if best_turn > 0 else 0
                    
                    input_dict["outline"] = input_dict[f"outline_turn_{outline_turn}"]
                    input_dict["search_result"] = input_dict[f"search_result_turn_{outline_turn}"]
                    input_dict["search_result_map"] = input_dict[f"search_result_map_turn_{outline_turn}"]
                    
                    input_dict["judge"] = input_dict[f"judge_turn_{best_turn}"]
                    input_dict["blueprint"] = input_dict[f"blueprint_turn_{outline_turn}"]
                    logging.info(f"selected best outline from turn {outline_turn} with judge score {best_score} from judge turn {best_turn}")
            

            if use_hierarchical_writer and "outline" in input_dict:
                chunks = outline_generator.divide_outline_into_chunks(input_dict["outline"])
                num_chunks = len(chunks)
                for chunk_id, chunk in enumerate(chunks):
                    input_dict[f"outline_chunk_{chunk_id}"] = chunk
                input_dict["num_chunks"] = num_chunks
            else:
                input_dict["num_chunks"] = num_chunks = 1
            logging.info(f"finish divide outline into chunks")
            input_dict["num_turns"] = num_turns
            
            if "outline" in input_dict:
                input_dict = memory_bank.get_docs_with_reference(
                    input_dict=input_dict,
                    num_chunks=num_chunks,
                )
            logging.info(f"finish finsih get references for outline")
            
            # 保存最终结果
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=final_cache_path)
                logging.info(f"finish cache outline generator final: {final_cache_path}")
    else:
        num_turns = num_chunks = 1

    # breakpoint()
    # report writer
    if not disable_report_writer:
        cache_path = os.path.join(cache_dir, f"{job_name}_report_writer.json")
        if skip_report_writer and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load report writer results from cache: {cache_path}")
            num_turns = input_dict.get("num_turns", 1)
            num_chunks = input_dict.get("num_chunks", 1)
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
    
    if not disable_note_selector:
        cache_path = os.path.join(cache_dir, f"{job_name}_note_selector.json")
        if skip_note_selector and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load note selector results from cache: {cache_path}")
        else:
            note_selector_st = time.time()
            note_selector = NoteSelector(use_zh=use_zh)
            input_dict = await note_selector.act(input_dict=input_dict, turn_id=-1)
            note_selector_et = time.time()
            logging.info(f"note_selector costs: {note_selector_et-note_selector_st}")

            image_manager_st = time.time()
            image_manger = RednoteImageManager(cache_dir=download_dir)
            input_dict = await image_manger.act(input_dict=input_dict, input_key="search_result")
            image_manager_et = time.time()
            logging.info(f"image manager costs: {image_manager_et-image_manager_st}")

            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"finish cache note selector: {cache_path}")
        
    if not disable_image_selector:
        cache_path = os.path.join(cache_dir, f"{job_name}_image_selector.json")
        if skip_image_selector and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load image selector from cache: {cache_path}")
        else:
            image_selector_st = time.time()
            image_selector = ImageSelector(use_zh=use_zh)
            input_dict = await image_selector.act(input_dict=input_dict, turn_id=-1)
            image_selector_et = time.time()
            logging.info(f"image_selector costs: {image_selector_et-image_selector_st}")

            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"cache image selector to: {cache_path}")
    
    if not disable_html_render:
        cache_path = os.path.join(cache_dir, f"{job_name}_html_render.json")
        if skip_html_render and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load html render results from cache: {cache_path}")
        else:
            html_render_st = time.time()
            html_render = HTMLRender(use_zh=use_zh, render_with_image=render_with_image and not disable_image_selector)
            input_dict = await html_render.act(
                input_dict=input_dict,
                input_key="rendered_report",
            )
            html_render_et = time.time()
            logging.info(f"finish html render costs: {html_render_et-html_render_st}")

            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"cache html render to: {cache_path}")

    if not disable_slide_content_planner:
        cache_path = os.path.join(cache_dir, f"{job_name}_slide_content_planner.json")
        if skip_slide_content_planner and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load slide content planner from cache: {cache_path}")
        else:
            slide_content_planner_st = time.time()
            slide_content_planner = PosterSlideContentGenerator(
                output_type="xhs_slides",
            )
            input_dict = await slide_content_planner.act(
                input_dict=input_dict,
                input_key="rendered_report",
            )
            slide_content_planner_et = time.time()
            logging.info(f"finish slide content planning costs: {slide_content_planner_et-slide_content_planner_st}")

            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"cache slide content planner to: {cache_path}")

    breakpoint()
    if not disable_slide_image_render:
        cache_path = os.path.join(cache_dir, f"{job_name}_slide_image_render.json")
        if skip_slide_image_render and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load slide image render from cache: {cache_path}")
        else:
            slide_image_render_st = time.time()
            slide_output_dir = os.path.join(cache_dir, f"{job_name}_slide_images")
            slide_image_render = SlideImageRender(use_zh=use_zh)
            input_dict = await slide_image_render.act(
                input_dict=input_dict,
                input_key="slide_content_plan",
                output_dir=slide_output_dir,
            )
            slide_image_render_et = time.time()
            logging.info(f"finish slide image render costs: {slide_image_render_et-slide_image_render_st}")

            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"cache slide image render to: {cache_path}")
    breakpoint()
    if not disable_slide_render:
        cache_path = os.path.join(cache_dir, f"{job_name}_slide_render.json")
        if skip_slide_render and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load slide render from cache: {cache_path}")
        else:
            slide_render_st = time.time()
            slide_render = SlideRender(use_zh=use_zh)
            input_dict = await slide_render.act(
                input_dict=input_dict,
                input_key="rendered_report",
            )
            slide_render_et = time.time()
            logging.info(f"finish slide render (xhs copy + html) costs: {slide_render_et-slide_render_st}")

            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"cache slide render to: {cache_path}")

    if not disable_slide_video_render:
        cache_path = os.path.join(cache_dir, f"{job_name}_slide_video_render.json")
        if skip_slide_video_render and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load slide video render from cache: {cache_path}")
        else:
            slide_video_render_st = time.time()
            slide_video_output_dir = os.path.join(cache_dir, f"{job_name}_slide_video")
            slide_video_render = SlideVideoRender(use_zh=use_zh)
            input_dict = await slide_video_render.act(
                input_dict=input_dict,
                input_key="xhs_slide_images",
                output_dir=slide_video_output_dir,
            )
            slide_video_render_et = time.time()
            logging.info(f"finish slide video render costs: {slide_video_render_et-slide_video_render_st}")

            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"cache slide video render to: {cache_path}")

    breakpoint()
    if not disable_report_polish:
        cache_path = os.path.join(cache_dir, f"{job_name}_report_polish.json")
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
    
    output_path_reference_json = None
    if output_path_reference_json is None:
        base, ext = os.path.splitext(input_path)
        output_path_reference_json = base + f"_{job_name}_reference" + '.json'
    
    logging.info(f"start output reference data at {output_path_reference_json}")
    with open(output_path_reference_json, "w", encoding="utf-8") as f:
        json.dump(input_dict.get("search_result_map", {}), f, ensure_ascii=False, indent=2)
    
    output_path_json = None
    if output_path_json is None:
        base, ext = os.path.splitext(input_path)
        output_path_json = base + f"_{job_name}_output" + '.json'

    logging.info(f"start output data at {output_path_json}")
    with open(output_path_json, "w", encoding="utf-8") as f:
        json.dump(input_dict, f, ensure_ascii=False, indent=2)
    
    logging.info(f"finish output data at {output_path_json}")
    # breakpoint()
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
