import os
import gin
import json
import argparse
import logging
import time
import multiprocessing as mp
import asyncio
import sys
import pandas as pd

sys.path.append('./api')

from typing import Dict, Optional, List, Union
from tqdm import tqdm
from datetime import datetime
from api.BaiduSearchService import BaiduImageSearch
from api.GoogleSearchService import GoogleTextImageSearch
from api.EntryJudgeService import EntryJudge
from api.ResponseGeneratorService import ResponseGenerator
from api.QueryGeneratorService import QueryGenerator
from api.RednoteTextSearchService import RedNoteTextSearch
from api.BochaSearchService import BochaTextSearch
from api.RednoteImageSearchService import RedImageSearch
from api.ImageManagerService import RednoteImageManager
from api.NoteJudgeService import NoteJudge
from api.NoteRerankerService import NoteReranker
from api.NoteSummaryService import NoteSummary
from api.ResponseJudgeService import ResponseJudge
from api.CommentSummaryService import CommentSummary
from api.ReferenceRenderService import ReferenceRender
from api.KnowledgeBaseSearchService import KnowledgeBaseSearch


def select_search(
    input_dict: dict,
    judge_thresh: float = 0.6,
    score_thresh: float = 0.,
    top_k_thresh: int = 10,
    like_thresh: int = 0,
):
    for scored_note, search_note in zip(input_dict["scored_search_results"], input_dict["search_results"]):
        if "judge" in search_note:
            scored_note["judge"] = search_note["judge"]
    input_dict["scored_search_results"] = sorted(
        input_dict["scored_search_results"], key=lambda x: x["score"], reverse=True
    )
    notes = input_dict["scored_search_results"]
    selected_notes = []
    for note in notes:
        if "judge" not in note or not isinstance(note["judge"], float):
            selected_notes.append(note)
            continue
        if judge_thresh > 0. and "judge" in note:
            if note["judge"] < judge_thresh:
                continue
        if score_thresh > 0. and "score" in note:
            if note["score"] < score_thresh:
                continue
        if like_thresh > 0. and "like_count" in note:
            if note["like_count"] != -1 and note["like_count"] < like_thresh:
                continue
        selected_notes.append(note)
        
    if top_k_thresh > 0:
        selected_notes = selected_notes[:top_k_thresh]
    
    input_dict["selected_search_results"] = selected_notes
    return input_dict



def merge_search_results(
    input_dict: dict
):
    input_dict["search_results"] = []
    search_notes = input_dict.get("search_notes", [])
    input_dict["search_results"].extend(search_notes)
    if len(search_notes) == 0:
        logging.info("No available search notes")
    
    search_webs = input_dict.get("search_bocha", [])
    input_dict["search_results"].extend(search_webs)
    if len(search_webs) == 0:
        logging.info("No available search bocha")
    
    search_knowledge = input_dict.get("search_knowledge", [])
    input_dict["search_results"].extend(search_knowledge)
    if len(search_knowledge) == 0:
        logging.info("No available search knowledge")
    
    search_images = input_dict.get("search_images", [])
    input_dict["search_results"].extend(search_images)
    if len(search_images) == 0:
        logging.info("No available search images")
    
    search_google_webs = input_dict.get("search_google_webs", [])
    input_dict["search_results"].extend(search_google_webs)
    if len(search_google_webs) == 0:
        logging.info("No available search google webs")
    
    search_google_images = input_dict.get("search_google_images", [])
    input_dict["search_results"].extend(search_google_images)
    if len(search_google_images) == 0:
        logging.info("No available search google images")
    
    
    if len(search_notes) > 0 and len(search_images) > 0:
        new_search_results = []
        note_id = set()
        for note in input_dict["search_results"]:
            if note["id"] in note_id:
                continue
            else:
                note_id.add(note["id"])
            new_search_results.append(note)
        input_dict["search_results"] = new_search_results
    return input_dict



def convert_golden_query_as_query(
    input_dict: dict,
    generate_type: str = "both"
):
    if generate_type in ("note", "both") and "golden_subquery" in input_dict:
        input_dict["note_subquery"] = input_dict["golden_subquery"]
    
    if generate_type in ("web", "both") and "golden_subquery" in input_dict:
        input_dict["note_subquery"] = input_dict["golden_subquery"]
    return input_dict



def load_data_from_cache(
    cache_path: str,
    data_index: int = 0,
    return_list: bool = False,
    load_from_excel: bool = False,
):
    if load_from_excel:
        df = pd.read_excel(cache_path)
        data_list = df.to_dict('records')
    else:
        data_list = []
        with open(cache_path, "r", encoding='utf-8') as f:
            try:
                # 如果文件包含单个JSON对象（多行格式）
                data = f.read().strip()
                if data:
                    data = json.loads(data)
                    if isinstance(data, dict):
                        data_list = [data]
                    elif isinstance(data, list):
                        data_list = data
                    else:
                        raise ValueError(f"Unsupported JSON format: {type(data)}")
            except json.JSONDecodeError as e:
                # 如果是JSONL格式，逐行解析
                f.seek(0)  # 重置文件指针
                for line in f:
                    data_list.append(json.loads(line.strip()))
                
    if return_list:
        return data_list
    input_dict = data_list[data_index]
    return input_dict


def cache_data(
    input_dict: Optional[Union[Dict, List]],
    cache_path: str,
    use_list: bool = False,
):
    if not use_list:
        data_list = [input_dict]
    else:
        data_list = input_dict
    with open(cache_path, "w", encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')  # ensure_ascii=False可以正确显示中文



def set_logger(
    log_dir: str,
    job_name: str
):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{job_name}.log")
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 定义过滤器 - 过滤掉特定的路由日志
    class NoServiceRouteFilter(logging.Filter):
        def filter(self, record):
            message = record.getMessage()
            # 过滤掉interactioncore-service-main的路由未变化日志
            if "interactioncore-service-main" in message and "路由未发生变化" in message:
                return False
            # 过滤其他不需要的服务日志
            if 'reddataservice-service-default' in message:
                return False
            return True
    
    # 创建handler并添加过滤器
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    # 给每个handler添加过滤器
    filter_instance = NoServiceRouteFilter()
    file_handler.addFilter(filter_instance)
    console_handler.addFilter(filter_instance)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[file_handler, console_handler]
    )
    logging.info(f"setup logger at: {log_path}")



def parser_judges(
    input_dict: dict,
    turn_id: int,
):
    # Make a static copy of the items for iteration
    for key, value in list(input_dict.items()):
        if key in ("query_text", "query_image", "note_subquery_judge", "web_subquery_judge"):
            continue
        input_dict[f"turn{turn_id}_{key}"] = value
    
    input_dict["note_subquery"] = input_dict["note_subquery_judge"]
    input_dict["web_subquery"] = input_dict["web_subquery_judge"]
    return input_dict



@gin.configurable()
async def run_batch_workflow(
    job_name: str,
    input_path: str,
    log_dir: str,
    cache_dir: str,
    output_path: str = "",
    max_concurrent: int = 10,
    require_cache_data: bool = False,
    load_from_excel: bool = False,
):
    input_list = load_data_from_cache(cache_path=input_path, return_list=True, load_from_excel=load_from_excel)
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(input_list), desc="Processing workflows")
    
    async def run_workflow_with_semaphore(input_dict, index):
        async with semaphore:
            try:
                result = await run_workflow(
                    input_dict=input_dict,
                    job_name=f"{job_name}_{index}",
                    require_cache_data=require_cache_data,
                    log_dir=f"{log_dir}/{job_name}_{index}",
                    return_search_results=False,
                    return_response_results=False,
                    cache_dir=f"{cache_dir}/{job_name}_{index}"
                )
                pbar.update(1)
                return index, result, None
            except Exception as e:
                pbar.update(1)
                return index, None, e
    
    tasks = [
        run_workflow_with_semaphore(input_dict=input_dict, index=i)
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
async def run_workflow(
    job_name: str,
    input_path: str,
    cache_dir: str,
    download_dir: str,
    log_dir: str,
    use_query_modality: str,
    use_note_modality: str,
    use_golden_query_as_query: bool,
    skip_entry_judge: bool,
    skip_query_generation: bool,
    skip_search: bool,
    skip_download_data: bool,
    skip_note_judge: bool,
    skip_score_note: bool,
    skip_select_note: bool,
    skip_note_summary: bool,
    skip_comment_summary: bool,
    skip_response_generation: bool,
    skip_response_judge: bool,
    disable_entry_judge: bool,
    disable_query_generation: bool,
    disable_search_image: bool,
    disable_search_web: bool,
    disable_search_knowledge: bool,
    disable_search_note: bool,
    disable_search_google_web: bool,
    disable_search_google_image: bool,
    disable_search_baidu_image: bool,
    disable_comment: bool,
    disable_video: bool,
    disable_note_judge: bool,
    disable_note_summary: bool,
    disable_comment_summary: bool,
    disable_response_judge: bool,
    disable_research: bool,
    select_top_k: int,
    output_path: str = "",
    input_dict: Dict = None,
    require_cache_data: bool = True,
    image_key_type: str = "path",
    require_download_data: bool = False,
    return_search_results: bool = False,
    return_response_results: bool = False,
    like_thresh: int = 0,
    include_search: bool = True,
    response_key: str = "response",
    need_render_reference: str = True,
    use_debug: bool = False,
):
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = job_name + "_" + cur_time if not use_debug else job_name
    set_logger(log_dir=log_dir, job_name=job_name)
    if require_cache_data:
        os.makedirs(cache_dir, exist_ok=True)
    
    job_queue = [job_name]
    prev_job_queue = []
    prev_response_queue, prev_note_subquery, prev_web_subquery = [], [], []
    turn_id = 0
    
    input_dict = load_data_from_cache(cache_path=input_path) if input_dict is None else input_dict
    logging.info(f"successfully load input dict from: {input_path}")
    
    
    if "query_text" not in input_dict:
        input_dict["query_text"] = input_dict["query"]
        input_dict["query_image"] = ""
    
    while job_queue:
        job_name = job_queue.pop()
        logging.info(f"start with job name: {job_name}")
        
        # entry judge
        if not disable_entry_judge:
            cache_path = os.path.join(cache_dir, f"{job_name}_entry_judge.json")
            if skip_entry_judge and os.path.exists(cache_path):
                input_dict = load_data_from_cache(cache_path=cache_path)
                logging.info(f"load entry judge results from cache: {cache_path}")
            else:
                entry_judge_st = time.time()
                entry_judge = EntryJudge(
                    use_query_modality=use_query_modality,
                )
                input_dict = await entry_judge.act(input_dict=input_dict)
                entry_judge_et = time.time()
                logging.info(f"entry judge costs: {entry_judge_et-entry_judge_st}")
                if require_cache_data:
                    cache_data(input_dict=input_dict, cache_path=cache_path)
                    logging.info(f"finish cache entry judge: {cache_path}")
        
        # query generation
        if not disable_query_generation:
            cache_path = os.path.join(cache_dir, f"{job_name}_query_generation.json")
            if skip_query_generation and os.path.exists(cache_path):
                input_dict = load_data_from_cache(cache_path=cache_path)
                logging.info(f"load query generation results from cache: {cache_path}")
            else:
                if not disable_search_note:
                    query_generator_note_st = time.time()
                    query_generator = QueryGenerator(
                        use_query_modality=use_query_modality,
                        generate_type="note"
                    )
                    input_dict = await query_generator.act(input_dict=input_dict)
                    query_generator_note_et = time.time()
                    logging.info(f"query generation for note costs: {query_generator_note_et-query_generator_note_st}")
                
                if not disable_search_web:
                    query_generator_web_st = time.time()
                    query_generator = QueryGenerator(use_query_modality=use_query_modality, generate_type="web")
                    input_dict = await query_generator.act(input_dict=input_dict)
                    query_generator_web_et = time.time()
                    logging.info(f"query generation for web costs: {query_generator_web_et-query_generator_web_st}")
                
                if not disable_search_knowledge:
                    query_generator_knowledge_st = time.time()
                    query_generator = QueryGenerator(
                        use_query_modality=use_query_modality,
                        generate_type="note"
                    )
                    input_dict = await query_generator.act(input_dict=input_dict)
                    query_generator_knowledge_et = time.time()
                    logging.info(f"query generation for note costs: {query_generator_knowledge_et-query_generator_knowledge_st}")
                    input_dict["knowledge_subquery"] = input_dict["note_subquery"]
                
                if require_cache_data:
                    cache_data(input_dict=input_dict, cache_path=cache_path)
                    logging.info(f"finish cache query generation: {cache_path}")
        
        
        if use_golden_query_as_query:
            input_dict = convert_golden_query_as_query(input_dict=input_dict)
        
        # Executor
        # query search
        cache_path = os.path.join(cache_dir, f"{job_name}_search_results.json")
        if skip_search and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load search results from cache: {cache_path}")
        else:
            if not disable_search_note:
                note_search_st = time.time()
                note_search = RedNoteTextSearch(
                    user_id="life",
                    disable_comment=disable_comment,
                    disable_video=disable_video,
                    use_note_modality=use_note_modality,
                )
                input_dict = await note_search.act(input_dict=input_dict)
                note_search_et = time.time()
                logging.info(f"search notes costs: {note_search_et-note_search_st}")
            
            if not disable_search_web:
                web_search_st = time.time()
                web_search = BochaTextSearch()
                input_dict = await web_search.act(input_dict=input_dict)
                web_search_et = time.time()
                logging.info(f"search webs costs: {web_search_et-web_search_st}")
            
            if not disable_search_knowledge:
                knowledge_search_st = time.time()
                knowledge_search = KnowledgeBaseSearch()
                input_dict = await knowledge_search.act(input_dict=input_dict)
                knowledge_search_et = time.time()
                logging.info(f"search knowledge costs: {knowledge_search_et-knowledge_search_st}")

            if not disable_search_image:
                image_search_st = time.time()
                image_search = RedImageSearch(
                    user_id="life",
                    disable_comment=disable_comment,
                    disable_video=disable_video,
                    use_note_modality=use_note_modality,
                )
                input_dict = await image_search.act(input_dict=input_dict)
                image_search_et = time.time()
                logging.info(f"search images costs: {image_search_et-image_search_st}")
            
            if not disable_search_google_web:
                google_search_web_st = time.time()
                google_search_web = GoogleTextImageSearch(
                    search_type="text"
                )
                input_dict = await google_search_web.act(input_dict=input_dict)
                google_search_web_et = time.time()
                logging.info(f"google search webs costs: {google_search_web_et-google_search_web_st}")
            
            if not disable_search_google_image:
                google_search_image_st = time.time()
                google_search_image = GoogleTextImageSearch(
                    search_type="image"
                )
                input_dict = await google_search_image.act(input_dict=input_dict)
                google_search_image_et = time.time()
                logging.info(f"google search image costs: {google_search_image_et-google_search_image_st}")
            
            if not disable_search_baidu_image:
                baidu_search_image_st = time.time()
                baidu_search_image = BaiduImageSearch()
                input_dict = await baidu_search_image.act(input_dict=input_dict)
                baidu_search_image_et = time.time()
                logging.info(f"baidu search image costs: {baidu_search_image_et-baidu_search_image_st}")
            
            input_dict = merge_search_results(input_dict=input_dict)
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"finish cache search results: {cache_path}")
        
        breakpoint()
        # download
        if image_key_type == "path" or require_download_data:
            cache_path = os.path.join(cache_dir, f"{job_name}_downloaded_search_results.json")
            if skip_download_data and os.path.exists(cache_path):
                input_dict = load_data_from_cache(cache_path=cache_path)
                logging.info(f"load downloaded search results from cache: {cache_path}")
            else:
                image_download_st = time.time()
                image_manager = RednoteImageManager(
                    cache_dir=download_dir,
                    require_convert_url_domain=True,
                    download_type="image"
                )
                input_dict = await image_manager.act(input_dict=input_dict)
                image_download_et = time.time()
                logging.info(f"image download costs: {image_download_et-image_download_st}")
                if require_cache_data:
                    cache_data(input_dict=input_dict, cache_path=cache_path)
                    logging.info(f"finish cache downloaded image results: {cache_path}")
                
                if not disable_video:
                    video_download_st = time.time()
                    video_manager = RednoteImageManager(
                        cache_dir=download_dir,
                        require_convert_url_domain=False,
                        download_type="video"
                    )
                    input_dict = await video_manager.act(input_dict=input_dict)
                    video_download_et = time.time()
                    logging.info(f"video download costs: {video_download_et-video_download_st}")
                    if require_cache_data:
                        cache_data(input_dict=input_dict, cache_path=cache_path)
                        logging.info(f"finish cache downloaded video results: {cache_path}")
        
        # note reranker
        cache_path = os.path.join(cache_dir, f"{job_name}_scored_results.json")
        if skip_score_note and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load scored note results from cache: {cache_path}")
        else:
            note_reranker_st = time.time()
            note_reranker = NoteReranker(
                use_query_modality=use_query_modality,
                use_note_modality=use_note_modality
            )
            input_dict = await note_reranker.act(input_dict=input_dict)
            note_reranker_et = time.time()
            logging.info(f"rerank search results costs: {note_reranker_et-note_reranker_st}")
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"finish cache score notes: {cache_path}")
        
        # note judge
        if not disable_note_judge:
            cache_path = os.path.join(cache_dir, f"{job_name}_judged_search_results.json")
            if skip_note_judge and os.path.exists(cache_path):
                input_dict = load_data_from_cache(cache_path=cache_path)
                logging.info(f"load judged search results from cache: {cache_path}")
            else:
                note_judge_st = time.time()
                note_judge = NoteJudge(
                    use_query_modality=use_query_modality,
                    use_note_modality=use_note_modality,
                    image_key_type=image_key_type
                )
                input_dict = await note_judge.act(input_dict=input_dict)
                note_judge_et = time.time()
                logging.info(f"judge search results costs: {note_judge_et-note_judge_st}")
                if require_cache_data:
                    cache_data(input_dict=input_dict, cache_path=cache_path)
                    logging.info(f"finish cache judged search results: {cache_path}")
        
        # note select
        cache_path = os.path.join(cache_dir, f"{job_name}_selected_results.json")
        if skip_select_note and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load selected note results from cache: {cache_path}")
        else:
            note_select_st = time.time()
            input_dict = select_search(
                input_dict=input_dict,
                top_k_thresh=select_top_k,
                like_thresh=0. if not disable_search_image or not disable_search_web else like_thresh
            )
            note_select_et = time.time()
            logging.info(f"select note results costs: {note_select_et-note_select_st}")
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"finish cache selected notes: {cache_path}")
        
        
        # note summary
        if not disable_note_summary:
            cache_path = os.path.join(cache_dir, f"{job_name}_summary_search_results.json")
            if skip_note_summary and os.path.exists(cache_path):
                input_dict = load_data_from_cache(cache_path=cache_path)
                logging.info(f"load judged search results from cache: {cache_path}")
            else:
                note_summary_st = time.time()
                note_summary = NoteSummary(
                    use_query_modality=use_query_modality,
                    use_note_modality=use_note_modality,
                    image_key_type=image_key_type
                )
                input_dict = await note_summary.act(input_dict=input_dict)
                note_summary_et = time.time()
                logging.info(f"summary search results costs: {note_summary_et-note_summary_st}")
                if require_cache_data:
                    cache_data(input_dict=input_dict, cache_path=cache_path)
                    logging.info(f"finish cache selected summary notes: {cache_path}")
        
        if return_search_results:
            return input_dict["search_results"]
        
        # comment summary
        if not disable_comment_summary:
            cache_path = os.path.join(cache_dir, f"{job_name}_comment_summary_search_results.json")
            if skip_comment_summary and os.path.exists(cache_path):
                input_dict = load_data_from_cache(cache_path=cache_path)
                logging.info(f"load comment summary results from cache: {cache_path}")
            else:
                comment_summary_st = time.time()
                comment_summary = CommentSummary(
                    use_query_modality=use_query_modality,
                    use_note_modality=use_note_modality,
                    image_key_type=image_key_type,
                )
                input_dict = await comment_summary.act(input_dict=input_dict)
                comment_summary_et = time.time()
                logging.info(f"comment summary results costs: {comment_summary_et-comment_summary_st}")
                if require_cache_data:
                    cache_data(input_dict=input_dict, cache_path=cache_path)
                    logging.info(f"finish cache comment summary notes: {cache_path}")
        
        
        # response
        cache_path = os.path.join(cache_dir, f"{job_name}_response.json")
        if skip_response_generation and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load response from cache: {cache_path}")
        else:
            response_generate_st = time.time()
            response_generator = ResponseGenerator(
                use_query_modality=use_query_modality,
                use_note_modality=use_note_modality,
                include_search=include_search,
                include_summary=False,
                image_key_type=image_key_type,
                include_comment=not disable_comment,
            )
            input_dict = await response_generator.act(
                input_dict=input_dict,
                response_key=response_key,
            )
            response_generate_et = time.time()
            logging.info(f"response costs: {response_generate_et-response_generate_st}")
            
            reference_render = ReferenceRender()
            reference_render_st = time.time()
            input_dict = reference_render.act(
                input_dict=input_dict,
                input_key="response",
                output_key="rendered_response"
            )
            reference_render_et = time.time()
            logging.info(f"finish reference render costs: {reference_render_et-reference_render_st}")
            
            
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"finish cache generated response: {cache_path}")
        
        if return_response_results:
            return input_dict["response"]
        
        # Planner
        # response judge
        if not disable_response_judge:
            cache_path = os.path.join(cache_dir, f"{job_name}_judge.json")
            if skip_response_judge and os.path.exists(cache_path):
                input_dict = load_data_from_cache(cache_path=cache_path)
                logging.info(f"load response judge from cache: {cache_path}")
            else:
                if not disable_search_note:
                    judge_generate_note_st = time.time()
                    judge_generator = ResponseJudge(
                        use_query_modality=use_query_modality,
                        generate_type="note",
                    )
                    input_dict = await judge_generator.act(input_dict=input_dict, response_key=response_key,
                                                           response_queue=prev_response_queue,
                                                           note_subquery_queue=prev_note_subquery,
                                                           web_subquery_queue=prev_web_subquery)
                    judge_generate_note_et = time.time()
                    logging.info(f"judge note subquery costs: {judge_generate_note_et-judge_generate_note_st}")
                
                if not disable_search_web:
                    judge_generate_web_st = time.time()
                    judge_generator = ResponseJudge(
                        use_query_modality=use_query_modality,
                        generate_type="web",
                    )
                    input_dict = await judge_generator.act(input_dict=input_dict, response_key=response_key,
                                                           response_queue=prev_response_queue,
                                                           note_subquery_queue=prev_note_subquery,
                                                           web_subquery_queue=prev_web_subquery)
                    judge_generate_web_et = time.time()
                    logging.info(f"judge web subquery costs: {judge_generate_web_et-judge_generate_web_st}")
                
                if require_cache_data:
                    cache_data(input_dict=input_dict, cache_path=cache_path)
                    logging.info(f"finish cache judged response: {cache_path}")
        
        if not disable_research:
            input_dict = parser_judges(input_dict=input_dict, turn_id=turn_id)
            if len(input_dict["note_subquery"]) > 0 or len(input_dict["note_subquery"]) > 0:
                prev_response_queue.append(input_dict[f"turn{turn_id}_response"])
                prev_note_subquery.append(input_dict[f"turn{turn_id}_note_subquery"])
                prev_web_subquery.append(input_dict[f"turn{turn_id}_web_subquery"])
                turn_id += 1
                
                disable_query_generation = True
                disable_search_note = len(input_dict["note_subquery"]) == 0
                disable_search_web = len(input_dict["web_subquery"]) == 0
                
                job_name = job_name + f"_turn_{turn_id}"
                job_queue.append(job_name)
                logging.info("need re-search!")
                logging.info(f"new job name is {job_name}")
            
            else:
                logging.info(f"finish re-search with num of turns == {turn_id+1}")
        
        else:
            logging.info("no re-search")
        prev_job_queue.append(job_name)
    
    if len(output_path) == 0 and len(input_path) > 0:
        base, ext = os.path.splitext(input_path)
        output_path = base + f"_{job_name}_output" + ".md"
    
    if not disable_search_knowledge: # 知识库搜索没有url
        need_render_reference = False
    # breakpoint()
    with open(output_path, "w", encoding="utf-8") as f:
        if need_render_reference:
            f.write(input_dict[f"rendered_{response_key}"])
        else:
            f.write(input_dict[response_key])
    
    output_path = None
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + f"_{job_name}_output" + '.json'
    logging.info(f"start output data at {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(input_dict["selected_search_results"], f, ensure_ascii=False, indent=2)
    
    if require_cache_data and len(output_path) > 0:
        cache_data(input_dict=input_dict, cache_path=output_path)
    
    logging.info(f"finish output data at {output_path}")
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
        asyncio.run(run_batch_workflow())
    else:
        asyncio.run(run_workflow())