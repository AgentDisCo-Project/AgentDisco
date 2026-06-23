import gin
import asyncio
import os
import logging
import time
import argparse
import multiprocessing as mp
import numpy as np
import sys

sys.path.append('./api')

from tqdm import tqdm
from run_workflow import set_logger, load_data_from_cache, cache_data, save_by_extension
from api.EntryJudgeService import EntryJudge
from api.RednoteTextSearchService import RedNoteTextSearch
from api.ImageManagerService import RednoteImageManager
from api.NoteRerankerService import NoteReranker
from api.QueryGeneratorService import QueryGenerator
from synthesis.QuerySynthesisService import QuerySynthesis



def aggregate_subquery_score(
    input_dict: dict,
    input_key: str,
):
    notes = input_dict[input_key]
    scores = [note["score"] for note in notes]
    average_all = float(np.mean(scores))
    average_1 = float(np.mean(scores[:1]))
    average_2 = float(np.mean(scores[:2]))
    average_3 = float(np.mean(scores[:3]))
    average_5 = float(np.mean(scores[:5]))
    average_10 = float(np.mean(scores[:10]))
    average_15 = float(np.mean(scores[:15]))
    average_20 = float(np.mean(scores[:20]))
    metrics = {
        "average_1": average_1,
        "average_2": average_2,
        "average_3": average_3,
        "average_5": average_5,
        "average_10": average_10,
        "average_15": average_15,
        "average_20": average_20,
        "average_all": average_all,
    }
    input_dict[f"{input_key}_metrics"] = metrics
    return input_dict
    
    

@gin.configurable()
async def run_batch_subquery(
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
    
    async def run_subquery_with_semaphore(input_dict, index):
        async with semaphore:
            try:
                result = await run_subquery(
                    input_dict=input_dict,
                    job_name=f"{job_name}_{index}",
                    require_cache_data=require_cache_data,
                    log_dir=f"{log_dir}/{job_name}_{index}",
                    return_search_results=False,
                    return_response_results=True,
                    cache_dir=f"{cache_dir}/{job_name}_{index}"
                )
                pbar.update(1)
                return index, result, None
            except Exception as e:
                pbar.update(1)
                return index, None, e
    
    tasks = [
        run_subquery_with_semaphore(input_dict=input_dict, index=i)
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
    save_by_extension(data_list=merged_list, file_path=output_path)
    return merged_list


@gin.configurable()
async def run_subquery(
    job_name: str,
    input_path: str,
    log_dir: str,
    cache_dir: str,
    download_dir: str,
    use_query_modality: str,
    use_note_modality: str,
    skip_entry_judge: bool,
    skip_query_generation: bool,
    skip_search: bool,
    skip_download_data: bool,
    skip_score_note: bool,
    disable_entry_judge: bool,
    disable_query_generation: bool,
    disable_search_note: bool,
    disable_comment: bool,
    disable_video: bool,
    num_samples: int = 5,
    input_dict: dict = None,
    require_cache_data: bool = True,
    image_key_type: str = "path",
    require_download_data: bool = False,
    use_similar_query_as_content: bool = True,
):
    set_logger(log_dir=log_dir, job_name=job_name)
    
    input_dict = load_data_from_cache(cache_path=input_path) if input_dict is None else input_dict
    logging.info(f"successfully load input dict from: {input_path}")
    
    # entry judge
    if not disable_entry_judge:
        cache_path = os.path.join(cache_dir, f"{job_name}_entry_judge.json")
        if skip_entry_judge and os.path.exists(cache_path):
            input_dict = await load_data_from_cache(cache_path=cache_path)
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
    
    if not disable_query_generation:
        # query generation
        cache_path = os.path.join(cache_dir, f"{job_name}_query_generation.json")
        if skip_query_generation and os.path.exists(cache_path):
            input_dict = await load_data_from_cache(cache_path=cache_path)
            logging.info(f"load query generation results from cache: {cache_path}")
        else:
            if not disable_search_note:
                query_generator_note_st = time.time()
                if use_similar_query_as_content:
                    query_generator = QuerySynthesis(use_query_modality=use_query_modality)
                else:
                    query_generator = QueryGenerator(
                        use_query_modality=use_query_modality,
                        generate_type="note"
                    )
                for sample_id in range(num_samples):
                    input_dict = await query_generator.act(
                        input_dict=input_dict,
                        output_key=f"note_subquery_{sample_id}",
                    )
                query_generator_note_et = time.time()
                logging.info(f"query generation for note costs: {query_generator_note_et-query_generator_note_st}")
            
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"finish cache query generation: {cache_path}")
        
        
        # Executor
        # query search
        cache_path = os.path.join(cache_dir, f"{job_name}_search_results.json")
        if skip_search and os.path.exists(cache_path):
            input_dict = await load_data_from_cache(cache_path=cache_path)
            logging.info(f"load search results from cache: {cache_path}")
        else:
            if not disable_search_note:
                note_search_st = time.time()
                note_search = RedNoteTextSearch(
                    disable_comment=disable_comment,
                    disable_video=disable_video,
                    use_note_modality=use_note_modality,
                )
                for sample_id in range(num_samples):
                    input_dict = await note_search.act(
                        input_dict=input_dict,
                        input_key=f"note_subquery_{sample_id}",
                        output_key=f"search_notes_{sample_id}"
                    )
                note_search_et = time.time()
                logging.info(f"search notes costs: {note_search_et-note_search_st}")
                        
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"finish cache search results: {cache_path}")
        
        
        # download
        if image_key_type == "path" or require_download_data:
            cache_path = os.path.join(cache_dir, f"{job_name}_downloaded_search_results.json")
            if skip_download_data and os.path.exists(cache_path):
                input_dict = await load_data_from_cache(cache_path=cache_path)
                logging.info(f"load downloaded search results from cache: {cache_path}")
            else:
                image_download_st = time.time()
                image_manager = RednoteImageManager(
                    cache_dir=download_dir,
                    require_convert_url_domain=True,
                    download_type="image"
                )
                for sample_id in range(num_samples):
                    input_dict = await image_manager.act(
                        input_dict=input_dict,
                        input_key=f"search_notes_{sample_id}"
                    )
                image_download_et = time.time()
                logging.info(f"image download costs: {image_download_et-image_download_st}")
                if require_cache_data:
                    cache_data(input_dict=input_dict, cache_path=cache_path)
                    logging.info(f"finish cache downloaded image results: {cache_path}")
                
        # note reranker
        cache_path = os.path.join(cache_dir, f"{job_name}_scored_results.json")
        if skip_score_note and os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load scored note results from cache: {cache_path}")
        else:
            note_reranker_st = time.time()
            note_reranker = NoteReranker(
                use_query_modality=use_query_modality,
                use_note_modality=use_note_modality,
                compute_type=["text2text"],
            )
            for sample_id in range(num_samples):
                input_dict = note_reranker.act(
                    input_dict=input_dict,
                    input_key=f"search_notes_{sample_id}",
                    output_key=f"scored_search_results_{sample_id}"
                )
            note_reranker_et = time.time()
            logging.info(f"rerank search results costs: {note_reranker_et-note_reranker_st}")
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"finish cache score notes: {cache_path}")
        
        # aggregate scores
        cache_path = os.path.join(cache_dir, f"{job_name}_aggregated_results.json")
        aggregate_st = time.time()
        for sample_id in range(num_samples):
            input_dict = aggregate_subquery_score(input_dict=input_dict, input_key=f"search_notes_{sample_id}")
        aggregate_et = time.time()
        logging.info(f"aggregate results costs: {aggregate_et-aggregate_st}")
        if require_cache_data:
            cache_data(input_dict=input_dict, cache_path=cache_path)
            logging.info(f"finish cache aggregated notes: {cache_path}")
        
        
        
 
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
        asyncio.run(run_batch_subquery())
    else:
        asyncio.run(run_subquery())