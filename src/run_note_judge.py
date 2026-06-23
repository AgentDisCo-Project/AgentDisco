
import gin
import asyncio
import os
import logging
import time
import argparse
import multiprocessing as mp
import json

from typing import List
from run_workflow import set_logger, load_data_from_cache, cache_data, merge_search_results
from api.NoteJudgeService import NoteJudge


# def convert_data_org(
#     input_list: List[dict],
#     input_key: str,
#     output_path: str = None,
#     use_samples: bool = False,
#     num_samples: int = 3,
#     require_cache_data: bool = False,
# ):
#     note_dict = dict()
#     for input_dict in input_list:
#         if use_samples:
#             for sample_id in range(num_samples):
#                 query = input_dict["query_text"]
#                 subquery = input_dict[f"note_subquery_{sample_id}"]
#                 tuple_key = (query, subquery)
#                 if tuple_key not in note_dict:
#                     note_dict[tuple_key] = []
#
#                 notes = input_dict[input_key]
#                 note_dict[tuple_key].extend(notes)
#         else:
#             query = input_dict["query_text"]
#             subquery = input_dict["note_subquery"]
#             tuple_key = (query, subquery)
#             if tuple_key not in note_dict:
#                 note_dict[tuple_key] = []
#
#             notes = input_dict[input_key]
#             note_dict[tuple_key].extend(notes)
#
#     if require_cache_data and output_path is not None:
#         with open(output_path, 'w', encoding='utf-8') as f:
#             for (query, subquery), dict_list in note_dict.items():
#                 line = f"{repr((query, subquery))}: {json.dumps(dict_list, ensure_ascii=False)}"
#                 f.write(line + '\n')
#
#     return note_dict


# @gin.configurable()
# async def run_batch_note_judge(
#     job_name: str,
#     input_path: str,
#     log_dir: str,
#     cache_dir: str,
#     output_path: str = None,
#     max_concurrent: int = 10,
#     require_cache_data: bool = False,
#     load_from_excel: bool = False,
# ):
#     input_list = load_data_from_cache(cache_path=input_path, return_list=True, load_from_excel=load_from_excel)
#     if load_from_excel:
#         for idx, input_dict in enumerate(input_list):
#             idx = input_dict.get("id", idx)
#             search_from = input_dict.get("search_from", "unknown")
#             title = input_dict["title"]
#             content = input_dict["content"]
#             url = input_dict.get("url", "unknown")
#             date = input_dict.get("date", {})
#             note_type = input_dict.get("note_type", "unknown"),
#             video = input_dict.get("video", {})
#             images = input_dict.get("images", [])
#             like_count = input_dict.get("like_count", -1)
#             collect_count = input_dict.get("collect_count", -1)
#             view_count = input_dict.get("view_count", -1)
#             comments = input_dict.get("comments", [])
#
#             doc = {
#                 "id": idx,
#                 "search_from": search_from,
#                 "content": content,
#                 "title": title,
#                 "url": url,
#                 "date": date,
#                 "note_type": note_type,
#                 "video": video,
#                 "images": images,
#                 "like_count": info.get("likeCount", -1),
#                 "collect_count": info.get("collectCount", -1),
#                 "view_count": info.get("viewCount", -1),
#                 "comments": comments,
#             }
#
#
#     tasks = [run_note_judge(input_dict=input_dict) for input_dict in input_list]
#     await asyncio.gather(*tasks)



@gin.configurable()
async def run_note_judge(
    job_name: str,
    input_path: str,
    input_key: str,
    output_key: str,
    log_dir: str,
    cache_dir: str,
    use_query_modality: str,
    use_note_modality: str,
    skip_note_judge: bool,
    disable_note_judge: bool,
    input_dict: dict = None,
    require_cache_data: bool = True,
    image_key_type: str = "path",
    export_to_excel: bool = False,
):
    set_logger(log_dir=log_dir, job_name=job_name)
    
    input_dict = load_data_from_cache(cache_path=input_path) if input_dict is None else input_dict
    logging.info(f"successfully load input dict from: {input_path}")
    
    # note judge
    if not disable_note_judge:
        cache_path = os.path.join(cache_dir, f"{job_name}_judged_search_results.json")
        if skip_note_judge and os.path.exists(cache_path):
            input_dict = await load_data_from_cache(cache_path=cache_path)
            logging.info(f"load judged search results from cache: {cache_path}")
        else:
            note_judge_st = time.time()
            note_judge = NoteJudge(
                use_query_modality=use_query_modality,
                use_note_modality=use_note_modality,
                image_key_type=image_key_type,
            )
            input_dict = await note_judge.act(
                input_dict=input_dict,
                input_key=input_key,
                output_key=output_key
            )
            note_judge_et = time.time()
            logging.info(f"judge search results costs: {note_judge_et-note_judge_st}")
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)
                logging.info(f"finish cache judged search results: {cache_path}")




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
        asyncio.run(run_batch_note_judge())
    else:
        asyncio.run(run_note_judge())