import argparse
import asyncio
import logging
import multiprocessing as mp
import os
import json
import time
import gin
import sys
sys.path.append('.')
sys.path.append('./api')
sys.path.append('./agent')

from typing import Dict
from datetime import datetime

from pipeline import (
    QueryMinerPipeline,
    OutlineGeneratorPipeline,
    OutlineJudgePipeline,
    ReportGeneratorPipeline,
    ReportRenderPipeline,
)
from run_workflow import set_logger, load_data_from_cache, cache_data


def find_last_completed_turn(cache_dir: str, job_name: str, max_turns: int) -> int:
    last_completed_turn = -1
    for turn_id in range(max_turns):
        turn_cache_path = os.path.join(cache_dir, f"{job_name}_outline_generator_turn_{turn_id}.json")
        if os.path.exists(turn_cache_path):
            last_completed_turn = turn_id
        else:
            break
    return last_completed_turn


def load_turn_data(cache_dir: str, job_name: str, turn_id: int) -> Dict:
    turn_cache_path = os.path.join(cache_dir, f"{job_name}_outline_generator_turn_{turn_id}.json")
    return load_data_from_cache(cache_path=turn_cache_path)


def save_turn_data(input_dict: Dict, cache_dir: str, job_name: str, turn_id: int):
    turn_cache_path = os.path.join(cache_dir, f"{job_name}_outline_generator_turn_{turn_id}.json")
    cache_data(input_dict=input_dict, cache_path=turn_cache_path)
    return turn_cache_path


@gin.configurable()
async def run_pipeline(
    job_name: str,
    input_path: str,
    cache_dir: str,
    download_dir: str,
    log_dir: str,
    use_zh: bool = False,
    use_input_query: bool = True,
    include_summary: bool = False,
    use_hierarchical_writer: bool = False,
    use_evidence_as_key: bool = False,
    use_polish_before_render: bool = False,
    use_debug: bool = False,
    render_with_image: bool = True,
    max_outline_generator_turns: int = 10,
    min_outline_generator_turns: int = 2,
    outline_judge_threshold: int = 8,
    force_restart_outline_generator: bool = False,
    disable_images: bool = False,
    disable_multi_images: bool = False,
    disable_comment: bool = True,
    disable_video: bool = True,
    enable_query_miner: bool = True,
    enable_outline_generator: bool = True,
    enable_report_generator: bool = True,
    enable_note_selector: bool = True,
    enable_image_selector: bool = True,
    enable_html: bool = True,
    enable_slides: bool = False,
    enable_xhs: bool = False,
    enable_video: bool = False,
    enable_polish: bool = False,
    require_cache_data: bool = True,
    output_path: str = None,
    input_dict: Dict = None,
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
    logging.info(f"processing query: {input_query}")

    if enable_polish:
        save_key = "polished_rendered_report"
    else:
        save_key = "rendered_report"

    # ---- Step 1: Query Mining ----
    if enable_query_miner:
        cache_path = os.path.join(cache_dir, f"{job_name}_query_miner.json")
        if os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            logging.info(f"load query miner from cache: {cache_path}")
        else:
            qm = QueryMinerPipeline(
                use_zh=use_zh,
                use_input_query=use_input_query,
                include_summary=include_summary,
                disable_video=disable_video,
                disable_images=disable_images,
                disable_multi_images=disable_multi_images,
                disable_comment=disable_comment,
            )
            qm_st = time.time()
            input_dict = await qm.act(input_dict=input_dict)
            logging.info(f"query miner + intent planner costs: {time.time() - qm_st:.1f}s")
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)

    # ---- Step 2: Outline Loop (judge <-> generator) ----
    num_turns = 1
    num_chunks = 1

    if enable_outline_generator:
        final_cache_path = os.path.join(cache_dir, f"{job_name}_outline_generator_final.json")

        if os.path.exists(final_cache_path):
            input_dict = load_data_from_cache(cache_path=final_cache_path)
            num_turns = input_dict["num_turns"]
            num_chunks = input_dict["num_chunks"]
            logging.info(f"load outline generator final from cache: {final_cache_path}")
        else:
            last_completed_turn = -1
            if not force_restart_outline_generator:
                last_completed_turn = find_last_completed_turn(
                    cache_dir, job_name, max_outline_generator_turns,
                )
                logging.info(f"last completed turn: {last_completed_turn}")

            if last_completed_turn >= 0:
                input_dict = load_turn_data(cache_dir, job_name, last_completed_turn)
                logging.info(f"loaded data from turn {last_completed_turn}")

            oj = OutlineJudgePipeline(
                use_zh=use_zh,
                use_evidence_as_key=use_evidence_as_key,
                use_hierarchical_writer=use_hierarchical_writer,
                outline_judge_threshold=outline_judge_threshold,
                max_outline_generator_turns=max_outline_generator_turns,
                min_outline_generator_turns=min_outline_generator_turns,
            )
            og = OutlineGeneratorPipeline(
                use_zh=use_zh,
                use_evidence_as_key=use_evidence_as_key,
            )

            start_turn = last_completed_turn + 1
            turn_id = start_turn
            logging.info(f"starting outline generator from turn {start_turn}")

            while turn_id < max_outline_generator_turns:
                logging.info(f"starting turn {turn_id}")

                turn_cache_path = os.path.join(
                    cache_dir, f"{job_name}_outline_generator_turn_{turn_id}.json",
                )
                if os.path.exists(turn_cache_path) and not force_restart_outline_generator:
                    input_dict = load_data_from_cache(cache_path=turn_cache_path)
                    logging.info(f"turn {turn_id} loaded from cache")
                    turn_id += 1
                    continue

                if f"is_finish_turn_{turn_id-1}" in input_dict and input_dict[f"is_finish_turn_{turn_id-1}"]:
                    break

                judge_st = time.time()
                is_finish, input_dict = await oj.act(
                    input_dict=input_dict, turn_id=turn_id,
                )
                logging.info(f"outline judge+memory turn {turn_id} costs: {time.time() - judge_st:.1f}s")

                if require_cache_data:
                    save_turn_data(input_dict, cache_dir, job_name, turn_id)

                if is_finish:
                    logging.info(f"outline generator finished at turn {turn_id}")
                    break

                gen_st = time.time()
                input_dict = await og.act(input_dict=input_dict, turn_id=turn_id)
                logging.info(f"outline generator turn {turn_id} costs: {time.time() - gen_st:.1f}s")

                if require_cache_data:
                    save_turn_data(input_dict, cache_dir, job_name, turn_id)

                turn_id += 1

            num_turns = turn_id + 1

            # Select best outline by judge score
            if turn_id >= 0:
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
                    outline_turn = max(0, best_turn - 1) if best_turn > 0 else 0
                    input_dict["outline"] = input_dict[f"outline_turn_{outline_turn}"]
                    input_dict["search_result"] = input_dict[f"search_result_turn_{outline_turn}"]
                    input_dict["search_result_map"] = input_dict[f"search_result_map_turn_{outline_turn}"]
                    input_dict["judge"] = input_dict[f"judge_turn_{best_turn}"]
                    input_dict["blueprint"] = input_dict[f"blueprint_turn_{outline_turn}"]
                    logging.info(
                        f"selected outline from turn {outline_turn} "
                        f"with judge score {best_score} from judge turn {best_turn}"
                    )

            if use_hierarchical_writer and "outline" in input_dict:
                chunks = og.divide_outline_into_chunks(input_dict["outline"])
                num_chunks = len(chunks)
                for chunk_id, chunk in enumerate(chunks):
                    input_dict[f"outline_chunk_{chunk_id}"] = chunk
                input_dict["num_chunks"] = num_chunks
            else:
                input_dict["num_chunks"] = num_chunks = 1

            input_dict["num_turns"] = num_turns

            if "outline" in input_dict:
                input_dict = oj.get_docs_with_reference(
                    input_dict=input_dict, num_chunks=num_chunks,
                )
            logging.info("finish get references for outline")

            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=final_cache_path)

    # ---- Step 3: Report Generation ----
    if enable_report_generator:
        cache_path = os.path.join(cache_dir, f"{job_name}_report_writer.json")
        if os.path.exists(cache_path):
            input_dict = load_data_from_cache(cache_path=cache_path)
            num_turns = input_dict.get("num_turns", 1)
            num_chunks = input_dict.get("num_chunks", 1)
            logging.info(f"load report writer from cache: {cache_path}")
        else:
            rg = ReportGeneratorPipeline(
                use_zh=use_zh,
                use_hierarchical_writer=use_hierarchical_writer,
                use_evidence_as_key=use_evidence_as_key,
            )
            rg_st = time.time()
            input_dict = await rg.act(input_dict=input_dict, num_chunks=num_chunks)
            logging.info(f"report generator costs: {time.time() - rg_st:.1f}s")
            if require_cache_data:
                cache_data(input_dict=input_dict, cache_path=cache_path)

    # ---- Step 4: Report Rendering ----
    rr = ReportRenderPipeline(
        use_zh=use_zh,
        render_with_image=render_with_image,
        enable_note_selector=enable_note_selector,
        enable_image_selector=enable_image_selector,
        enable_html=enable_html,
        enable_slides=enable_slides,
        enable_xhs=enable_xhs,
        enable_video=enable_video,
        enable_polish=enable_polish,
        use_polish_before_render=use_polish_before_render,
        download_dir=download_dir,
    )
    rr_st = time.time()
    input_dict = await rr.act(
        input_dict=input_dict,
        output_dir=cache_dir,
        cache_dir=cache_dir,
        job_name=job_name,
    )
    logging.info(f"report render costs: {time.time() - rr_st:.1f}s")

    # ---- Save Output ----
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + f"_{job_name}_output.md"

    if save_key in input_dict:
        logging.info(f"saving report to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(input_dict[save_key])

    output_ref_path = os.path.splitext(output_path)[0] + "_reference.json"
    logging.info(f"saving references to {output_ref_path}")
    with open(output_ref_path, "w", encoding="utf-8") as f:
        json.dump(input_dict.get("search_result_map", {}), f, ensure_ascii=False, indent=2)

    output_json_path = os.path.splitext(output_path)[0] + ".json"
    logging.info(f"saving full output to {output_json_path}")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(input_dict, f, ensure_ascii=False, indent=2)

    logging.info("pipeline complete")
    return input_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gin-config-file",
        type=str,
        required=False,
        default="./config/test_demo.gin",
    )
    args = parser.parse_args()

    gin.parse_config_file(args.gin_config_file)
    mp.set_start_method("spawn", force=True)
    asyncio.run(run_pipeline())
