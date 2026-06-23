import pandas as pd
import argparse

from api.utils import convert_jsonl_to_list


def convert_list_to_excel_results(data: list, out_name: str):
    out_dir = f"/mnt/ali-sh-1/usr/tusen/search-agent-dev/outs/{out_name}.xlsx"
    with pd.ExcelWriter(out_dir) as writer:
        for qa in data:
            query = qa['query']
            results = qa.get('search_results', [])
            if results:  # 只导出有结果的
                # 筛选需要的字段，如果可能有字段丢失建议用 pd.DataFrame(results)
                df = pd.DataFrame(results)
                df.to_excel(writer, sheet_name=query, index=False)


def convert_list_to_excel_response(data: list, out_name: str):
    out_dir = f"/mnt/ali-sh-1/usr/tusen/search-agent-dev/outs/{out_name}.xlsx"
    df = pd.DataFrame(data)
    df.to_excel(out_dir, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=False,
        default="inputs",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        required=False,
        default="out",
    )
    parser.add_argument(
        "--process-type",
        type=str,
        required=False,
        default="results"
    )
    args = parser.parse_args()

    args.input_dir = "/mnt/ali-sh-1/usr/tusen/search-agent-dev/cache/sigmaai_badcase_full_search_72b_para_local_query_responses.json"
    args.out_name = "0709_sigmaai_badcase_full_search_72b_para_local_query_results"
    args.process_type = "results"

    data_list = convert_jsonl_to_list(args.input_dir)
    if args.process_type == "results":
        convert_list_to_excel_results(data_list, args.out_name)
    else:
        convert_list_to_excel_response(data_list, args.out_name)

    
