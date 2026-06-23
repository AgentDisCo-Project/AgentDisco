import os
import gin
import jinja2
import asyncio
import re
import httpx
import logging
import json
import time
import sys
sys.path.append('.')

from tqdm import tqdm
from typing import Union, Optional, List, Dict
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.key_operator import ApiKeyCycler
from api.EvidenceMergeService import EvidenceMerger
from api.EvidenceGeneratorService import EvidenceGenerator
# from api.JudgeAndEvidenceGeneratorService import JudgeAndEvidenceGenerator
from api.JudgeAndEvidenceQAGeneratorService import JudgeAndEvidenceQAGenerator 
from api.SummaryQAGeneratorService import SummaryQAGenerator
from api.utils.string_operator import json_fix
from agent.BaseAgent import BasicAgent
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")


@gin.configurable()
class MemoryBankManager(BasicAgent):
    def __init__(
        self,
        name: Optional[str] = "memory_bank",
        description_en: Optional[str] = "A manager for memory bannk.",
        description_zh: Optional[str] = "记忆管理模块。",
        tool_bank: Optional[List[Union[str, Dict]]] = "",
        use_zh: bool = False,
        model_name: str = "",
        max_retries: int = 5,
        max_retries_jina: int = 1,
        retry_delay: int = 3,
        use_customize_url: bool = False,
        customize_url: str = "",
        max_summary_len: int = 128,
        max_evidence_len: int = 256,
        max_evidence_num: int = 10,
        use_api_key: bool = True,
        include_user_query: bool = False,
        include_outline: bool = False,
        include_search_query: bool = False,
        use_evidence_as_key: bool = False,
        use_hierarchical_writer: bool = False,
        max_concurrent: int = 50,
        store_prev_references: bool = True,
        combine_judge_and_evidence: bool = False,
    ):
        super().__init__(
            name=name,
            description_en=description_en,
            description_zh=description_zh,
            tool_bank=tool_bank,
            use_zh=use_zh
        )
        
        self.evidence_generator = EvidenceGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_zh=use_zh,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
            max_evidence_len=max_evidence_len,
            include_user_query=include_user_query,
            include_outline=include_outline,
            include_search_query=include_search_query,
            use_evidence_as_key=use_evidence_as_key,
            max_summary_len=max_summary_len,
            max_evidence_num=max_evidence_num,
            max_concurrent=max_concurrent,
        )

        self.use_evidence_as_key = use_evidence_as_key
        self.use_hierarchical_writer = use_hierarchical_writer
        self.evidence_merger = EvidenceMerger(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_zh=use_zh,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
            max_evidence_len=max_evidence_len,
        )
        
        # self.judge_and_evidence_generator = JudgeAndEvidenceQAGenerator(
        #     model_name=model_name,
        #     max_retries=max_retries,
        #     retry_delay=retry_delay,
        #     use_zh=use_zh,
        #     use_customize_url=use_customize_url,
        #     customize_url=customize_url,
        #     use_api_key=use_api_key,
        #     max_evidence_len=max_evidence_len,
        # )
        self.judge_and_evidence_generator = SummaryQAGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_zh=use_zh,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
        )
        
        self.combine_judge_and_evidence = combine_judge_and_evidence
        
        self.store_prev_references = store_prev_references

    
    async def act(
        self,
        input_dict: Dict,
        turn_id: int,
    ):
        if not self.combine_judge_and_evidence:
            # self.document_bank[input_dict["query_text"]] = []
            generate_summary_st = time.time()
            input_dict = await self.evidence_generator.act(input_dict=input_dict, turn_id=turn_id)
            generate_summary_et = time.time()
            logging.info(f"generate summary costs: {generate_summary_et-generate_summary_st}")
            
            if self.use_evidence_as_key:
                merge_evidence_st = time.time()
                input_dict = await self.evidence_merger.act(input_dict=input_dict, turn_id=turn_id)
                merge_evidence_et = time.time()
                logging.info(f"merge evidence costs: {merge_evidence_et-merge_evidence_st}")
                # self.add_evidences_with_query(input_dict=input_dict, turn_id=turn_id)
        else:
            generate_judge_and_evidence_st = time.time()
            input_dict = await self.judge_and_evidence_generator.act(input_dict=input_dict, turn_id=turn_id)
            generate_judge_and_evidence_et = time.time()
            logging.info(f"generate summary costs: {generate_judge_and_evidence_et-generate_judge_and_evidence_st}")
            
            document_map = dict()
            for doc in input_dict[f"search_result_turn_{turn_id}"]:
                doc_id = doc["id"]
                # evidence_map = dict()
                # for idx, evidence in enumerate(doc["evidences"]):
                #     evidence_id = f"{doc_id}_{idx}"
                #     evidence_map[evidence_id] = evidence
                # doc["evidence_map"] = evidence_map
                document_map[doc_id] = doc
            input_dict[f"search_result_map_turn_{turn_id}"] = document_map
            
            # self.add_docs_with_query(input_dict=input_dict, turn_id=turn_id)
        if turn_id > 0 and self.store_prev_references:
            input_dict = self.get_next_step_reference(input_dict=input_dict, turn_id=turn_id)
            # self.add_docs_with_query(input_dict=input_dict, turn_id=turn_id)
        
        return input_dict
    
    
    # def add_docs_with_query(
    #     self,
    #     input_dict: Dict,
    #     num_turns: int = None,
    #     turn_id: int = None,
    # ):
    #     if turn_id is not None:
    #         for doc in input_dict.get(f"search_result_turn_{turn_id}", []):
    #             self.add_doc(input_dict["query_text"], doc)
    #     else:
    #         for turn_id in range(num_turns):
    #             for doc in input_dict.get(f"search_result_turn_{turn_id}", []):
    #                 self.add_doc(input_dict["query_text"], doc)
    
    
    # def add_evidences_with_query(
    #     self,
    #     input_dict: Dict,
    #     num_turns: int = None,
    #     turn_id: int = None,
    # ):
    #     if turn_id is not None:
    #         for evidence in input_dict.get(f"search_evidence_turn_{turn_id}", []):
    #             self.add_evidence(input_dict["query_text"], evidence)
    #     else:
    #         for turn_id in range(num_turns):
    #             for evidence in input_dict.get(f"search_evidence_turn_{turn_id}", []):
    #                 self.add_evidence(input_dict["query_text"], evidence)
        
        
    def get_next_step_reference(
        self,
        input_dict: Dict,
        turn_id: int,
    ):
        input_dict = self.get_docs_with_reference(input_dict=input_dict, turn_id=turn_id, num_chunks=-1)
        input_dict[f"record_search_result_turn_{turn_id}"] = input_dict[f"search_result_turn_{turn_id}"]
        input_dict[f"search_result_turn_{turn_id}"] = input_dict["outline"]["references"] + input_dict[f"search_result_turn_{turn_id}"]
        logging.info(f"number of search results in use is {len(input_dict['outline']['references'])}")

        reference_urls = set()
        new_results = []
        for doc in input_dict[f"search_result_turn_{turn_id}"]:
            if ("knowledge" not in doc["search_from"]) and ("url" not in doc or doc["url"] in reference_urls):
                continue
            reference_urls.add(doc["url"])
            new_results.append(doc)
        input_dict[f"search_result_turn_{turn_id}"] = new_results

        reference_map, search_result_map = {}, {}
        for idx, doc in enumerate(input_dict[f"search_result_turn_{turn_id}"]):
            new_idx = f"turn_{turn_id}_{idx}"
            reference_map[doc["id"]] = new_idx
            doc["id"] = new_idx
            # evidence_map = {}
            # for evidence_id, (key, value) in enumerate(doc["evidence_map"].items()):
            #     new_key = f"{new_idx}" + "_" + f"{evidence_id}"
            #     evidence_map[new_key] = value
            # doc["evidence_map"] = evidence_map
            search_result_map[doc["id"]] = doc 
        input_dict[f"search_result_map_turn_{turn_id}"] = search_result_map

        logging.info(f"number of search results after replacement is {len(input_dict[f'search_result_turn_{turn_id}'])}")
        # replace reference map in outline
        input_dict[f"record_outline_turn_{turn_id-1}"] = input_dict[f"outline_turn_{turn_id-1}"]
        outline = input_dict[f"outline_turn_{turn_id-1}"]
        def replace_reference(text):
            indices = text.group(1)
            indices_list = re.findall(r'[a-zA-Z0-9_]+', indices)
            
            replaced_indices = []
            for idx in indices_list:
                # 如果是原始ID，通过reference_map转换为新ID
                if idx in reference_map:
                    new_idx = reference_map[idx]  # 转换为turn_id_idx格式
                    replaced_indices.append(new_idx)
                else:
                    # 如果不在映射中，保持原样
                    replaced_indices.append(idx)
            # 重新组合并保持原始格式
            return f"<cite>{','.join(replaced_indices)}</cite>"
        
        pattern = r'<cite>(.*?)</cite>'
        
        outline = re.sub(pattern, replace_reference, outline)
        input_dict[f"outline_turn_{turn_id-1}"] = outline
        return input_dict
    
    
    def get_docs_with_reference(
        self,
        input_dict: Dict,
        num_chunks: int = -1,
        turn_id: int = -1,
    ):
        if turn_id == -1:
            if self.use_hierarchical_writer:
                assert num_chunks > 0, f"Unsupported num_chunks < 0 if use_hierarchical writer!"
                for chunk_id in range(num_chunks):
                    chunk = dict()
                    chunk["content"] = input_dict[f"outline_chunk_{chunk_id}"]["content"]
                    reference_ids = self.parser_response_with_reference(chunk["content"])
                    references = []
                    for reference_id in reference_ids:
                        reference = input_dict[f"search_result_map"].get(f"{reference_id}")
                        references.append(reference)
                    chunk["references"] = references
                    input_dict[f"outline_chunk_{chunk_id}"] = chunk
            else:
                outline = dict()
                outline["content"] = input_dict[f"outline"]
                reference_ids = self.parser_response_with_reference(outline["content"])
                references = []
                for reference_id in reference_ids:
                    reference = input_dict[f"search_result_map"].get(f"{reference_id}")
                    references.append(reference)
                outline["references"] = references
                input_dict["outline"] = outline
        else:
            outline = dict()
            outline["content"] = input_dict[f"outline_turn_{turn_id-1}"]
            reference_ids = self.parser_response_with_reference(outline["content"])
            references = []
            for reference_id in reference_ids:
                reference = input_dict[f"search_result_map_turn_{turn_id-1}"].get(f"{reference_id}", {})
                if len(reference) > 0:
                    references.append(reference)
            outline["references"] = references
            input_dict["outline"] = outline
        return input_dict
    
    
    # def get_evidences_with_reference(
    #     self,
    #     input_dict: Dict,
    #     turn_id: int = -1,
    #     num_chunks: int = -1,
    # ):
    #     if self.use_hierarchical_writer:
    #         for chunk_id in range(num_chunks):
    #             chunk = dict()
    #             chunk["content"] = input_dict[f"outline_chunk_{chunk_id}"]["content"]
    #             reference_ids = self.parser_response_with_reference(chunk["content"])
    #             references = []
    #             for reference_id in reference_ids:
    #                 reference = self.get_evidence(
    #                     query_text=input_dict["query_text"],
    #                     evidence_id=reference_id,
    #                     return_evidence_list=False,
    #                 )
    #                 references.append(reference)
    #             chunk["references"] = references
    #             input_dict[f"outline_chunk_{chunk_id}"] = chunk
    #
    #     else:
    #         outline = dict()
    #         outline["content"] = input_dict[f"outline_turn_{turn_id-1}"]
    #         reference_ids = self.parser_response_with_reference(outline["content"])
    #         references = []
    #         for reference_id in reference_ids:
    #             reference = self.get_evidence(
    #                 query_text=input_dict["query_text"],
    #                 evidence_id=reference_id,
    #                 return_evidence_list=False,
    #             )
    #             references.append(reference)
    #         outline["references"] = references
    #         input_dict["outline"] = outline
    #     return input_dict
    
    
    @staticmethod
    def parser_response_with_reference(
        response: str
    ):
        if not response:
            return set()
        
        # 先提取cite标签内容
        cite_pattern = r'<cite>(.*?)</cite>'
        cite_contents = re.findall(cite_pattern, response)
        
        reference_ids = set()
        for content in cite_contents:
            # 使用字符模式匹配提取所有ID
            indices = re.findall(r'[a-zA-Z0-9_]+', content)
            for idx in indices:
                # 检查是否符合 turn_数字_数字_数字 的格式
                if re.match(r'turn_\d+_\d+', idx):
                    reference_ids.add(idx)
                    
        return list(reference_ids)
    
    
    # def add_doc(
    #     self,
    #     query_text: str,
    #     document: Dict,
    # ):
    #     if query_text not in self.document_bank:
    #         self.document_bank[query_text] = dict()
    #     doc_id = document["id"]
    #     self.document_bank[query_text][doc_id] = document # edit

    # def add_evidence(
    #     self,
    #     query_text: str,
    #     evidence: Dict,
    # ):
    #     if query_text not in self.evidence_bank:
    #         self.evidence_bank[query_text] = dict()
    #     evidence_id = evidence["id"]
    #     if evidence_id not in self.evidence_bank[query_text]:
    #         self.evidence_bank[query_text][evidence_id] = evidence
    
    # def get_doc(
    #     self,
    #     query_text: str,
    #     doc_id: str = "",
    #     return_doc_list: bool = False
    # ):
    #     if return_doc_list:
    #         if query_text not in self.document_bank:
    #             logging.info(f"missing query {query_text}")
    #             return {}
    #         else:
    #             return self.document_bank[query_text]
    #     else:
    #         if query_text not in self.document_bank:
    #             logging.info(f"missing query {query_text}")
    #             return {}
    #         else:
    #             if doc_id not in self.document_bank[query_text]:
    #                 logging.info(f"missing doc id {doc_id}")
    #                 return {}
    #             else:
    #                 return self.document_bank[query_text][doc_id]
    #
    # def get_evidence(
    #     self,
    #     query_text: str,
    #     evidence_id: str = "",
    #     return_evidence_list: bool = False,
    # ):
    #     if return_evidence_list:
    #         if query_text not in self.evidence_bank:
    #             logging.info(f"missing query {query_text}")
    #             return {}
    #         else:
    #             return self.evidence_bank[query_text]
    #     else:
    #         if query_text not in self.evidence_bank:
    #             logging.info(f"missing query {query_text}")
    #             return {}
    #         else:
    #             if evidence_id not in self.evidence_bank[query_text]:
    #                 logging.info(f"missing doc id {evidence_id}")
    #                 return {}
    #             else:
    #                 return self.evidence_bank[query_text][evidence_id]
    #
    # def edit_doc(
    #     self,
    #     query_text: str,
    #     doc_id: str,
    #     document: Dict,
    # ):
    #     self.document_bank[query_text][doc_id] = document
    #
    # def edit_evidence(
    #     self,
    #     query_text: str,
    #     evidence_id: str,
    #     evidence: Dict,
    # ):
    #     self.evidence_bank[query_text][evidence_id] = evidence
        
