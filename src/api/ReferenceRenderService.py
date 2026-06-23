import gin
import re
import logging

from collections import OrderedDict
from typing import Dict



@gin.configurable()
class ReferenceRender:
    def __init__(
        self,
        file_type: str = "md",
        use_evidence_as_key: bool = True,
        use_restrict_match: bool = False,
    ):
        self.file_type = file_type
        self.use_evidence_as_key = use_evidence_as_key
        self.use_restrict_match = use_restrict_match
        
        
    def render_report(
        self,
        report: str,
        reference_map: Dict,
    ):
        reference_ids = OrderedDict()
        cnt_references = 1
        
        def render_reference(text):
            nonlocal cnt_references
            indices = text.group(1)
            
            # 提取所有数字
            # indices = re.findall(r'\d+', indices)
            indices = re.findall(r'[a-zA-Z0-9_]+', indices)
            
            links = []
            for idx in indices:
                if self.use_evidence_as_key:
                    if self.use_restrict_match and not re.match(r'turn_\d+_\d+_\d+', idx):
                        continue
                else:
                    if self.use_restrict_match and not re.match(r'turn_\d+_\d+', idx):
                        continue
                    
                if idx not in reference_ids:
                    reference_ids[idx] = cnt_references
                    cnt_references += 1
                
                ref_num = reference_ids[idx]
                if idx not in reference_map:
                    logging.info(f"missing doc idx {idx}")
                    continue
                ref_url = reference_map[idx]["url"]
                links.append(f"[<sup>[{ref_num}]</sup>]({ref_url})")
            return "".join(links)
        
        pattern = r'<cite>(.*?)</cite>'
        processed_md = re.sub(pattern, render_reference, report)
        return processed_md
    
    
    def act(
        self,
        input_dict: Dict,
        input_key: str = "report",
        output_key: str = "rendered_report",
        allow_compatible: bool = True,
    ):
        report = input_dict[f"{input_key}"]
        reference_map = dict()
        if self.use_evidence_as_key:
            for ref_id, ref in input_dict.get(f"search_result_map").items():
                evidences = ref["evidence_map"]
                for evidence_id, evidence in evidences.items():
                    reference_map[evidence_id] = ref
                # compatible
                reference_map[ref_id] = ref
        else:
            # for ref in input_dict.get(f"search_result_turn"):
            #     reference_map[ref["id"]] = ref
            for ref_id, ref in input_dict.get(f"search_result_map").items():
                reference_map[ref_id] = ref
        input_dict["reference_map"] = reference_map

        reference_map = input_dict["reference_map"]
        report = self.render_report(
            report=report,
            reference_map=reference_map,
        )
        input_dict[f"{output_key}"] = report
        return input_dict
