import os
import gin
import asyncio
import jinja2
import json
import logging
import sys
sys.path.append('.')

from typing import List,Dict
from dotenv import load_dotenv
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.url_operator import compress_and_convert_base64, compress_url
from api.utils.key_operator import ApiKeyCycler
from api.utils.string_operator import json_fix

load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


@gin.configurable()
class EvidenceMerger:
    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        retry_delay: int = 3,
        use_zh: bool = False,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file: str = "EvidenceMerger_EN.jinja2",
        system_template_zh_file: str = "EvidenceMerger_ZH.jinja2",
        max_evidence_len: int = 512,
    ):
        self.model = CustomizeChatGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
        )
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.jinja_file = system_template_zh_file if use_zh else system_template_en_file
        self.use_zh = use_zh
        self.max_evidence_len = max_evidence_len
    
    
    def get_system(
        self
    ):
        template_vars = {}
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    def get_system_prompt(
        self
    ):
        template_vars = {
            "max_evidence_len": self.max_evidence_len,
        }
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
        
    def check_func(
        self,
        response: str,
    ):
        return self.parser_response(response)
    
    
    def parser_response(
        self,
        response: str,
    ):
        response = json_fix(response)
        response = json.loads(response)
        
        if not isinstance(response, List):
            raise ValueError()
        for r in response:
            if "evidence" not in r or "references" not in r:
                raise ValueError()
            if len(r["evidence"]) > self.max_evidence_len or len(r["references"]) < 1:
                raise ValueError()
        return response
        
    
    async def post_request(
        self,
        evidences: List[Dict],
        evidence_map: Dict,
        turn_id: int,
    ):
        if "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            system_prompt = [
                {
                    "type": "text",
                    "text": self.get_system_prompt()
                }
            ]
            user_prompt = []
            
            for evidence in evidences:
                idx, content, doc_id = evidence["id"], evidence["content"], evidence["doc_id"]
                if self.use_zh:
                    _user_prompt = f"""
## 外源搜索观点{idx}的内容如下
观点ID：{idx}
内容：{content}
"""
                else:
                    _user_prompt = f"""
## External Search Document Claim {idx}
Claim ID: {idx}
Content: {content}
"""
                
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt,
                    }
                )
            
            cycler = ApiKeyCycler(api_key_list=list(DIRECTLLM_API_KEY_USER.values()))
            response = await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                cycler=cycler,
                return_cot=False,
            )
        
        elif "gemini" in self.model.model_name:
            system_prompt = [{"text": self.get_system_prompt()}]
            user_prompt = []
            
            for evidence in evidences:
                idx, content, doc_id = evidence["id"], evidence["content"], evidence["doc_id"]
                if self.use_zh:
                    _user_prompt = f"""
## 外源搜索观点{idx}的内容如下
观点ID：{idx}
内容：{content}
"""
                else:
                    _user_prompt = f"""
## External Search Document Claim {idx}
Claim ID: {idx}
Content: {content}
"""
                
                user_prompt.append({"text": _user_prompt})
            
            response = await self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                gemini_api_key=GEMINI_API_KEY,
                directllm_api_key=DIRECTLLM_API_KEY_USER["tusen"],
                return_cot=False,
            )        
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        evidences = []
        if len(response) > 0:
            for idx, r in enumerate(response):
                evidence = dict()
                evidence["content"] = r["evidence"]
                evidence["references"] = r["references"]
                evidence["id"] = f"turn_{turn_id}_{idx}"
                evidence["doc_ids"] = []
                evidence["doc_urls"] = []
                for ref in r["references"]:
                    if ref in evidence_map.keys():
                        if evidence_map[ref]["doc_id"] not in evidence["doc_ids"]:
                            evidence["doc_ids"].append(evidence_map[ref]["doc_id"])
                            evidence["doc_urls"].append(evidence_map[ref]["doc_url"])
                evidences.append(evidence)
        return evidences
    
    
    async def act(
        self,
        input_dict: Dict,
        turn_id: int,
    ):
        evidences = []
        evidence_map = dict()
        documents = input_dict[f"search_result_turn_{turn_id}"]
        for d in documents:
            for e in d["evidences"]:
                evidence = {
                    "content": e,
                    "doc_id": d["id"],
                    "doc_url": d["url"],
                }
                evidences.append(evidence)
        for idx, evidence in enumerate(evidences):
            evidence["id"] = f"turn_{turn_id}_{idx}"
            evidence_map[evidence["id"]] = evidence

        evidences = await self.post_request(
            evidences=evidences,
            evidence_map=evidence_map,
            turn_id=turn_id,
        )
        input_dict[f"search_evidence_turn_{turn_id}"] = evidences
        return input_dict
    
    
    
if __name__ == "__main__":
    async def main():
        service = EvidenceMerger(
            model_name="gemini-2.5-pro",
            use_zh=True,
        )
        input_dict = dict()
        input_dict["query_text"] = input_dict["query"] = '请整理消费娱乐机器人、格斗机器人的行业政策，包括国家和地方政策，重点关注与工匠社科技公司高度相关的规划、意见、措施、细则等。最后用一张表格的形式呈现。'
        input_dict["outline_turn_0"] = '# 消费娱乐与格斗机器人行业政策分析报告\n\n## 1. 行业背景与政策环境概述\n- **1.1 消费娱乐与格斗机器人行业发展现状**：简述该细分领域的市场规模、技术特点及主要参与者（如工匠社）。\n- **1.2 政策对行业发展的重要性**：分析政策在引导技术创新、规范市场秩序、推动产业化应用方面扮演的关键角色。\n\n## 2. 国家层面相关政策梳理\n- **2.1 机器人产业顶层设计**：分析国家级机器人产业发展规划中，与消费服务、娱乐体验相关的指导方针。\n- **2.2 人工智能发展战略**：解读人工智能国家战略中，关于AI技术在文娱、教育领域应用的鼓励措施。\n- **2.3 科技、教育、体育部门专项政策**：梳理与机器人竞赛、青少年科技教育（STEAM）、全民科学素质提升相关的支持文件。\n\n## 3. 地方层面相关政策解读\n- **3.1 重点省市（如广东、北京、上海）支持措施**：分析主要经济和科技强省市发布的机器人或AI产业行动计划，寻找对消费级机器人企业的扶持条款。\n- **3.2 地方科创示范区（如深圳）专项细则**：研究工匠社所在地或潜在发展区域的政府补贴、人才引进、研发资助等具体优惠政策。\n\n## 4. 聚焦工匠社：高度相关政策深度剖析\n- **4.1 机器人赛事活动相关政策**：重点分析政府对举办或参与格斗机器人赛事的审批、资助和推广政策。\n- **4.2 “AI+教育”融合政策**：剖析将机器人作为教具引入中小学课堂或课后服务的相关指导意见和采购标准。\n- **4.3 企业扶持政策**：梳理适用于工匠社的高新技术企业认定、税收优惠、知识产权保护、融资支持等普适性及专项政策。\n\n## 5. 政策汇总与影响分析（表格呈现）\n- **5.1 核心政策汇总表**\n    - 表格将包含以下列：政策名称、发布机构、发布时间、政策层级（国家/地方）、核心内容摘要、与工匠社业务关联点。\n- **5.2 政策环境综合评价与未来趋势研判**：基于汇总信息，评估当前政策环境的利好与挑战，并预测未来政策走向。\n\n'
        input_dict["is_finish_turn_0"] = False
        input_dict["search_query_turn_0"] = ['机器人产业 发展规划 政策', '娱乐机器人 行业标准', '格斗机器人 赛事 政策支持', '工匠社 政府补贴', '青少年 科技教育 机器人 政策', '请整理消费娱乐机器人、格斗机器人的行业政策，包括国家和地方政策，重点关注与工匠社科技公司高度相关的规划、意见、措施、细则等。最后用一张表格的形式呈现。']
        input_dict["search_result_turn_0"] = []
        input_dict["search_result_turn_0"].append(
            {'id': 'turn_0_6', 'search_from': 'search_note', 'content': '·\n🏎️我们正站在一个新时代的起点。人形机器人（Humanoid Robot），这个曾经只存在于科幻中的产物，正在加速走进现实。\n\t\n🗂️中美等大国都将人工智能和机器人视为战略制高点。中国出台《“十四五”机器人产业发展规划》等政策，提供研发支持与应用示范。\n\t\n🛁巨头林立，创业公司扎堆，行业存在明显泡沫：传统机器人企业、新兴人形机器人企业、车企、消费电子企业、互联网大厂纷纷布局人形机器人。\n\t\n📈若想在这个领域取得长远发展，\xa0企业需要找到清晰的、可规模化的应用场景，并围绕场景打造产品，而非追求技术的炫技；需要有技术整合与工程化的能力、供应链管理与成本控制能力、持续融资与商业化落地能力。\n\t\n#行业分析[话题]# #产业发展趋势[话题]# #商业分析[话题]# #人工智能发展[话题]# #具身智能[话题]# #人形机器人[话题]# #特斯拉[话题]# #小米[话题]# #人工智能[话题]#', 'title': '一天吃透一条产业链：NO.12 人形机器人', 'url': 'https://www.xiaohongshu.com/explore/68d0ce55000000001201489d', 'date': '2025-09-22 12:19:33', 'note_type': 'images', 'video': {'noteId': '68d0ce55000000001201489d', 'url': ''}, 'images': [], 'like_count': '282', 'collect_count': 368, 'view_count': '5072', 'comments': [], 'confidence': -1, 'detail': '', 'summary': '人形机器人正从科幻加速走进现实，中美等大国已将其视为战略制高点并出台政策支持。当前行业巨头与创业公司纷纷涌入，存在明显泡沫。企业若想长远发展，需找到可规模化的应用场景，并具备技术整合、成本控制及商业化落地等多方面能力。', 'evidences': ['我们正站在一个新时代的起点。人形机器人（Humanoid Robot），这个曾经只存在于科幻中的产物，正在加速走进现实。', '中美等大国都将人工智能和机器人视为战略制高点。中国出台《“十四五”机器人产业发展规划》等政策，提供研发支持与应用示范。', '巨头林立，创业公司扎堆，行业存在明显泡沫：传统机器人企业、新兴人形机器人企业、车企、消费电子企业、互联网大厂纷纷布局人形机器人。', '若想在这个领域取得长远发展， 企业需要找到清晰的、可规模化的应用场景，并围绕场景打造产品，而非追求技术的炫技；需要有技术整合与工程化的能力、供应链管理与成本控制能力、持续融资与商业化落地能力。']}
        )
        input_dict["search_result_turn_0"].append(
            {'id': 'turn_0_8', 'search_from': 'search_note', 'content': '2025年，具身智能被首次写入中国政府工作报告，标志着这一领域正式成为国家战略重点。具身智能的核心包括感知、决策和行动能力，能够通过与环境的交互，执行复杂任务，尤其在人形机器人等技术的推动下，正在加速发展。\n\t\n1.产业现状与挑战 🔧\n市场规模持续扩大：随着资本投入的增加，人形机器人在工业、服务和家庭等多个场景的应用逐步推进。然而，技术的泛化能力仍不足以应对复杂场景，且量产成本较高，商业化回报周期较长。\n政府与资本支持：政府持续出台配套政策，各地纷纷提出具身智能的发展规划，北京、上海、深圳等地已设立专项资金与发展计划，推动产业集聚与技术创新。\n\t\n2.应用场景洞察 🌍\n科研与工业场景：目前具身智能应用已在工业生产中实现了初步商业化，如机器人在自动化生产线、仓储物流中的应用。未来将进一步向多样化任务场景扩展。\n商业服务场景：具身智能机器人开始进入酒店、餐饮、零售等行业，提供智能导览、送餐、清洁等服务，提升效率并减少人力成本。\n家庭应用场景：智能养老机器人正快速进入市场，解决老龄化社会的挑战，提升家庭护理质量。\n\t\n3.未来发展路径 🔮\n具身智能的未来将在短期、中期和长期三个阶段逐步推进。短期内专注于科研和初步的工业应用，中期拓展至商业服务和家庭服务领域，长期则实现全面普及，成为生活和工作的重要助手。技术标准和安全评测的不断完善，将为行业的规模化应用奠定基础。\n\t\n结论 💡\n随着技术的成熟和市场的开拓，具身智能将在多个领域发挥重要作用。政府的政策支持、资本的推动以及技术的不断突破，为这一产业的发展提供了强大动力。未来几年，具身智能将迎来爆发式增长，成为产业升级和技术创新的核心引擎。\n#具身智能[话题]# #机器人[话题]# #智能化时代[话题]# #行业报告[话题]# #市场洞察[话题]# #投资趋势[话题]#', 'title': '中国具身智能产业发展规划与应用前景 🚀', 'url': 'https://www.xiaohongshu.com/explore/68d7535d000000001301996b', 'date': '2025-09-27 11:00:45', 'note_type': 'images', 'video': {'noteId': '68d7535d000000001301996b', 'url': ''}, 'images': [], 'like_count': '31', 'collect_count': 38, 'view_count': '688', 'comments': [], 'confidence': -1, 'detail': '', 'summary': '中国已将具身智能提升为国家战略重点，其产业在政府政策、专项资金及资本市场的推动下正加速发展。目前，具身智能已在工业生产、仓储物流等领域初步商业化，并逐步拓展至酒店、零售等商业服务和智能养老等家庭场景。尽管面临技术泛化能力不足、量产成本高等挑战，但未来规划通过短、中、长期三阶段，从工业应用扩展至全面普及，有望迎来爆发式增长，成为产业升级的核心引擎。', 'evidences': ['2025年，具身智能被首次写入中国政府工作报告，标志着这一领域正式成为国家战略重点。具身智能的核心包括感知、决策和行动能力，能够通过与环境的交互，执行复杂任务，尤其在人形机器人等技术的推动下，正在加速发展。', '市场规模持续扩大：随着资本投入的增加，人形机器人在工业、服务和家庭等多个场景的应用逐步推进。然而，技术的泛化能力仍不足以应对复杂场景，且量产成本较高，商业化回报周期较长。', '政府与资本支持：政府持续出台配套政策，各地纷纷提出具身智能的发展规划，北京、上海、深圳等地已设立专项资金与发展计划，推动产业集聚与技术创新。', '科研与工业场景：目前具身智能应用已在工业生产中实现了初步商业化，如机器人在自动化生产线、仓储物流中的应用。未来将进一步向多样化任务场景扩展。商业服务场景：具身智能机器人开始进入酒店、餐饮、零售等行业，提供智能导览、送餐、清洁等服务，提升效率并减少人力成本。家庭应用场景：智能养老机器人正快速进入市场，解决老龄化社会的挑战，提升家庭护理质量。', '具身智能的未来将在短期、中期和长期三个阶段逐步推进。短期内专注于科研和初步的工业应用，中期拓展至商业服务和家庭服务领域，长期则实现全面普及，成为生活和工作的重要助手。']}
        )
        
        results = await service.act(
            input_dict=input_dict,
            turn_id=0,
        )
        print(results)
    
    asyncio.run(main())

            








