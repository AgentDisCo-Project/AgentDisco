import gin
import jinja2
import re
import os
import json
import sys
sys.path.append('.')

from typing import Optional, Union, Dict, List
from agent.BaseAgent import BasicAgent
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.key_operator import ApiKeyCycler
from api.utils.string_operator import markdown_fix
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")


@gin.configurable()
class ReportWriter(BasicAgent):
    def __init__(
        self,
        name: Optional[str] = "report_writer",
        description_en: Optional[str] = "A hierarchical writer for report generation.",
        description_zh: Optional[str] = "分层报告生成器。",
        tool_bank: Optional[List[Union[str, Dict]]] = "",
        use_zh: bool = False,
        model_name: str = "",
        max_retries: int = 5,
        retry_delay: int = 3,
        max_concurrent: int = 50,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file: str = "ReportWriter_EN.jinja2",
        system_template_zh_file: str = "ReportWriter_ZH.jinja2",
        system_template_en_file_hierarchical: str = "HierarchicalWriter_EN.jinja2",
        system_template_zh_file_hierarchical: str = "HierarchicalWriter_ZH.jinja2",
        system_template_en_file_evidence: str = "ReportWriterEvidence_EN.jinja2",
        system_template_zh_file_evidence: str = "ReportWriterEvidence_ZH.jinja2",
        system_template_en_file_evidence_hierarchical: str = "HierarchicalWriterEvidence_EN.jinja2",
        system_template_zh_file_evidence_hierarchical: str = "HierarchicalWriterEvidence_ZH.jinja2",
        system_template_en_file_style: str = "ReportWriterStyle_EN.jinja2",
        system_template_zh_file_style: str = "ReportWriterStyle_ZH.jinja2",
        system_template_en_file_style_hierarchical: str = "HierarchicalWriterStyle_EN.jinja2",
        system_template_zh_file_style_hierarchical: str = "HierarchicalWriterStyle_ZH.jinja2",
        use_hierarchical_writer: bool = False,
        use_evidence_as_key: bool = False,
        use_mask_evidence: bool = False,
        use_response_style: bool = True,
    ):
        super().__init__(
            name=name,
            description_en=description_en,
            description_zh=description_zh,
            tool_bank=tool_bank,
            use_zh=use_zh
        )
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
        self.use_hierarchical_writer = use_hierarchical_writer
        self.use_evidence_as_key = use_evidence_as_key
        self.use_response_style = use_response_style
        
        if use_hierarchical_writer:
            if use_evidence_as_key:
                self.jinja_file = system_template_zh_file_evidence_hierarchical if self.use_zh else system_template_en_file_evidence_hierarchical
            else:
                if use_response_style:
                    self.jinja_file = system_template_zh_file_style_hierarchical if self.use_zh else system_template_en_file_style_hierarchical
                else:
                    self.jinja_file = system_template_zh_file_hierarchical if self.use_zh else system_template_en_file_hierarchical
        else:
            if use_evidence_as_key:
                self.jinja_file = system_template_zh_file_evidence if self.use_zh else system_template_en_file_evidence
            else:
                if use_response_style:
                    self.jinja_file = system_template_zh_file_style if self.use_zh else system_template_en_file_style
                else:
                    self.jinja_file = system_template_zh_file if self.use_zh else system_template_en_file
        
        self.max_concurrent = max_concurrent
        self.use_mask_evidence = use_mask_evidence
    
    
    def get_system_prompt(
        self,
    ):
        template_vars = {}
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
        
    
    @staticmethod
    def check_func(
        response: str,
    ):
        return response
    
    
    async def act(
        self,
        input_dict: dict,
        chunk_id: int = -1,
        num_turns: int = -1,
    ):
        query_text = input_dict["query_text"]
        if self.use_hierarchical_writer:
            outline = input_dict[f"outline_chunk_{chunk_id}"]
        else:
            outline = input_dict["outline"]
        content, references = outline["content"], outline["references"]
        blueprint = input_dict["blueprint"]
        
        if "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            system_prompt = self.get_system_prompt()
            user_prompt = ""
            
            if self.use_zh:
                _user_prompt = f"""
# 报告大纲
{content}
"""
            else:
                _user_prompt = f"""
# Report Outline
{content}
"""
                user_prompt += _user_prompt
            
            if self.use_zh:
                _user_prompt = f"""
# 大纲要点列表
{blueprint}
"""
            else:
                _user_prompt = f"""
# Report Outline Blueprints
{blueprint}
"""
            user_prompt += _user_prompt
                
            for _, reference in enumerate(references):
                if reference is not None and "id" in reference and "title" in reference and "content" in reference:
                    idx, title, evidence = reference["id"], reference["title"], reference.get("evidence", reference["content"])
                    summary = reference.get("summary", reference["title"])
                    if self.use_zh:
                        _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
摘要：{summary}
内容：{evidence}
"""
                    
                    else:
                        _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Summary: {summary}
Content: {evidence}
"""
                    
                    user_prompt += _user_prompt
                

            if self.use_zh:
                _user_prompt = f"""
# 用户提问
{query_text}
"""
            else:
                _user_prompt = f"""
# User Question
{query_text}
"""
                    
            user_prompt += _user_prompt

            if self.use_response_style:
                response_style = input_dict['response_style']
                if self.use_zh:
                    _user_prompt = f"""
# 回复风格
{response_style}
"""
                else:
                    _user_prompt = f"""
# Response Style
{response_style}
"""
                user_prompt += _user_prompt
            
            if self.use_hierarchical_writer:
                previous_chunk = ""
                for _chunk_id in range(chunk_id):
                    previous_chunk += input_dict[f"writer_chunk_{_chunk_id}"]
                    previous_chunk += "\n\n"
                    
                if self.use_zh:
                    _user_prompt = f"""
# 前章已经写的内容
{previous_chunk}
"""
                else:
                    _user_prompt = f"""
# Previous Written Content
{previous_chunk}
"""
            
                user_prompt += _user_prompt
                
            response = await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                return_cot=False,
            )
            
            
        elif "gemini" in self.model.model_name:
            system_prompt = [{"text": self.get_system_prompt()}]
            user_prompt = []
            
            if self.use_zh:
                _user_prompt = f"""
# 报告大纲
{content}
"""
            else:
                _user_prompt = f"""
# Report Outline
{content}
"""
            user_prompt.append({"text": _user_prompt})
            
            if self.use_zh:
                _user_prompt = f"""
# 大纲要点列表
{blueprint}
"""
            else:
                _user_prompt = f"""
# Report Outline Blueprints
{blueprint}
"""
            user_prompt.append({"text": _user_prompt})
            
            for _, reference in enumerate(references):
                if reference == {}:
                    continue
                idx, title, evidence = reference["id"], reference["title"], reference.get("evidence", reference["content"])
                if self.use_zh:
                    _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
内容：{evidence}
"""
                
                else:
                    _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Content: {evidence}
"""
                
                user_prompt.append({"text": _user_prompt})
                
            if self.use_zh:
                _user_prompt = f"""
# 用户提问
{query_text}
"""
            else:
                _user_prompt = f"""
# User Question
{query_text}
"""
            
            user_prompt.append({"text": _user_prompt})

            if self.use_response_style:
                response_style = input_dict['response_style']
                if self.use_zh:
                    _user_prompt = f"""
# 回复风格
{response_style}
"""
                else:
                    _user_prompt = f"""
# Response Style
{response_style}
"""
                user_prompt.append({"text": _user_prompt})
            
            if self.use_hierarchical_writer:
                previous_chunk = ""
                for _chunk_id in range(chunk_id):
                    previous_chunk += input_dict[f"writer_chunk_{_chunk_id}"]
                    previous_chunk += "\n\n"
            
                if self.use_zh:
                    _user_prompt = f"""
# 前章已经写的内容
{previous_chunk}
"""
                else:
                    _user_prompt = f"""
# Previous Written Content
{previous_chunk}
"""
            
                user_prompt.append({"text": _user_prompt})
            
            response = await self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                return_cot=False,
            )
        
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        response = markdown_fix(response)
        if self.use_hierarchical_writer:
            input_dict[f"writer_chunk_{chunk_id}"] = response
        else:
            input_dict["report"] = response 
        return input_dict
    
    
    def merge_chunks_into_report(
        self,
        input_dict: Dict,
        num_chunks: int,
    ):
        report = ""
        for chunk_id in range(num_chunks):
            _report = input_dict[f"writer_chunk_{chunk_id}"]
            report += _report
            report += "\n\n"
        input_dict["report"] = report
        return input_dict
    



if __name__ == "__main__":
    async def main():
        service = ReportWriter(
            model_name="gemini-2.5-pro",
            use_zh=True,
            use_hierarchical_writer=False,
            use_evidence_as_key=True,
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
        input_dict["outline_turn_1"] = '# 关于消费娱乐与格斗机器人行业政策分析报告\n\n## 1. 报告核心摘要\n- **研究目的**：系统梳理与消费娱乐、格斗机器人相关的国家及地方性产业政策，并重点分析其对代表性企业“工匠社科技”的潜在影响。\n- **核心发现**：目前尚无专门针对“格斗机器人”的垂直政策，相关支持和规范主要体现在更宏观的机器人产业（如《“十四五”机器人产业发展规划》<cite>turn_0_0</cite>）、人工智能（如“人工智能+”行动<cite>turn_0_0</cite>）、以及“专精特新”企业扶持<cite>turn_0_1</cite>等政策框架下。\n- **核心结论**：企业需重点关注服务机器人、智能硬件、以及文体娱科教融合等领域的宏观政策导向，积极申请“专精特新”等资质，并利用地方性补贴<cite>turn_0_2</cite>，以获取发展机遇。\n\n## 2. 国家层面相关政策解读\n- **宏观战略规划**：《“十四五”机器人产业发展规划》提出到2025年成为全球机器人技术创新策源地等目标<cite>turn_0_0</cite>。近期“人工智能+”行动及将“具身智能”列为未来产业的规划，为消费娱乐机器人提供了顶层设计和战略牵引<cite>turn_0_0</cite>。\n- **细分领域指引**：关注《人形机器人创新发展指导意见》中对2025年初步建立创新体系的目标<cite>turn_0_0</cite>，以及工信部、科技部等部委发布的关于人工智能、数字文化等产业的指导意见，挖掘与娱乐机器人应用场景相关的支持性条款。\n- **企业资质与标准**：深入理解“专精特新”企业梯度培育体系<cite>turn_0_1</cite>。该资质是获取专项资金扶持、政府采购优先、融资授信等系列政策红利的关键<cite>turn_0_1</cite>，对工匠社这类深耕细分领域的中小企业至关重要。\n\n## 3. 地方层面相关政策分析（以深圳、广州等为例）\n- **省级产业规划**：分析广东省在机器人、人工智能、战略性新兴产业等方面的中长期发展规划和行动方案。\n- **市级扶持措施**：重点研究深圳、广州等城市，其政策通过提供“真金白银”的补贴，覆盖从技术研发到场景应用的全产业链<cite>turn_0_2</cite>。具体措施包括：\n  - **研发创新支持**：对重大技术攻关项目提供高达数千万元的补贴<cite>turn_0_2</cite>。\n  - **成本降低**：发放算力券以降低企业AI模型训练成本<cite>turn_0_2</cite>。\n  - **应用推广**：对“首台套”示范应用项目给予奖励<cite>turn_0_2</cite>。\n  - **金融与资本支持**：设立数十亿至百亿规模的产业基金，为企业提供融资渠道<cite>turn_0_2</cite>。\n- **区级精准支持**：探讨工匠社所在行政区（如深圳南山区、宝安区）的特色化支持，如为初创企业提供免费办公空间和启动资金等<cite>turn_0_2</cite>。\n\n## 4. 政策对工匠社科技的机遇与挑战\n- **发展机遇**：\n  - **研发创新激励**：可申请国家、省、市各级科研项目经费，以及地方性的重大技术攻关补贴<cite>turn_0_2</cite>。\n  - **市场与品牌认可**：凭借“专精特新”资质，提升品牌公信力，在政府采购与大型合作项目中获得优势<cite>turn_0_1</cite>。\n  - **产业生态支持**：融入地方产业集群，享受产业基金<cite>turn_0_2</cite>、完善的供应链和人才配套服务。商业化可从B端（商用服务）切入，逐步拓展至C端（家庭）<cite>turn_0_4</cite>。\n- **潜在挑战**：\n  - **技术与商业化**：行业整体仍处于商业化初期，面临技术泛化能力不足等挑战<cite>turn_0_3</cite>。企业需探索从“卖硬件”到“租赁服务”等多元商业模式<cite>turn_0_3</cite>。\n  - **政策方向不确定性**：由于缺乏直接对口政策，需持续解读宏观政策，判断发展风向，将AI大模型、先进运动控制等技术与产品结合<cite>turn_0_3</cite>。\n  - **合规性要求**：满足日益严格的产品安全、数据安全和隐私保护标准。\n\n## 5. 相关政策汇总表\n\n| 政策名称 | 发布机构 | 发布时间 | 政策层级 | 核心内容与相关性（摘要） | 与工匠社关联分析 |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n| 《“十四五”机器人产业发展规划》<cite>turn_0_0</cite> | 工业和信息化部等 | 2021 | 国家级 | 目标到2025年成为全球机器人技术创新策源地，制造业机器人密度翻番。<cite>turn_0_0</cite> | 为公司发展提供宏观战略指引，利好整体产业环境。 |\n| 《人形机器人创新发展指导意见》<cite>turn_0_0</cite> | 工业和信息化部 | 2023 | 国家级 | 目标到2025年初步建立人形机器人创新体系，2027年综合实力达到世界先进水平。<cite>turn_0_0</cite> | 虽非直接针对格斗机器人，但人形机器人相关的技术突破（如运动控制）可为公司所借鉴。 |\n| “人工智能+”行动 | 国务院 | 2024 | 国家级 | 将AI提升至国家战略，明确智能终端普及率目标，将“具身智能”列为未来产业。<cite>turn_0_0</cite> | 极大利好，为格斗机器人的智能化、场景化应用提供了顶层政策背书。 |\n| “专精特新”中小企业培育政策<cite>turn_0_1</cite> | 工业和信息化部等 | 持续 | 国家级 | 为入选企业提供资金、融资、人才、采购等一揽子政策红利。<cite>turn_0_1</cite> | 工匠社作为细分领域企业，是重点培育对象，可直接申请以获得实质性支持。 |\n| 地方性机器人与AI产业扶持政策<cite>turn_0_2</cite> | 深圳、广州等地方政府 | 近年 | 地方级 | 提供研发补贴、算力券、产业基金、首台套奖励、免费办公空间等“真金白银”支持。<cite>turn_0_2</cite> | 可直接申请所在地市、区的各项补贴和奖励，降低研发和运营成本，加速商业化进程。 |'
        input_dict['is_finish_turn_1'] = True

        
        

            