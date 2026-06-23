import os
import gin
import asyncio
import jinja2
import json
import sys
import time
import logging
sys.path.append('.')

from typing import Dict, List
from api.utils.key_operator import ApiKeyCycler
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.string_operator import json_fix, list_fix
from api.utils.url_operator import compress_and_convert_base64, compress_url
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


@gin.configurable()
class NoteSelector:
    def __init__(
        self,
        model_name: str,
        use_zh: bool = True,
        max_retries: int = 5,
        retry_delay: int = 3,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        max_document_num: int = 100,
        batch_size: int = 100,
        system_template_dir: str = "./template",
        system_template_en_file: str = "NoteSelector_EN.jinja2",
        system_template_zh_file: str = "NoteSelector_ZH.jinja2",
        include_search_query: bool = False,
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
        self.jinja_file = system_template_en_file if not use_zh else system_template_zh_file
        self.use_zh = use_zh
        self.max_document_num = max_document_num
        self.include_search_query = include_search_query
        self.batch_size = batch_size


    def get_system_prompt(
        self
    ):
        template_vars = {
            "max_document_num": self.max_document_num
        }
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    def check_func(
        self,
        response: str,
    ):
        response = list_fix(response)
        return response
    

    async def post_request(
        self,
        query_text: str,
        search_query: str,
        documents: List,
        blueprints: List,
        outline: str = "",
    ):
        note_selector_st = time.time()
        if "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            system_prompt = self.get_system_prompt()
            user_prompt = ""

            for document in documents:
                idx, title, content, search_from = document["id"], document["title"], document["content"], document["search_from"]
                if "google" in search_from:
                    if self.use_zh:
                        _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
摘要（不是全部内容）：{content}
"""
                    else:
                        _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Abstract (Not Full Content): {content}
"""

                else:
                    if self.use_zh:
                        _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
内容：{content}
"""
                    else:
                        _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Content: {content}
"""
                user_prompt += _user_prompt
            
            if self.use_zh:
                _user_prompt = f"""
# 报告大纲
{outline}
"""
            else:
                _user_prompt = f"""
# Report Outline
{outline}
"""
                
            user_prompt += _user_prompt

            if self.use_zh:
                _user_prompt = f"""
# 大纲要点列表
{blueprints}
"""
            else:
                 _user_prompt = f"""
# Report Outline Blueprints
{blueprints}
"""
            user_prompt += _user_prompt

            if self.include_search_query:
                if self.use_zh:
                    _user_prompt = f"""
# 搜索词
{search_query}
"""
                else:
                    _user_prompt = f"""
# Search Query Terms
{search_query}
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

            for document in documents:
                idx, title, content, search_from = document["id"], document["title"], document["content"], document["search_from"]
                if "google" in search_from:
                    if self.use_zh:
                        _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
摘要（不是全部内容）：{content}
"""
                    else:
                        _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Abstract (Not Full Content): {content}
"""

                else:
                    if self.use_zh:
                        _user_prompt = f"""
## 外源搜索结果{idx}的内容如下
文档ID：{idx}
标题：{title}
内容：{content}
"""
                    else:
                        _user_prompt = f"""
## External Search Document {idx}
Document ID: {idx}
Title: {title}
Content: {content}
"""
                user_prompt.append({"text": _user_prompt})
            
            if self.use_zh:
                _user_prompt = f"""
# 报告大纲
{outline}
"""
            else:
                _user_prompt = f"""
# Report Outline
{outline}
"""
                
            user_prompt.append({"text": _user_prompt})
            
            if self.use_zh:
                _user_prompt = f"""
# 大纲要点列表
{blueprints}
"""
            else:
                 _user_prompt = f"""
# Report Outline Blueprints
{blueprints}
"""
            user_prompt.append({"text": _user_prompt})
        
                
            if self.include_search_query:
                if self.use_zh:
                    _user_prompt = f"""
# 搜索词
{search_query}
"""
                else:
                    _user_prompt = f"""
# Search Query Terms
{search_query}
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
            
            response = await self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                return_cot=False,
            )

        else:
            raise ValueError(f"Unsupported {self.model.model_name}")

        note_selector_et = time.time()
        logging.info(f"note_selector costs: {note_selector_et-note_selector_st}")
        
        new_documents = []
        for doc in documents:
            if doc["id"] in response:
                new_documents.append(doc)
        return new_documents


    async def act(
        self,
        input_dict: Dict,
        turn_id: int,
    ):
        query_text = input_dict.get("input_text", "") or input_dict.get("query", "")
        if turn_id == -1:
            search_query = []
            documents = list(input_dict['reference_map'].values())
            outline = input_dict["outline"]
            blueprints = input_dict["blueprint"]
        else:
            search_query = input_dict.get(f"search_query_turn_{turn_id}", [])
            documents = input_dict[f"search_result_turn_{turn_id}"]
            outline = input_dict.get(f"outline_turn_{turn_id}", "")
            blueprints = input_dict.get(f"blueprints_turn_{turn_id}", [])

        batches = [documents[i:i + self.batch_size] for i in range(0, len(documents), self.batch_size)]
        tasks = [
            self.post_request(
                query_text=query_text,
                search_query=search_query,
                documents=batch,
                outline=outline,
                blueprints=blueprints,
            )
            for batch in batches
        ]
        batch_results = await asyncio.gather(*tasks)
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        if turn_id == -1:
            input_dict[f"search_result"] = results
        else:
            input_dict[f"record_search_result_turn_before_selector_{turn_id}"] = documents
            input_dict[f"search_result_turn_{turn_id}"] = results
        return input_dict

if __name__ == "__main__":
    DEBUG_SEARCH_RESULT = [{'id': 'turn_0_0', 'search_from': 'search_note', 'content': '[派对R]视黄醇是什么？\n[害羞R]视黄醇的适用人群\n[赞R]视黄醇的功效\n[喝奶茶R]视黄醇的作用机理\n[偷笑R]视黄醇的常见误区\n[哇R]视黄醇的使用注意事项\n\ufeff#每天一个成长小知识[话题]#\ufeff \ufeff#变美[话题]#\ufeff \ufeff#皮肤[话题]#\ufeff \ufeff#视黄醇[话题]#\ufeff \ufeff#护肤[话题]#\ufeff \ufeff#知识科普[话题]#\ufeff', 'title': '每天一个小知识：视黄醇', 'url': 'https://www.xiaohongshu.com/explore/687ee7ac0000000010027809', 'date': '2025-07-22 12:03:26', 'note_type': 'images', 'video': {'noteId': '687ee7ac0000000010027809', 'url': ''}, 'images': [{'fileId': 'spectrum/1040g34o31k7jusbc3q605odiv47k1ecl5evdgqg', 'url': 'http://ci.xiaohongshu.com/spectrum/1040g34o31k7jusbc3q605odiv47k1ecl5evdgqg?imageView2/2/w/1080/format/jpg', 'width': 1242, 'height': 1660}, {'fileId': 'spectrum/1040g34o31k7jusbc3q6g5odiv47k1eclve9103o', 'url': 'http://ci.xiaohongshu.com/spectrum/1040g34o31k7jusbc3q6g5odiv47k1eclve9103o?imageView2/2/w/1080/format/jpg', 'width': 1242, 'height': 1660}, {'fileId': 'spectrum/1040g34o31k7jusbc3q705odiv47k1eclp9stmp0', 'url': 'http://ci.xiaohongshu.com/spectrum/1040g34o31k7jusbc3q705odiv47k1eclp9stmp0?imageView2/2/w/1080/format/jpg', 'width': 1242, 'height': 1660}, {'fileId': 'spectrum/1040g34o31k7jusbc3q7g5odiv47k1eclpv8jb1o', 'url': 'http://ci.xiaohongshu.com/spectrum/1040g34o31k7jusbc3q7g5odiv47k1eclpv8jb1o?imageView2/2/w/1080/format/jpg', 'width': 1242, 'height': 1660}, {'fileId': 'spectrum/1040g34o31k7jusbc3q805odiv47k1ecls20s1n8', 'url': 'http://ci.xiaohongshu.com/spectrum/1040g34o31k7jusbc3q805odiv47k1ecls20s1n8?imageView2/2/w/1080/format/jpg', 'width': 1242, 'height': 1660}, {'fileId': 'spectrum/1040g34o31k7jusbc3q8g5odiv47k1ecla9tjm90', 'url': 'http://ci.xiaohongshu.com/spectrum/1040g34o31k7jusbc3q8g5odiv47k1ecla9tjm90?imageView2/2/w/1080/format/jpg', 'width': 1242, 'height': 1660}], 'like_count': '230', 'collect_count': 145, 'view_count': '21176', 'comments': [], 'confidence': -1, 'detail': ''}, {'id': 'turn_0_1', 'search_from': 'search_note', 'content': '很多人用到这个程度就很害怕了\n就停用了，停用了也好，不用遭那罪\n-\n这种情况是a醇正在开始起效的标志\n角质开始收干，由于角质没有水分，在做表情的时候拉到表皮，从而就会出现这种看上去像细纹的东西\n-\n我是耐受之后（已用a类7-8年），隔着很长一段时间没有系统的用，今年开始慢慢捡回来，一般停用a醇或者a酸三个月后，要重新建立耐受，耐受的方法有很多种\n-\n我自己是老玩家，重新耐受是每天晚上1a醇，一般是一个月就可以耐受，再过一段时间再看表观和仪器检测的变化#A醇[话题]#', 'title': '用了a醇后细纹更深了……', 'url': 'https://www.xiaohongshu.com/explore/69a30353000000001a022c13', 'date': '2026-02-28 23:01:39', 'note_type': 'images', 'video': {'noteId': '69a30353000000001a022c13', 'url': ''}, 'images': [{'fileId': 'notes_pre_post/1040g3k031t4s51o8lm604a6gir86g9r2r5p1bkg', 'url': 'http://ci.xiaohongshu.com/notes_pre_post/1040g3k031t4s51o8lm604a6gir86g9r2r5p1bkg?imageView2/2/w/1080/format/jpg', 'width': 1920, 'height': 2560}], 'like_count': '1527', 'collect_count': 666, 'view_count': '346137', 'comments': [], 'confidence': -1, 'detail': ''}, {'id': 'turn_0_2', 'search_from': 'search_note', 'content': '选对成分，抗老效果翻倍！\n简单易懂，建议直接收藏⭐\n💡 一、 三大成分一句话速看\n🤍 胜肽：“温柔信号兵”。主攻淡化表情纹（抬头纹、鱼尾纹），温和促进胶原蛋白，修护维稳。适合所有肤质，尤其敏感肌、新手和孕妇，早晚可用，是安心牌选手。\n🌙 A醇：“高效突击队”。主打强效淡纹、紧致肌肤、收缩毛孔，抗老效果最猛。刺激性强，仅限健康耐受皮夜间使用，必须严格防晒。新手需建立耐受，不可与酸类混搭。\n💦 玻色因：“温和填充师”。核心在于充盈饱满，改善皮肤松弛、干瘪，同时保湿修护屏障。温和不挑皮，无需避光。适合追求嘭弹感、肌肤有干瘪感的熟龄肌。\n💡 二、 超实用搭配公式（不翻车版）\n✅ 经典抗老王炸组合（早C晚A变体）\n- ☀️ 白天（防御修护）： 爽肤水 → 胜肽精华 → 玻色因面霜 → 防晒（必须！）\n- 🌙 晚上（抗老猛攻）： 爽肤水 → A醇精华/乳 → 保湿面霜（干皮可加玻色因）\n✅ 温和抗老组合（敏感肌/新手专属）\n- 早晚均可： 爽肤水 → 胜肽精华 → 玻色因精华/面霜\n- 思路： 胜肽主攻纹路，玻色因负责充盈修护，强强联合，温和有效。\n🚫 【禁忌搭配】牢记！\n- A醇 ❌ 酸类（果酸、水杨酸等）：烂脸预警！\n- A醇 ❌ 高浓度原型VC：刺激性叠加，皮肤可能吃不消。\n- A醇 ❌ 其他强效去角质产品：屏障受损警告！\n- （如果想用，务必分早晚或分天使用）\n💡 三、 一句话选购指南\n- 想淡化表情纹，皮肤敏感/是新手 → 首选胜肽。\n- 追求强效淡纹紧致，皮肤耐受不敏感 → 挑战A醇。\n- 解决松弛、干瘪，追求饱满弹润 → 锁定玻色因。\n💎 总结：\n抗老没有唯一解，了解成分，认清自己的皮肤需求和耐受度，才能找到最适合你的“本命成分”！\n循序渐进，坚持防晒，才是抗老的终极奥义！\n#抗老成分[话题]# #护肤干货[话题]# #A醇[话题]# #玻色因[话题]# #胜肽[话题]# #保姆级教程[话题]# #精简护肤笔记[话题]#\n希望这份笔记能帮你清晰梳理，轻松搞定抗老护肤！❤️', 'title': '抗老成分“三巨头”｜一篇看懂怎么选', 'url': 'https://www.xiaohongshu.com/explore/69cefdfd00000000210382db', 'date': '2026-04-03 07:38:37', 'note_type': 'images', 'video': {'noteId': '69cefdfd00000000210382db', 'url': ''}, 'images': [{'fileId': 'notes_pre_post/1040g3k831ufqnk1biqdg5ovs8e2jqc0p2trjq6g', 'url': 'http://ci.xiaohongshu.com/notes_pre_post/1040g3k831ufqnk1biqdg5ovs8e2jqc0p2trjq6g?imageView2/2/w/1080/format/jpg', 'width': 2728, 'height': 4096}], 'like_count': '187', 'collect_count': 228, 'view_count': '12773', 'comments': [], 'confidence': -1, 'detail': ''}, {'id': 'turn_0_3', 'search_from': 'search_note', 'content': '1⃣ A醇是什么？\n2⃣A醇有什么功效与作用\n3⃣ A醇需要注意哪些事项？\n4⃣ 如何正确使用A醇？\n5⃣ A醇的CP组合和使用公式\n6⃣ A醇适用人群\n评论区分享一下，你们使用A醇产品一般搭配什么产品一起用吧！！！\n#A醇[话题]# #A醇护肤[话题]# #早C晚A[话题]# #护肤[话题]# #护肤小知识[话题]# #护肤成分[话题]#', 'title': '每天一个护肤知识：A醇', 'url': 'https://www.xiaohongshu.com/explore/696781d5000000002202f39c', 'date': '2026-01-14 19:45:25', 'note_type': 'images', 'video': {'noteId': '696781d5000000002202f39c', 'url': ''}, 'images': [{'fileId': 'notes_pre_post/1040g3k831raoqvkmg0dg5q7oidltt9kvuv4ccto', 'url': 'http://ci.xiaohongshu.com/notes_pre_post/1040g3k831raoqvkmg0dg5q7oidltt9kvuv4ccto?imageView2/2/w/1080/format/jpg', 'width': 1242, 'height': 1656}, {'fileId': 'notes_pre_post/1040g3k031raoqvmdng505q7oidltt9kvmu854tg', 'url': 'http://ci.xiaohongshu.com/notes_pre_post/1040g3k031raoqvmdng505q7oidltt9kvmu854tg?imageView2/2/w/1080/format/jpg', 'width': 1242, 'height': 1656}, {'fileId': 'notes_pre_post/1040g3k031raoqvmdng5g5q7oidltt9kvl40ik88', 'url': 'http://ci.xiaohongshu.com/notes_pre_post/1040g3k031raoqvmdng5g5q7oidltt9kvl40ik88?imageView2/2/w/1080/format/jpg', 'width': 1242, 'height': 1656}, {'fileId': 'notes_pre_post/1040g3k031raoqvmdng605q7oidltt9kvsnia2u0', 'url': 'http://ci.xiaohongshu.com/notes_pre_post/1040g3k031raoqvmdng605q7oidltt9kvsnia2u0?imageView2/2/w/1080/format/jpg', 'width': 1242, 'height': 1656}], 'like_count': '79', 'collect_count': 29, 'view_count': '4133', 'comments': [], 'confidence': -1, 'detail': ''}, {'id': 'turn_0_4', 'search_from': 'search_note', 'content': '我只要出门几乎每个人都会说我看起来很小是不是00后[捂脸R]知道我32岁且有个儿子后都对我的身材面貌不可置信，但我确实也经历过产后爆肥和断崖衰老，所以对于抗老我确实挺有话语权。抗老一定得多维度，就下面几点能做到，吊打任何项目\n\t\n1️⃣睡多久因人而异\n这个因人而异，有的人就是6小时或者8小时就足够，但如果是高敏感人群，平时因为思虑过多，消耗了大量能量，需要睡10小时才够，那如果没睡够就会老得快，这就是为什么思虑重压力大的人更显老憔悴的原因，就是你消耗掉得能量没有补够回来。再就是尽量11点前睡，皮肤真正的修护时间只有23:00–03:00\n\t\n2️⃣想老的慢就一定要做抗阻力\n从30岁开始，每年流失0.5–1%肌肉，40岁后加速。肌肉越少 也就代表代谢越低 ，越容易胖、松、垮、没精神。抗阻力训练是唯一能增加肌肉的运动。但不要一上来用力过猛，循序渐进，边练边补\n\t\n3️⃣面部专项护理\n如果垮脸严重，那一方面需要提升筋膜，比如面部瑜伽、面部刮痧和拨筋、美容仪这些都可以，不必短时间内用力太猛，重点是细水长流一直坚持，是一定会有用的\n另一方面，坚持面霜厚敷。我自己就是长期厚敷的受益者，利用睡眠时间厚敷来深度修护滋养。如果你试过一次觉得闷脸，一个是没有建立耐受，再一个就是选的面霜不对\n\t\n4️⃣无时无刻保持舌头的位置摆放\n外貌特征不仅仅只是被基因决定的，还会被后天习惯影响。舌头掉了，上颌骨就掉了，鼻基底就凹了，法令纹就出来了！试试吞一口口水，舌头就会自然贴上去了，然后一直保持住\n\t\n5️⃣平衡防晒和光照\n无论晴天还是阴天都要防晒。哪怕在室内或者车内，因为引起光老化的长波紫外线（UVA）可以穿透玻璃！！但我们又每天需要光照来补充维生素D，那如果防晒做的全面，就要内服维生素D，这是大家担心的如何平衡防晒和晒太阳的方法\n\t\n6️⃣内调\n这个七七八八的分享过许多了，这里就不多说了。不管你有没有时间做内调，至少一定要控糖和养肝，这是女人漂亮皮肤好显年轻的关键\n\t\n7️⃣修心\n人的精力是有限的，要学会精神内守，别想太多管太多，尤其不要到处攀比，欲望过强。一定要向内求，专注本心。你会发现那些生活中看起来没心没肺大大咧咧或内心平和好心态的人会更年轻，整个人得状态也会更好\n\t\n#保持年轻的秘诀[话题]##有效抗老化[话题]##恢复年轻状态[话题]##我的美丽秘诀[话题]##女性保养[话题]##如何面对衰老[话题]##抗老[话题]#', 'title': '32岁被说像00后，抗老总结：', 'url': 'https://www.xiaohongshu.com/explore/69edd4b8000000001f0034cb', 'date': '2026-04-26 17:02:48', 'note_type': 'images', 'video': {'noteId': '69edd4b8000000001f0034cb', 'url': ''}, 'images': [{'fileId': 'notes_pre_post/1040g3k031ve2g8tbjq4g49n62j171bihsbth6eg', 'url': 'http://ci.xiaohongshu.com/notes_pre_post/1040g3k031ve2g8tbjq4g49n62j171bihsbth6eg?imageView2/2/w/1080/format/jpg', 'width': 1504, 'height': 2008}, {'fileId': 'notes_pre_post/1040g3k031vdp3cka2a5g49n62j171bih4koobq0', 'url': 'http://ci.xiaohongshu.com/notes_pre_post/1040g3k031vdp3cka2a5g49n62j171bih4koobq0?imageView2/2/w/1080/format/jpg', 'width': 1920, 'height': 2560}, {'fileId': 'notes_pre_post/1040g3k031vdp3cka2a6049n62j171bihnksdo0g', 'url': 'http://ci.xiaohongshu.com/notes_pre_post/1040g3k031vdp3cka2a6049n62j171bihnksdo0g?imageView2/2/w/1080/format/jpg', 'width': 1080, 'height': 1440}, {'fileId': 'notes_pre_post/1040g3k031ve2g8tbjq5049n62j171bihqqhnk4g', 'url': 'http://ci.xiaohongshu.com/notes_pre_post/1040g3k031ve2g8tbjq5049n62j171bihqqhnk4g?imageView2/2/w/1080/format/jpg', 'width': 2148, 'height': 2864}, {'fileId': 'notes_pre_post/1040g3k031vduemcuia0049n62j171bih7qd4f80', 'url': 'http://ci.xiaohongshu.com/notes_pre_post/1040g3k031vduemcuia0049n62j171bih7qd4f80?imageView2/2/w/1080/format/jpg', 'width': 2160, 'height': 2880}, {'fileId': 'notes_pre_post/1040g3k031ve2g8tbjq5g49n62j171bih5jqv3bg', 'url': 'http://ci.xiaohongshu.com/notes_pre_post/1040g3k031ve2g8tbjq5g49n62j171bih5jqv3bg?imageView2/2/w/1080/format/jpg', 'width': 1828, 'height': 2436}], 'like_count': '1861', 'collect_count': 1268, 'view_count': '108451', 'comments': [], 'confidence': -1, 'detail': ''}]

    async def main():
        service = NoteSelector(
            model_name="gemini-3.1-pro",
        )
        input_dict = dict()
        input_dict["query"] = '我想开始护肤，但对视黄醇和玻色因不太了解，想知道哪个更适合30岁左右、有初步抗老需求、肤质偏混合性的人群。请帮我对比一下这两个成分的抗衰老效果、使用后的肌肤反应（如是否会刺激、是否会蜕皮）、以及推荐几款市面上口碑较好的、适合新手入门的眼霜或精华产品。我的预算在300-500元之间，希望产品使用感温和，能改善细纹和初步的松弛感。最终目标是明确选择视黄醇还是玻色因，并能确定一到两款具体产品开始尝试。'
        input_dict["blueprint_turn_0"] = []
        input_dict["search_result_turn_0"] = DEBUG_SEARCH_RESULT
        input_dict = await service.act(
            input_dict=input_dict,
            turn_id=0,
        )
        print(input_dict)
    
    asyncio.run(main())