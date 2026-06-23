import os
import gin
import asyncio
import jinja2
import json
import time
import sys
import logging
sys.path.append('.')

from PIL import Image, ImageDraw, ImageOps
from typing import Dict, List
from api.utils.key_operator import ApiKeyCycler
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.string_operator import json_fix, list_fix
from api.utils.url_operator import compress_and_convert_base64, compress_url, convert_pil_to_base64
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


@gin.configurable()
class ImageSelector:
    def __init__(
        self,
        model_name: str,
        use_zh: bool = True,
        max_retries: int = 5,
        retry_delay: int = 3,
        max_concurrent: int = 50,
        use_customize_url: bool = False,
        customize_url: str = "",
        max_summary_len: int = 256,
        image_key_type: str = "url",
        max_num_images: int = 14,
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file: str = "ImageSelector_EN.jinja2",
        system_template_zh_file: str = "ImageSelector_ZH.jinja2",
        max_long_edge: int = 2048,
        max_vlm_candidates: int = 18,
        max_vlm_collage_edge: int = 1536,
        max_image_num: int = 5,
        max_image_num_per_document: int = 5,
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
        self.max_long_edge = max_long_edge
        self.max_vlm_candidates = max_vlm_candidates
        self.max_vlm_collage_edge = max_vlm_collage_edge
        self.max_image_num = max_image_num
        self.max_image_num_per_document = max_image_num_per_document

    def get_system_prompt(self):
        template_vars = {
            "max_image_num": self.max_image_num
        }
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    def check_func(
        self,
        response: str,
    ):
        return list_fix(response)

    def _make_vlm_collage(
        self,
        images: List,
    ):
        cell = 420
        cols = 3 if len(images) <= 9 else 4
        rows = (len(images) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * cell, rows * cell), "white")
        draw = ImageDraw.Draw(canvas)
        image_idx_to_url = {}
        image_url_to_idx = {}

        for idx, image in enumerate(images):
            image_idx = str(idx + 1)
            image_idx_to_url[image_idx] = image["url"]
            image_url_to_idx[image["url"]] = image_idx

            with Image.open(image["path"]) as _image:
                _image = ImageOps.exif_transpose(_image).convert("RGB")
                fitted = ImageOps.contain(_image, (cell, cell), Image.Resampling.LANCZOS)

            x = (idx % cols) * cell + (cell - fitted.width) // 2
            y = (idx // cols) * cell + (cell - fitted.height) // 2
            canvas.paste(fitted, (x, y))

            label_x = (idx % cols) * cell + 12
            label_y = (idx // cols) * cell + 12
            draw.rectangle((label_x, label_y, label_x + 58, label_y + 46), fill=(0, 0, 0))
            draw.text((label_x + 16, label_y + 10), image_idx, fill=(255, 255, 255))

        canvas = self._resize_long_edge(canvas)
        return canvas, image_idx_to_url, image_url_to_idx

    def _resize_long_edge(
        self,
        image: Image.Image, 
    ):
        long_edge = max(image.size)
        if long_edge <= self.max_vlm_collage_edge:
            return image.copy()
        scale = self.max_vlm_collage_edge / long_edge
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        return image.resize(size, Image.Resampling.LANCZOS)
    

    async def post_request(
        self,
        query_text: str,
        images: List,
        blueprints: List,
        outline: str,
    ):
        canvas, image_idx_to_url, image_url_to_idx = self._make_vlm_collage(images)
        if "gemini" in self.model.model_name:
            system_prompt = [{"text": self.get_system_prompt()}]
            user_prompt = []

            if self.use_zh:
                _user_prompt = f"""
# 用户提问
{query_text}
"""
            else:
                _user_prompt = f"""
# User question
{query_text}
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

            for image in images:
                idx = image_url_to_idx[image["url"]]
                width, height = image["width"], image["height"]
                if self.use_zh:
                    _user_prompt = f"""
# 拼图中图片{idx}
单图ID：{idx}
原图尺寸：{width}x{height}
"""
                else:
                    _user_prompt = f"""
# Image {idx}
Number: {idx}
Original image size: {width}x{height}
"""
                
                user_prompt.append({"text": _user_prompt})
            
            canvas = convert_pil_to_base64(canvas)
            user_prompt.append(
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": canvas,
                    }
                }
            )

            response = await self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                return_cot=False,
            )
            response = response[:self.max_image_num]
        
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")

        selected_images = []
        for image in images:
            if image_url_to_idx[image["url"]] in response:
                selected_images.append(image)

        return selected_images 
    

    async def act(
        self,
        input_dict: Dict,
        turn_id: int,
    ):
        query_text = input_dict.get("query_text", "") or input_dict.get("query", "")
        if turn_id == -1:
            outline = input_dict["outline"]
            blueprints = input_dict["blueprint"]
            documents = input_dict[f"search_result"]
        else:
            outline = input_dict[f"outline_turn_{turn_id}"]
            blueprints = input_dict[f"blueprint_turn_{turn_id}"]
            documents = input_dict[f"search_result_turn_{turn_id}"]
        
        images = []
        for document in documents:
            _images = []
            for image in document['images']:
                if image['status'] != 'valid':
                    continue
                _images.append(image)
            images.extend(_images[:self.max_image_num_per_document])
        images = images[:self.max_vlm_candidates]
        
        selected_images = await self.post_request(
            query_text=query_text,
            images=images,
            outline=outline,
            blueprints=blueprints,
        )
        input_dict["selected_images"] = selected_images
        return input_dict
    


if __name__ == "__main__":
    DEBUG_SEARCH_RESULT = []
    async def main():
        service = ImageSelector(
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