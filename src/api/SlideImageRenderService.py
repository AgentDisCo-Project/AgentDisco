import os
import gin
import base64
import json
import logging
import jinja2
import sys
sys.path.append('.')

from pathlib import Path
from typing import Dict, List, Optional
from api.utils.string_operator import json_fix
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


@gin.configurable()
class SlideImageRender:
    VALID_ASPECT_RATIOS = {"9:16", "3:4", "16:9", "4:3"}
    PORTRAIT_RATIOS = {"9:16", "3:4"}
    LANDSCAPE_RATIOS = {"16:9", "4:3"}

    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        retry_delay: int = 3,
        use_zh: bool = True,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        max_slides: int = 0,
        system_template_dir: str = "./template",
        system_template_en_file: str = "SlideImageRender_EN.jinja2",
        system_template_zh_file: str = "SlideImageRender_ZH.jinja2",
        style: str = "handdrawn",
        orientation: str = "portrait",
        aspect_ratio: str = "",
    ):
        self.style = style
        self.orientation = orientation
        if aspect_ratio and aspect_ratio in self.VALID_ASPECT_RATIOS:
            self.aspect_ratio = aspect_ratio
            self.orientation = "portrait" if aspect_ratio in self.PORTRAIT_RATIOS else "landscape"
        else:
            self.aspect_ratio = "9:16" if orientation == "portrait" else "16:9"
        self.use_zh = use_zh
        self.max_slides = max_slides
        self.max_retries = max_retries
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
            lstrip_blocks=True,
        )
        self.jinja_file = system_template_zh_file if use_zh else system_template_en_file

    def get_system_prompt(self):
        template = self.jinja_env.get_template(self.jinja_file)
        return template.render(
            style=self.style,
            orientation=self.orientation,
            aspect_ratio=self.aspect_ratio,
        )

    async def act(
        self,
        input_dict: Dict,
        input_key: str = "slide_content_plan",
        output_dir: str = "",
    ) -> Dict:
        plan_data = input_dict.get(input_key)
        if not plan_data:
            logging.warning("[SlideImageRender] no slide_content_plan found, skipping")
            input_dict["xhs_slide_images"] = []
            return input_dict

        if isinstance(plan_data, str):
            plan_data = json_fix(plan_data)
            if isinstance(plan_data, str):
                plan_data = json.loads(plan_data)
        if isinstance(plan_data, dict):
            sections = plan_data.get("slides", plan_data.get("sections", []))
        elif isinstance(plan_data, list):
            sections = plan_data
        else:
            sections = []

        if not sections:
            logging.warning("[SlideImageRender] content plan has no sections, skipping")
            input_dict["xhs_slide_images"] = []
            return input_dict

        if self.max_slides > 0:
            sections = sections[:self.max_slides]

        if not output_dir:
            output_dir = input_dict.get("output_dir", "")
        if not output_dir:
            cache_dir = input_dict.get("cache_dir", "/tmp")
            job_id = input_dict.get("id", "default")
            output_dir = os.path.join(cache_dir, f"{job_id}_slide_images")
        os.makedirs(output_dir, exist_ok=True)

        system_prompt = self.get_system_prompt()
        total = len(sections)

        all_sections_md = "\n\n---\n\n".join(
            f"## {s.get('title', '')}\n\n{s.get('content', '')}" for s in sections
        )

        image_paths: List[str] = []
        style_ref_image: Optional[Dict] = None

        for i, section in enumerate(sections):
            section_id = section.get("id", f"slide_{i:02d}")
            section_type = "opening" if i == 0 else ("ending" if i == total - 1 else "content")

            section_md = f"## {section.get('title', 'Untitled')}\n\n{section.get('content', '')}"
            for tbl in section.get("tables", []):
                section_md += f"\n\n**{tbl.get('table_id', 'Table')}**:\n{tbl.get('extract', '')}"

            user_prompt = []
            user_prompt.append({"text": f"Slide {i + 1} of {total}, type: {section_type}"})
            user_prompt.append({"text": f"---\nThis slide:\n{section_md}"})
            user_prompt.append({"text": f"---\nFull context:\n{all_sections_md}"})

            if style_ref_image:
                user_prompt.append({"text": f"Reference - {style_ref_image['figure_id']}: {style_ref_image['caption']}"})
                user_prompt.append({
                    "inlineData": {
                        "mimeType": style_ref_image.get("mime_type", "image/png"),
                        "data": style_ref_image["base64"],
                    }
                })

            if i == 0:
                for sel_img in input_dict.get("selected_images", [])[:3]:
                    img_path = sel_img.get("path", "")
                    if img_path and os.path.exists(img_path):
                        try:
                            img_data = base64.b64encode(Path(img_path).read_bytes()).decode("utf-8")
                            user_prompt.append({"text": "Reference - Style reference from report images"})
                            user_prompt.append({
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": img_data,
                                }
                            })
                        except Exception:
                            pass

            try:
                response = await self.model.chat_gemini_image(
                    system_prompt=[{"text": system_prompt}],
                    user_prompt=user_prompt,
                    aspect_ratio=self.aspect_ratio,
                )
            except Exception as e:
                logging.error(f"[SlideImageRender] FAILED slide {section_id}: {e}")
                continue

            if not response or not response.get("image_bytes"):
                logging.warning(f"[SlideImageRender] no image in response for slide {section_id}")
                continue

            image_bytes = response["image_bytes"]
            mime_type = response.get("mime_type", "image/png")
            ext = ".png" if "png" in mime_type else ".jpg"
            fpath = os.path.join(output_dir, f"{section_id}{ext}")
            Path(fpath).write_bytes(image_bytes)
            image_paths.append(fpath)
            logging.info(f"[SlideImageRender] saved slide {i+1}/{total}: {fpath}")

            if style_ref_image is None and i >= 1:
                style_ref_image = {
                    "figure_id": "Reference Slide",
                    "caption": "Maintain consistent style",
                    "base64": base64.b64encode(image_bytes).decode("utf-8"),
                    "mime_type": mime_type,
                }

        input_dict["xhs_slide_images"] = image_paths
        input_dict["output_dir"] = output_dir
        logging.info(f"[SlideImageRender] generated {len(image_paths)}/{total} slide images")
        return input_dict
