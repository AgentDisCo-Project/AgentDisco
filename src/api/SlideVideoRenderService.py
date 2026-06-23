import os
import gin
import json
import logging
import jinja2
import sys
sys.path.append('.')

from pathlib import Path
from typing import Dict, List
from api.utils.string_operator import json_fix
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.QwenTTSDeployService import QwenTTSDeploy
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')


@gin.configurable()
class SlideVideoRender:
    def __init__(
        self,
        model_name: str,
        tts_model_name: str = "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        tts_voice: str = "Vivian",
        max_retries: int = 5,
        retry_delay: int = 3,
        use_zh: bool = True,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_api_key: bool = True,
        fps: int = 24,
        system_template_dir: str = "./template",
        system_template_en_file: str = "SlideVideoRender_EN.jinja2",
        system_template_zh_file: str = "SlideVideoRender_ZH.jinja2",
    ):
        self.use_zh = use_zh
        self.fps = fps
        self.max_retries = max_retries
        self.model = CustomizeChatGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
        )
        self.tts = QwenTTSDeploy(
            model_name=tts_model_name,
            voice=tts_voice,
        )
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.jinja_file = system_template_zh_file if use_zh else system_template_en_file

    def get_system_prompt(self):
        template = self.jinja_env.get_template(self.jinja_file)
        return template.render()

    async def act(
        self,
        input_dict: Dict,
        input_key: str = "xhs_slide_images",
        output_dir: str = "",
    ) -> Dict:
        image_paths = input_dict.get(input_key, [])
        if not image_paths:
            logging.warning("[SlideVideoRender] no slide images found, skipping")
            input_dict["slide_video_path"] = ""
            return input_dict

        plan_data = input_dict.get("slide_content_plan")
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

        if not output_dir:
            output_dir = input_dict.get("output_dir", "")
        if not output_dir:
            cache_dir = input_dict.get("cache_dir", "/tmp")
            job_id = input_dict.get("id", "default")
            output_dir = os.path.join(cache_dir, f"{job_id}_slide_video")
        os.makedirs(output_dir, exist_ok=True)

        system_prompt = self.get_system_prompt()
        total = len(image_paths)

        audio_paths: List[str] = []
        for i, img_path in enumerate(image_paths):
            section_content = ""
            if i < len(sections):
                s = sections[i]
                section_content = f"## {s.get('title', '')}\n\n{s.get('content', '')}"

            narration = await self._generate_narration(
                system_prompt, section_content, i, total,
            )
            if not narration:
                logging.warning(f"[SlideVideoRender] no narration for slide {i+1}/{total}, using fallback")
                narration = sections[i].get("title", f"Slide {i+1}") if i < len(sections) else f"Slide {i+1}"

            slide_id = f"slide_{i:02d}"
            audio_bytes = await self.tts.synthesize(narration)
            if not audio_bytes:
                logging.error(f"[SlideVideoRender] TTS failed for slide {i+1}/{total}")
                continue

            audio_path = os.path.join(output_dir, f"{slide_id}.wav")
            Path(audio_path).write_bytes(audio_bytes)
            audio_paths.append(audio_path)
            logging.info(f"[SlideVideoRender] TTS done slide {i+1}/{total}: {audio_path}")

        if not audio_paths:
            logging.error("[SlideVideoRender] no audio generated, skipping video assembly")
            input_dict["slide_video_path"] = ""
            return input_dict

        video_path = os.path.join(output_dir, "slide_video.mp4")
        self._assemble_video(image_paths, audio_paths, video_path)
        input_dict["slide_video_path"] = video_path
        logging.info(f"[SlideVideoRender] video saved: {video_path}")
        return input_dict

    async def _generate_narration(
        self,
        system_prompt: str,
        section_content: str,
        slide_index: int,
        total_slides: int,
    ) -> str:
        if self.use_zh:
            user_text = f"请为以下幻灯片内容生成讲解旁白（第 {slide_index+1} 页，共 {total_slides} 页）：\n\n{section_content}"
        else:
            user_text = f"Generate narration for this slide (slide {slide_index+1} of {total_slides}):\n\n{section_content}"

        system_parts = [{"text": system_prompt}]
        user_parts = [{"text": user_text}]

        response = await self.model.chat_gemini(
            system_prompt=system_parts,
            user_prompt=user_parts,
            check_func=lambda r: r,
            return_cot=False,
        )
        return response.strip() if response else ""

    def _assemble_video(
        self,
        image_paths: List[str],
        audio_paths: List[str],
        output_path: str,
    ):
        from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

        clips = []
        for img_path, aud_path in zip(image_paths, audio_paths):
            if not os.path.exists(img_path) or not os.path.exists(aud_path):
                continue
            audio = AudioFileClip(aud_path)
            clip = ImageClip(img_path, duration=audio.duration)
            clip = clip.with_audio(audio)
            clips.append(clip)

        if not clips:
            logging.error("[SlideVideoRender] no valid clips to assemble")
            return

        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(
            output_path,
            fps=self.fps,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )

        for clip in clips:
            clip.close()
        final.close()
