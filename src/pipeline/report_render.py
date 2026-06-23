import os
import logging
import gin
import sys
sys.path.append('.')

from typing import Dict

from api.NoteSelectorService import NoteSelector
from api.ImageManagerService import RednoteImageManager
from api.ImageSelectorService import ImageSelector
from api.HTMLRenderService import HTMLRender
from api.PosterSlideContentGeneratorService import PosterSlideContentGenerator
from api.SlideImageRenderService import SlideImageRender
from api.SlideRenderService import SlideRender
from api.SlideVideoRenderService import SlideVideoRender
from agent.ReportPolishAgent import ReportPolish
from api.ReferenceRenderService import ReferenceRender


@gin.configurable()
class ReportRenderPipeline:
    """Unified render pipeline supporting multiple output modes.

    Output modes (each is independently toggleable):
      - html:   rendered HTML report
      - slides: slide content plan + slide images (poster/slides)
      - xhs:    XHS copywriting + XHS HTML page (requires slides)
      - video:  narrated MP4 video (requires slides + xhs)

    Dependency chain: video -> xhs -> slides.
    Enabling a downstream mode auto-enables its prerequisites.
    """

    def __init__(
        self,
        use_zh: bool = False,
        render_with_image: bool = True,
        enable_note_selector: bool = True,
        enable_image_selector: bool = True,
        enable_html: bool = True,
        enable_slides: bool = False,
        enable_xhs: bool = False,
        enable_video: bool = False,
        enable_polish: bool = False,
        use_polish_before_render: bool = False,
        download_dir: str = "./cache",
    ):
        # Enforce dependency chain: video -> xhs -> slides
        if enable_video:
            enable_xhs = True
        if enable_xhs:
            enable_slides = True

        self.use_zh = use_zh
        self.render_with_image = render_with_image
        self.enable_note_selector = enable_note_selector
        self.enable_image_selector = enable_image_selector
        self.enable_html = enable_html
        self.enable_slides = enable_slides
        self.enable_xhs = enable_xhs
        self.enable_video = enable_video
        self.enable_polish = enable_polish
        self.use_polish_before_render = use_polish_before_render
        self.download_dir = download_dir

        if enable_note_selector:
            self.note_selector = NoteSelector(use_zh=use_zh)
            self.image_manager = RednoteImageManager(cache_dir=download_dir)

        if enable_image_selector:
            self.image_selector = ImageSelector(use_zh=use_zh)

        if enable_html:
            self.html_render = HTMLRender(
                use_zh=use_zh,
                render_with_image=render_with_image and enable_image_selector,
            )

        if enable_slides:
            self.slide_content_planner = PosterSlideContentGenerator(
                output_type="xhs_slides",
            )
            self.slide_image_render = SlideImageRender(use_zh=use_zh)

        if enable_xhs:
            self.slide_render = SlideRender(use_zh=use_zh)

        if enable_video:
            self.slide_video_render = SlideVideoRender(use_zh=use_zh)

        if enable_polish:
            self.report_polish = ReportPolish(
                use_zh=use_zh,
                use_polish_before_render=use_polish_before_render,
            )
            self.reference_render = ReferenceRender()

    async def act(
        self,
        input_dict: Dict,
        output_dir: str = "",
        cache_dir: str = "",
        job_name: str = "",
    ) -> Dict:
        # ---- Supporting steps (image selection for HTML) ----
        if self.enable_note_selector:
            logging.info("[ReportRenderPipeline] running note selector")
            input_dict = await self.note_selector.act(
                input_dict=input_dict, turn_id=-1,
            )
            input_dict = await self.image_manager.act(
                input_dict=input_dict, input_key="search_result",
            )

        if self.enable_image_selector:
            logging.info("[ReportRenderPipeline] running image selector")
            input_dict = await self.image_selector.act(
                input_dict=input_dict, turn_id=-1,
            )

        # ---- Output: HTML report ----
        if self.enable_html:
            logging.info("[ReportRenderPipeline] running html render")
            input_dict = await self.html_render.act(
                input_dict=input_dict,
                input_key="rendered_report",
            )

        # ---- Step 1: Slides (content plan + images) ----
        if self.enable_slides:
            logging.info("[ReportRenderPipeline] step 1: slide content planner")
            input_dict = await self.slide_content_planner.act(
                input_dict=input_dict,
                input_key="rendered_report",
            )

            slide_output_dir = os.path.join(
                cache_dir or output_dir, f"{job_name}_slide_images",
            )
            logging.info("[ReportRenderPipeline] step 1: slide image render")
            input_dict = await self.slide_image_render.act(
                input_dict=input_dict,
                input_key="slide_content_plan",
                output_dir=slide_output_dir,
            )

        # ---- Step 2: XHS (copywriting + HTML page) ----
        if self.enable_xhs:
            logging.info("[ReportRenderPipeline] step 2: xhs copy + html")
            input_dict = await self.slide_render.act(
                input_dict=input_dict,
                input_key="rendered_report",
            )

        # ---- Step 3: Video (narration + TTS + MP4) ----
        if self.enable_video:
            slide_video_output_dir = os.path.join(
                cache_dir or output_dir, f"{job_name}_slide_video",
            )
            logging.info("[ReportRenderPipeline] step 3: video render")
            input_dict = await self.slide_video_render.act(
                input_dict=input_dict,
                input_key="xhs_slide_images",
                output_dir=slide_video_output_dir,
            )

        # ---- Report polish ----
        if self.enable_polish:
            logging.info("[ReportRenderPipeline] running report polish")
            if self.use_polish_before_render:
                input_key = "rendered_report"
                output_key = "polished_rendered_report"
            else:
                input_key = "report"
                output_key = "polished_report"

            input_dict = await self.report_polish.act(
                input_dict=input_dict,
                input_key=input_key,
                output_key=output_key,
            )

            if not self.use_polish_before_render:
                input_dict = self.reference_render.act(
                    input_dict=input_dict,
                    input_key="polished_report",
                    output_key="polished_rendered_report",
                )

        return input_dict
