import gin
import sys
sys.path.append('.')

from typing import Dict

from agent.ReportWriterAgent import ReportWriter
from api.ReferenceRenderService import ReferenceRender


@gin.configurable()
class ReportGeneratorPipeline:
    def __init__(
        self,
        use_zh: bool = False,
        use_hierarchical_writer: bool = False,
        use_evidence_as_key: bool = False,
    ):
        self.writer = ReportWriter(
            use_zh=use_zh,
            use_hierarchical_writer=use_hierarchical_writer,
            use_evidence_as_key=use_evidence_as_key,
        )
        self.reference_render = ReferenceRender()
        self.use_hierarchical_writer = use_hierarchical_writer

    async def act(self, input_dict: Dict, num_chunks: int = 1) -> Dict:
        if self.use_hierarchical_writer:
            for chunk_id in range(num_chunks):
                input_dict = await self.writer.act(
                    input_dict=input_dict,
                    chunk_id=chunk_id,
                )
            input_dict = self.writer.merge_chunks_into_report(
                input_dict=input_dict,
                num_chunks=num_chunks,
            )
        else:
            input_dict = await self.writer.act(input_dict=input_dict)

        input_dict = self.reference_render.act(
            input_dict=input_dict,
            input_key="report",
            output_key="rendered_report",
        )
        return input_dict
