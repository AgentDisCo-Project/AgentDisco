import gin
import sys
sys.path.append('.')

from typing import Dict

from api.DisentangledOutlineGeneratorBlueprintService import DisentangledOutlineGeneratorBlueprint


@gin.configurable()
class OutlineGeneratorPipeline:
    def __init__(
        self,
        use_zh: bool = False,
        use_evidence_as_key: bool = False,
    ):
        self.generator = DisentangledOutlineGeneratorBlueprint(
            use_zh=use_zh,
            use_evidence_as_key=use_evidence_as_key,
        )

    async def act(self, input_dict: Dict, turn_id: int) -> Dict:
        return await self.generator.act(input_dict=input_dict, turn_id=turn_id)

    def divide_outline_into_chunks(self, outline):
        return self.generator.divide_outline_into_chunks(outline)
