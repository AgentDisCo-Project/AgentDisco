import gin
import sys
sys.path.append('.')

from typing import Dict, Tuple

from api.DisentangledOutlineJudgeBlueprintService import DisentangledOutlineJudgeBlueprint
from agent.MemoryBankAgent import MemoryBankManager


@gin.configurable()
class OutlineJudgePipeline:
    def __init__(
        self,
        use_zh: bool = False,
        use_evidence_as_key: bool = False,
        use_hierarchical_writer: bool = False,
        outline_judge_threshold: int = 8,
        max_outline_generator_turns: int = 10,
        min_outline_generator_turns: int = 2,
    ):
        self.judge = DisentangledOutlineJudgeBlueprint(
            use_zh=use_zh,
            use_evidence_as_key=use_evidence_as_key,
            need_filter=False,
            outline_judge_threshold=outline_judge_threshold,
            max_outline_generator_turns=max_outline_generator_turns,
            min_outline_generator_turns=min_outline_generator_turns,
        )
        self.memory_bank = MemoryBankManager(
            use_zh=use_zh,
            use_evidence_as_key=use_evidence_as_key,
            use_hierarchical_writer=use_hierarchical_writer,
        )

    async def act(self, input_dict: Dict, turn_id: int) -> Tuple[bool, Dict]:
        """Run one iteration of judge + memory bank.

        Returns (is_finish, input_dict). When is_finish=True the memory bank
        step is skipped because no further evidence gathering is needed.
        """
        is_finish, input_dict = await self.judge.act(
            input_dict=input_dict,
            turn_id=turn_id,
        )
        input_dict[f"is_finish_turn_{turn_id}"] = is_finish
        if not is_finish:
            input_dict = await self.memory_bank.act(
                input_dict=input_dict,
                turn_id=turn_id,
            )
        return is_finish, input_dict

    def get_docs_with_reference(self, input_dict: Dict, num_chunks: int = -1, turn_id: int = -1) -> Dict:
        return self.memory_bank.get_docs_with_reference(
            input_dict=input_dict,
            num_chunks=num_chunks,
            turn_id=turn_id,
        )
