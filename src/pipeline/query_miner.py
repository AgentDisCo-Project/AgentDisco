import gin
import sys
sys.path.append('.')

from typing import Dict

from agent.QueryMinerAgent import QueryMiner
from api.IntentPlannerService import IntentPlanner


@gin.configurable()
class QueryMinerPipeline:
    def __init__(
        self,
        use_zh: bool = False,
        use_input_query: bool = True,
        include_summary: bool = False,
        disable_video: bool = True,
        disable_images: bool = False,
        disable_multi_images: bool = False,
        disable_comment: bool = True,
    ):
        self.query_miner = QueryMiner(
            use_zh=use_zh,
            disable_video=disable_video,
            disable_images=disable_images,
            disable_multi_images=disable_multi_images,
            disable_comment=disable_comment,
            use_input_query=use_input_query,
            include_summary=include_summary,
        )
        self.intent_planner = IntentPlanner(use_zh=use_zh)

    async def act(self, input_dict: Dict) -> Dict:
        input_dict = await self.query_miner.act(
            input_dict=input_dict,
            response_key="query_text",
        )
        input_dict = await self.intent_planner.act(input_dict=input_dict)
        return input_dict
