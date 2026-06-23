import gin

from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from typing import Optional, List, Union, Dict
from agent.BaseAgent import BasicAgent


@gin.configurable()
class ReportRender(BasicAgent):
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
        system_template_en_file: str = "HierarchicalWriter_EN.jinja2",
        system_template_zh_file: str = "HierarchicalWriter_ZH.jinja2",
        include_user_query: bool = False,
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
    
    
    async def act(self):
        pass