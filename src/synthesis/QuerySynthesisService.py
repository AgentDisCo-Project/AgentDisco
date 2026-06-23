import gin
import json
import jinja2

from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.keys import DIRECTLLM_API_KEY_USER
from api.utils import ApiKeyCycler



@gin.configurable()
class QuerySynthesis:
    def __init__(
        self,
        model_name: str,
        max_retries: int = 5,
        retry_delay: int = 3,
        max_concurrent: int = 50,
        use_query_modality: str = "text",
        generate_type: str = "note",
        use_customize_url: bool = False,
        customize_url: str = "",
        max_query_len: int = 5,
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_file: str = "QuerySynthesis.jinja2"
    ):
        self.model = CustomizeChatGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
        )
        assert use_query_modality in ("text", "both"), f"Unsupported use_query_modality {use_query_modality}"
        self.use_query_modality = use_query_modality
        assert generate_type in ("note", "web"), f"Unsupported generate_type {generate_type}"
        self.generate_type = generate_type
        self.max_concurrent = max_concurrent
        self.max_query_len = max_query_len
        
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.jinja_file = system_template_file
    
    
    def get_system_prompt(self):
        try:
            template_vars = {
                "generate_type": self.generate_type,
                'use_query_modality': self.use_query_modality
            }
            template = self.jinja_env.get_template(self.jinja_file)
            system_prompt = template.render(**template_vars)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Template rendering error: {e}")
        return system_prompt
    
    
    def check_func(
        self,
        response: str,
    ):
        response = json.loads(response)
        assert len(response) <= self.max_query_len, f"Unsupported {self.max_query_len}"
        return response
    
    
    async def act(
        self,
        input_dict: dict,
        output_key: str = "subquery"
    ):
        query_text = input_dict["query_text"]
        similar_query = input_dict["similar_query"]
        
        if "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            system_prompt = [
                {
                    "type": "text",
                    "text": self.get_system_prompt()
                }
            ]
            
            
            user_prompt = []
            _user_prompt = f"""
# 用户提问
{query_text}\n
"""
            user_prompt.append(
                {
                    "type": "text",
                    "text": _user_prompt
                }
            )
            
            _user_prompt = f"""
# 相关小红书常见搜索词
{json.dumps(similar_query, ensure_ascii=False)}\n
"""
            user_prompt.append(
                {
                    "type": "text",
                    "text": _user_prompt
                }
            )
            
            cycler = ApiKeyCycler(api_key_list=list(DIRECTLLM_API_KEY_USER.values()))
            response = await self.model.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=self.check_func,
                cycler=cycler,
                return_cot=False,
            )
        
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")
        
        input_dict[f"{output_key}"] = response
        return input_dict