import os

from typing import Optional, Dict, Union
from tool.BaseToolService import register_tool, BasicTool
from api.utils.file_operator import read_text_from_file



@register_tool("storage")
class Storage(BasicTool):
    # Reference: https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/tools/storage.py
    name = "storage"
    description_en = "A specific tool for data storage"
    description_zh = "存储和读取数据的工具"
    parameters = {
        'type': 'object',
        'properties': {
            'operate': {
                'description': '数据操作类型，可选项为["put", "get", "delete", "scan"]之一，分别为存数据、取数据、删除数据、遍历数据',
                'type': 'string',
            },
            'key': {
                'description': '数据的路径，类似于文件路径，是一份数据的唯一标识，不能为空，默认根目录为`/`。存数据时，应该合理的设计路径，保证路径含义清晰且唯一。',
                'type': 'string',
                'default': '/'
            },
            'value': {
                'description': '数据的内容，仅存数据时需要',
                'type': 'string',
            },
        },
        'required': ['operate'],
    }
    def __init__(
        self,
        cfg: Optional[Dict] = None,
        use_zh: bool = False,
    ):
        super().__init__(
            cfg=cfg,
            use_zh=use_zh
        )
        self.root_path = self.cfg["root_path"]
        os.makedirs(self.root_path, exist_ok=True)
        
    
    def call(
        self,
        params: Union[str, Dict]
    ):
        params = self.verify_json_format_args(params)
        operate = params["operate"]
        key = params.get('key', '/')
        if key.startswith('/'):
            key = key[1:]
        
        if operate == 'put':
            if 'value' in params:
                return self.put(key, params['value'])
            return "No available value"
        elif operate == 'get':
            return self.get(key)
        elif operate == 'delete':
            return self.delete(key)
        else:
            return self.scan(key)
    
    def put(
        self,
        key: str,
        value: str,
        path: Optional[str] = None
    ):
        path = path or self.root_path
        
        # one file for one key value pair
        path = os.path.join(path, key)
        
        path_dir = path[:path.rfind('/') + 1]
        if path_dir:
            os.makedirs(path_dir, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as fp:
            fp.write(value)
        # return f'Successfully saved {key}.'
    
    
    def get(
        self,
        key: str,
        path: Optional[str] = None
    ):
        path = path or self.root_path
        if not os.path.exists(os.path.join(path, key)):
            # return f'Get Failed: {key} does not exist'
            return ""
        return read_text_from_file(os.path.join(path, key))
    
    
    def delete(
        self, key,
        path: Optional[str] = None
    ):
        path = path or self.root_path
        path = os.path.join(path, key)
        if os.path.exists(path):
            os.remove(path)
            # return f'Successfully deleted {key}'
        # else:
        #     return f'Delete Failed: {key} does not exist'
    
    
    def scan(
        self,
        key: str,
        path: Optional[str] = None
    ):
        path = path or self.root_path
        path = os.path.join(path, key)
        if os.path.exists(path):
            if not os.path.isdir(path):
                return 'Scan Failed: The scan operation requires passing in a folder path as the key.'
            # All key-value pairs
            kvs = {}
            for root, dirs, files in os.walk(path):
                for file in files:
                    k = os.path.join(root, file)[len(path):]
                    if not k.startswith('/'):
                        k = '/' + k
                    v = read_text_from_file(os.path.join(root, file))
                    kvs[k] = v
            return '\n'.join([f'{k}: {v}' for k, v in kvs.items()])
        # else:
        #     return f'Scan Failed: {key} does not exist.'

    
    