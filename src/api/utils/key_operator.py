import asyncio

from itertools import cycle
from typing import List, Dict, Any, Union


class ApiKeyCycler:
    def __init__(self, api_key_list: List[str]):
        self.api_keys = api_key_list
        self._lock = asyncio.Lock()
        self._cycler = cycle(self.api_keys)
    
    async def get_key(self):
        async with self._lock:
            return next(self._cycler)