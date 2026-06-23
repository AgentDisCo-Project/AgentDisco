import os
import jinja2
import json
import logging
import time
import asyncio
import requests
import datetime
import urllib
import sys
sys.path.append('.')
sys.path.append('./api')

from typing import Union, Optional, Dict, List
from api.RednoteTextSearchService import RedNoteTextSearch
from api.GoogleSearchService import GoogleTextImageSearch
from api.KnowledgeBaseSearchService import KnowledgeBaseSearch
from tool.BaseToolService import register_tool, BasicTool
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")


@register_tool("get_weather", allow_overwrite=True)
class GetWeather(BasicTool):
    name = 'get_weather'
    description_en = 'Get weather for the city and date from the internet.'
    description_zh = '从互联网中得到天气信息。'
    parameters = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
            },
            'searcher': {
                'type': 'string',
            },
            'num_searches': {
                'type': 'integer',
            },
        },
        'required': ['query'],
    }
    
    
    async def poi_to_latlng(
        self,
        poi_name: str,
        city: str,
        page_size: int = 10,
        page_index: int = 1,
        output_format: str ='json',
        poi_type: str = None,
    ):
        # 调用API
        api_url = "https://apis.map.qq.com/ws/place/v1/search"
        boundary = f"region({city})"
        params = {
            "key": WEATHER_API_KEY,
            "keyword": urllib.parse.quote(poi_name),
            "boundary": boundary,
            "page_size": page_size,
            "page_index": page_index,
            "output": output_format
        }
        if poi_type:
            params["tag"] = poi_type
        
        try:
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result["status"] == 0 and result["data"]:
                first_poi = result["data"][0]
                lat = first_poi["location"]["lat"]
                lng = first_poi["location"]["lng"]
                location = f"{lat},{lng}"
                return location
            else:
                return f"API错误：{result.get('message', '未找到POI')}"
        except requests.exceptions.RequestException as e:
            return f"请求失败：{str(e)}"
    
    
    async def get_weather(
        self,
        location: str,
        weather_type: str = 'future',
        get_md: int = 1,
        output_format = 'json',
    ):
        api_url = "https://apis.map.qq.com/ws/weather/v1/"
        params = {
            "key": WEATHER_API_KEY,
            "type": weather_type,
            "get_md": get_md,
            "output": output_format
        }
        if location is not None:
            params["location"] = location
        
        try:
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result["status"] == 0:
                return result["result"]
            else:
                return f"API错误：{result['message']}"
        
        except requests.exceptions.RequestException as e:
            return f"请求失败：{str(e)}"
        except Exception as e:
            return f"未知错误：{str(e)}"
    
    
    async def act(
        self,
        params: Union[str, Dict],
    ):
        """
        根据时间范围、地点、城市查询天气
        
        参数：
        - start_date: str 开始时间(yyyy-mm-dd)
        - end_date: str 结束时间(yyyy-mm-dd)
        - poi_name: str 地点名称
        - city: str 城市名称
        - key: str API密钥
        
        返回：list 天气信息列表，包含日期、温度、湿度、天气等信息
        """
        params = self.verify_json_format_args(params)
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        poi_name = params.get("poi_name")
        city = params.get("city")
        
        # 获取POI经纬度（自动使用缓存）
        location = await self.poi_to_latlng(poi_name=poi_name, city=city)
        if not location or "错误" in location or "API" in location or "失败" in location:
            return f"获取地点经纬度失败：{location}"
        
        # 获取天气数据（未来7天）
        weather_data = await self.get_weather(location=location, weather_type='future', get_md=1)
        if isinstance(weather_data, str):
            return f"获取天气数据失败：{weather_data}"
        
        # 解析日期范围
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            return f"日期格式错误：{str(e)}，请使用yyyy-mm-dd格式"
        
        if start > end:
            return "错误：开始时间不能晚于结束时间"
        
        # 提取天气预报数据
        try:
            forecast_infos = weather_data['forecast'][0]['infos']
        except (KeyError, IndexError) as e:
            return f"天气数据格式错误：{str(e)}"
        
        # 筛选日期范围内的天气
        weather_list = []
        for info in forecast_infos:
            info_date = datetime.strptime(info['date'], '%Y-%m-%d')
            
            if start <= info_date <= end:
                weather_item = {
                    '日期': info['date'],
                    '星期': info['week'],
                    '白天天气': info['day']['weather'],
                    '白天温度': info['day']['temperature'],
                    '白天湿度': info['day']['humidity'],
                    '夜晚天气': info['night']['weather'],
                    '夜晚温度': info['night']['temperature'],
                    '夜晚湿度': info['night']['humidity'],
                    '风向': info['day']['wind_direction'],
                    '风力': info['day']['wind_power']
                }
                weather_list.append(weather_item)
        
        return weather_list if weather_list else "未找到指定日期范围内的天气数据"


    
if __name__ == "__main__":
    async def main():
        service = GetWeather()
        params = {
            "start_date": "2025-10-24",
            "end_date": "2025-10-24",
            "poi_name": "天安门",
            "city": "北京市",
        }
        results = await service.act(
            params=params
        )
        print(results)
    
    asyncio.run(main())
