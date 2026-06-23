import aiohttp
import time

from tqdm import tqdm
import interaction.service.InteractionService as InteractionService
from interaction.dto.ttypes import GetCommentByPageRequest
from poi_rpc.infra.rpc.base.ttypes import Context
from poi_rpc.reddata.service import RedDataService
from poi_rpc.reddata.dto.ttypes import MultiGetNoteDetailByOidRequest
from poi_rpc.thrift_rpc.red_rpc_util import create_thrift_client



class RedNoteSearch:
    def __init__(
        self,
        max_retries: int,
        retry_delay: int,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    
    async def get_note_details(
        self,
        note_id: str
    ):
        for attempt in range(self.max_retries):
            try:
                client2 = create_thrift_client(
                    service_name="reddataservice-service-default",
                    client_class=RedDataService,
                )
                request = MultiGetNoteDetailByOidRequest([note_id])
                response = client2.multiGetNoteDetailByOid(Context(), request)
                return response
            
            except (
                aiohttp.ClientError,
                Exception,
            ) as e:
                tqdm.write(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                
                if attempt < self.max_retries - 1:  # 如果不是最后一次尝试
                    time.sleep(self.retry_delay)  # 等待一段时间再重试
                    continue
                else:
                    tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                    return ""  # 重试多次后仍失败，返回None
    
    
    async def get_comment_info(
        self,
        note_id: int,
    ):
        for attempt in range(self.max_retries):
            try:
                client2 = create_thrift_client(
                    service_name="interactioncore-service-main",
                    client_class=InteractionService,
                )
                request = GetCommentByPageRequest(noteId=note_id, num=50)
                response = client2.getCommentByPage(Context(), request)
                return response
            
            except (
                aiohttp.ClientError,
                Exception,
            ) as e:
                tqdm.write(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                
                if attempt < self.max_retries - 1:  # 如果不是最后一次尝试
                    time.sleep(self.retry_delay)  # 等待一段时间再重试
                    continue
                else:
                    tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                    return None  # 重试多次后仍失败，返回None
    
    
    @staticmethod
    async def convert_oid_to_long(
        oid: str
    ):
        if len(oid) != 24:
            raise ValueError(f"Invalid oid length {len(oid)}")
        
        timestamp = int(oid[0:8], 16)
        mid = int(oid[8:14], 16)
        pid = int(oid[14:18], 16)
        seq = int(oid[18:24], 16)
        id_val = 3 << 56           # type (byte)3 << 56
        
        if timestamp >= 1544803200:
            id_val |= 36028797018963968    # 0x80000000000000
            id_val |= timestamp << 24
            id_val |= pid << 18
            id_val |= seq & 262143         # 0x3FFFF
            return id_val
        else:
            return -1
            # raise ValueError("timestamp too small (< 1544803200)")
    
    
    @staticmethod
    async def convert_long_to_oid(
        id: int
    ):
        if (id & 0x80000000000000) == 0:
            raise FileNotFoundError(f"Note id {id} not support!")
        else:
            part1 = (id >> 24) & 0x7FFFFFFF
            part2 = ((id >> 18) & 0x3F) << 24 | (id & 0x3FFFF)
            return f"{part1:08x}00000000{part2:08x}"
    
    
    @staticmethod
    async def convert_note_id_to_video_id(
        note_id: str
    ):
        video_url  = f'https://media-video.devops.xiaohongshu.com/media/video/'
        video_url += f'getVideoByNoteId?caller=test&bizName=110&noteId={note_id}'
        video_url += f'&fillType=7'
        return video_url
    
