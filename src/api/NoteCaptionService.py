import os
import gin
import asyncio
import time
import jinja2
import json
import logging
import base64
import io
import sys
sys.path.append('.')

from PIL import Image
from dotenv import load_dotenv
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.url_operator import compress_and_convert_base64, compress_url
from api.utils.key_operator import ApiKeyCycler


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)


@gin.configurable()
class NoteCaption:
    def __init__(
        self,
        model_name: str = "qwen2.5-7b-instruct",
        max_retries: int = 5,
        retry_delay: int = 3,
        max_concurrent: int = 50,
        use_customize_url: bool = False,
        customize_url: str = "",
        use_query_modality: str = "",
        use_note_modality: str = "",
        image_key_type: str = "url",
        max_num_images: int = 14,
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_file: str = "NoteCaption_ZH.jinja2",
        max_num_frames: int = 16,
        max_summary_len: int = 1024,
    ):
        self.model = CustomizeChatGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
        )
        assert image_key_type in ("path", "url"), f"Unsupported image_key_type {image_key_type}"
        self.image_key_type = image_key_type
        assert use_query_modality in ("text", "both"), f"Unsupported {use_query_modality}"
        self.use_query_modality = use_query_modality
        assert use_note_modality in ("one_image", "all_images", "text")
        self.use_note_modality = use_note_modality
        self.max_num_images = max_num_images
        self.max_concurrent = max_concurrent
        
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.jinja_file = system_template_file
        self.max_num_frames = max_num_frames
        self.max_summary_len = max_summary_len
    
    
    def get_system_prompt(self):
        template_vars = {
            'max_summary_len': self.max_summary_len,
        }
        template = self.jinja_env.get_template(self.jinja_file)
        system_prompt = template.render(**template_vars)
        return system_prompt
    
    
    @staticmethod
    def check_func(response: str):
        return response
    
    
    def extract_frames_with_imageio(self, video_path: str):
        """使用imageio提取帧（修复异常总帧数问题）"""
        logging.info(f"使用imageio提取视频帧: {video_path}")
        
        try:
            import imageio.v2 as imageio
        except ImportError:
            try:
                import imageio
            except ImportError:
                raise ImportError("需要安装imageio: pip install imageio[ffmpeg]")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        frames = []
        
        try:
            # 打开视频文件
            reader = imageio.get_reader(video_path)
            
            try:
                # 获取视频信息
                meta = reader.get_meta_data()
                fps = meta.get('fps', 30)
                
                # 尝试获取总帧数，但要检查是否合理
                total_frames = None
                try:
                    if hasattr(reader, '__len__'):
                        frame_count = len(reader)
                        # 检查帧数是否合理（超过1000万帧认为异常）
                        if frame_count > 0 and frame_count < 10_000_000:
                            total_frames = frame_count
                        else:
                            logging.info(f"检测到异常的总帧数: {frame_count}，将使用固定间隔采样")
                            total_frames = None
                except Exception as e:
                    logging.info(f"无法获取总帧数: {e}")
                    total_frames = None
                
                logging.info(f"视频FPS: {fps}, 总帧数: {total_frames}")
                
                if total_frames:
                    # 如果能获取合理的总帧数，进行均匀采样
                    step = max(1, total_frames // self.max_num_frames)
                    # 确保步长不会太大
                    step = min(step, total_frames // 2) if total_frames > 2 else 1
                    logging.info(f"采样步长: {step}")
                    
                    for i in range(0, total_frames, step):
                        if len(frames) >= self.max_num_frames:
                            break
                        try:
                            frame = reader.get_data(i)
                            pil_frame = Image.fromarray(frame).convert('RGB')
                            frames.append(pil_frame)
                            logging.info(f"提取第 {i} 帧")
                        except Exception as e:
                            logging.warning(f"跳过帧 {i}: {e}")
                            continue
                
                # 如果无法获取总帧数或者均匀采样失败，使用固定间隔采样
                if not frames:
                    logging.info("使用固定间隔采样")
                    frame_idx = 0
                    sample_rate = max(30, int(fps * 2))  # 至少每30帧采样一次，或每2秒采样一次
                    max_frames_to_process = 50000  # 最多处理5万帧，防止无限循环
                    
                    try:
                        for frame in reader:
                            if len(frames) >= self.max_num_frames:
                                break
                            if frame_idx >= max_frames_to_process:
                                logging.info(f"达到最大处理帧数限制 {max_frames_to_process}，停止采样")
                                break
                            if frame_idx % sample_rate == 0:
                                pil_frame = Image.fromarray(frame).convert('RGB')
                                frames.append(pil_frame)
                                logging.info(f"固定间隔采样：第 {frame_idx} 帧 (每{sample_rate}帧采样)")
                            frame_idx += 1
                    except Exception as e:
                        logging.info(f"固定间隔采样出错: {e}")
                    
            except Exception as e:
                logging.info(f"获取视频元数据失败，尝试简单采样: {e}")
                # 备用方案：最简单的采样方法
                frame_idx = 0
                sample_rate = 90  # 每90帧采样一次
                max_frames_to_process = 20000  # 最多处理2万帧
                
                try:
                    for frame in reader:
                        if len(frames) >= self.max_num_frames:
                            break
                        if frame_idx >= max_frames_to_process:
                            logging.info(f"备用方案达到最大处理帧数限制，停止采样")
                            break
                        if frame_idx % sample_rate == 0:
                            pil_frame = Image.fromarray(frame).convert('RGB')
                            frames.append(pil_frame)
                            logging.info(f"备用采样：第 {frame_idx} 帧")
                        frame_idx += 1
                except Exception as e:
                    logging.info(f"备用方案失败: {e}")
            
            finally:
                reader.close()
            
            if frames:
                logging.info(f"成功提取 {len(frames)} 帧")
                return frames
            else:
                raise RuntimeError("未能提取到任何帧")
                
        except Exception as e:
            logging.info(f"imageio处理失败: {e}")
            raise

    
    def extract_frames_with_moviepy(self, video_path: str):
        """使用moviepy提取帧"""
        logging.info(f"使用moviepy提取视频帧: {video_path}")
        
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            raise ImportError("需要安装moviepy: pip install moviepy")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        frames = []
        clip = None
        
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            
            logging.info(f"视频时长: {duration} 秒")
            
            # 计算采样时间点
            time_points = [i * duration / (self.max_num_frames + 1) for i in range(1, self.max_num_frames + 1)]
            
            for i, t in enumerate(time_points):
                try:
                    frame_array = clip.get_frame(t)
                    pil_frame = Image.fromarray(frame_array.astype('uint8')).convert('RGB')
                    frames.append(pil_frame)
                    logging.info(f"提取第 {i+1} 帧 (时间: {t:.2f}s)")
                except Exception as e:
                    logging.info(f"跳过时间点 {t}: {e}")
                    continue
            
            if frames:
                logging.info(f"moviepy成功提取 {len(frames)} 帧")
                return frames
            else:
                raise RuntimeError("未能提取到任何帧")
                
        except Exception as e:
            logging.info(f"moviepy处理失败: {e}")
            raise
        finally:
            if clip:
                clip.close()
    
    
    def extract_frames(self, video_path: str):
        """多种方法尝试提取视频帧"""
        logging.info(f"开始提取视频帧: {video_path}")
        
        # 方法1: imageio
        try:
            return self.extract_frames_with_imageio(video_path)
        except Exception as e:
            logging.info(f"imageio方法失败: {e}")
        
        # 方法2: moviepy
        try:
            return self.extract_frames_with_moviepy(video_path)
        except Exception as e:
            logging.info(f"moviepy方法失败: {e}")
        
        # 所有方法都失败
        raise RuntimeError(f"所有视频处理方法都失败: {video_path}")
    
    
    async def act(
        self,
        input_dict: dict,
        input_key: str = "search_results",
        output_key: str = "caption",
        print_concurrent: bool = False
    ):
        query_text = input_dict["query_text"]
        
        sem = asyncio.Semaphore(self.max_concurrent)
        num_active_task = 0
        active_lock = asyncio.Lock()
        
        async def post_and_judge(note):
            nonlocal num_active_task
            async with sem:
                async with active_lock:
                    num_active_task += 1
                if print_concurrent:
                    print(f"number of active tasks: {num_active_task}")
                try:
                    response = await self.post_request(
                        query_text=query_text,
                        note=note
                    )
                    note[output_key] = response
                finally:
                    async with active_lock:
                        num_active_task -= 1
            return note
        
        tasks = [
            asyncio.create_task(post_and_judge(note=note))
            for note in input_dict[input_key]
        ]
        
        await asyncio.gather(*tasks)
        return input_dict
    
    async def post_request(self, query_text: str, note: dict):
        if "qwen" in self.model.model_name or "deepseek" in self.model.model_name:
            system_prompt = [
                {
                    "type": "text",
                    "text": self.get_system_prompt()
                }
            ]
        
            user_prompt = []
            
            # 处理文字内容
            title, content = note.get("title", ""), note.get("content", "")
            if len(title) > 0 or len(content) > 0:
                _user_prompt = f"""
# 外源搜索文档文字内容
标题：{title}
内容：{content}
"""
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt,
                    }
                )

            # 处理图片类型
            if note.get("note_type") == "images":
                _user_prompt = f"""
# 外源搜索文档图片内容
"""
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt,
                    }
                )
                
                for image in note.get("images", [])[:self.max_num_images]:
                    if self.image_key_type == "url":
                        image_url = compress_url(image["url"])
                        user_prompt.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                },
                            }
                        )
                    
                    elif self.image_key_type == "path":
                        if image.get("status") != "valid":
                            continue
                        image_data = compress_and_convert_base64(path=image["path"])
                        user_prompt.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}",
                                },
                            }
                        )
            
            # 处理视频类型
            elif note.get("note_type") == "video":
                _user_prompt = f"""
# 外源搜索文档视频内容
"""
                
                user_prompt.append(
                    {
                        "type": "text",
                        "text": _user_prompt,
                    }
                )
                
                try:
                    video_path = note["video_meta"]["path"]
                    frames = self.extract_frames(video_path=video_path)
                    
                    for i, frame in enumerate(frames):
                        # 将PIL图像转换为base64
                        img_buffer = io.BytesIO()
                        frame.save(img_buffer, format='JPEG', quality=85)
                        img_str = base64.b64encode(img_buffer.getvalue()).decode()
                        
                        user_prompt.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}",
                                },
                            }
                        )
                        
                except Exception as e:
                    error_msg = f"视频处理失败: {str(e)}"
                    logging.info(error_msg)
                    user_prompt.append(
                        {
                            "type": "text",
                            "text": f"视频处理失败: {str(e)}，无法提取视频帧进行分析。"
                        }
                    )
            
            # 添加用户问题
            _user_prompt = f"""
# 用户提问
文本：{query_text}
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
            
            return response



if __name__ == "__main__":
    async def main():
        # video = {
        #     "url": "http://vjs.zencdn.net/v/oceans.mp4",
        #     "path": "/mnt/tidalfs-bdsz01/dataset/llm_dataset/jinjr_data/download_cache/6144af705a83c56cd859517ec2585c50ea53198c4c95dba4620d350c11bdabb1/oceans.mp4"
        # }
        
        service = NoteCaption(
            model_name="qwen3-vl-235b-a22b-instruct",
            use_api_key=True,
            use_query_modality="text",
            use_note_modality="text",
            max_num_frames=8,
            image_key_type="path",
        )

        images = [
            {
                "url": "https://cds.chinadaily.com.cn/dams/capital/image/202503/10/67ce5629e4b0ccc959925a1e_m.jpeg",
                "path": "/mnt/tidalfs-bdsz01/dataset/llm_dataset/jinjr_data/download_cache/2dea306d5e30b734c49339f62dd18441680e7fbfb59701f47597ff27dd386844/67ce5629e4b0ccc959925a1e_m.jpeg",
                "status": "valid"
            }
        ]
        
        note = {
            "content": "",
            "title": "",
            # "video_meta": video,
            "note_type": "images",
            "images": images
        }
        
        st = time.time()
        try:
            response = await service.post_request(
                query_text="请描述这个视频的内容",
                note=note,
            )
            print("Response:", response)
        except Exception as e:
            logging.error(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
        
        et = time.time()
        logging.info(f"captioning notes costs {et-st}")
        breakpoint()
    
    asyncio.run(main())
