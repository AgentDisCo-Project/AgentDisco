import base64
import re
import urllib
import os
import requests
import logging
import time
import shutil

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from PIL import Image
from io import BytesIO


def convert_btye_to_base64(
    path: str
):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def convert_pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buf = BytesIO()
    image.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def is_http_url(
    path_or_url: str
):
    if path_or_url.startswith('https://') or path_or_url.startswith('http://'):
        return True
    return False



def get_basename_from_url(
    path_or_url: str
) -> str:
    if re.match(r'^[A-Za-z]:\\', path_or_url):
        # "C:\\a\\b\\c" -> "C:/a/b/c"
        path_or_url = path_or_url.replace('\\', '/')
    
    # "/mnt/a/b/c" -> "c"
    # "https://github.com/here?k=v" -> "here"
    # "https://github.com/" -> ""
    basename = urllib.parse.urlparse(path_or_url).path
    basename = os.path.basename(basename)
    basename = urllib.parse.unquote(basename)
    basename = basename.strip()
    
    # "https://github.com/" -> "" -> "github.com"
    if not basename:
        basename = [x.strip() for x in path_or_url.split('/') if x.strip()][-1]
    
    return basename


def compress_url(
    url: str,
    height: int = 256,
    width: int = 256,
    quality: int = 75,
    format: str = "png"
):
    base_url = re.split(r'\?', url)[0]
    param = f"imageView2/1/w/{width}/h/{height}/{format}/png/q/{quality}"
    return f"{base_url}?{param}"



def compress_and_convert_base64(
    path: str,
    max_size = (512, 512),
    format = "PNG",
    quality = 75,
):
    with Image.open(path) as image:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)  # 修正这里
        if format.upper() == "JPEG":
            image = image.convert('RGB')
        buffer = BytesIO()
        if format.upper() == 'PNG':
            image.save(buffer, format=format)
        elif format.upper() == 'JPEG':
            image.save(buffer, format=format, quality=quality)
        else:
            raise Exception(f"Unsupported format: {format}")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return img_base64


def get_content_type_by_head_request(
    url: str
):
    try:
        response = requests.head(url, timeout=5)
        content_type = response.headers.get('Content-Type', '')
        return content_type
    except requests.RequestException:
        return 'unk'


def contains_html_tags(
    text: str
):
    pattern = r'<(p|span|div|li|html|script)[^>]*?'
    return bool(re.search(pattern, text))
    
    
def sanitize_chrome_file_path(file_path: str) -> str:
    if os.path.exists(file_path):
        return file_path
    
    # Dealing with "file:///...":
    new_path = urllib.parse.urlparse(file_path)
    new_path = urllib.parse.unquote(new_path.path)
    new_path = sanitize_windows_file_path(new_path)
    if os.path.exists(new_path):
        return new_path
    
    return sanitize_windows_file_path(file_path)
    
    
def sanitize_windows_file_path(file_path: str) -> str:
    # For Linux and macOS.
    if os.path.exists(file_path):
        return file_path
    
    # For native Windows, drop the leading '/' in '/C:/'
    win_path = file_path
    if win_path.startswith('/'):
        win_path = win_path[1:]
    if os.path.exists(win_path):
        return win_path
    
    # For Windows + WSL.
    if re.match(r'^[A-Za-z]:/', win_path):
        wsl_path = f'/mnt/{win_path[0].lower()}/{win_path[3:]}'
        if os.path.exists(wsl_path):
            return wsl_path
    
    # For native Windows, replace / with \.
    win_path = win_path.replace('/', '\\')
    if os.path.exists(win_path):
        return win_path
    
    return file_path


def save_url_to_local_work_dir(
    url: str,
    save_dir: str,
    save_filename: str = '',
    max_retries: int = 3,
    timeout: tuple = (30, 300),  # (连接超时, 读取超时)
    chunk_size: int = 8192
):
    if not save_filename:
        save_filename = get_basename_from_url(url)
    new_path = os.path.join(save_dir, save_filename)
    
    if os.path.exists(new_path):
        os.remove(new_path)
    
    logging.info(f'Downloading {url} to {new_path}...')
    start_time = time.time()
    
    if not is_http_url(url):
        url = sanitize_chrome_file_path(url)
        shutil.copy(url, new_path)
    else:
        # 设置重试策略
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        # 创建session并配置重试
        session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        try:
            # 流式下载，避免内存问题
            response = session.get(
                url, 
                headers=headers, 
                timeout=timeout,
                stream=True,
                allow_redirects=True
            )
            response.raise_for_status()  # 抛出HTTP错误
            
            # 获取文件大小（如果有）
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(new_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # 可选：显示下载进度
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            if downloaded_size % (chunk_size * 100) == 0:  # 每800KB显示一次
                                logging.info(f'Download progress: {progress:.1f}%')
            
            logging.info(f'Download completed. Total size: {downloaded_size} bytes')
            
        except requests.exceptions.RequestException as e:
            logging.error(f'Failed to download {url}: {str(e)}')
            # 清理可能的部分下载文件
            if os.path.exists(new_path):
                os.remove(new_path)
            raise ValueError(f'Cannot download this file: {str(e)}. Please check your network or the file link.')
        
        except Exception as e:
            logging.error(f'Unexpected error when downloading {url}: {str(e)}')
            # 清理可能的部分下载文件
            if os.path.exists(new_path):
                os.remove(new_path)
            raise
        
        finally:
            session.close()
    
    end_time = time.time()
    logging.info(f'Finished downloading {url} to {new_path}. Time spent: {end_time - start_time:.2f} seconds.')
    return new_path
