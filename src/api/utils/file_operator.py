import pandas as pd
import json
import re
import logging
import sys
import traceback

from pathlib import Path
from typing import List, Dict, Any, Union
from api.utils.url_operator import is_http_url, get_basename_from_url, get_content_type_by_head_request, contains_html_tags



class ExcelProcessor:
    @staticmethod
    def clean_text_for_excel(text: str, max_length: int = None) -> str:
        """清理文本中的非法Excel字符"""
        if not isinstance(text, str):
            text = str(text)
        cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
        # 如果指定了最大长度，则截断
        if max_length:
            cleaned = cleaned[:max_length]
        return cleaned
    
    
    @staticmethod
    def clean_sheet_name(name: str, max_length: int = 31) -> str:
        """清理Excel工作表名称"""
        # 移除Excel不支持的字符
        cleaned = re.sub(r'[\[\]\*:/\\\?]', ' ', str(name))
        return cleaned[:max_length]



class FileLoader:
    @staticmethod
    def load_data(file_path: str) -> List[Dict]:
        """根据文件扩展名自动选择加载方式"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension in ['.xlsx', '.xls', '.csv']:
            return FileLoader._load_tabular(file_path)
        elif extension == '.jsonl':
            return FileLoader._load_jsonl(file_path)
        elif extension == '.json':
            return FileLoader._load_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    
    @staticmethod
    def _load_tabular(file_path: str) -> List[Dict]:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        return df.to_dict(orient="records")
    
    
    @staticmethod
    def _load_jsonl(file_path: str) -> List[Dict]:
        data_list = []
        with open(file_path, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data_list.append(json.loads(line))
        return data_list
    
    
    @staticmethod
    def _load_json(file_path: str) -> List[Dict]:
        """加载JSON文件"""
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        # 如果是单个对象，转换为列表
        return data if isinstance(data, list) else [data]



class FileSaver:
    @staticmethod
    def save_data(data: List[Dict], file_path: str, **kwargs):
        """根据文件扩展名自动选择保存方式"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        print(f"Saving to {file_path} (format: {extension})")
        
        try:
            if extension in ['.xlsx', '.xls']:
                FileSaver._save_excel(data, file_path, **kwargs)
            elif extension == '.jsonl':
                FileSaver._save_jsonl(data, file_path)
            elif extension == '.json':
                FileSaver._save_json(data, file_path)
            elif extension == '.csv':
                FileSaver._save_csv(data, file_path)
            else:
                print(f"Unknown extension '{extension}', defaulting to JSON")
                FileSaver._save_json(data, file_path.rsplit('.', 1)[0] + '.json')
        
        except Exception as e:
            print(f"❌ Error saving file: {str(e)}")
            # 备份保存
            backup_path = path.stem + '_backup.json'
            FileSaver._save_json(data, backup_path)
    
    
    @staticmethod
    def _save_json(data: List[Dict], file_path: str):
        """保存为JSON格式"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ JSON saved: {file_path} ({len(data)} items)")
    
    
    @staticmethod
    def _save_jsonl(data: List[Dict], file_path: str):
        """保存为JSONL格式"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        print(f"✓ JSONL saved: {file_path} ({len(data)} lines)")
    
    
    @staticmethod
    def _save_csv(data: List[Dict], file_path: str):
        """保存为CSV格式"""
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"✓ CSV saved: {file_path} ({len(data)} rows)")
    
    
    @staticmethod
    def _save_excel(
        data: List[Dict],
        file_path: str,
        sheet_name: str = 'Sheet1',
        auto_adjust_width: bool = True,
        flatten_nested: bool = True,
        **kwargs
    ):
        if flatten_nested:
            data = FileSaver._flatten_nested_data(data)
        
        df = pd.DataFrame(data)
        
        # 清理数据中的非法字符
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(lambda x: ExcelProcessor.clean_text_for_excel(x) if x is not None else x)
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            clean_sheet_name = ExcelProcessor.clean_sheet_name(sheet_name)
            df.to_excel(writer, sheet_name=clean_sheet_name, index=False)
            
            if auto_adjust_width:
                FileSaver._adjust_column_width(writer.sheets[clean_sheet_name])
        
        print(f"✓ Excel saved: {file_path} ({len(data)} rows, {len(df.columns)} columns)")
    
    
    
    @staticmethod
    def _flatten_nested_data(data: List[Dict]) -> List[Dict]:
        """展开嵌套的JSON数据"""
        flattened_data = []
        
        for item in data:
            flattened_item = {}
            
            for key, value in item.items():
                if isinstance(value, str):
                    try:
                        # 尝试解析JSON字符串
                        parsed_value = json.loads(value)
                        if isinstance(parsed_value, dict):
                            # 展开嵌套字段
                            for nested_key, nested_value in parsed_value.items():
                                flattened_item[f"{key}_{nested_key}"] = nested_value
                            # 保留原始JSON字符串
                            flattened_item[key] = value
                        else:
                            flattened_item[key] = value
                    except (json.JSONDecodeError, TypeError):
                        flattened_item[key] = value
                else:
                    flattened_item[key] = value
            
            flattened_data.append(flattened_item)
        
        return flattened_data
    
    
    
    @staticmethod
    def _adjust_column_width(worksheet, max_width: int = 50):
        """自动调整Excel列宽"""
        for column in worksheet.columns:
            max_length = 0
            column_name = column[0].column_letter
            
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            
            adjusted_width = min(max_length + 2, max_width)
            worksheet.column_dimensions[column_name].width = adjusted_width



def read_text_from_file(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as file:
            file_content = file.read()
    except UnicodeDecodeError:
        tb = ''.join(traceback.format_exception(*sys.exc_info(), limit=3))
        logging.info(tb)
        from charset_normalizer import from_path
        results = from_path(path)
        file_content = str(results.best())
    return file_content
    
    
def df_to_md(df) -> str:
    def replace_long_dashes(text):
        if text.replace('-', '').replace(':', '').strip():
            return text
        pattern = r'-{6,}'
        replaced_text = re.sub(pattern, '-----', text)
        return replaced_text
    
    from tabulate import tabulate
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    df = df.fillna('')
    md_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
    
    md_table = '\n'.join([
        '|'.join(replace_long_dashes(' ' + cell.strip() + ' ' if cell else '')
                 for cell in row.split('|'))
        for row in md_table.split('\n')
    ])
    return md_table



def get_file_type(
    path: str
):
    f_type = get_basename_from_url(path).split('.')[-1].lower()
    if f_type in ['pdf', 'docx', 'pptx', 'csv', 'tsv', 'xlsx', 'xls']:
        # Specially supported file types
        return f_type
    
    if is_http_url(path):
        # The HTTP header information for the response is obtained by making a HEAD request to the target URL,
        # where the Content-type field usually indicates the Type of Content to be returned
        content_type = get_content_type_by_head_request(path)
        if 'application/pdf' in content_type:
            return 'pdf'
        elif 'application/msword' in content_type:
            return 'docx'
        
        # Assuming that the URL is HTML by default,
        # because the file downloaded by the request may contain html tags
        return 'html'
    else:
        # Determine by reading local HTML file
        try:
            content = read_text_from_file(path)
        except Exception as ex:
            tb = ''.join(traceback.format_exception(*sys.exc_info(), limit=3))
            logging.info(tb)
            return 'unk'
        
        if contains_html_tags(content):
            return 'html'
        else:
            return 'txt'


