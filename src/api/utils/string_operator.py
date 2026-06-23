import re
import logging
import traceback
import sys
import json 

from typing import Any


CHINESE_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')
PARAGRAPH_SPLIT_SYMBOL = '\n'


def has_chinese_chars(data: Any) -> bool:
    text = f'{data}'
    return bool(CHINESE_CHAR_RE.search(text))

def clean_paragraph(text: str):
    text = rm_cid(text)
    text = rm_hexadecimal(text)
    text = rm_continuous_placeholders(text)
    return text

def rm_newlines(text: str):
    if text.endswith('-\n'):
        text = text[:-2]
        return text.strip()
    rep_c = ' '
    if has_chinese_chars(text):
        rep_c = ''
    text = re.sub(r'(?<=[^\.。:：\d])\n', rep_c, text)
    return text.strip()


def rm_cid(text: str):
    text = re.sub(r'\(cid:\d+\)', '', text)
    return text


def rm_hexadecimal(text: str):
    text = re.sub(r'[0-9A-Fa-f]{21,}', '', text)
    return text


def rm_continuous_placeholders(text: str):
    text = re.sub(r'[.\- —。_*]{7,}', '\t', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def json_fix(text: str):
    if len(text) == 0:
        return text
    
    text = text.strip()
    
    # 提取 <json_output> 标签内的内容
    if '<json_output>' in text and '</json_output>' in text:
        start = text.index('<json_output>') + len('<json_output>')
        end = text.index('</json_output>')
        text = text[start:end]
    
    # 提取文本中间的 ```json 代码块（内容前有其他文字的情况）
    elif '```json' in text:
        start = text.index('```json') + len('```json')
        # 从 ```json 之后寻找结束的 ```
        rest = text[start:]
        if '```' in rest:
            end = rest.index('```')
            text = rest[:end]
        else:
            text = rest

    else:
        # 移除开头的 markdown 代码块标记
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
    
    return text.strip()


def list_fix(text: str):
    match = re.search(r"```(?:\w+)?\s*([\s\S]*?)\s*```", text)
    if match:
        json_str = match.group(1)
    else:
        json_str = text.strip()
    
    return json.loads(json_str)

    
def markdown_fix(text: str):
    if len(text) == 0:
        return text
    
    text = text.strip()
    # 移除markdown代码块标记
    if text.startswith('```markdown'):
        text = text[11:]
    elif text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    return text.strip()

