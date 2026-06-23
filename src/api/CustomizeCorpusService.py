import re


class CustomizeCorpus:
    def __init__(
        self,
        need_chunk: bool = False,
        chunk_size: int = None,
        max_num_chunks_per_request: int = None,
        use_chunk_size: int = None,
        need_overlap: bool = False
    ):
        self.need_chunk = need_chunk # 使用分chunk操作
        self.chunk_size = chunk_size
        self.max_num_chunks_per_request = max_num_chunks_per_request
        self.use_chunk_size = use_chunk_size
        self.need_overlap = need_overlap
    
    
    @staticmethod
    def remove_empty_texts(
        note: dict,
    ):
        content = note.get("content", "")
        content = "" if content is None else content
        content = re.sub(r"#.*?#\s*", "", content, flags=re.DOTALL)  # 删除话题
        content = content.strip()
        return len(content) == 0
    
    
    def divide_into_chunks(
        self,
        input_text: str,
    ):
        if len(input_text) < self.use_chunk_size:
            return input_text
        
        para_seg = re.split(r'\n{1,2}', input_text)
        results = []
        
        for para in para_seg:
            para = para.strip()
            if not para:
                continue
            
            sent_seg = re.split(r'([。！？；])', para)
            full_sentences = []
            for i in range(0, len(sent_seg), 2):
                if i+1 < len(sent_seg): # 如果有分隔符就拼在一起
                    full_sentences.append(sent_seg[i] + sent_seg[i+1])
                else:
                    full_sentences.append(sent_seg[i])
            
            for sent in full_sentences:
                sent = sent.strip()
                if not sent:
                    continue
                
                comma_seg = [s.strip() for s in sent.split("，") if s.strip()]
                i = 0
                chunk = []
                while i < len(comma_seg):
                    chunk = [comma_seg[i]] # 当前块初始为当前片段
                    chunk_size = len(chunk)
                    j = i + 1
                    
                    # 尝试加入下一个逗号片段，若不超过max_len则继续，否则停止
                    while j < len(comma_seg) and chunk_size + len(comma_seg[j]) + 1 <= self.chunk_size:
                        chunk.append(comma_seg[j])
                        chunk_size += len(comma_seg[j]) + 1  # +1为逗号
                        j += 1
                    
                    # 把这块按逗号拼接，存入结果列表
                    results.append("，".join(chunk))
                    i = j
            
            
            if self.need_overlap:
                new_results = []
                for k in range(len(results)):
                    if k == 0:
                        new_results.append(results[0])
                    else:
                        # 上一块最后一个逗号分句，作为重叠内容
                        overlap_unit = results[k-1].split('，')[-1]
                        curr_chunk = results[k]
                        agg = overlap_unit + "，" + curr_chunk # 尝试将overlap拼接到当前块头部
                        if len(agg) > self.chunk_size: # 如果超长，则只保留当前块
                            new_results.append(curr_chunk)
                        else: # 没超长，则把重叠片段+当前块做新块
                            new_results.append(agg)
                results = new_results
            
            return results
        