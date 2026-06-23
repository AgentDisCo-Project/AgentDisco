import json
import numpy as np
import aiohttp
import asyncio
import requests

from io import BytesIO
from tqdm import tqdm
from typing import Optional, Union
from PIL import Image
from api.utils.url_operator import convert_btye_to_base64



class CustomizeScoreGenerator:
    def __init__(
        self,
        model_name: str = "jina-reranker-m0",
        max_retries: int = 5,
        retry_delay: int = 3,
        use_customize_url: bool = True,
        customize_url: str = "",
        use_query_modality: str = "",
        use_note_modality: str = "",
        image_pool_type: str = "max",
        image_key_type: str = "url",
        timeout: int = 60,
        compute_type: list = None,
        use_chunk: bool = False,
    ):
        self.model_name = model_name
        assert image_pool_type in ("max", "mean"), f"Unsupported image_pool_type {image_pool_type}"
        assert image_key_type in ("path", "url"), f"Unsupported image_key_type {image_key_type}"
        self.image_pool_type = image_pool_type
        self.image_key_type = image_key_type
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        assert use_customize_url and customize_url
        self.use_customize_url = use_customize_url
        self.customize_url = customize_url
        assert use_query_modality in ("image", "text", "both"), f"Unsupported use_query_modality {use_query_modality}"
        self.use_query_modality = use_query_modality
        assert use_note_modality in ("one_image", "all_images", "text"), f"Unsupported use_note_modality {use_note_modality}"
        self.use_note_modality = use_note_modality
        
        self.compute_type = [] if compute_type is None else compute_type
        if use_query_modality in ("text", "both"):
            if compute_type is None:
                self.compute_type.append("text2text")
        if use_query_modality in ("text", "both") and use_note_modality in ("one_image", "all_images"):
            if compute_type is None:
                self.compute_type.append("text2image")
        if use_query_modality in ("image", "both"):
            if compute_type is None:
                self.compute_type.append("image2text")
        if use_query_modality in ("image", "both") and use_note_modality in ("one_image", "all_images"):
            if compute_type is None:
                self.compute_type.append("image2image")
        self.use_chunk = use_chunk
    
    
    def get_text_from_note(
        self,
        note: dict
    ):
        if self.use_chunk and "chunk" in note:
            return note["chunk"]
        else:
            title, content = note["title"], note["content"]
            text = "" if title is None else title + "\n"
            content = "" if content is None else content
            text += content
            return text
    
    
    def get_image_from_note(
        self,
        note:dict
    ):
        if len(note["images"]) == 0:
            return []
        
        content = []
        for image_meta in note["images"]:
            if self.image_key_type == "url":
                if (image_meta["width"] < 28 and image_meta["width"] > 0) or (image_meta["height"] < 28 and image_meta["height"] > 0):
                    continue
                
                try:
                    response = requests.get(image_meta["url"])
                    response.raise_for_status()
                    
                    # 将图片数据加载到内存
                    image = Image.open(BytesIO(response.content))
                    width, height = image.size
                    if width < 28 or height < 28:
                        continue
                    image.load()
                
                except Exception as e:
                    print("error is ", e)
                    continue
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_meta[self.image_key_type]
                    }
                })
            
            elif self.image_key_type == "path":
                if (image_meta["width"] < 28 and image_meta["width"] > 0) or (image_meta["height"] < 28 and image_meta["height"] > 0):
                    continue
                if image_meta["status"] != "valid":
                    continue
                
                try:
                    with Image.open(image_meta["path"]) as image:
                        width, height = image.size
                        if width < 28 or height < 28:
                            continue
                        image.load()
                except Exception as e:
                    print("error is ", e)
                    continue
                
                image = convert_btye_to_base64(image_meta["path"])
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}",
                    }
                })
            
            if self.use_note_modality == "one_image":
                break

        return content
    
    
    @staticmethod
    def dist_based_norm(
        scores_list: list,
    ):
        invalid_indices = [idx for idx, score in enumerate(scores_list) if score == -1]
        valid_scores_list = [score for score in scores_list if score != -1]
        if len(valid_scores_list) == 0:
            return scores_list
        
        # Reference: https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18
        mean_score = np.mean(valid_scores_list)
        std_dev = (
                      sum((x - mean_score) ** 2 for x in valid_scores_list) / len(valid_scores_list)
                  ) ** 0.5
        min_score = mean_score - 3 * std_dev
        max_score = mean_score + 3 * std_dev
        
        if max_score == min_score:
            norm_scores_list = [0. for _ in range(len(valid_scores_list))]
        else:
            norm_scores_list = []
            for score in valid_scores_list:
                new_score = (score - min_score) / (max_score - min_score)
                norm_scores_list.append(new_score)
        
        new_scores_list = []
        norm_idx = 0
        for i in range(len(scores_list)):
            if i in invalid_indices:
                new_scores_list.append(-1)
            else:
                new_scores_list.append(norm_scores_list[norm_idx])
                norm_idx += 1
        return new_scores_list
    
    
    async def post_requests(
        self,
        query_text: str = "",
        query_image: str = "",
        input_texts: list[str] = None,
        input_images: list[dict] = None,
        query_type: str = "text",
        doc_type:str = "image"
    ):
        for attempt in range(self.max_retries):
            try:
                query = query_text if query_type == "text" else {"content": query_image}
                documents = input_texts if doc_type == "text" else {"content": input_images}
                headers = {'content-type': 'application/json'}
                llm_req = {
                    "model": "/workspace/models/jinaai",
                    "query" : query,
                    "documents": documents
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.customize_url,
                        data=json.dumps(llm_req),
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        response.raise_for_status()
                        response = await response.json()
                        response = response["results"]
                        return response
            
            except (
                Exception,
            ) as e:
                tqdm.write(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    tqdm.write(f"达到最大重试次数 {self.max_retries}，放弃重试")
                    return ""
    
    
    async def act(
        self,
        notes: list[dict],
        query_text: str = "",
        query_image: str = "",
        input_texts: list[str] = None,
        input_images: Union[list[dict], list[list[dict]], None] = None
    ):
        if "text2text" in self.compute_type:
            response = await self.post_requests(query_text=query_text, input_texts=input_texts, query_type="text", doc_type="text")
            if response == "":
                query_candidate_text_to_texts = norm_query_candidate_text_to_texts = [-1.] * len(input_texts)
            else:
                query_candidate_text_to_texts = sorted(response, key=lambda x: x["index"])
                query_candidate_text_to_texts = [x["relevance_score"] for x in query_candidate_text_to_texts]
                if not isinstance(query_candidate_text_to_texts, list):
                    query_candidate_text_to_texts = [query_candidate_text_to_texts]
                norm_query_candidate_text_to_texts = self.dist_based_norm(query_candidate_text_to_texts)
            for idx, note in enumerate(notes):
                note["text2text_score"] = query_candidate_text_to_texts[idx]
                note["norm_text2text_score"] = norm_query_candidate_text_to_texts[idx]
        
        if "text2image" in self.compute_type:
            if self.use_note_modality == "one_images":
                response = await self.post_requests(query_text=query_text, input_images=input_images, query_type="text", doc_type="image")
                if response == "":
                    query_candidate_text_to_image = norm_query_candidate_text_to_image = [-1.] * len(input_images)
                else:
                    query_candidate_text_to_image = sorted(response, key=lambda x: x["index"])
                    query_candidate_text_to_image = [x["relevance_score"] for x in query_candidate_text_to_image]
                    if not isinstance(query_candidate_text_to_image, list):
                        query_candidate_text_to_image = [query_candidate_text_to_image]
                    norm_query_candidate_text_to_image = self.dist_based_norm(query_candidate_text_to_image)
                for idx, note in enumerate(notes):
                    note["text2image_score"] = query_candidate_text_to_image[idx]
                    note["norm_text2image_score"] = norm_query_candidate_text_to_image[idx]
            
            elif self.use_note_modality == "all_images":
                # query_candidate_text_to_images = []
                # for _input_images in input_images:
                #     if len(_input_images) > 0:
                #         headers = {'content-type': 'application/json'}
                #         llm_req = {
                #             "model": "/workspace/models/jinaai",
                #             "query" : query_text,
                #             "documents": {"content": _input_images}
                #         }
                #         response = requests.post(self.customize_url, data=json.dumps(llm_req), headers=headers)
                #         response = response.json()
                #         response = response["results"]
                #         query_candidate_text_to_image = sorted(response, key=lambda x: x["index"])
                #         query_candidate_text_to_image = [x["relevance_score"] for x in query_candidate_text_to_image]
                #         if self.image_pool_type == "mean":
                #             query_candidate_text_to_image = np.mean(query_candidate_text_to_image)
                #         elif self.image_pool_type == "max":
                #             query_candidate_text_to_image = np.max(query_candidate_text_to_image)
                #         query_candidate_text_to_images.append(query_candidate_text_to_image)
                #     else:
                #         query_candidate_text_to_images.append(-1.)
                # if not isinstance(query_candidate_text_to_images, list):
                #     query_candidate_text_to_images = [query_candidate_text_to_images]
                # norm_query_candidate_text_to_images = self.dist_based_norm(query_candidate_text_to_images)
                async def worker(_query_text, _input_images, _semaphore):
                    async with _semaphore:
                        response = await self.post_requests(
                            query_text=query_text,
                            input_images=input_images,
                            query_type="text",
                            doc_type="image"
                        )
                        return response
                
                semaphore = asyncio.Semaphore(self.max_concurrency)
                tasks = [
                    worker(query_text, _input_images, semaphore)
                    for _input_images in input_images
                ]
                responses = await asyncio.gather(*tasks)
                query_candidate_text_to_images = []
                for response in responses:
                    if response == "":
                        query_candidate_text_to_images.append(-1.)
                    else:
                        query_candidate_text_to_image = sorted(response, key=lambda x: x["index"])
                        query_candidate_text_to_image = [x["relevance_score"] for x in query_candidate_text_to_image]
                        if self.image_pool_type == "mean":
                            query_candidate_text_to_image = float(np.mean(query_candidate_text_to_image))
                        elif self.image_pool_type == "max":
                            query_candidate_text_to_image = float(np.max(query_candidate_text_to_image))
                        query_candidate_text_to_images.append(query_candidate_text_to_image)
                
                norm_query_candidate_text_to_images = self.dist_based_norm(query_candidate_text_to_images)
                for idx, note in enumerate(notes):
                    note["text2images_score"] = query_candidate_text_to_images[idx]
                    note["norm_text2images_score"] = norm_query_candidate_text_to_images[idx]
        
        if "image2text" in self.compute_type:
            response = await self.post_requests(query_image=query_image, input_texts=input_texts, query_type="image", doc_type="text")
            if response == "":
                query_candidate_image_to_texts = norm_query_candidate_image_to_texts = [-1.] * len(input_texts)
            else:
                query_candidate_image_to_texts = sorted(response, key=lambda x: x["index"])
                query_candidate_image_to_texts = [x["relevance_score"] for x in query_candidate_image_to_texts]
                if not isinstance(query_candidate_image_to_texts, list):
                    query_candidate_image_to_texts = [query_candidate_image_to_texts]
                norm_query_candidate_image_to_texts = self.dist_based_norm(query_candidate_image_to_texts)
            for idx, note in enumerate(notes):
                note["image2text_score"] = query_candidate_image_to_texts[idx]
                note["norm_image2text_score"] = norm_query_candidate_image_to_texts[idx]
        
        if "image2image" in self.compute_type:
            if self.use_note_modality == "one_image":
                response = await self.post_requests(query_image=query_image, input_images=input_images, query_type="image", doc_type="image")
                if response == "":
                    query_candidate_image_to_image = norm_query_candidate_image_to_image = [-1.] * len(input_images)
                else:
                    query_candidate_image_to_image = sorted(response, key=lambda x: x["index"])
                    query_candidate_image_to_image = [x["relevance_score"] for x in query_candidate_image_to_image]
                    if not isinstance(query_candidate_image_to_image, list):
                        query_candidate_image_to_image = [query_candidate_image_to_image]
                    norm_query_candidate_image_to_image = self.dist_based_norm(query_candidate_image_to_image)
                for idx, note in enumerate(notes):
                    note["image2image_score"] = query_candidate_image_to_image
                    note["norm_image2image_score"] = norm_query_candidate_image_to_image
            
            elif self.use_note_modality == "all_images":
                async def worker(_query_image, _input_images, _semaphore):
                    async with _semaphore:
                        response = await self.post_requests(
                            query_image=query_image,
                            input_images=input_images,
                            query_type="image",
                            doc_type="image"
                        )
                        return response
                
                semaphore = asyncio.Semaphore(self.max_concurrency)
                tasks = [
                    worker(query_image, _input_images, semaphore)
                    for _input_images in input_images
                ]
                responses = await asyncio.gather(*tasks)
                query_candidate_image_to_images = []
                for response in responses:
                    if response == "":
                        query_candidate_image_to_images.append(-1.)
                    else:
                        query_candidate_image_to_image = sorted(response, key=lambda x: x["index"])
                        query_candidate_image_to_image = [x["relevance_score"] for x in query_candidate_image_to_image]
                        if self.image_pool_type == "mean":
                            query_candidate_image_to_image = float(np.mean(query_candidate_image_to_image))
                        elif self.image_pool_type == "max":
                            query_candidate_image_to_image = float(np.max(query_candidate_image_to_image))
                        query_candidate_image_to_images.append(query_candidate_image_to_image)
                
                norm_query_candidate_image_to_images = self.dist_based_norm(query_candidate_image_to_images)
                for idx, note in enumerate(notes):
                    note["image2images_score"] = query_candidate_image_to_images[idx]
                    note["norm_image2images_score"] = norm_query_candidate_image_to_images[idx]
        
        score_cnt = 0
        for note in notes:
            note["score"] = 0
        if "text2text" in self.compute_type:
            for note in notes:
                note["score"] += note["norm_text2text_score"]
            score_cnt += 1
        if "text2image" in self.compute_type:
            if self.use_note_modality == "one_image":
                for note in notes:
                    note["score"] += note["norm_text2image_score"]
                score_cnt += 1
            if self.use_note_modality == "all_images":
                for note in notes:
                    note["score"] += note["norm_text2images_score"]
                score_cnt += 1
        if "image2text" in self.compute_type:
            for note in notes:
                note["score"] += note["norm_image2text_score"]
            score_cnt += 1
        if "image2image" in self.compute_type:
            if self.use_note_modality == "one_image":
                for note in notes:
                    note["score"] += note["norm_image2image_score"]
                score_cnt += 1
            if self.use_note_modality == "all_images":
                for note in notes:
                    note["score"] = note["norm_image2images_score"]
                score_cnt += 1
        
        if score_cnt != 0:
            for note in notes:
                note["score"] /= score_cnt
        
        return notes
            
