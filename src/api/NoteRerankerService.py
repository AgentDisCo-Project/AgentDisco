import os
import copy
import gin
import sys
sys.path.append('.')

from api.CustomizeScoreGeneratorService import CustomizeScoreGenerator
from api.CustomizeCorpusService import CustomizeCorpus
from utils.urls import RERANKER_JINA_URL


@gin.configurable()
class NoteReranker:
    def __init__(
        self,
        need_chunk: bool = False,
        chunk_size: int = None,
        max_num_chunks_per_request: int = None,
        use_chunk_size: int = None,
        need_overlap: bool = False,
        use_query_modality: str = "",
        use_note_modality: str = "",
        image_pool_type: str = "max",
        image_key_type: str = "url",
        need_remove_empty_notes: bool = True,
        compute_type: list = None,
    ):
        self.reranker = CustomizeScoreGenerator(
            use_customize_url=True,
            customize_url=RERANKER_JINA_URL,
            use_query_modality=use_query_modality,
            use_note_modality=use_note_modality,
            image_pool_type=image_pool_type,
            image_key_type=image_key_type,
            compute_type=compute_type,
        )
        
        self.corpus = CustomizeCorpus(
            need_chunk=need_chunk,
            chunk_size=chunk_size,
            max_num_chunks_per_request=max_num_chunks_per_request,
            use_chunk_size=use_chunk_size,
            need_overlap=need_overlap,
        )
        
        self.need_remove_empty_notes = need_remove_empty_notes
    
    
    async def act(
        self,
        input_dict: dict,
        input_key: str = "search_results",
        output_key: str = "scored_search_results",
    ):
        query_text = input_dict.get("query_text", "")
        query_image = input_dict.get("query_image", "")
        notes = input_dict[input_key]
        
        if self.reranker.use_note_modality == "text" and self.need_remove_empty_notes:
            new_notes = []
            for note in notes:
                if self.corpus.remove_empty_texts(note):
                    continue
                new_notes.append(note)
            notes = new_notes
        
        if self.reranker.use_note_modality in ("text", "both") and self.corpus.need_chunk:
            new_notes = []
            input_texts = []
            for note in notes:
                input_text = self.reranker.get_text_from_note(note)
                chunks = self.corpus.divide_into_chunks(input_text)
                if not isinstance(chunks, list):
                    chunks = [chunks]
                input_texts.extend(chunks)
                for chunk_id, chunk in enumerate(chunks):
                    new_note = copy.deepcopy(note)
                    new_note["chunk_id"] = chunk_id
                    new_note["chunk"] = chunk
                    new_notes.append(new_note)
            notes = new_notes
        
        else:
            input_texts = [self.reranker.get_text_from_note(note) for note in notes]
        
        if self.reranker.use_note_modality in ("one_image", "all_images"):
            input_images = [self.reranker.get_image_from_note(note) for note in notes]
        else:
            input_images = None
        
        scored_notes = await self.reranker.act(
            notes=notes,
            query_text=query_text,
            query_image=query_image,
            input_texts=input_texts,
            input_images=input_images,
        )
        
        input_dict[output_key] = scored_notes
        return input_dict
