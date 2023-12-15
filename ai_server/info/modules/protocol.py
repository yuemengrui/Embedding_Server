# *_*coding:utf-8 *_*
# @Author : YueMengRui
from pydantic import BaseModel, Field
from typing import Dict, List


class ErrorResponse(BaseModel):
    object: str = "error"
    errcode: int
    errmsg: str


class ModelCard(BaseModel):
    embedding_type: str
    model_name: str
    max_seq_length: int
    embedding_dim: int


class ModelListResponse(BaseModel):
    object: str = "embedding_model_list"
    data: List[ModelCard] = []


class EmbeddingRequest(BaseModel):
    model_name: str = Field(default=None, description="模型名称")
    sentences: List[str] = Field(description="句子列表")


class EmbeddingResponse(BaseModel):
    object: str = "embedding"
    embedding_type: str
    model_name: str
    max_seq_length: int
    embedding_dim: int
    embeddings: List


class RerankRequest(BaseModel):
    sentence_pairs: List = Field(description="句子对列表")


class TokenCountResponse(BaseModel):
    object: str = 'token_count'
    token_counts: List[int]
