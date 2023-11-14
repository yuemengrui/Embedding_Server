# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os

FASTAPI_TITLE = 'Embedding_Server'
FASTAPI_HOST = '0.0.0.0'
FASTAPI_PORT = 5000

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

EMBEDDING_ENCODE_BATCH_SIZE = 8

EMBEDDING_MODEL_LIST = [
    {
        "embedding_type": "text",
        "model_name": "m3e_base",
        "max_seq_length": 512,
        "embedding_dim": 768,
        "model_name_or_path": "",
        "device": "cuda"
    },
    {
        "embedding_type": "text",
        "model_name": "text2vec_large_chinese",
        "max_seq_length": 512,
        "embedding_dim": 1024,
        "model_name_or_path": "",
        "device": "cuda"
    },
    {
        "embedding_type": "text",
        "model_name": "bge_large_zh",
        "max_seq_length": 512,
        "embedding_dim": 768,
        "instruction": "为这个句子生成表示以用于检索相关文章：",
        "model_name_or_path": "",
        "device": "cuda"
    },
    {
        "embedding_type": "text",
        "model_name": "gte_large_zh",
        "max_seq_length": 512,
        "embedding_dim": 1024,
        "model_name_or_path": "",
        "device": "cuda"
    }
]

BGE_RERANKER_MODEL_NAME_OR_PATH = ''

# API LIMIT
API_LIMIT = {
    "model_list": "120/minute",
    "text_embedding": "60/minute",
}
