# *_*coding:utf-8 *_*
import os
import sys
import time
from info.configs import *
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from copy import deepcopy
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from info.utils.logger import MyLogger
from sentence_transformers import SentenceTransformer

limiter = Limiter(key_func=lambda *args, **kwargs: '127.0.0.1')
app = FastAPI(title="Embedding Server")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

logger = MyLogger()


@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"start request {request.method} {request.url.path}")
    start = time.time()

    response = await call_next(request)

    cost = time.time() - start
    logger.info(f"end request {request.method} {request.url.path} {cost:.3f}s")
    return response


embedding_model_dict = {}
for embedding_config in deepcopy(EMBEDDING_MODEL_LIST):
    model_name_or_path = embedding_config.pop('model_name_or_path')
    device = embedding_config.pop('device')

    if os.path.exists(model_name_or_path):
        embedding_model = SentenceTransformer(model_name_or_path=model_name_or_path, device=device)

        embedding_config.update({"model": embedding_model})
        embedding_model_dict[embedding_config['model_name']] = embedding_config

if embedding_model_dict == {}:
    logger.error('embedding模型加载失败，程序退出！！！')
    sys.exit()

from info.modules import register_router

register_router(app)