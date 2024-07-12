# *_*coding:utf-8 *_*
import os
import sys
import time
from mylogger import logger
from configs import EMBEDDING_MODEL_LIST, BGE_RERANKER_MODEL_NAME_OR_PATH, MODEL_REGISTER
from fastapi.requests import Request
from starlette.middleware.cors import CORSMiddleware
from copy import deepcopy
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sentence_transformers import SentenceTransformer
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles
from info.utils.model_register import register_model_to_server
from info.libs.reranker.bge_reranker import BGEReRanker

limiter = Limiter(key_func=lambda *args, **kwargs: '127.0.0.1')
embedding_model_dict = {}
register_models = []
for embedding_config in deepcopy(EMBEDDING_MODEL_LIST):
    model_name_or_path = embedding_config.pop('model_name_or_path')
    device = embedding_config.pop('device')

    if os.path.exists(model_name_or_path):
        try:
            embedding_model = SentenceTransformer(model_name_or_path=model_name_or_path, device=device)
            logger.info(f"{embedding_config['model_name']} load successful!")
            register_models.append(deepcopy(embedding_config))

            embedding_config.update({"model": embedding_model})
            embedding_model_dict[embedding_config['model_name']] = embedding_config
        except Exception as e:
            logger.error({'EXCEPTION': e})
            continue

if embedding_model_dict == {}:
    logger.error('embedding模型全部加载失败，程序退出！！！')
    sys.exit()

if MODEL_REGISTER:
    register_model_to_server(register_models)

try:
    bge_reRanker = BGEReRanker(model_name_or_path=BGE_RERANKER_MODEL_NAME_OR_PATH)
except Exception as e:
    logger.error('bge_reranker load error!!! ' + str(e))
    bge_reRanker = None


def app_registry(app):
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @app.middleware("http")
    async def api_time_cost(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        cost = time.time() - start
        logger.info(f'end request "{request.method} {request.url.path}" - {cost:.3f}s')
        return response

    app.mount("/ai/embedding/static", StaticFiles(directory=f"static"), name="static")

    @app.get("/ai/embedding/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url="/ai/embedding/openapi.json",
            title=app.title + " - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="/ai/embedding/static/swagger-ui-bundle.js",
            swagger_css_url="/ai/embedding/static/swagger-ui.css",
        )

    @app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
    async def swagger_ui_redirect():
        return get_swagger_ui_oauth2_redirect_html()

    @app.get("/ai/embedding/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url="/ai/embedding/openapi.json",
            title=app.title + " - ReDoc",
            redoc_js_url="/ai/embedding/static/redoc.standalone.js",
        )

    from info.modules import register_router

    register_router(app)
