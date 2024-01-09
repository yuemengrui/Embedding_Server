# *_*coding:utf-8 *_*
from fastapi import FastAPI
from . import embedding, rerank, health


def register_router(app: FastAPI):
    app.include_router(router=embedding.router, prefix="", tags=['Embedding'])
    app.include_router(router=rerank.router, prefix="", tags=['reRank'])
    app.include_router(router=health.router, prefix="", tags=['health'])
