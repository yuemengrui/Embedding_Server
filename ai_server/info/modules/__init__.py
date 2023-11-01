# *_*coding:utf-8 *_*
from fastapi import FastAPI
from . import embedding


def register_router(app: FastAPI):
    app.include_router(router=embedding.router, prefix="", tags=['Embedding'])
