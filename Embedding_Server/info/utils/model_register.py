# *_*coding:utf-8 *_*
# @Author : YueMengRui
import requests
from configs import MODEL_REGISTER_URL, THIS_SERVER_HOST
from typing import List
from mylogger import logger


def register_model_to_server(models: List):
    logger.info(f'register model to server')
    for m in models:
        model_name = m.pop('model_name')
        req_data = {
            "type": "embedding",
            "model_name": model_name,
            "url_prefix": THIS_SERVER_HOST,
            "info": m
        }

        _ = requests.post(url=MODEL_REGISTER_URL, json=req_data)
