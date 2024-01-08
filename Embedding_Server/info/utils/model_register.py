# *_*coding:utf-8 *_*
# @Author : YueMengRui
import requests
from configs import MODEL_REGISTER_URL, THIS_SERVER_HOST
from typing import List


def register_model_to_server(models: List):
    for m in models:
        model_name = m.pop('model_name')
        req_data = {
            "type": "embedding",
            "model_name": model_name,
            "url_prefix": THIS_SERVER_HOST,
            "info": m
        }

        _ = requests.post(url=MODEL_REGISTER_URL, json=req_data)
