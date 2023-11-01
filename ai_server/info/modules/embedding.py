# *_*coding:utf-8 *_*
# @Author : YueMengRui
from fastapi import APIRouter, Request
from sklearn.preprocessing import normalize
from info.configs.base_configs import API_LIMIT, EMBEDDING_ENCODE_BATCH_SIZE
from info import embedding_model_dict, logger, limiter
from fastapi.responses import JSONResponse
from .protocol import ErrorResponse, EmbeddingRequest, ModelCard, ModelListResponse, EmbeddingResponse
from info.utils.response_code import RET, error_map

router = APIRouter()


@router.api_route(path='/ai/embedding/model/list', methods=['GET'], response_model=ModelListResponse,
                  summary="获取支持的embedding模型列表")
@limiter.limit(API_LIMIT['model_list'])
def support_embedding_model_list(request: Request):
    model_cards = []
    for i in embedding_model_dict.values():
        model_cards.append(ModelCard(embedding_type=i['embedding_type'],
                                     model_name=i['model_name'],
                                     max_seq_length=i['max_seq_length'],
                                     embedding_dim=i['embedding_dim']))
    return ModelListResponse(data=model_cards)


@router.api_route(path='/ai/embedding/text', methods=['POST'], summary="文本embedding")
@limiter.limit(API_LIMIT['text_embedding'])
def text_embedding(request: Request,
                   req: EmbeddingRequest
                   ):
    logger.info(str(req.dict()))
    embedding_model_name_list = list(embedding_model_dict.keys())
    if req.model_name is None or req.model_name not in embedding_model_name_list:
        req.model_name = embedding_model_name_list[0]

    embedding_model_config = embedding_model_dict[req.model_name]

    try:
        embeddings = embedding_model_config['model'].encode(sentences=req.sentences,
                                                            batch_size=EMBEDDING_ENCODE_BATCH_SIZE)
        embeddings = normalize(embeddings, norm='l2')
        embeddings = [x.tolist() for x in embeddings]
        return EmbeddingResponse(model_name=embedding_model_config['model_name'],
                                 embedding_type=embedding_model_config['embedding_type'],
                                 max_seq_length=embedding_model_config['max_seq_length'],
                                 embedding_dim=embedding_model_config['embedding_dim'],
                                 embeddings=embeddings)
    except Exception as e:
        logger.error(str({'EXCEPTION': e}) + '\n')
        return ErrorResponse(errcode=RET.SERVERERR, errmsg=error_map[RET.SERVERERR])

