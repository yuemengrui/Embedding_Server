# *_*coding:utf-8 *_*
# @Author : YueMengRui
from mylogger import logger
from fastapi import APIRouter, Request
from configs import API_LIMIT, EMBEDDING_ENCODE_BATCH_SIZE
from info import embedding_model_dict, limiter
from fastapi.responses import JSONResponse
from .protocol import ErrorResponse, EmbeddingRequest, ModelCard, ModelListResponse, EmbeddingResponse, \
    TokenCountResponse
from info.utils.response_code import RET, error_map

router = APIRouter()


@router.api_route(path='/model/list', methods=['GET'], response_model=ModelListResponse,
                  summary="获取支持的embedding模型列表")
@limiter.limit(API_LIMIT['model_list'])
def support_embedding_model_list(request: Request):
    model_cards = []
    for i in embedding_model_dict.values():
        model_cards.append(ModelCard(embedding_type=i['embedding_type'],
                                     model_name=i['model_name'],
                                     max_seq_length=i['max_seq_length'],
                                     embedding_dim=i['embedding_dim']))
    return JSONResponse(ModelListResponse(data=model_cards).dict())


@router.api_route(path='/text', methods=['POST'], response_model=EmbeddingResponse,
                  summary="文本embedding")
@limiter.limit(API_LIMIT['text_embedding'])
def text_embedding(request: Request,
                   req: EmbeddingRequest
                   ):
    logger.info(str(req.dict()))

    if req.model_name not in list(embedding_model_dict.keys()):
        return JSONResponse(ErrorResponse(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR]).dict(),
                            status_code=412)

    embedding_model_config = embedding_model_dict[req.model_name]

    try:
        embeddings = embedding_model_config['model'].encode(sentences=req.sentences,
                                                            batch_size=EMBEDDING_ENCODE_BATCH_SIZE,
                                                            normalize_embeddings=True)

        embeddings = [x.tolist() for x in embeddings]
        return JSONResponse(EmbeddingResponse(model_name=embedding_model_config['model_name'],
                                              embedding_type=embedding_model_config['embedding_type'],
                                              max_seq_length=embedding_model_config['max_seq_length'],
                                              embedding_dim=embedding_model_config['embedding_dim'],
                                              embeddings=embeddings).dict())
    except Exception as e:
        logger.error(str({'EXCEPTION': e}))
        return JSONResponse(ErrorResponse(errcode=RET.SERVERERR, errmsg=error_map[RET.SERVERERR]).dict(),
                            status_code=500)


@router.api_route(path='/token/count', methods=['POST'], response_model=TokenCountResponse,
                  summary="Embedding token count")
@limiter.limit(API_LIMIT['text_embedding'])
def text_embedding_token_count(request: Request,
                               req: EmbeddingRequest
                               ):
    logger.info(str(req.dict()))

    if req.model_name not in list(embedding_model_dict.keys()):
        return JSONResponse(ErrorResponse(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR]).dict(),
                            status_code=412)

    embedding_model_config = embedding_model_dict[req.model_name]

    try:
        token_counts = [len(embedding_model_config['model'].tokenizer.tokenize(s)) for s in req.sentences]
        return JSONResponse(TokenCountResponse(model_name=embedding_model_config['model_name'],
                                               token_counts=token_counts,
                                               max_seq_length=embedding_model_config['max_seq_length']
                                               ).dict())
    except Exception as e:
        logger.error(str({'EXCEPTION': e}))
        return JSONResponse(ErrorResponse(errcode=RET.SERVERERR, errmsg=error_map[RET.SERVERERR]).dict(),
                            status_code=500)
