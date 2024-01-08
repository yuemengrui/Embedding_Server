# *_*coding:utf-8 *_*
# @Author : YueMengRui
from mylogger import logger
from fastapi import APIRouter, Request
from configs import API_LIMIT
from info import limiter, bge_reRanker
from fastapi.responses import JSONResponse
from .protocol import ErrorResponse, RerankRequest
from info.utils.response_code import RET, error_map

router = APIRouter()


@router.api_route(path='/ai/retrieval/rerank', methods=['POST'], summary="rerank")
@limiter.limit(API_LIMIT['text_embedding'])
def text_rerank(request: Request,
                req: RerankRequest
                ):
    logger.info(str(req.dict()))
    if bge_reRanker is not None:
        try:
            scores = bge_reRanker.rerank(req.sentence_pairs)
            return JSONResponse({'scores': scores})
        except Exception as e:
            logger.error(str({'EXCEPTION': e}))

    return JSONResponse(ErrorResponse(errcode=RET.SERVERERR, errmsg=error_map[RET.SERVERERR]).dict(),
                        status_code=500)
