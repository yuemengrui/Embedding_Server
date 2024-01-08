#!/bin/bash
cd /workspace/Embedding_Server && CUDA_VISIBLE_DEVICES=0 nohup python manage_embedding_server.py >/dev/null 2>&1 &
