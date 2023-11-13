# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
from tqdm.autonotebook import trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEReRanker:

    def __init__(self, model_name_or_path: str, device='cuda', batch_size=16):
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if device == 'cuda':
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).cuda()
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

        self.model.eval()

    def rerank(self, sentence_pairs, **kwargs):
        all_scores = []
        for start_index in trange(0, len(sentence_pairs), self.batch_size, disable=True):
            sentences_batch = sentence_pairs[start_index:start_index + self.batch_size]

            with torch.no_grad():
                inputs = self.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors='pt',
                                        max_length=512)
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().numpy().tolist()
                all_scores.extend(scores)

        return all_scores
