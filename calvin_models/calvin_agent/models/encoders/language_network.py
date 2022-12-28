from typing import List

from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn


class SBert(nn.Module):
    def __init__(self, nlp_model: str):
        #  choose model from https://www.sbert.net/docs/pretrained_models.html
        super().__init__()
        assert isinstance(nlp_model, str)
        self.model = SentenceTransformer(nlp_model)

    def forward(self, x: List) -> torch.Tensor:
        emb = self.model.encode(x, convert_to_tensor=True)
        return torch.unsqueeze(emb, 1)