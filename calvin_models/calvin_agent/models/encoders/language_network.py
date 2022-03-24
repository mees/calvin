from typing import List

from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn


class SBert(nn.Module):
    def __init__(self, nlp_model):
        super().__init__()
        if nlp_model == "mpnet":
            weights = "paraphrase-mpnet-base-v2"
        elif nlp_model == "multi":
            weights = "paraphrase-multilingual-mpnet-base-v2"
        else:
            weights = "paraphrase-MiniLM-L6-v2"
        self.model = SentenceTransformer(weights)

    def forward(self, x: List) -> torch.Tensor:
        emb = self.model.encode(x, convert_to_tensor=True)
        return torch.unsqueeze(emb, 1)
