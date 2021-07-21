from typing import Tuple

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer


class Bert(nn.Module):
    def __init__(self, freeze_bert=True):
        super().__init__()
        config = BertConfig()
        config.return_dict = True
        self.bert = BertModel(config).from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.MAX_LEN = 64
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, x: str) -> torch.Tensor:
        inp, att = self.preprocessing(x)
        output = self.bert(inp, att)
        return output["pooler_output"]  # (batch, 768)

    def preprocessing(self, x: str) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = []
        attention_masks = []
        for sent in x:
            encoded_sent = self.tokenizer.encode_plus(
                text=sent,  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=self.MAX_LEN,  # Max length to truncate/pad
                padding="max_length",  # Pad sentence to max length
                truncation=True,
                return_attention_mask=True,  # Return attention mask
            )
            input_ids.append(encoded_sent.get("input_ids"))
            attention_masks.append(encoded_sent.get("attention_mask"))

        input_ids_tensor = torch.tensor(input_ids).to(self.bert.device)
        attention_masks_tensor = torch.tensor(attention_masks).to(self.bert.device)

        return input_ids_tensor, attention_masks_tensor
