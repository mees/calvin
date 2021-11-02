from typing import Dict, List

import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
import torch
from torch.utils.data import DataLoader
import torchvision


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, n_examples: int = 64, window_size: int = 32, split: str = "train", transforms: List = []):
        self.n_examples = n_examples
        self.split = split
        self.data = [
            dict(
                images=torch.rand(window_size, 3, 200, 200),
                observations=torch.rand(window_size, 8),
                actions=torch.rand(window_size, 7),
            )
            for x in range(n_examples)
        ]
        self.transform = torchvision.transforms.Compose(transforms)

    def __getitem__(self, idx):
        x = self.data[idx]
        seq_acts = x["actions"]
        seq_rgb_obs = (x["images"],)
        seq_depth_obs = (x["images"],)
        seq_state_obs = x["observations"]
        seq_lang = torch.empty(0)
        info = {}
        return seq_state_obs, seq_rgb_obs, seq_depth_obs, seq_acts, seq_lang, info, idx

    def __len__(self):
        return self.n_examples


class RandomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, train_transforms: List = [], val_transforms: List = [], **kwargs: Dict):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = RandomDataset(n_examples=32, window_size=16, split="train", transforms=train_transforms)
        self.val_dataset = RandomDataset(n_examples=32, window_size=16, split="val", transforms=val_transforms)
        self.modalities = ["vis"]

    def train_dataloader(self):
        return {"vis": DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=0)}

    def val_dataloader(self):
        val_dataloader = {"vis": DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)}
        return CombinedLoader(val_dataloader, "max_size_cycle")

    @property
    def len_train(self):
        return len(self.train_dataset)

    @property
    def len_valid(self):
        return len(self.val_dataset)
