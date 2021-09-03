import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Sized, Union

import hydra
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, Sampler, SequentialSampler
import torchvision

import calvin
from calvin.datasets.utils.episode_utils import load_dataset_statistics

logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})
ONE_EP_DATASET_URL = "http://www.informatik.uni-freiburg.de/~meeso/50steps.tar.xz"


class PlayDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        root_data_dir: str = "data",
        num_workers: int = 8,
        transforms: DictConfig = DEFAULT_TRANSFORM,
        shuffle_val: bool = False,
        **kwargs: Dict,
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.train_datasets = None
        self.val_datasets = None
        self.train_sampler = None
        self.val_sampler = None
        self.num_workers = num_workers
        root_data_path = Path(root_data_dir)
        if not root_data_path.is_absolute():
            root_data_path = Path(calvin.__file__).parent / root_data_path
        self.training_dir = root_data_path / "training"
        self.val_dir = root_data_path / "validation"
        self.shuffle_val = shuffle_val
        self.modalities: List[str] = []

        transforms = load_dataset_statistics(self.training_dir, self.val_dir, transforms)

        self.train_transforms = {
            cam: [hydra.utils.instantiate(transform) for transform in transforms.train[cam]] for cam in transforms.train
        }

        self.val_transforms = {
            cam: [hydra.utils.instantiate(transform) for transform in transforms.val[cam]] for cam in transforms.val
        }

    def prepare_data(self, *args, **kwargs):
        # check if files already exist
        dataset_exist = np.any([len(list(self.training_dir.glob(extension))) for extension in ["*.npz", "*.pkl"]])

        # download and unpack images
        if not dataset_exist:
            logger.info(f"downloading dataset to {self.training_dir} and {self.val_dir}")
            torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, self.training_dir)
            torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, self.val_dir)

    def setup(self, stage=None):
        self.train_transforms = {key: torchvision.transforms.Compose(val) for key, val in self.train_transforms.items()}
        self.val_transforms = {key: torchvision.transforms.Compose(val) for key, val in self.val_transforms.items()}
        self.train_datasets, self.train_sampler, self.val_datasets, self.val_sampler = {}, {}, {}, {}
        for _, dataset in self.datasets_cfg.items():
            train_dataset = hydra.utils.instantiate(
                dataset, datasets_dir=self.training_dir, transforms=self.train_transforms
            )
            val_dataset = hydra.utils.instantiate(dataset, datasets_dir=self.val_dir, transforms=self.val_transforms)
            train_sampler = get_sampler(train_dataset, shuffle=True)
            val_sampler = get_sampler(val_dataset, shuffle=self.shuffle_val)
            key = dataset.key
            self.train_datasets[key] = train_dataset
            self.train_sampler[key] = train_sampler
            self.val_datasets[key] = val_dataset
            self.val_sampler[key] = val_sampler
            self.modalities.append(key)

    def train_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                sampler=self.train_sampler[key],
                batch_size=dataset.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            for key, dataset in self.train_datasets.items()
        }

    def val_dataloader(self):
        val_dataloaders = {
            key: DataLoader(
                dataset,
                sampler=self.val_sampler[key],
                batch_size=dataset.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            for key, dataset in self.val_datasets.items()
        }
        combined_val_loaders = CombinedLoader(val_dataloaders, "max_size_cycle")
        return combined_val_loaders


def get_sampler(dataset: Dataset, shuffle: bool) -> Sampler:
    if dist.is_available() and dist.is_initialized():
        return DistributedSampler(dataset=dataset, shuffle=shuffle, seed=int(os.environ["PL_GLOBAL_SEED"]))
    elif shuffle:
        return RandomSampler(dataset, generator=torch.Generator().manual_seed(int(os.environ["PL_GLOBAL_SEED"])))  # type: ignore
    else:
        return SequentialSampler(dataset)
