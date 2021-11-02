from abc import abstractmethod
import logging
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Union

import numpy as np
from omegaconf import DictConfig
import pyhash
import torch
from torch.utils.data import Dataset

hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


def get_validation_window_size(idx, min_window_size, max_window_size):
    window_range = max_window_size - min_window_size + 1
    return min_window_size + hasher(str(idx)) % window_range


class BaseDataset(Dataset):
    """Common dataset loader class"""

    def __init__(
        self,
        datasets_dir: Path,
        obs_space: DictConfig,
        proprio_state: DictConfig,
        key: str,
        lang_folder: str,
        save_format: str = None,
        transforms: Dict = {},
        batch_size: int = 32,
        min_window_size: int = 16,
        max_window_size: int = 32,
        pad: bool = True,
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.save_format = save_format
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        if self.save_format == "pkl":
            self.load_episode = load_pkl
        elif self.save_format == "npz":
            self.load_episode = load_npz
        else:
            raise NotImplementedError
        self.pad = pad
        self.batch_size = batch_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.abs_datasets_dir = datasets_dir
        self.lang_folder = lang_folder  # if self.with_lang else None
        assert "validation" in self.abs_datasets_dir.as_posix() or "training" in self.abs_datasets_dir.as_posix()
        self.validation = "validation" in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir()
        self.episode_lookup: List[int] = []
        logger.info(f"loading dataset at {self.abs_datasets_dir}")
        logger.info("finished loading dataset")

    @property
    def is_varying(self) -> bool:
        return self.min_window_size != self.max_window_size and not self.pad

    @abstractmethod
    def get_sequences(self, idx: int, window_size: int) -> Dict:
        pass

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        if isinstance(idx, int):
            # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
            # acts like Constant dataset. Currently, used for language data
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                if self.validation:
                    window_size = get_validation_window_size(idx, self.min_window_size, self.max_window_size)
                else:
                    window_size = np.random.randint(self.min_window_size, self.max_window_size + 1)
            else:
                logger.error(f"min_window_size {self.min_window_size} > max_window_size {self.max_window_size}")
                raise ValueError
        else:
            idx, window_size = idx
        sequence = self.get_sequences(idx, window_size)
        if self.pad:
            sequence = self.pad_sequence(sequence, window_size)
        return sequence

    def __len__(self) -> int:
        """
        returns
        ----------
        number of possible starting frames
        """
        return len(self.episode_lookup)

    def pad_sequence(
        self,
        seq: Dict,
        window_size: int,
    ) -> Dict:
        pad_size = self.max_window_size - window_size
        seq.update({"robot_obs": self.pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update({"rgb_obs": {k: self.pad_with_repetition(v, pad_size) for k, v in seq["rgb_obs"].items()}})
        seq.update({"depth_obs": {k: self.pad_with_repetition(v, pad_size) for k, v in seq["depth_obs"].items()}})
        #  todo: find better way of distinguishing rk and play action spaces
        if self.save_format == "npz" and not self.relative_actions:
            # repeat action for world coordinates action space
            seq.update({"actions": self.pad_with_repetition(seq["actions"], pad_size)})
        elif self.save_format == "npz" and self.relative_actions:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            seq_acts = torch.cat(
                [
                    self.pad_with_zeros(seq["actions"][..., :-1], pad_size),
                    self.pad_with_repetition(seq["actions"][..., -1:], pad_size),
                ],
                dim=-1,
            )
            seq.update({"actions": seq_acts})
        else:
            # set action to zero for joints action space
            seq.update({"actions": self.pad_with_zeros(seq["actions"], pad_size)})
        seq.update({"state_info": {k: self.pad_with_repetition(v, pad_size) for k, v in seq["state_info"].items()}})
        return seq

    @staticmethod
    def pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """repeats the last element with pad_size"""
        last_repeated = torch.repeat_interleave(torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0)
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """repeats the last element with pad_size"""
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded
