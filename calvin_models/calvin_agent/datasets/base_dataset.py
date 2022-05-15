import logging
from pathlib import Path
from typing import Dict, Tuple, Union

from calvin_agent.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    process_language,
    process_rgb,
    process_state,
)
import numpy as np
from omegaconf import DictConfig
import pyhash
import torch
from torch.utils.data import Dataset

hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)


def get_validation_window_size(idx: int, min_window_size: int, max_window_size: int) -> int:
    """
    In validation step, use hash function instead of random sampling for consistent window sizes across epochs.

    Args:
        idx: Sequence index.
        min_window_size: Minimum window size.
        max_window_size: Maximum window size.

    Returns:
        Window size computed with hash function.
    """
    window_range = max_window_size - min_window_size + 1
    return min_window_size + hasher(str(idx)) % window_range


class BaseDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
    """

    def __init__(
        self,
        datasets_dir: Path,
        obs_space: DictConfig,
        proprio_state: DictConfig,
        key: str,
        lang_folder: str,
        num_workers: int,
        transforms: Dict = {},
        batch_size: int = 32,
        min_window_size: int = 16,
        max_window_size: int = 32,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]

        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.abs_datasets_dir = datasets_dir
        self.lang_folder = lang_folder  # if self.with_lang else None
        self.aux_lang_loss_window = aux_lang_loss_window
        assert "validation" in self.abs_datasets_dir.as_posix() or "training" in self.abs_datasets_dir.as_posix()
        self.validation = "validation" in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir()
        logger.info(f"loading dataset at {self.abs_datasets_dir}")
        logger.info("finished loading dataset")

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
            # acts like Constant dataset. Currently, used for language data
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                window_size = self._get_window_size(idx)
            else:
                logger.error(f"min_window_size {self.min_window_size} > max_window_size {self.max_window_size}")
                raise ValueError
        else:
            idx, window_size = idx
        sequence = self._get_sequences(idx, window_size)
        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size)
        return sequence

    def _get_sequences(self, idx: int, window_size: int) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx, window_size)

        seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {**seq_state_obs, **seq_rgb_obs, **seq_depth_obs, **seq_acts, **info, **seq_lang}  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int:
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif self.episode_lookup[idx + window_diff] != self.episode_lookup[idx] + window_diff:
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(self.max_window_size, (self.min_window_size + steps_to_next_episode - 1))
        else:
            max_window = self.max_window_size

        if self.validation:
            # in validation step, repeat the window sizes for each epoch.
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update({"rgb_obs": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["rgb_obs"].items()}})
        seq.update({"depth_obs": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["depth_obs"].items()}})
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            # repeat action for world coordinates action space
            seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            seq_acts = torch.cat(
                [
                    self._pad_with_zeros(seq["actions"][..., :-1], pad_size),
                    self._pad_with_repetition(seq["actions"][..., -1:], pad_size),
                ],
                dim=-1,
            )
            seq.update({"actions": seq_acts})
        seq.update({"state_info": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["state_info"].items()}})
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        last_repeated = torch.repeat_interleave(torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0)
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info
