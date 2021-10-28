import logging
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from calvin.datasets.base_dataset import BaseDataset
from calvin.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    process_rgb,
    process_state,
)

logger = logging.getLogger(__name__)


class NpzDataset(BaseDataset):
    """
    Dataset Loader that uses a shared memory cache

    parameters
    ----------

    datasets_dir:       path of folder containing episode files (string must contain 'validation' or 'training')
    save_format:        format of episodes in datasets_dir (.pkl or .npz)
    obs_space:          DictConfig of the observation modalities of the dataset
    max_window_size:    maximum length of the episodes sampled from the dataset
    """

    def __init__(self, *args, skip_frames: int = 0, n_digits: Optional[int] = None, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)
        self.skip_frames = skip_frames
        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_lookup,
                self.max_batched_length_per_demo,
                self.lang_ann,
            ) = self.load_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup, self.max_batched_length_per_demo = self.load_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = self.lookup_naming_pattern(n_digits)

    def lookup_naming_pattern(self, n_digits):
        it = os.scandir(self.abs_datasets_dir)
        while True:
            filename = Path(next(it))
            if self.save_format in filename.suffix:
                break
        aux_naming_pattern = re.split(r"\d+", filename.stem)
        naming_pattern = [filename.parent / aux_naming_pattern[0], filename.suffix]
        n_digits = n_digits if n_digits is not None else len(re.findall(r"\d+", filename.stem)[0])
        assert len(naming_pattern) == 2
        assert n_digits > 0
        return naming_pattern, n_digits

    def get_episode_name(self, idx: int) -> Path:
        """
        Convert frame idx to file name
        """
        return Path(f"{self.naming_pattern[0]}{idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def zip_sequence(self, start_idx: int, end_idx: int, idx: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive individual frames saved as npy files and combine to episode dict
        parameters:
        -----------
        start_idx: index of first frame
        end_idx: index of last frame

        returns:
        -----------
        episode: dict of numpy arrays containing the episode where keys are the names of modalities
        """
        episodes = [self.load_episode(self.get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx)]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key, _ in episodes[0].items()}
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]  # TODO check  [0]
        return episode

    def get_sequences(self, idx: int, window_size: int) -> Dict:
        """
        parameters
        ----------
        idx: index of starting frame
        window_size:    length of sampled episode

        returns
        ----------
        seq_state_obs:  numpy array of state observations
        seq_rgb_obs:    tuple of numpy arrays of rgb observations
        seq_depth_obs:  tuple of numpy arrays of depths observations
        seq_acts:       numpy array of actions
        """

        start_file_indx = self.episode_lookup[idx]
        end_file_indx = start_file_indx + window_size

        episode = self.zip_sequence(start_file_indx, end_file_indx, idx)

        seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = {"lang": torch.from_numpy(episode["language"]) if self.with_lang else torch.empty(0)}
        seq_dict = {**seq_state_obs, **seq_rgb_obs, **seq_depth_obs, **seq_acts, **info, **seq_lang}
        seq_dict["idx"] = idx

        return seq_dict

    def load_file_indices_lang(self, abs_datasets_dir: Path) -> Tuple[List, List, List, np.ndarray]:

        """
        this method builds the mapping from index to file_name used for loading the episodes

        parameters
        ----------
        abs_datasets_dir:               absolute path of the directory containing the dataset

        returns
        ----------
        episode_lookup:                 list for the mapping from training example index to episode (file) index
        max_batched_length_per_demo:    list of possible starting indices per episode
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        try:
            print("trying to load lang data from: ", abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).reshape(
                -1
            )[0]
        except Exception:
            print("Exception, trying to load lang data from: ", abs_datasets_dir / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True).reshape(-1)[0]

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["emb"]  # length total number of annotations
        lang_lookup = []
        max_batched_length_per_demo = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            assert end_idx >= self.max_window_size
            cnt = 0
            for idx in range(start_idx, end_idx + 1 - self.max_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1
            possible_indices = end_idx + 1 - start_idx - self.max_window_size  # TODO: check it for skip_frames
            max_batched_length_per_demo.append(possible_indices)

        return episode_lookup, lang_lookup, max_batched_length_per_demo, lang_ann

    def load_file_indices(self, abs_datasets_dir: Path) -> Tuple[List, List]:
        """
        this method builds the mapping from index to file_name used for loading the episodes

        parameters
        ----------
        abs_datasets_dir:               absolute path of the directory containing the dataset

        returns
        ----------
        episode_lookup:                 list for the mapping from training example index to episode (file) index
        max_batched_length_per_demo:    list of possible starting indices per episode
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        max_batched_length_per_demo = []
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.max_window_size):
                episode_lookup.append(idx)
            possible_indices = end_idx + 1 - start_idx - self.max_window_size
            max_batched_length_per_demo.append(possible_indices)
        return episode_lookup, max_batched_length_per_demo
