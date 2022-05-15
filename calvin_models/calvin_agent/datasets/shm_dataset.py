import logging
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional

from calvin_agent.datasets.base_dataset import BaseDataset
import numpy as np

logger = logging.getLogger(__name__)


class ShmDataset(BaseDataset):
    """
    Dataset that loads episodes from shared memory.
    """

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)
        self.episode_lookup_dict: Dict[str, List] = {}
        self.episode_lookup: Optional[np.ndarray] = None
        self.lang_lookup = None
        self.lang_ann = None
        self.shapes = None
        self.sizes = None
        self.dtypes = None
        self.dataset_type = None
        self.shared_memories = None

    def setup_shm_lookup(self, shm_lookup: Dict) -> None:
        """
        Initialize episode lookups.

        Args:
            shm_lookup: Dictionary containing precomputed lookups.
        """
        if self.with_lang:
            self.episode_lookup_dict = shm_lookup["episode_lookup_lang"]
            self.lang_lookup = shm_lookup["lang_lookup"]
            self.lang_ann = shm_lookup["lang_ann"]
        else:
            self.episode_lookup_dict = shm_lookup["episode_lookup_vision"]
        key = list(self.episode_lookup_dict.keys())[0]
        self.episode_lookup = np.array(self.episode_lookup_dict[key])[:, 1]
        self.shapes = shm_lookup["shapes"]
        self.sizes = shm_lookup["sizes"]
        self.dtypes = shm_lookup["dtypes"]
        self.dataset_type = "train" if "training" in self.abs_datasets_dir.as_posix() else "val"
        # attach to shared memories
        self.shared_memories = {
            key: SharedMemory(name=f"{self.dataset_type}_{key}") for key in self.episode_lookup_dict
        }

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames from shared memory and combine to episode dict.

        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.

        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        episode = {}
        for key, lookup in self.episode_lookup_dict.items():
            offset, j = lookup[idx]
            shape = (window_size + j,) + self.shapes[key]
            array = np.ndarray(shape, dtype=self.dtypes[key], buffer=self.shared_memories[key].buf, offset=offset)[j:]  # type: ignore
            episode[key] = array
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]  # TODO check  [0]
        return episode
