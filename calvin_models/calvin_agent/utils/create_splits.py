import argparse
from pathlib import Path
import shutil
from typing import Dict, List, Tuple

import calvin_agent
import numpy as np
from tqdm import tqdm

TRAINING_DIR: str = "training"
VAL_DIR: str = "validation"


def slice_split(
    ep_lens: np.ndarray,
    ep_start_end_ids: np.ndarray,
    eps_list: List[int],
    idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    val_indx = eps_list[-idx:]
    train_indx = [ep for ep in list(range(0, len(ep_lens))) if ep not in val_indx]
    val_ep_lens = ep_lens[val_indx]
    train_ep_lens = ep_lens[train_indx]
    val_ep_start_end_ids = ep_start_end_ids[val_indx, :]
    train_ep_start_end_ids = ep_start_end_ids[train_indx, :]
    return val_ep_lens, train_ep_lens, val_ep_start_end_ids, train_ep_start_end_ids


def main(input_params: Dict) -> None:

    dataset_root_str, last_k = (
        input_params["dataset_root"],
        input_params["last_K"],
    )
    module_root_path = Path(calvin_agent.__file__)
    dataset_root = module_root_path.parent / Path(dataset_root_str)
    split_data_path = dataset_root
    (split_data_path / TRAINING_DIR).mkdir(parents=True, exist_ok=True)
    (split_data_path / VAL_DIR).mkdir(parents=True, exist_ok=True)
    ep_lens = np.load(dataset_root / "ep_lens.npy")
    ep_start_end_ids = np.load(dataset_root / "ep_start_end_ids.npy")
    eps_list = list(range(0, len(ep_lens)))

    if last_k > 0:
        assert last_k < len(eps_list)
        splits = slice_split(ep_lens, ep_start_end_ids, eps_list, last_k)
        (
            val_ep_lens,
            train_ep_lens,
            val_ep_start_end_ids,
            train_ep_start_end_ids,
        ) = splits
    elif last_k == 0:
        rand_perm = np.random.permutation(len(ep_lens))
        val_size = round(len(eps_list) * 0.1)
        splits = slice_split(ep_lens[rand_perm], ep_start_end_ids[rand_perm], eps_list, val_size)
        (
            val_ep_lens,
            train_ep_lens,
            val_ep_start_end_ids,
            train_ep_start_end_ids,
        ) = splits
    else:
        raise NotImplementedError

    np.save(split_data_path / VAL_DIR / "ep_start_end_ids.npy", val_ep_start_end_ids)
    np.save(split_data_path / TRAINING_DIR / "ep_start_end_ids.npy", train_ep_start_end_ids)
    np.save(split_data_path / VAL_DIR / "ep_lens.npy", val_ep_lens)
    np.save(split_data_path / TRAINING_DIR / "ep_lens.npy", train_ep_lens)

    # copy hydra config folder to training and validation dataset
    shutil.copytree(dataset_root / ".hydra", split_data_path / TRAINING_DIR / ".hydra", dirs_exist_ok=True)
    shutil.copytree(dataset_root / ".hydra", split_data_path / VAL_DIR / ".hydra", dirs_exist_ok=True)

    print("moving files to play_data/validation")
    for x in tqdm(val_ep_start_end_ids):
        range_ids = np.arange(x[0], x[1] + 1)  # to include end frame
        for frame_id in range_ids:
            filename = f"episode_{frame_id:07d}.npz"
            (dataset_root / filename).rename(split_data_path / VAL_DIR / filename)
    print("moving files to play_data/training")
    for x in tqdm(train_ep_start_end_ids):
        range_ids = np.arange(x[0], x[1] + 1)  # to include end frame
        for frame_id in range_ids:
            filename = f"episode_{frame_id:07d}.npz"
            (dataset_root / filename).rename(split_data_path / TRAINING_DIR / filename)
    print("finished creating splits!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="data", help="directory where raw dataset is allocated")
    parser.add_argument(
        "--last_K",
        type=int,
        default="0",
        help="number of last episodes used for validation split, set to 0 for random splits",
    )
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
