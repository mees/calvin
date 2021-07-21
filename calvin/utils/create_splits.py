import argparse
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import torch

import lfp

TRAINING_DIR: str = "training"
VAL_DIR: str = "validation"
MyLangDictType = TypedDict("MyLangDictType", {"language": torch.Tensor, "indx": List[int], "eps": List[int]})


def add_eps_lang(lang_data: Dict[str, Any], ep_start_end_ids: np.ndarray) -> MyLangDictType:
    lang_eps: List = []
    for (start, end) in lang_data["indx"]:
        done = False
        for k, lims in enumerate(ep_start_end_ids):
            if done:
                break
            if (start >= lims[0]) & (start <= lims[1]):
                lang_eps.append(k)
                done = True
                break
        # lang_data["eps"] = lang_eps
    assert isinstance(lang_data["language"], torch.Tensor)
    assert isinstance(lang_data["indx"], list)
    new_typed_lang_dict: MyLangDictType = {
        "language": lang_data["language"],
        "indx": lang_data["indx"],
        "eps": lang_eps,
    }
    return new_typed_lang_dict


def slice_split(
    ep_lens: np.ndarray,
    ep_start_end_ids: np.ndarray,
    eps_list: List[int],
    lang_data: Optional[MyLangDictType],
    idx: int,
    with_lang: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[Dict[str, Union[List, torch.Tensor]]],
    Optional[Dict[str, Union[List, torch.Tensor]]],
]:
    val_indx = eps_list[-idx:]
    train_indx = [ep for ep in list(range(0, len(ep_lens))) if ep not in val_indx]
    val_ep_lens = ep_lens[val_indx]
    train_ep_lens = ep_lens[train_indx]
    val_ep_start_end_ids = ep_start_end_ids[val_indx, :]
    train_ep_start_end_ids = ep_start_end_ids[train_indx, :]
    val_lang_data: Optional[Dict[str, Union[List, torch.Tensor]]]
    train_lang_data: Optional[Dict[str, Union[List, torch.Tensor]]]
    if with_lang:
        assert lang_data
        val_lang_indx = [i for i, ep in enumerate(lang_data["eps"]) if ep in val_indx]
        train_lang_indx = [i for i, ep in enumerate(lang_data["eps"]) if ep not in val_indx]
        val_lang_data = {
            "language": lang_data["language"][torch.Tensor(val_lang_indx).long()],
            "indx": [lang_data["indx"][i] for i in val_lang_indx],
        }
        train_lang_data = {
            "language": lang_data["language"][torch.Tensor(train_lang_indx).long()],
            "indx": [lang_data["indx"][i] for i in train_lang_indx],
        }
    else:
        val_lang_data = None
        train_lang_data = None
    return val_ep_lens, train_ep_lens, val_ep_start_end_ids, train_ep_start_end_ids, val_lang_data, train_lang_data


def main(input_params: Dict) -> None:

    dataset_root_str, last_k, with_lang, filename_lang_ann = (
        input_params["dataset_root"],
        input_params["last_K"],
        input_params["lang"],
        input_params["file_name_lang"],
    )
    module_root_path = Path(calvin.__file__)
    dataset_root = module_root_path.parent / Path(dataset_root_str)
    split_data_path = dataset_root
    (split_data_path / TRAINING_DIR).mkdir(parents=True, exist_ok=True)
    (split_data_path / VAL_DIR).mkdir(parents=True, exist_ok=True)
    ep_lens = np.load(dataset_root / "ep_lens.npy")
    ep_start_end_ids = np.load(dataset_root / "ep_start_end_ids.npy")
    lang_data: Optional[MyLangDictType]
    if with_lang:
        lang_data_ = np.load(dataset_root / filename_lang_ann, allow_pickle=True).reshape(-1)[0]
        lang_data = add_eps_lang(lang_data_, ep_start_end_ids)
        eps_list = list(set(lang_data["eps"]))
    else:
        eps_list = list(range(0, len(ep_lens)))
        lang_data = None

    if last_k > 0:
        assert last_k < len(eps_list)
        splits = slice_split(ep_lens, ep_start_end_ids, eps_list, lang_data, last_k, with_lang)
        (
            val_ep_lens,
            train_ep_lens,
            val_ep_start_end_ids,
            train_ep_start_end_ids,
            val_lang_data,
            train_lang_data,
        ) = splits
    elif last_k == 0:
        rand_perm = np.random.permutation(len(ep_lens))
        val_size = round(len(eps_list) * 0.1)
        splits = slice_split(ep_lens[rand_perm], ep_start_end_ids[rand_perm], eps_list, lang_data, val_size, with_lang)
        (
            val_ep_lens,
            train_ep_lens,
            val_ep_start_end_ids,
            train_ep_start_end_ids,
            val_lang_data,
            train_lang_data,
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

    if with_lang:
        assert val_lang_data
        assert train_lang_data
        np.save(split_data_path / VAL_DIR / filename_lang_ann, val_lang_data)
        np.save(split_data_path / TRAINING_DIR / filename_lang_ann, train_lang_data)

    print("moving files to play_data/validation")
    print("-------")
    for x in val_ep_start_end_ids:
        range_ids = np.arange(x[0], x[1] + 1)  # to include end frame
        for frame_id in range_ids:
            filename = f"episode_{frame_id:06d}.npz"
            (dataset_root / filename).rename(split_data_path / VAL_DIR / filename)
    print("moving files to play_data/training")
    print("-------")
    for x in train_ep_start_end_ids:
        range_ids = np.arange(x[0], x[1] + 1)  # to include end frame
        for frame_id in range_ids:
            filename = f"episode_{frame_id:06d}.npz"
            (dataset_root / filename).rename(split_data_path / TRAINING_DIR / filename)
    print("finished creating splits!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="data", help="directory where raw dataset is allocated")
    parser.add_argument("--lang", action="store_true", help="Use flag with lang datasets, default no language")
    parser.add_argument(
        "--file_name_lang", type=str, default="lang_emb_ann.npy", help="language annotations .npy filename"
    )
    parser.add_argument(
        "--last_K",
        type=int,
        default="0",
        help="number of last episodes used for validation split, set to 0 for random splits",
    )
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
