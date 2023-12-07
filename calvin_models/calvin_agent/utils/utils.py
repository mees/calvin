import os
from pathlib import Path
import shutil
import time
from typing import Dict, List, Union

import cv2
import git
import hydra
import numpy as np
import pytorch_lightning
import torch
import tqdm


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def get_git_commit_hash(repo_path: Path) -> str:
    try:
        repo = git.Repo(search_parent_directories=True, path=repo_path.parent)
    except git.exc.InvalidGitRepositoryError:
        return "Not a git repository. Are you using pycharm remote interpreter?"

    changed_files = [item.a_path for item in repo.index.diff(None)]
    if changed_files:
        print("WARNING uncommitted modified files: {}".format(",".join(changed_files)))
    return repo.head.object.hexsha


def get_checkpoints_for_epochs(experiment_folder: Path, epochs: Union[List, str]) -> List:
    if isinstance(epochs, str):
        epochs = epochs.split(",")
        epochs = list(map(int, epochs))
    ep = lambda s: int(s.stem.split("=")[1])
    return [chk for chk in get_all_checkpoints(experiment_folder) if ep(chk) in epochs]


def get_all_checkpoints(experiment_folder: Path) -> List:
    if not experiment_folder.is_dir():
        return []
    checkpoint_folder = experiment_folder / "saved_models"
    if checkpoint_folder.is_dir():
        return get_all_checkpoints(checkpoint_folder)

    checkpoints = sorted(Path(experiment_folder).iterdir(), key=lambda chk: chk.stat().st_mtime)
    if len(checkpoints):
        return [chk for chk in checkpoints if chk.suffix == ".ckpt"]
    return []


def get_last_checkpoint(experiment_folder: Path) -> Union[Path, None]:
    # return newest checkpoint according to creation time
    checkpoints = get_all_checkpoints(experiment_folder)
    if len(checkpoints):
        return checkpoints[-1]
    return None


def info_cuda() -> Dict[str, Union[str, List[str]]]:
    return {
        "GPU": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        # 'nvidia_driver': get_nvidia_driver_version(run_lambda),
        "available": str(torch.cuda.is_available()),
        "version": torch.version.cuda,
    }


def info_packages() -> Dict[str, str]:
    return {
        "numpy": np.__version__,
        "pyTorch_version": torch.__version__,
        "pyTorch_debug": str(torch.version.debug),
        "pytorch-lightning": pytorch_lightning.__version__,
        "tqdm": tqdm.__version__,
    }


def nice_print(details: Dict, level: int = 0) -> List:
    lines = []
    LEVEL_OFFSET = "\t"
    KEY_PADDING = 20
    for k in sorted(details):
        key = f"* {k}:" if level == 0 else f"- {k}:"
        if isinstance(details[k], dict):
            lines += [level * LEVEL_OFFSET + key]
            lines += nice_print(details[k], level + 1)
        elif isinstance(details[k], (set, list, tuple)):
            lines += [level * LEVEL_OFFSET + key]
            lines += [(level + 1) * LEVEL_OFFSET + "- " + v for v in details[k]]
        else:
            template = "{:%is} {}" % KEY_PADDING
            key_val = template.format(key, details[k])
            lines += [(level * LEVEL_OFFSET) + key_val]
    return lines


def print_system_env_info():
    details = {
        "Packages": info_packages(),
        "CUDA": info_cuda(),
    }
    lines = nice_print(details)
    text = os.linesep.join(lines)
    return text


def get_portion_of_batch_ids(percentage: float, batch_size: int) -> np.ndarray:
    """
    Select percentage * batch_size indices spread out evenly throughout array
    Examples
    ________
     >>> get_portion_of_batch_ids(percentage=0.5, batch_size=32)
     array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
     >>> get_portion_of_batch_ids(percentage=0.2, batch_size=32)
     array([ 0,  5, 10, 16, 21, 26])
     >>> get_portion_of_batch_ids(percentage=0.01, batch_size=64)
     array([], dtype=int64)
    """
    num = int(batch_size * percentage)
    if num == 0:
        return np.array([], dtype=np.int64)
    indices = np.arange(num).astype(float)
    stretch = batch_size / num
    indices *= stretch
    return np.unique(indices.astype(np.int64))


def add_text(img, lang_text):
    height, width, _ = img.shape
    if lang_text != "":
        coord = (1, int(height - 10))
        font_scale = (0.7 / 500) * width
        thickness = 1
        cv2.putText(
            img,
            text=lang_text,
            org=coord,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(0, 0, 0),
            thickness=thickness * 3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text=lang_text,
            org=coord,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


def format_sftp_path(path):
    """
    When using network mount from nautilus, format path
    """
    if path.as_posix().startswith("sftp"):
        uid = os.getuid()
        path = Path(f"/run/user/{uid}/gvfs/sftp:host={path.as_posix()[6:]}")
    return path
