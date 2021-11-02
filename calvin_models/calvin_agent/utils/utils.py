import os
from pathlib import Path
import shutil
import time
from typing import Dict, List, Union

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
    repo = git.Repo(search_parent_directories=True, path=repo_path.parent)
    assert repo, "not a repo"
    changed_files = [item.a_path for item in repo.index.diff(None)]
    if changed_files:
        print("WARNING uncommitted modified files: {}".format(",".join(changed_files)))
    return repo.head.object.hexsha


def get_last_checkpoint(experiment_folder: Path) -> Union[Path, None]:
    if experiment_folder.is_dir():
        checkpoint_folder = experiment_folder / "saved_models"
        if checkpoint_folder.is_dir():
            checkpoints = sorted(Path(checkpoint_folder).iterdir(), key=lambda chk: chk.stat().st_mtime)
            if len(checkpoints):
                # return newest checkpoint according to creation time
                assert checkpoints[-1].suffix == ".ckpt"
                return checkpoints[-1]

    return None


def save_executed_code() -> None:
    print(hydra.utils.get_original_cwd())
    print(os.getcwd())
    shutil.copytree(
        os.path.join(hydra.utils.get_original_cwd(), "calvin_agent"),
        os.path.join(hydra.utils.get_original_cwd(), f"{os.getcwd()}/code/calvin_agent"),
    )


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
