from datetime import timedelta
import logging
import os
from pathlib import Path
import sys
from typing import List, Union

# This is for using the locally installed repo clone when using slurm
import calvin_agent
from lightning_lite.accelerators.cuda import num_cuda_devices
from pytorch_lightning.strategies import DDPStrategy

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import calvin_agent.models.mcil as models_m
from calvin_agent.utils.utils import get_git_commit_hash, get_last_checkpoint, print_system_env_info
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    This is called to start a training.

    Args:
        cfg: hydra config
    """
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(cfg.seed, workers=True)  # type: ignore
    datamodule = hydra.utils.instantiate(cfg.datamodule, training_repo_root=Path(calvin_agent.__file__).parents[2])
    chk = get_last_checkpoint(Path.cwd())

    # Load Model
    if chk is not None:
        model = getattr(models_m, cfg.model["_target_"].split(".")[-1]).load_from_checkpoint(chk.as_posix())
    else:
        model = hydra.utils.instantiate(cfg.model)

    log_rank_0(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0("Repo commit hash: {}".format(get_git_commit_hash(Path(hydra.utils.to_absolute_path(__file__)))))
    log_rank_0(print_system_env_info())

    train_logger = setup_logger(cfg, model)
    callbacks = setup_callbacks(cfg.callbacks)

    trainer_args = {
        **cfg.trainer,
        "logger": train_logger,
        "callbacks": callbacks,
        "benchmark": False,
    }

    # Configure multi-GPU training
    if is_multi_gpu_training(trainer_args["devices"]):
        # increase default timeout for loading data into shared memory
        trainer_args["strategy"] = DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=3600))
        if not cfg.slurm:
            modify_argv_hydra()

    trainer = Trainer(**trainer_args)

    # Start training
    trainer.fit(model, datamodule=datamodule, ckpt_path=chk)  # type: ignore


def setup_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """
    Instantiate all training callbacks.

    Args:
        callbacks_cfg: DictConfig with all callback params

    Returns:
        List of instantiated callbacks.
    """
    callbacks = [hydra.utils.instantiate(cb) for cb in callbacks_cfg.values()]
    return callbacks


def setup_logger(cfg: DictConfig, model: LightningModule) -> Logger:
    """
    Set up the logger (tensorboard or wandb) from hydra config.

    Args:
        cfg: Hydra config
        model: LightningModule

    Returns:
        logger
    """
    pathlib_cwd = Path.cwd()
    if "group" in cfg.logger:
        cfg.logger.group = pathlib_cwd.parent.name
        cfg.logger.name = pathlib_cwd.parent.name + "/" + pathlib_cwd.name
        cfg.logger.id = cfg.logger.name.replace("/", "_")
        train_logger = hydra.utils.instantiate(cfg.logger)
        train_logger.watch(model)
    else:
        train_logger = hydra.utils.instantiate(cfg.logger)
    return train_logger


def modify_argv_hydra() -> None:
    """
    To make hydra work with pytorch-lightning and ddp, we modify sys.argv for the child processes spawned with ddp.
    This is only used when NOT using slurm.
    """
    cwd = Path.cwd().as_posix()
    cwd = f'"{cwd}"'
    sys.argv = sys.argv[:1]
    sys.argv.extend(
        [
            f"hydra.run.dir={cwd}",
            "hydra/hydra_logging=disabled",
            "hydra/job_logging=disabled",
        ]
    )
    overrides = OmegaConf.load(".hydra/overrides.yaml")
    for o in overrides:
        if "hydra/sweeper" in o:  # type: ignore
            continue

        if "hydra/launcher" in o:  # type: ignore
            continue

        sys.argv.append(o)  # type: ignore


def is_multi_gpu_training(devices: Union[int, str, ListConfig]) -> bool:
    """
    Check if training on multiple GPUs.
    See https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#devices

     Args:
        devices: int, str or ListConfig specifying devices

    Returns:
        True if multi-gpu training (ddp), False otherwise.
    """
    num_gpu_available = num_cuda_devices()
    if isinstance(devices, int):
        return devices > 1 or (devices == -1 and num_gpu_available > 1)
    elif isinstance(devices, str) and devices == "auto":
        return num_gpu_available > 1
    elif isinstance(devices, str):
        return len(devices) > 1
    elif isinstance(devices, ListConfig):
        return len(devices) > 1
    else:
        raise ValueError


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


if __name__ == "__main__":
    train()
