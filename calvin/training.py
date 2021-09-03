import logging
import os
from pathlib import Path
import sys
from typing import Callable, List, Union

from pytorch_lightning.plugins import DDPPlugin

sys.path.insert(0, Path(__file__).parents[1].as_posix())
import hydra
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

import calvin.models.play_lmp as models_m
from calvin.utils.utils import get_git_commit_hash, get_last_checkpoint, print_system_env_info

logger = logging.getLogger(__name__)


def wrap_train(config_name):
    @hydra.main(config_path="../conf", config_name=f"{config_name}.yaml")
    def train(cfg: DictConfig) -> None:
        # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
        seed_everything(cfg.seed, workers=True)  # type: ignore
        datamodule = hydra.utils.instantiate(cfg.datamodule)
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
        lr_logger = LearningRateMonitor(logging_interval="step")
        callbacks.append((lr_logger))

        trainer_args = {
            **cfg.trainer,
            "logger": train_logger,
            "callbacks": callbacks,
            "resume_from_checkpoint": chk,
            "benchmark": False,
        }

        # Configure multi-GPU training
        if is_multi_gpu_training(trainer_args["gpus"]):  # type: ignore
            # trainer_args["accelerator"] = "ddp"
            trainer_args["plugins"] = DDPPlugin(find_unused_parameters=False)
            # trainer_args["plugins"] = "ddp_sharded"

            if not cfg.slurm:
                modify_argv_hydra()

        trainer = Trainer(**trainer_args)

        # Start training
        trainer.fit(model, datamodule=datamodule)

    train()


def setup_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    callbacks = [hydra.utils.instantiate(cb) for cb in callbacks_cfg.values()]
    return callbacks


def setup_logger(cfg: DictConfig, model: LightningModule) -> LightningLoggerBase:
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


def is_multi_gpu_training(gpus: Union[int, str, ListConfig]) -> bool:
    return (
        (isinstance(gpus, int) and (gpus > 1 or gpus == -1))
        or (isinstance(gpus, str) and len(gpus) > 1)
        or (isinstance(gpus, ListConfig) and len(gpus) > 1)
    )


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


def setup_config():
    config_str = next((x for x in sys.argv if "config_name" in x), None)
    if config_str is not None:
        config_name = config_str.split("=")[1]
        sys.argv.remove(config_str)
        os.environ["HYDRA_CONFIG_NAME"] = config_name
        return config_name
    elif "HYDRA_CONFIG_NAME" in os.environ:
        return os.environ["HYDRA_CONFIG_NAME"]
    else:
        return "config"


if __name__ == "__main__":
    conf = setup_config()
    wrap_train(conf)
