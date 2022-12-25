from collections import Counter
from functools import reduce
import logging
from operator import add
import os
from pathlib import Path
from typing import Any, Dict, Optional

import calvin_agent
from calvin_agent.training import log_rank_0
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.distributed as dist
from torch.nn import Linear

"""This script will collect data snt store it with a fixed window size"""

logger = logging.getLogger(__name__)


def merge_data(list_of_data):
    merged_data = {
        "language": {"ann": [], "task": [], "emb": []},
        "info": {"episodes": [], "indx": []},
    }
    for d in list_of_data:
        for k in d:
            for k2, v2 in d[k].items():
                if isinstance(v2, list):
                    merged_data[k][k2] += v2
                elif isinstance(v2, np.ndarray) and len(merged_data[k][k2]) == 0:
                    merged_data[k][k2] = v2
                elif isinstance(v2, np.ndarray) and len(merged_data[k][k2]) != 0:
                    merged_data[k][k2] = np.concatenate((merged_data[k][k2], v2), axis=0)
                else:
                    print(type(v2))
                    raise ValueError
    return merged_data


class Annotator(Callback):
    def __init__(self, cfg):
        self.envs = None  # type: Any
        self.cfg = cfg
        self.device = None
        self.lang_folder = cfg.lang_folder
        self.tasks = hydra.utils.instantiate(cfg.callbacks.rollout_lh.tasks)
        self.demo_task_counter_train = Counter()  # type: Counter[str]
        self.demo_task_counter_val = Counter()  # type: Counter[str]
        self.train_dataset = None
        self.val_dataset = None
        self.file_name = "auto_lang_ann.npy"  # + save_format
        self.train_lang_folder = None
        self.val_lang_folder = None
        self.collected_data_train = {
            "language": {"ann": [], "task": [], "emb": []},
            "info": {"episodes": [], "indx": []},
        }  # type: Dict
        self.collected_data_val = {
            "language": {"ann": [], "task": [], "emb": []},
            "info": {"episodes": [], "indx": []},
        }  # type: Dict
        self.lang_model = None
        self.num_samples_train = None
        self.num_samples_val = None
        self.finished_annotation_val = False
        self.scene_idx_info = None

    @rank_zero_only
    def create_folders(self):
        self.train_lang_folder = self.train_dataset.abs_datasets_dir / self.lang_folder
        self.train_lang_folder.mkdir(parents=True, exist_ok=True)

        self.val_lang_folder = self.val_dataset.abs_datasets_dir / self.lang_folder
        self.val_lang_folder.mkdir(parents=True, exist_ok=True)

    @rank_zero_only
    def compute_val_embeddings(self):
        val_sent = self.cfg.val_instructions
        embeddings = {}
        for task, ann in val_sent.items():
            embeddings[task] = {}
            language_embedding = self.lang_model(list(ann))
            embeddings[task]["emb"] = language_embedding.cpu().numpy()
            embeddings[task]["ann"] = ann
        np.save(self.val_lang_folder / "embeddings", embeddings)
        logger.info("Done saving val language embeddings for Rollouts !")

    def init_vars(self, trainer, pl_module):
        self.device = pl_module.device
        self.val_dataset = trainer.val_dataloaders[0].dataset.datasets["vis"]  # type: ignore
        self.train_dataset = trainer.train_dataloader.dataset.datasets["vis"]
        self.scene_idx_info = np.load(self.train_dataset.abs_datasets_dir / "scene_info.npy", allow_pickle=True).item()

        self.envs = {
            scene: hydra.utils.instantiate(
                self.cfg.callbacks.rollout_lh.env_cfg, self.val_dataset, pl_module.device, scene=scene
            )
            for scene, _ in self.scene_idx_info.items()
        }
        if self.cfg.validation_scene not in self.envs:
            self.envs[self.cfg.validation_scene] = hydra.utils.instantiate(
                self.cfg.callbacks.rollout_lh.env_cfg,
                self.val_dataset,
                pl_module.device,
                scene=self.cfg.validation_scene,
                cameras=(),
            )

        self.create_folders()
        self.lang_model = hydra.utils.instantiate(self.cfg.model)
        self.compute_val_embeddings()
        self.num_samples_train = int(self.cfg.eps * len(self.train_dataset) / len(self.cfg.train_instructions.keys()))
        self.num_samples_val = int(self.cfg.eps * len(self.val_dataset) / len(self.cfg.train_instructions.keys()))

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the validation loop begins."""
        if self.envs is None:
            self.init_vars(trainer, pl_module)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.envs is None:
            self.init_vars(trainer, pl_module)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        batch = batch["vis"] if isinstance(batch, dict) else batch
        self.collected_data_val, self.demo_task_counter_val, current_task_counter = self.annotate(
            batch,
            self.val_dataset,
            self.collected_data_val,
            self.demo_task_counter_val,
            self.num_samples_val,
        )
        if dist.is_available() and dist.is_initialized():
            global_counters = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(global_counters, current_task_counter)
            current_task_counter = reduce(add, global_counters)
        self.demo_task_counter_val += current_task_counter
        if self.check_done(
            self.demo_task_counter_val, self.num_samples_val, batch_idx, trainer.num_val_batches[0], "val"
        ):
            print()
            print()
            print()
            logger.info("Finished annotating val dataset")
            print()
            print()
            print()
            self.finished_annotation_val = True

    def on_train_batch_end(  # type: ignore
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        batch = batch["vis"] if isinstance(batch, dict) else batch

        self.collected_data_train, self.demo_task_counter_train, current_task_counter = self.annotate(
            batch, self.train_dataset, self.collected_data_train, self.demo_task_counter_train, self.num_samples_train
        )
        if dist.is_available() and dist.is_initialized():
            global_counters = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(global_counters, current_task_counter)
            current_task_counter = reduce(add, global_counters)
        self.demo_task_counter_train += current_task_counter
        if self.check_done(
            self.demo_task_counter_train, self.num_samples_train, batch_idx, trainer.num_training_batches, "train"
        ):
            print()
            print()
            print()
            log_rank_0("Finished annotating train dataset")
            print()
            print()
            print()
            pl_module.finished_annotation_train = True  # type: ignore

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, unused: Optional[int] = None) -> None:
        self.save_and_postprocess(self.collected_data_train, self.train_lang_folder, "train", len(self.train_dataset))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.save_and_postprocess(self.collected_data_val, self.val_lang_folder, "val", len(self.val_dataset))

    def save_and_postprocess(self, collected_data, lang_folder, mod, length):
        if dist.is_available() and dist.is_initialized():
            global_collected_data = [None for _ in range(dist.get_world_size())]
            torch.distributed.all_gather_object(global_collected_data, collected_data)
            if dist.get_rank() == 0:
                global_collected_data = merge_data(global_collected_data)
                np.save("lang_ann", global_collected_data)
        else:
            np.save("lang_ann", collected_data)
        if self.cfg.postprocessing:
            language = collected_data["language"]["ann"]
            language_embedding = self.lang_model(language)
            collected_data["language"]["emb"] = language_embedding.cpu().numpy()
            logger.info(f"Done extracting {mod} language embeddings !")

        if dist.is_available() and dist.is_initialized():
            global_collected_data = [None for _ in range(dist.get_world_size())]
            torch.distributed.all_gather_object(global_collected_data, collected_data)
            if dist.get_rank() != 0:
                return
            collected_data = merge_data(global_collected_data)

        np.save(self.file_name, collected_data)
        np.save(lang_folder / self.file_name, collected_data)
        logger.info(f"Done saving {mod} language annotations !")

        lang_length = float(len(collected_data["language"]["ann"]))
        logger.info(
            f"\nVision Dataset contains  {length} datapoints "
            f"\nLanguage Dataset contains {lang_length} datapoints "
            f"\n    VISION --> {100.0 * length / (length + lang_length):.3f} %"
            f"\n    LANGUAGE --> {100.0 * lang_length / (length + lang_length):.3f} %"
        )

    def check_done(self, counter, num_samples, batch_idx, num_batches, mode):
        if batch_idx % 10 == 0:
            log_rank_0(f"{mode} Tasks Objective: {num_samples}")
            log_rank_0(f"Tasks Lang: {self.cfg.train_instructions.keys()}")
            log_rank_0(f"Tasks Annotations Progress: {counter}")
            log_rank_0(
                "Progress [ "
                + "=" * int(0.5 * 100 * batch_idx / num_batches)
                + ">"
                + "-" * int(0.5 * 100 * (num_batches - batch_idx) / num_batches)
                + str(round(100 * batch_idx / num_batches))
                + "%"
                + "]"
            )
        return len(counter.values()) >= len(self.cfg.train_instructions) and min(counter.values()) >= num_samples

    def select_env(self, dataset, idx):
        if "validation" in dataset.abs_datasets_dir.as_posix():
            return self.envs[self.cfg.validation_scene]
        seq_idx = dataset.episode_lookup[idx]
        for scene, interval in self.scene_idx_info.items():
            if interval[0] <= seq_idx <= interval[1]:
                return self.envs[scene]
        raise ValueError

    def annotate(self, episode, dataset, collected_data, global_task_counter, num_samples):
        state_obs = episode["robot_obs"]
        reset_info = episode["state_info"]
        idx = episode["idx"]
        batch_size, seq_length = state_obs.shape[0], state_obs.shape[1]
        current_task_counter = Counter()
        for i in range(batch_size):
            env = self.select_env(dataset, idx[i])
            # reset env to state of last step in the episode (goal state)
            env.reset(reset_info, i, -1)
            goal_info = env.get_info()

            prior_steps = np.random.randint(16, 32)
            env.reset(reset_info, i, prior_steps)
            middle_info = env.get_info()

            env.reset(reset_info, i, seq_length - 16)
            close_to_end_info = env.get_info()

            # check if task was achieved in sequence
            task_info = self.tasks.get_task_info(middle_info, goal_info)
            if (
                len(task_info) != 1
                or not task_info <= self.cfg.train_instructions.keys()
                or len(self.tasks.get_task_info_for_set(middle_info, close_to_end_info, task_info))
            ):
                continue
            task = list(task_info)[0]
            if global_task_counter[task] + current_task_counter[task] >= num_samples:
                continue
            # reset self.env to state of first step in the episode
            env.reset(reset_info, i, 0)
            start_info = env.get_info()

            env.reset(reset_info, i, 32)
            middle_info2 = env.get_info()

            if len(self.tasks.get_task_info_for_set(start_info, goal_info, task_info)) and not len(
                self.tasks.get_task_info(start_info, middle_info2)
            ):
                start_idx = idx[i]
                window_size = seq_length
            else:
                start_idx = idx[i] + prior_steps
                window_size = seq_length - prior_steps

            # seq_length = torch.unique(actions[i], dim=0).shape[0]
            current_task_counter += Counter(task_info)
            collected_data = self.label_seq(collected_data, dataset, window_size, start_idx, task)
        return collected_data, global_task_counter, current_task_counter

    def label_seq(self, collected_data, dataset, seq_length, idx, task):
        seq_idx = dataset.episode_lookup[idx]
        collected_data["info"]["indx"].append((seq_idx, seq_idx + seq_length))
        task_lang = self.cfg.train_instructions[task]
        lang_ann = task_lang[np.random.randint(len(task_lang))]
        collected_data["language"]["ann"].append(lang_ann)
        collected_data["language"]["task"].append(task)
        return collected_data


class LangAnnotationModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.finished_annotation_train = False
        self.dummy_net = Linear(1, 1)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:  # type: ignore
        if self.finished_annotation_train:
            return -1

    def training_step(self, batch, batch_idx):
        return self.dummy_net(torch.Tensor([0.0]).to(self.device))

    def validation_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


@hydra.main(config_path="../../conf", config_name="lang_ann.yaml")
def main(cfg: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(cfg.seed)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    callbacks = Annotator(cfg)

    dummy_model = LangAnnotationModel()

    trainer_args = {
        **cfg.trainer,
        "callbacks": callbacks,
        "num_sanity_val_steps": 0,
        "max_epochs": 1,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    # Configure multi-GPU training
    if trainer_args["devices"] > 1:  # type: ignore
        trainer_args["strategy"] = DDPStrategy(find_unused_parameters=False)

    trainer = Trainer(**trainer_args)

    trainer.fit(dummy_model, datamodule=datamodule)
    trainer.validate(dummy_model, datamodule=datamodule)  # type: ignore


if __name__ == "__main__":
    main()
