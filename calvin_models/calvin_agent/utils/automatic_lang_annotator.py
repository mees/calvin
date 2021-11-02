from collections import Counter
import logging
from pathlib import Path
import typing

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
import torch

from calvin_models import calvin_agent

"""This script will collect data snt store it with a fixed window size"""

logger = logging.getLogger(__name__)


def label_seq(collected_data, dataloader, seq_length, idx, task, tasks_lang):
    seq_idx = dataloader.dataset.episode_lookup[idx]
    collected_data["info"]["indx"].append((seq_idx, seq_idx + seq_length))
    task_lang = tasks_lang[task]
    lang_ann = task_lang[np.random.randint(len(task_lang))]
    collected_data["language"]["ann"].append(lang_ann)
    collected_data["language"]["task"].append(task)
    return collected_data


def annotator(episode, env, tasks, demo_task_counter, dataloader, collected_data, tasks_lang, num_samples):
    state_obs, rgb_obs, depth_obs, actions, _, reset_info, idx = episode
    batch_size, seq_length = state_obs.shape[0], state_obs.shape[1]
    for i in range(batch_size):
        # reset env to state of last step in the episode (goal state)
        env.reset(reset_info, i, -1)
        goal_info = env.get_info()
        # reset env to state of first step in the episode
        env.reset(reset_info, i, 0)
        start_info = env.get_info()
        # check if task was achieved in sequence
        task_info = tasks.get_task_info(start_info, goal_info)
        if len(task_info) != 1:
            continue
        task = task_info.pop()
        if task not in tasks_lang.keys():
            continue
        if "slide_" in task:
            env.reset(reset_info, i, seq_length // 2)
            inter_info = env.get_info()
            # check if task was achieved in sequence
            task_info = tasks.get_task_info(start_info, inter_info)
            if len(task_info) > 0:
                inter_task = task_info.pop()
                if inter_task != task:
                    logger.warning(f"Conflict sub task {inter_task} of {task}")
                    continue

        if demo_task_counter[task] < num_samples:
            seq_length = torch.unique(actions[i], dim=0).shape[0]
            demo_task_counter += Counter(({task}))
            collected_data = label_seq(collected_data, dataloader, seq_length, idx[i], task, tasks_lang)
    return collected_data, demo_task_counter


@hydra.main(config_path="../../conf", config_name="lang_ann.yaml")
def main(cfg: DictConfig) -> None:
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(cfg.seed)
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=4)
    data_module.prepare_data()
    data_module.setup()
    dataloaders = {
        "val": data_module.val_dataloader().loaders["vis"].loader,
        "train": data_module.train_dataloader()["vis"],
    }
    env = None
    for mod, dataloader in dataloaders.items():
        if env is None:
            env = hydra.utils.instantiate(
                cfg.callbacks.rollout.env_cfg, dataloader.dataset, torch.device("cuda:0"), show_gui=False
            )
            tasks = hydra.utils.instantiate(cfg.callbacks.rollout.tasks)

        data_path = dataloader.dataset.abs_datasets_dir
        lang_folder = data_path / cfg.lang_folder
        lang_folder.mkdir(parents=True, exist_ok=True)
        file_name = "auto_lang_ann.npy"  # + save_format

        demo_task_counter = Counter()  # type: typing.Counter[str]
        length = len(dataloader.dataset.episode_lookup)
        tasks_lang = cfg.annotations
        num_samples = int(cfg.eps * length / len(tasks_lang.keys())) if "train" in mod else 1
        collected_data = {
            "language": {"ann": [], "task": [], "emb": []},
            "info": {"episodes": [], "indx": []},
        }  # type: typing.Dict

        if mod == "val":
            model = hydra.utils.instantiate(cfg.model)
            val_sent = OmegaConf.load(
                Path(calvin_agent.__file__).parent / f"../conf/annotations/{cfg.rollout_sentences}.yaml"
            )
            embeddings = {}  # type: typing.Dict
            for task, ann in val_sent.items():
                embeddings[task] = {}
                language_embedding = model(list(ann))
                embeddings[task]["emb"] = language_embedding
                embeddings[task]["ann"] = ann
            np.save(lang_folder / "embeddings", embeddings)
            logger.info(f"Done saving {mod} language embeddings for Rollouts !")

        for i, episode in enumerate(dataloader):
            collected_data, demo_task_counter = annotator(
                episode, env, tasks, demo_task_counter, dataloader, collected_data, tasks_lang, num_samples
            )
            if i % 10 == 0:
                logger.info(f"Tasks Objective: {num_samples}")
                logger.info(f"Tasks Lang: {tasks_lang.keys()}")
                logger.info(f"Tasks Annotations Progress: {demo_task_counter}")
                logger.info(
                    "Progress [ "
                    + "=" * int(0.5 * 100 * i / len(dataloader))
                    + ">"
                    + "-" * int(0.5 * 100 * (len(dataloader) - i) / len(dataloader))
                    + str(round(100 * i / len(dataloader)))
                    + "%"
                    + "]"
                )
            if len(demo_task_counter.values()) >= len(tasks_lang):
                if min(demo_task_counter.values()) >= num_samples:
                    break
        np.save("lang_ann", collected_data)
        if cfg.postprocessing:
            model = hydra.utils.instantiate(cfg.model)
            language = collected_data["language"]["ann"]
            language_embedding = model(language)
            collected_data["language"]["emb"] = language_embedding
            logger.info(f"Done extracting {mod} language embeddings !")

        np.save(file_name, collected_data)
        np.save(lang_folder / file_name, collected_data)
        logger.info(f"Done saving {mod} language annotations !")

        lang_length = float(len(collected_data["language"]["ann"]))
        logger.info(
            f"\nVision Dataset contains  {length} datapoints "
            f"\nLanguage Dataset contains {lang_length} datapoints "
            f"\n    VISION --> {100.0 * length / (length + lang_length):.3f} %"
            f"\n    LANGUAGE --> {100.0 * lang_length / (length + lang_length):.3f} %"
        )


if __name__ == "__main__":
    main()
