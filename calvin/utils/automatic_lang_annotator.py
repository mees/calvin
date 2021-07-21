from collections import Counter
import logging
import typing

import hydra
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
import torch

"""This script will collect data snt store it with a fixed window size"""

logger = logging.getLogger(__name__)


def label_seq(collected_data, dataloader, seq_length, idx, task_info, tasks_lang):
    seq_idx = dataloader.dataset.episode_lookup[idx]
    collected_data["info"]["indx"].append((seq_idx, seq_idx + seq_length))
    if list(task_info)[0] in tasks_lang.keys():
        task_lang = tasks_lang[list(task_info)[0]]
        lang_ann = [task_lang[np.random.randint(len(task_lang))]]
        collected_data["language"]["ann"].append(lang_ann)
        collected_data["language"]["task"].append(list(task_info)[0])
    return collected_data


def annotator(episode, env, tasks, demo_task_counter, dataloader, collected_data, tasks_lang, num_samples):
    state_obs, rgb_obs, depth_obs, actions, _, reset_info, idx = episode
    batch_size = state_obs.shape[0]
    for i in range(batch_size):
        # reset env to state of last step in the episode (goal state)
        env.reset(reset_info, i, -1)
        goal_info = env.get_info()
        # reset env to state of first step in the episode
        env.reset(reset_info, i, 0)
        start_info = env.get_info()
        # check if task was achieved in sequence
        task_info = tasks.get_task_info(start_info, goal_info)
        if len(task_info) == 0:
            continue
        if demo_task_counter[list(task_info)[0]] < num_samples:
            seq_length = torch.unique(actions, dim=1).shape[1]  # Compute windows_size
            demo_task_counter += Counter(task_info)
            logger.info(f"Tasks Objective: {num_samples}")
            logger.info(f"Tasks Annotations Progress: {demo_task_counter}")
            collected_data = label_seq(collected_data, dataloader, seq_length, idx, task_info, tasks_lang)
    return collected_data, demo_task_counter


@hydra.main(config_path="../../conf", config_name="lang_ann.yaml")
def main(cfg: DictConfig) -> None:
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(cfg.seed)
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=4)
    data_module.prepare_data()
    data_module.setup()
    if cfg.train:
        dataloader = data_module.train_dataloader()["vis"]
    else:
        dataloader = data_module.val_dataloader()["vis"]

    env = hydra.utils.instantiate(
        cfg.callbacks.rollout.env_cfg, dataloader.dataset, torch.device("cpu"), show_gui=False
    )
    tasks = hydra.utils.instantiate(cfg.callbacks.rollout.tasks)

    data_path = dataloader.dataset.abs_datasets_dir
    file_name = "auto_lang_ann.npy"  # + save_format
    path_to_filename = data_path / file_name

    demo_task_counter = Counter()  # type: typing.Counter[str]
    length = len(dataloader)
    tasks_lang = cfg.annotations
    num_samples = int(float(cfg.eps / 3) * length // len(tasks_lang.keys()))
    collected_data = {
        "language": {"ann": [], "task": [], "emb": []},
        "info": {"episodes": [], "indx": []},
    }  # type: typing.Dict

    for i, episode in enumerate(dataloader):
        collected_data, demo_task_counter = annotator(
            episode, env, tasks, demo_task_counter, dataloader, collected_data, tasks_lang, num_samples
        )
        if len(demo_task_counter.values()) > 0:
            if min(demo_task_counter.values()) >= num_samples:
                break
    if cfg.postprocessing:
        bert = hydra.utils.instantiate(cfg.model)
        language = [item for sublist in collected_data["language"]["ann"] for item in sublist]
        language_embedding = bert(language)
        collected_data["language"]["emb"] = language_embedding.unsqueeze(1).numpy()
        logger.info("Done extracting language embeddings !")
    np.save(path_to_filename, collected_data)


if __name__ == "__main__":
    main()
