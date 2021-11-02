from collections import Counter
import logging

import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from tqdm import tqdm

logger = logging.getLogger(__name__)


def count_tasks(batch, env, tasks, task_counter):
    state_obs, rgb_obs, depth_obs, actions, _, reset_info, idx = batch
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
        task_counter += Counter(task_info)


@hydra.main(config_path="../../conf", config_name="config")
def compute_dataset_statistics(cfg: DictConfig) -> None:
    """"""
    seed_everything(cfg.seed)

    # since we don't use the trainer during inference, manually set up datamodule
    data_module = hydra.utils.instantiate(cfg.dataset, batch_size=32, num_workers=4)
    data_module.prepare_data()
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    env = hydra.utils.instantiate(cfg.rollout.env_cfg, train_dataloader.dataset.dataset_loader, "cpu")
    tasks = hydra.utils.instantiate(cfg.rollout.task_cfg)

    task_counter = Counter()  # type: ignore
    logger.info(
        f"training dataset with {len(train_dataloader.dataset.dataset_loader.max_batched_length_per_demo)} "
        f"episodes and {len(train_dataloader.dataset.dataset_loader.episode_lookup)} frames"
    )

    for batch in tqdm(train_dataloader):
        count_tasks(batch, env, tasks, task_counter)
    logger.info(f"training tasks: {task_counter}")

    task_counter = Counter()
    logger.info(
        f"training dataset with {len(val_dataloader.dataset.dataset_loader.max_batched_length_per_demo)} "
        f"episodes and {len(val_dataloader.dataset.dataset_loader.episode_lookup)} frames"
    )
    for batch in tqdm(val_dataloader):
        count_tasks(batch, env, tasks, task_counter)
    logger.info(f"validation tasks: {task_counter}")


if __name__ == "__main__":
    compute_dataset_statistics()
