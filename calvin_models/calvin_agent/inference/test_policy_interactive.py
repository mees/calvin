from collections import Counter
import logging
from pathlib import Path
import typing

from calvin_agent.evaluation.utils import imshow_tensor
from calvin_agent.models.encoders.language_network import SBert
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pytorch_lightning import seed_everything

logger = logging.getLogger(__name__)


def rollout(model, episode, env, tasks, demo_task_counter, live_task_counter, cfg, sbert):
    """
    Args:
        model: MCIL model
        episode: Batch from dataloader
             state_obs: Tensor,
             rgb_obs: tuple(Tensor, ),
             depth_obs: tuple(Tensor, ),
             actions: Tensor,
             lang: Tensor,
             reset_info: Dict
             idx: int
        env: play_lmp_wrapper(play_table_env)
        tasks: Tasks
        demo_task_counter: Counter[str]
        live_task_counter: Counter[str]
        visualize: visualize images
    """
    state_obs, rgb_obs, depth_obs, actions, _, reset_info, idx = episode
    batch_size = state_obs.shape[0]
    seq_len_max = state_obs.shape[1] - 1
    for i in range(batch_size):
        # reset env to state of last step in the episode (goal state)
        env.reset(reset_info, i, -1)
        goal_info = env.get_info()
        # reset env to state of first step in the episode
        obs = env.reset(reset_info, i, 0)
        # noise = torch.ones(obs['state_obs'].shape) * eps
        obs["state_obs"] = obs["state_obs"]  # + noise
        start_info = env.get_info()

        # check if task was achieved in sequence
        task_info = tasks.get_task_info(start_info, goal_info)
        if len(task_info) == 0:
            continue
        neutral_init = input("Start rollout from Neutral Position ? [y/n] \n")
        if "y" in neutral_init:
            obs = env.reset()
        demo_task_counter += Counter(task_info)
        current_img_obs = obs["rgb_obs"]
        current_state_obs = obs["state_obs"]
        # goal image is last step of the episode
        logger.info(f"Recognized task: {task_info}")
        # logger.info(f"Noise rate: {eps}")
        lang_input = [input("What should I do? \n")]
        goal_lang = sbert(lang_input)
        goal_img = rgb_obs[0][i, -1].unsqueeze(0)

        for step in range(cfg.ep_len):
            #  replan every replan_freq steps (default 30 i.e every second)
            if step % cfg.replan_freq == 0:
                plan, latent_goal = model.get_pp_plan_lang(
                    current_img_obs, current_state_obs, goal_lang
                )  # type: ignore
            # kps = model.vision.conv_model[6].coords.cpu().numpy()
            kps = None
            if cfg.visualize:
                imshow_tensor(lang_input[0], goal_img, wait=1)
                imshow_tensor("current_img", current_img_obs[0], wait=1, keypoints=kps)
                imshow_tensor("dataset_img", rgb_obs[0][i, np.clip(step, 0, seq_len_max)], wait=1)

            # use plan to predict actions with current observations
            action = model.predict_with_plan(current_img_obs, current_state_obs, latent_goal, plan)
            obs, _, _, current_info = env.step(action)
            # check if current step solves a task
            current_task_info = tasks.get_task_info(start_info, current_info)
            # check if a task was achieved and if that task is a subset of the original tasks
            # we do not just want to solve any task, we want to solve the task that was proposed
            if len(current_task_info) > 0 and current_task_info <= task_info:
                live_task_counter += Counter(current_task_info)
                # skip current sequence if task was achieved
                break
            # update current observation
            current_img_obs = obs["rgb_obs"]
            current_state_obs = obs["state_obs"]

        print_task_log(demo_task_counter, live_task_counter)


def print_task_log(demo_task_counter, live_task_counter):
    print()
    for task in demo_task_counter:
        logger.info(
            f"{task}: SR = {(live_task_counter[task] / demo_task_counter[task]) * 100:.0f}%"
            + f"|  {live_task_counter[task]} of {demo_task_counter[task]}"
        )
    logger.info(
        "Average Success Rate = "
        + f"{(sum(live_task_counter.values()) / s if (s := sum(demo_task_counter.values())) > 0 else 0) * 100:.0f}% "
    )


def get_checkpoint(cfg):
    try:
        checkpoint = cfg.load_checkpoint
    except MissingMandatoryValue:
        checkpoint = sorted((Path(cfg.train_folder) / "saved_models").glob("*.ckpt"), reverse=True)[0]
    return checkpoint


def format_sftp_path(cfg):
    """
    When using network mount from nautilus, format path
    """
    if cfg.train_folder.startswith("sftp"):
        cfg.train_folder = "/run/user/9984/gvfs/sftp:host=" + cfg.train_folder[7:]


@hydra.main(config_path="../../conf/inference", config_name="config_inference")
def test_policy(input_cfg: DictConfig) -> None:
    """
    Run inference on trained policy.
     Arguments:
        train_folder (str): path of trained model.
        load_checkpoint (str): optional model checkpoint. If not specified, the last checkpoint is taken by default.
        +dataset.root_data_dir (str): /path/dataset when running inference on another machine than were it was trained
        visualize (bool): wether to visualize the policy rollouts (default True).
    """
    # when mounting remote folder with sftp, format path
    format_sftp_path(input_cfg)
    # load config used during training
    train_cfg_path = Path(input_cfg.train_folder) / ".hydra/config.yaml"
    train_cfg = OmegaConf.load(train_cfg_path)

    # merge configs to keep current cmd line overrides
    cfg = OmegaConf.merge(train_cfg, input_cfg)
    seed_everything(cfg.seed)

    # since we don't use the trainer during inference, manually set up data_module
    data_module = hydra.utils.instantiate(cfg.dataset, num_workers=2)
    data_module.datasets["lang"].batch_size = 1
    data_module.datasets["vis"].batch_size = 1
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader(shuffle=False)
    # dataloader = data_module.train_dataloader()

    env = hydra.utils.instantiate(
        cfg.rollout.env_cfg, dataloader.dataset.datasets["lang"].dataset_loader, "cpu", show_gui=False
    )
    sbert = SBert("mpnet")
    tasks = hydra.utils.instantiate(cfg.rollout.task_cfg)
    checkpoint = get_checkpoint(cfg)
    logger.info("Loading model from checkpoint.")
    model = hydra.utils.instantiate(cfg.model)
    model = model.load_from_checkpoint(checkpoint)
    model.freeze()
    model.action_decoder._setup_action_bounds(cfg.dataset.root_data_dir, None, None)
    logger.info("Successfully loaded model.")
    demo_task_counter = Counter()  # type: typing.Counter[str]
    live_task_counter = Counter()  # type: typing.Counter[str]
    for i, episode in enumerate(dataloader):
        rollout(model, episode["vis"], env, tasks, demo_task_counter, live_task_counter, cfg, sbert)


if __name__ == "__main__":
    test_policy()
