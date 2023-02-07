import logging
from pathlib import Path

from calvin_agent.evaluation.utils import imshow_tensor
from calvin_agent.models.mcil import MCIL
from calvin_agent.utils.utils import get_last_checkpoint
import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pytorch_lightning import seed_everything
import torch

logger = logging.getLogger(__name__)


def get_checkpoint(cfg):
    try:
        checkpoint = cfg.load_checkpoint
    except MissingMandatoryValue:
        checkpoint = get_last_checkpoint(Path(cfg.train_folder))
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
        +datamodule.root_data_dir (str): /path/dataset when running inference on another machine than were it was trained
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
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=4)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["vis"]
    env = hydra.utils.instantiate(cfg.callbacks.rollout.env_cfg, dataset, torch.device("cuda:0"), show_gui=False)

    tasks = hydra.utils.instantiate(cfg.callbacks.rollout.tasks)
    checkpoint = get_checkpoint(cfg)
    logger.info("Loading model from checkpoint.")
    model = MCIL.load_from_checkpoint(checkpoint)
    model.freeze()
    # model.action_decoder._setup_action_bounds(cfg.datamodule.root_data_dir, None, None)
    model = model.cuda(0)
    logger.info("Successfully loaded model.")

    ep_start_end_ids = np.sort(np.load(dataset.abs_datasets_dir / "ep_start_end_ids.npy"), axis=0)

    for s, e in ep_start_end_ids:
        i = start_i = s
        file = dataset.abs_datasets_dir / f"episode_{i:06d}.npz"
        data = np.load(file)
        obs = env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
        start_info = env.get_info()
        current_img_obs = start_img_obs = obs["rgb_obs"]
        start_state_obs = obs["state_obs"]
        goal_imgs = obs["rgb_obs"]
        goal_state = obs["state_obs"]
        scene_obs = data["scene_obs"]
        robot_obs = data["robot_obs"]
        while 1:
            imshow_tensor("current_img", current_img_obs[0], wait=1)
            imshow_tensor("start", start_img_obs[0], wait=1)
            imshow_tensor("goal", goal_imgs[0], wait=1)
            cv2.imshow("keylistener", np.zeros((300, 300)))
            k = cv2.waitKey(0) % 256
            if k == ord("s"):
                start_info = env.get_info()
                start_img_obs = obs["rgb_obs"]
                start_state_obs = obs["state_obs"]
                scene_obs = data["scene_obs"]
                robot_obs = data["robot_obs"]
                start_i = i
            elif k == ord("w"):
                end_info = env.get_info()
                print(tasks.get_task_info(start_info, end_info))
                goal_imgs = obs["rgb_obs"]
                goal_state = obs["state_obs"]
                print(f"steps: {i - start_i}")
            elif k == ord("r"):
                file = dataset.abs_datasets_dir / f"episode_{i:06d}.npz"
                data = np.load(file)
                obs = env.reset(scene_obs=data["scene_obs"])
                current_img_obs = obs["rgb_obs"]
            elif k == ord("a"):
                i -= 1
                i = np.clip(i, s, e)
                file = dataset.abs_datasets_dir / f"episode_{i:06d}.npz"
                data = np.load(file)
                obs = env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                current_img_obs = obs["rgb_obs"]

            elif k == ord("d"):
                i += 1
                i = np.clip(i, s, e)
                file = dataset.abs_datasets_dir / f"episode_{i:06d}.npz"
                data = np.load(file)
                obs = env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                current_img_obs = obs["rgb_obs"]
            elif k == ord("q"):
                i -= 100
                i = np.clip(i, s, e)
                file = dataset.abs_datasets_dir / f"episode_{i:06d}.npz"
                data = np.load(file)
                obs = env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                current_img_obs = obs["rgb_obs"]

            elif k == ord("e"):
                i += 100
                i = np.clip(i, s, e)
                file = dataset.abs_datasets_dir / f"episode_{i:06d}.npz"
                data = np.load(file)
                obs = env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                current_img_obs = obs["rgb_obs"]

            elif k == ord("f"):
                env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
                rollout(model, env, tasks, cfg, start_info, start_img_obs, start_state_obs, goal_imgs, goal_state)
                obs = env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
                current_img_obs = obs["rgb_obs"]
                i = start_i
            elif k == ord("n"):  # ESC
                break


def rollout(model, env, tasks, cfg, start_info, current_img_obs, current_state_obs, goal_imgs, goal_state):
    # goal image is last step of the episode
    # goal_imgs = [goal_img.unsqueeze(0).cuda() for goal_img in goal_imgs]
    goal_imgs = goal_imgs[0].contiguous()
    for step in range(cfg.ep_len):
        #  replan every replan_freq steps (default 30 i.e every second)
        if step % cfg.replan_freq == 0:
            plan, latent_goal = model.get_pp_plan_vision(
                current_img_obs, goal_imgs, current_state_obs, goal_state
            )  # type: ignore
        imshow_tensor("current_img", current_img_obs[0], wait=1)

        # use plan to predict actions with current observations
        action = model.predict_with_plan(current_img_obs, current_state_obs, latent_goal, plan)
        obs, _, _, current_info = env.step(action)
        # check if current step solves a task
        current_task_info = tasks.get_task_info(start_info, current_info)
        if len(current_task_info) > 0:
            print(current_task_info)
        # update current observation
        current_img_obs = obs["rgb_obs"]
        current_state_obs = obs["state_obs"]


if __name__ == "__main__":
    test_policy()
