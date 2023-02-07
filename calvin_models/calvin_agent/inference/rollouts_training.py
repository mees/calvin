from collections import Counter
import logging
from pathlib import Path
import typing

from calvin_agent.evaluation.utils import imshow_tensor
from calvin_agent.models.mcil import MCIL
from calvin_agent.utils.utils import get_last_checkpoint
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pytorch_lightning import seed_everything
import torch

logger = logging.getLogger(__name__)


def rollout(
    model,
    episode,
    env,
    tasks,
    demo_task_counter,
    live_task_counter,
    modalities,
    cfg,
    id_to_task_dict=None,
    embeddings=None,
):
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
    seq_len_max = state_obs.shape[0] - 1
    for mod in modalities:
        groundtruth_task = id_to_task_dict[int(idx)]
        # reset env to state of first step in the episode
        obs = env.reset(robot_obs=reset_info["robot_obs"][0], scene_obs=reset_info["scene_obs"][0])
        start_info = env.get_info()
        demo_task_counter += Counter(groundtruth_task)
        current_img_obs = obs["rgb_obs"]
        current_state_obs = obs["state_obs"]

        # goal image is last step of the episode
        if mod == "lang":
            _task = np.random.choice(list(groundtruth_task))
            task_embeddings = embeddings[_task]
            goal_lang = torch.tensor(task_embeddings[np.random.randint(task_embeddings.shape[0])]).unsqueeze(0)
        else:
            # goal image is last step of the episode
            goal_img = [rgb_ob[-1].unsqueeze(0).cuda() for rgb_ob in rgb_obs]
            goal_state = state_obs[-1].unsqueeze(0).cuda()

        for step in range(cfg.ep_len):
            #  replan every replan_freq steps (default 30 i.e every second)
            if step % cfg.replan_freq == 0:
                if mod == "lang":
                    plan, latent_goal = model.get_pp_plan_lang(
                        current_img_obs, current_state_obs, goal_lang
                    )  # type: ignore
                else:
                    plan, latent_goal = model.get_pp_plan_vision(
                        current_img_obs, goal_img, current_state_obs, goal_state
                    )  # type: ignore
            # kps = model.vision.conv_model[6].coords.cpu().numpy()
            kps = None
            if cfg.visualize:
                imshow_tensor("goal_img", goal_img[0], wait=1)
                imshow_tensor("current_img", current_img_obs[0], wait=1, keypoints=kps)
                imshow_tensor("dataset_img", rgb_obs[0][np.clip(step, 0, seq_len_max)], wait=1)

            # use plan to predict actions with current observations
            action = model.predict_with_plan(current_img_obs, current_state_obs, latent_goal, plan)
            obs, _, _, current_info = env.step(action)
            # check if current step solves a task
            current_task_info = tasks.get_task_info_for_set(start_info, current_info, groundtruth_task)
            # check if a task was achieved and if that task is a subset of the original tasks
            # we do not just want to solve any task, we want to solve the task that was proposed
            if len(current_task_info) > 0:
                live_task_counter += Counter(current_task_info)
                # skip current sequence if task was achieved
                break
            # update current observation
            current_img_obs = obs["rgb_obs"]
            current_state_obs = obs["state_obs"]

        print_task_log(demo_task_counter, live_task_counter, mod)


def print_task_log(demo_task_counter, live_task_counter, mod):
    print()
    logger.info(f"Modality: {mod}")
    for task in demo_task_counter:
        logger.info(
            f"{task}: SR = {(live_task_counter[task] / demo_task_counter[task]) * 100:.0f}%"
            + f" |  {live_task_counter[task]} of {demo_task_counter[task]}"
        )
    logger.info(
        f"Average Success Rate {mod} = "
        + f"{(sum(live_task_counter.values()) / s if (s := sum(demo_task_counter.values())) > 0 else 0) * 100:.0f}% "
    )
    logger.info(
        f"Success Rates averaged throughout classes = {np.mean([live_task_counter[task] / demo_task_counter[task] for task in demo_task_counter]) * 100:.0f}%"
    )


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
    checkpoint = get_checkpoint(cfg)
    id_to_task_dict = torch.load(checkpoint)["id_to_task_dict"]

    # since we don't use the trainer during inference, manually set up data_module
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=4)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["vis"]
    # dataset = val_dataloaders[0].dataset.datasets["vis"]  # type: ignore
    # dataloader = data_module.train_dataloader()
    env = hydra.utils.instantiate(cfg.callbacks.rollout.env_cfg, dataset, torch.device("cuda:0"), show_gui=False)

    try:
        embeddings = np.load(dataset.abs_datasets_dir / "embeddings.npy", allow_pickle=True,).reshape(
            -1
        )[0]
    except FileNotFoundError:
        embeddings = None

    tasks = hydra.utils.instantiate(cfg.callbacks.rollout.tasks)

    logger.info("Loading model from checkpoint.")
    model = MCIL.load_from_checkpoint(checkpoint)
    model.freeze()
    if train_cfg.model.action_decoder.get("load_action_bounds", False):
        model.action_decoder._setup_action_bounds(cfg.datamodule.root_data_dir, None, None)
    model = model.cuda(0)

    logger.info("Successfully loaded model.")
    demo_task_counter = Counter()  # type: typing.Counter[str]
    live_task_counter = Counter()  # type: typing.Counter[str]
    for i in id_to_task_dict:

        episode = dataset[int(i)]

        rollout(
            model=model,
            episode=episode,
            env=env,
            tasks=tasks,
            demo_task_counter=demo_task_counter,
            live_task_counter=live_task_counter,
            modalities=["vis"],
            cfg=cfg,
            id_to_task_dict=id_to_task_dict,
            embeddings=embeddings,
        )


if __name__ == "__main__":
    test_policy()
