from collections import Counter
import logging
from pathlib import Path
import typing

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
import torch

from calvin_models.calvin_agent.evaluation.utils import format_sftp_path, get_checkpoint, imshow_tensor, print_task_log
from calvin_models.calvin_agent.models.play_lmp import PlayLMP

logger = logging.getLogger(__name__)


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

    device = torch.device("cuda:0")

    checkpoint = get_checkpoint(cfg)
    task_to_id_dict = torch.load(checkpoint)["task_to_id_dict"]
    id_to_task_dict = torch.load(checkpoint)["id_to_task_dict"]

    # since we don't use the trainer during inference, manually set up data_module
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=4)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["vis"]
    lang_dataset = dataloader.dataset.datasets["lang"]
    env = hydra.utils.instantiate(cfg.callbacks.rollout.env_cfg, dataset, device, show_gui=False)

    embeddings = np.load(
        lang_dataset.abs_datasets_dir / lang_dataset.lang_folder / "embeddings.npy", allow_pickle=True
    ).item()

    task_checker = hydra.utils.instantiate(cfg.callbacks.rollout.tasks)

    logger.info("Loading model from checkpoint.")
    model = PlayLMP.load_from_checkpoint(checkpoint)
    model.freeze()
    if train_cfg.model.decoder.get("load_action_bounds", False):
        model.action_decoder._setup_action_bounds(cfg.datamodule.root_data_dir, None, None, True)
    model = model.cuda(device)

    logger.info("Successfully loaded model.")
    demo_task_counter = Counter()  # type: typing.Counter[str]
    live_task_counter = Counter()  # type: typing.Counter[str]
    for task_name, ids in task_to_id_dict.items():
        print()
        print(f"Evaluate {task_name}: {embeddings[task_name]['ann']}")
        print()
        for i in ids:
            episode = dataset[int(i)]
            rollout(
                model=model,
                episode=episode,
                env=env,
                tasks=task_checker,
                demo_task_counter=demo_task_counter,
                live_task_counter=live_task_counter,
                modalities=["lang"],
                cfg=cfg,
                device=device,
                id_to_task_dict=id_to_task_dict,
                embeddings=embeddings,
            )
        print_task_log(demo_task_counter, live_task_counter, "lang")


def rollout(
    model,
    episode,
    env,
    tasks,
    demo_task_counter,
    live_task_counter,
    modalities,
    cfg,
    device,
    id_to_task_dict=None,
    embeddings=None,
):
    """
    Args:
        model: PlayLMP model
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
        current_depth_obs = obs["depth_obs"]
        current_state_obs = obs["state_obs"]

        start_img_obs = [img.clone() for img in current_img_obs]

        # goal image is last step of the episode

        _task = np.random.choice(list(groundtruth_task))
        task_embeddings = embeddings[_task]["emb"]
        goal_lang = torch.from_numpy(embeddings[_task]["emb"]).to(device).squeeze(0)

        # goal image is last step of the episode
        goal_imgs = [rgb_ob[-1].unsqueeze(0).to(device) for rgb_ob in rgb_obs]
        goal_depths = [depth_ob[-1].unsqueeze(0).to(device) for depth_ob in depth_obs]
        goal_state = state_obs[-1].unsqueeze(0).to(device)

        for step in range(cfg.ep_len):
            #  replan every replan_freq steps (default 30 i.e every second)
            if step % cfg.replan_freq == 0:
                if mod == "lang":
                    plan, latent_goal = model.get_pp_plan_lang(
                        current_img_obs, current_depth_obs, current_state_obs, goal_lang
                    )  # type: ignore
                else:
                    plan, latent_goal = model.get_pp_plan_vision(
                        current_img_obs,
                        current_depth_obs,
                        goal_imgs,
                        goal_depths,
                        current_state_obs,
                        goal_state,
                    )  # type: ignore
            if cfg.visualize:
                imshow_tensor("start_img", start_img_obs[0], wait=1)
                imshow_tensor("goal_img", goal_imgs[0], wait=1)
                imshow_tensor("current_img", current_img_obs[0], wait=1)
                imshow_tensor("dataset_img", rgb_obs[0][np.clip(step, 0, seq_len_max)], wait=1)

            # use plan to predict actions with current observations
            action = model.predict_with_plan(current_img_obs, current_depth_obs, current_state_obs, latent_goal, plan)
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
            current_depth_obs = obs["depth_obs"]
            current_state_obs = obs["state_obs"]


if __name__ == "__main__":
    test_policy()
