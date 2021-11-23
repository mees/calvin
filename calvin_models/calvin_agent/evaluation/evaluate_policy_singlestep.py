import argparse
from collections import Counter
from pathlib import Path

from calvin_agent.evaluation.utils import DefaultLangEmbeddings, get_default_model_and_env, join_vis_lang
from calvin_agent.utils.utils import get_last_checkpoint
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch


def evaluate_policy(model, env, datamodule, lang_embeddings, args):
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    task_to_id_dict = torch.load(args.checkpoint)["task_to_id_dict"]
    dataset = datamodule.val_dataloader().dataset.datasets["vis"]

    results = Counter()

    for task, ids in task_to_id_dict.items():
        for i in ids:
            episode = dataset[int(i)]
            results[task] += rollout(env, model, episode, task_oracle, args, task, lang_embeddings, val_annotations)
        print(f"{task}: {results[task]} / {len(ids)}")

    print(f"SR: {sum(results.values()) / sum(len(x) for x in task_to_id_dict.values()) * 100:.1f}%")


def rollout(env, model, episode, task_oracle, args, task, lang_embeddings, val_annotations):
    state_obs, rgb_obs, depth_obs = episode["robot_obs"], episode["rgb_obs"], episode["depth_obs"]
    reset_info = episode["state_info"]
    idx = episode["idx"]
    obs = env.reset(robot_obs=reset_info["robot_obs"][0], scene_obs=reset_info["scene_obs"][0])
    # get lang annotation for subtask
    lang_annotation = val_annotations[task][0]
    # get language goal embedding
    goal = lang_embeddings.get_lang_goal(lang_annotation)
    model.reset()
    start_info = env.get_info()

    for step in range(args.ep_len):
        action = model.step(obs, goal)
        obs, _, _, current_info = env.step(action)
        if args.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task})
        if len(current_task_info) > 0:
            if args.debug:
                print(colored("S", "green"), end=" ")
            return True
    if args.debug:
        print(colored("F", "red"), end=" ")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Manually specify checkpoint path (default is latest). Only used for calvin_agent.",
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    args = parser.parse_args()

    # Do not change
    args.ep_len = 240
    model, env, datamodule = get_default_model_and_env(args.train_folder, args.dataset_path, args.checkpoint)

    if args.checkpoint is None:
        args.checkpoint = get_last_checkpoint(Path(args.train_folder))

    lang_embeddings = DefaultLangEmbeddings(args.dataset_path)  # type: ignore
    evaluate_policy(model, env, datamodule, lang_embeddings, args)
