import argparse
from collections import Counter
import logging
from pathlib import Path
import time

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    DefaultLangEmbeddings,
    get_default_model_and_env,
    get_eval_env_state,
    join_vis_lang,
)
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch

torch.backends.cudnn.deterministic = True
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env

logger = logging.getLogger(__name__)


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class CustomModel:
    def __init__(self):
        logger.warning("Please implement these methods as an interface to your custom model architecture.")
        raise NotImplementedError

    def reset(self):
        """
        This is called
        """
        raise NotImplementedError

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        raise NotImplementedError


class CustomLangEmbeddings:
    def __init__(self):
        logger.warning("Please implement these methods in order to use your own language embeddings")
        raise NotImplementedError

    def get_lang_goal(self, task_annotation):
        """
        Args:
             task_annotation: langauge annotation
        Returns:

        """
        raise NotImplementedError


def evaluate_policy(model, env, lang_embeddings, args):
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_sequences = get_sequences()

    results = {}

    if not args.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, eval_sequence, lang_embeddings, val_annotations, args)
        results[eval_sequence] = result
        if not args.debug:
            count = Counter(np.array(list(results.values())))
            eval_sequences.set_description(" ".join([f"{k}/5 : {v} |" for k, v in count.items()]) + "|")
    print(f"Average successful sequence length: {np.mean(list(results.values()))}")
    count = Counter(np.array(list(results.values())))
    for i in range(6):
        print(f"{i} successful tasks: {count[i]} / {len(eval_sequences)} sequences")


def evaluate_sequence(env, model, task_checker, eval_sequence, lang_embeddings, val_annotations, args):
    robot_obs, scene_obs = get_eval_env_state()
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if args.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        success = rollout(env, model, task_checker, args, subtask, lang_embeddings, val_annotations)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, args, subtask, lang_embeddings, val_annotations):
    if args.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
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
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if args.debug:
                print(colored("success", "green"), end=" ")
            return True
    if args.debug:
        print(colored("fail", "red"), end=" ")
    return False


if __name__ == "__main__":
    seed_everything(42, workers=True)
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

    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )
    parser.add_argument("--custom_lang_embeddings", action="store_true", help="Use custom language embeddings.")

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    args = parser.parse_args()

    # Do not change
    args.ep_len = 350

    # evaluate a custom model
    if args.custom_model:
        model = CustomModel()
        env = make_env(args.dataset_path)
    else:
        assert "train_folder" in args
        model, env, _ = get_default_model_and_env(args.train_folder, args.dataset_path, args.checkpoint)

    if args.custom_lang_embeddings:
        lang_embeddings = CustomLangEmbeddings()
    else:
        lang_embeddings = DefaultLangEmbeddings(args.dataset_path)  # type: ignore
    evaluate_policy(model, env, lang_embeddings, args)
