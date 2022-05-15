import argparse
from pathlib import Path
from typing import Dict

import calvin_agent
from calvin_agent.datasets.disk_dataset import load_npz, load_pkl
import numpy as np
import tqdm

TRAINING_DIR: str = "training"


def main(input_params: Dict) -> None:
    dataset_root_str, save_format = (input_params["dataset_root"], input_params["save_format"])
    module_root_path = Path(calvin_agent.__file__)
    dataset_root = module_root_path.parent / Path(dataset_root_str)
    training_folder = dataset_root / TRAINING_DIR
    if training_folder.is_dir():
        if save_format == "pkl":
            load_episode = load_pkl
        elif save_format == "npz":
            load_episode = load_npz
        glob_generator = training_folder.glob(f"*.{save_format}")
        file_names = [x for x in glob_generator if x.is_file()]
        print(f"found training folder {training_folder} with {len(file_names)} files")
        acc_robot_state = np.zeros((), "float64")
        acc_scene_state = np.zeros((), "float64")
        acc_actions = np.zeros((), "float64")
        for idx, f in enumerate(tqdm.tqdm(file_names)):
            episode = load_episode(f)
            if "observations" in episode:
                if idx == 0:
                    acc_robot_state = episode["observations"]
                else:
                    acc_robot_state = np.concatenate((acc_robot_state, episode["observations"]), axis=0)
            if "actions" in episode:
                if idx == 0:
                    acc_actions = np.expand_dims(episode["actions"], axis=0)
                else:
                    acc_actions = np.concatenate((acc_actions, np.expand_dims(episode["actions"], axis=0)), axis=0)
            else:
                print("no actions found!!")
                exit(0)
            #  our play table environment
            if "robot_obs" in episode:
                if idx == 0:
                    acc_robot_state = np.expand_dims(episode["robot_obs"], axis=0)
                    acc_scene_state = np.expand_dims(episode["scene_obs"], axis=0)
                else:
                    acc_robot_state = np.concatenate(
                        (acc_robot_state, np.expand_dims(episode["robot_obs"], axis=0)), axis=0
                    )
                    acc_scene_state = np.concatenate(
                        (acc_scene_state, np.expand_dims(episode["scene_obs"], axis=0)), axis=0
                    )
        np.set_printoptions(precision=6, suppress=True)
        print(f"final robot obs shape {acc_robot_state.shape}")
        mean_robot_obs = np.mean(acc_robot_state, 0)
        std_robot_obs = np.std(acc_robot_state, 0)
        print(f"mean: {repr(mean_robot_obs)} and std: {repr(std_robot_obs)}")

        print(f"final scene obs shape {acc_scene_state.shape}")
        mean_scene_obs = np.mean(acc_scene_state, 0)
        std_scene_obs = np.std(acc_scene_state, 0)
        print(f"mean: {repr(mean_scene_obs)} and std: {repr(std_scene_obs)}")

        print(f"final robot actions shape {acc_actions.shape}")
        act_max_bounds = np.max(acc_actions, 0)
        act_min_bounds = np.min(acc_actions, 0)
        print(f"min action bounds: {repr(act_min_bounds)}")
        print(f"max action bounds: {repr(act_max_bounds)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="data", help="directory where  dataset is allocated")
    parser.add_argument("--save_format", type=str, default="npz", help="file format")
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
