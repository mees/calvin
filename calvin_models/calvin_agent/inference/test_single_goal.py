import glob
import os
import pickle
import sys

import cv2
import gym
import matplotlib.pyplot as plt
from networks.play_lmp import PlayLMP
import numpy as np
import torch

sys.path.append("./relay-policy-learning/adept_envs/")
import argparse
import time

import adept_envs
import keyboard
import skvideo.io
from tqdm import tqdm
import utils.constants as constants
from utils.test_utils import init_env, reproduce_file_actions, viewer


# Reproduce actions from .pkl into environment and save video
def parse_reprod_act_vid():
    eval_filename = "./data/validation/friday_microwave_topknob_bottomknob_slide_0_path.pkl"
    demo_files = [
        "friday_topknob_bottomknob_switch_slide_0_path",
        "friday_microwave_topknob_bottomknob_hinge_0_path",
        "friday_microwave_kettle_topknob_switch_0_path",
        "friday_microwave_kettle_topknob_hinge_0_path",
        "friday_microwave_kettle_switch_slide_0_path",
        "friday_microwave_kettle_hinge_slide_0_path",
        "friday_microwave_kettle_bottomknob_slide_0_path",
        "friday_microwave_kettle_bottomknob_hinge_0_path",
        "friday_microwave_bottomknob_switch_slide_0_path",
        "friday_microwave_bottomknob_hinge_slide_0_path",
        "friday_kettle_topknob_switch_slide_0_path",
        "friday_kettle_topknob_bottomknob_slide_1_path",
        "friday_kettle_switch_hinge_slide_0_path",
        "friday_kettle_bottomknob_switch_slide_0_path",
        "friday_kettle_bottomknob_hinge_slide_0_path",
    ]

    for name in demo_files:
        # save training videos
        file_path = "./data/training/" + name + ".pkl"
        video_name = name[:-5] + "_demo.mp4"
        reproduce_file_actions(file_path, show_video=False, save_video=True, save_filename=video_name)

    # save evaluation file video
    reproduce_file_actions(
        eval_filename,
        show_video=False,
        save_video=True,
        save_filename="friday_microwave_topknob_bottomknob_slide_eval_demo.mp4",
    )


def test_model(
    model,
    goal_path,
    show_goal=False,
    env_steps=1000,
    new_plan_frec=20,
    show_video=False,
    save_video=False,
    save_folder="./analysis/videos/model_trials/",
    save_filename="video.mp4",
):
    # load goal
    goal = plt.imread(goal_path)  # read as RGB, blue shelfs
    if show_goal:
        plt.axis("off")
        plt.suptitle("Goal")
        plt.imshow(goal)
        plt.show()
    goal = np.rint(goal * 255).astype(int)  # change to model scale
    # Env init
    gym_env = gym.make("kitchen_relax-v1")
    env = gym_env.env

    s = env.reset()
    # init viewer utility
    FPS = 10
    render_skip = max(1, round(1.0 / (FPS * env.sim.model.opt.timestep * env.frame_skip)))
    viewer(env, mode="initialize")

    # take actions
    for i in tqdm(range(env_steps)):
        curr_img = env.render(mode="rgb_array")
        curr_img = cv2.resize(curr_img, (300, 300))

        current_and_goal = np.stack((curr_img, goal), axis=0)  # (2, 300, 300, 3)
        current_and_goal = np.expand_dims(current_and_goal.transpose(0, 3, 1, 2), axis=0)  # (1, 2, 3, 300, 300)
        current_obs = np.expand_dims(s[:9], axis=0)  # (1,9)

        # prediction
        if i % new_plan_frec == 0:
            plan = model.get_pp_plan_vision(current_obs, current_and_goal)
        action = model.predict_with_plan(current_obs, current_and_goal, plan).squeeze()  # (9)
        s, r, _, _ = env.step(action.cpu().detach().numpy())
        if i % render_skip == 0:
            viewer(env, mode="render", render=show_video)

    # Save model
    if save_video:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        viewer(env, mode="save", filename=save_folder + save_filename)
    env.close()


def test(
    model_name,
    n_mixtures,
    use_logistics,
    goals,
    save_parent_dir="./analysis/videos/model_trials/",
    sample_new_plan=1,
    n_runs=5,
    save_video=True,
    show_video=False,
):

    # model init
    model_file_path = "./models/%s.pth" % model_name
    model = PlayLMP(num_mixtures=n_mixtures, use_logistics=use_logistics)
    model.load(model_file_path)

    # name to store folder videos on save_parent_dir
    names = [g.split("/")[-1] for g in goals]

    # create folders
    for name in names:
        save_folder = save_parent_dir + name + "/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    # run loop
    for goal, name in zip(goals, names):
        for i in range(n_runs):  # save n_runs videos for each goal
            goal_file_path = "./data/goals/" + goal + ".png"
            video_name = "%s_npf_%d_" % (model_name, sample_new_plan) + name + "(%d).mp4" % i
            test_model(
                model,
                goal_file_path,
                env_steps=400,
                new_plan_frec=sample_new_plan,
                save_video=save_video,
                show_video=show_video,
                save_folder=save_parent_dir + name + "/",
                save_filename=video_name,
            )


if __name__ == "__main__":
    # ----------- Test ------------#
    # model init
    # model needs to be placed on "./models/"
    model_name = "10_logistic_multitask_bestacc_new"
    num_mixtures = 10
    use_logistics = True

    # name of goal to run.
    # Will look for goal.png file in ./data/goals/__goals__.png
    goals = ["microwave", "grip_microwave", "bottomknob", "grip_hinge", "subsequent/microwave_bottomknob_topknob"]
    n_runs = 10  # number of videos to run/save
    sample_new_plan = 10  # frequency to sample new plan from policy network

    # test
    # Will save run on "./analysis/viedeos/model_trials/__goal__" folder
    # video will be named as : model_name +  npf_[sample_new_plan] + goal_name .mp4
    save_video = True
    show_video = False
    test(
        model_name,
        num_mixtures,
        use_logistics,
        goals,
        sample_new_plan=sample_new_plan,
        n_runs=n_runs,
        save_video=save_video,
        show_video=show_video,
    )
