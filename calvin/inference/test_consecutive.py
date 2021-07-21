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
from utils.test_utils import viewer


def test_model_seq_goals(
    model,
    goal_lst,
    env_steps=1000,
    new_plan_frec=20,
    show_video=False,
    save_video=False,
    save_folder="./analysis/videos/model_trials/",
    save_filename="video.mp4",
):
    print("Test subsequent goals... press n to change goal")
    print("Goal list:", goal_lst)
    new_goal = goal_lst.pop(0)
    print("first goal: ", new_goal)
    # load goal
    goal = plt.imread(new_goal)  # read as RGB, blue shelfs
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

        # change goal
        if keyboard.is_pressed("n"):
            if goal_lst:  # list not empty
                new_goal = goal_lst.pop(0)
                print("Changing goal..")
                print("New goal", new_goal)
                print("Goals left:", goal_lst)
                goal = plt.imread(new_goal)  # read as RGB, blue shelfs
                goal = np.rint(goal * 255).astype(int)  # change to model scale
                time.sleep(1)
                print("Continue execution..")
            else:
                print("empty goal list")
        if keyboard.is_pressed("e"):
            print("stopping execution..")
            break
        ########################################

        if i % render_skip == 0:
            viewer(env, mode="render", render=show_video)
    # Save model
    if save_video:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        viewer(env, mode="save", filename=save_folder + save_filename)
    env.close()


def test_subsequent(model_name, n_mixtures, use_logistics, goal_list, video_name, sample_new_plan=1, n_runs=5):
    # model init
    model_file_path = "./models/%s.pth" % model_name
    model = PlayLMP(num_mixtures=n_mixtures, use_logistics=use_logistics)
    model.load(model_file_path)

    # run and save video
    # press n to switch to next goal in list#
    # press e to terminate run: i = i+1
    for i in range(n_runs):
        video_name = "%s%d.mp4" % (video_name, i)
        # goals to be exectured in sequence
        goal_lst = ["./data/goals/" + goal + ".png" for goal in goal_list]
        test_model_seq_goals(
            model,
            goal_lst,
            env_steps=600,
            new_plan_frec=sample_new_plan,
            show_video=True,
            save_video=True,
            save_filename=video_name,
        )


if __name__ == "__main__":
    # model init
    # model needs to be placed on "./models/"
    model_name = "10_logistic_multitask_bestacc_new"
    num_mixtures = 10
    use_logistics = True

    # video params
    goal_lst = ["grip_kettle", "kettle", "/subsequent/kettle_switch"]
    video_name = "kettle_switch"
    sample_new_plan = 2  # frequency to sample new plan from policy network
    n_runs = 5  # number of videos to run/save

    # test
    test_subsequent(model_name, num_mixtures, use_logistics, goal_lst, video_name, sample_new_plan, n_runs)
