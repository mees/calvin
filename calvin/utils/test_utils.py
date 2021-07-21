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


# Load actions from .pkl file. Use absolute path including extension name
def load_actions_data(file_name):
    path = {"actions": [], "init_qpos": [], "init_qvel": []}  # Only retrieve this keys
    if os.path.getsize(file_name) > 0:  # Check if the file is not empty
        with open(file_name, "rb") as f:
            data = pickle.load(f)
            for key in path.keys():
                if key == "observations":
                    path[key] = data[key][:, :9]
                else:
                    path[key] = data[key]
    return path


# Print imgs from validation packages. Use i to select the nth package.
# n_packages to select how many packages to load (i.e, val_data[i:i+n_packages])
def print_img_goals(data_dir="../data/validation/", save_folder="../data/goals/", i=0, n_packages=1, load_all=False):
    data_files = glob.glob(data_dir + "*.pkl")
    if not load_all:
        data_files = data_files[i : i + n_packages]
    print("Printing images...")
    print(data_files)

    data_img = []
    try:
        for i, file in enumerate(data_files):
            # load images of file
            with open(file, "rb") as f:
                if i == 0:
                    data_img = pickle.load(f)["images"]
                else:
                    data_img = np.concatenate(pickle.load(f)["images"], axis=0)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            for i, img in enumerate(data_img):
                save_path = save_folder + os.path.basename(file)[:-4] + "_img_" + str(i) + ".png"
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # save as blue shelfs
    except Exception as e:
        print(e)

    print("done!")


# init environment with pos and vel from given file
def init_env(env, file_name):
    if os.path.getsize(file_name) > 0:  # Check if the file is not empty
        with open(file_name, "rb") as f:
            data = pickle.load(f)
            env.sim.data.qpos[:] = data["init_qpos"].copy()
            env.sim.data.qvel[:] = data["init_qvel"].copy()
            env.sim.forward()
    return env


# Proxy function to either save or render a model
def viewer(env, mode="initialize", filename="video", render=False):
    global render_buffer
    if mode == "initialize":
        render_buffer = []
        mode = "render"

    if mode == "render":
        if render:
            env.render()
        curr_frame = env.render(mode="rgb_array")
        curr_frame = cv2.resize(curr_frame, (curr_frame.shape[1] // 3 - 1, curr_frame.shape[0] // 3))  # "852x640"
        render_buffer.append(curr_frame)

    if mode == "save":
        skvideo.io.vwrite(filename, np.asarray(render_buffer), outputdict={"-pix_fmt": "yuv420p"}, verbosity=1)
        print("\n Video saved", filename)


# Reproduced saved actions from file in the env. Optionally save a video.
# Load file should include file path + filename + extension
# save_filename is the name for the video to be saved. It is stored under './data/videos/'
def reproduce_file_actions(
    load_file,
    save_folder="../analysis/videos/reproduced_demonstrations/",
    save_filename="video.mp4",
    show_video=True,
    save_video=False,
):
    data = load_actions_data(load_file)
    # Env init
    gym_env = gym.make("kitchen_relax-v1")
    env = gym_env.env
    s = env.reset()

    # prepare env
    env.sim.data.qpos[:] = data["init_qpos"].copy()
    env.sim.data.qvel[:] = data["init_qvel"].copy()
    env.sim.forward()

    # Viewer
    FPS = 30
    render_skip = max(1, round(1.0 / (FPS * env.sim.model.opt.timestep * env.frame_skip)))
    viewer(env, mode="initialize")

    for i, action in enumerate(data["actions"]):
        s, r, _, _ = env.step(action)
        if i % render_skip == 0:
            viewer(env, mode="render", render=show_video)

    # save_video
    if save_video:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        viewer(env, mode="save", filename=save_folder + save_filename)
    env.close()


# ----------- Save videos from reproducing actions from files .pkl ------------#
# name = "friday_kettle_bottomknob_hinge_slide_1_path"
# file_path = "./data/training/"+ name +".pkl"
# video_name = "./analysis/videos/reproduced_demonstrations"
# reproduce_file_actions(file_path, show_video=True, save_video=False, save_filename = video_name)

# ----------- Print images from packages to select valid goals  ------------#
# print_img_goals(data_dir = "./data/validation/", i=0, n_packages=1)
