import logging
import os
from pathlib import Path
from typing import List, Set

import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.distributed as dist
from torchvision.transforms.functional import resize
import wandb
import wandb.util

log = logging.getLogger(__name__)

flatten = lambda t: [item for sublist in t for item in sublist]
flatten_list_of_dicts = lambda t: {k: v for d in t for k, v in d.items()}


def _unnormalize(img):
    return img / 2 + 0.5


def delete_tmp_video(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def add_modality(tasks, mod):
    return {f"{mod}/{task}" for task in tasks}


class RolloutVideo:
    def __init__(self, logger, empty_cache, log_to_file, save_dir):
        self.videos = []
        self.video_paths = {}
        self.task_names = []
        self.logger = logger
        self.empty_cache = empty_cache
        self.log_to_file = log_to_file
        self.save_dir = Path(save_dir)
        if self.log_to_file:
            os.makedirs(self.save_dir, exist_ok=True)
        if (
            isinstance(self.logger, TensorBoardLogger)
            and dist.is_available()
            and dist.is_initialized()
            and not self.log_to_file
        ):
            log.warning("Video logging with tensorboard and ddp can lead to OOM errors.")

    def new_video(self, initial_frame: torch.Tensor, tasks: Set[str], modality: str) -> None:
        """
        Begin a new video with the first frame of a rollout.
        Args:
             initial_frame: static camera RGB image
             tasks: set of tasks that were achieved
             modality: condition rollout modality ('vis' or 'lang')
        """
        # (1, 1, channels, height, width)
        self.videos.append(_unnormalize(initial_frame.detach().cpu()))
        tasks = add_modality(tasks, modality)
        self.task_names.append(tasks)

    def update(self, rgb_obs: torch.Tensor) -> None:
        """
        Add new frame to video.
        Args:
            rgb_obs: static camera RGB images
        """
        img = rgb_obs.detach().cpu()
        self.videos[-1] = torch.cat([self.videos[-1], _unnormalize(img)], dim=1)  # shape 1, t, c, h, w

    def add_goal_thumbnail(self, goal_img):
        size = self.videos[-1].shape[-2:]
        i_h = int(size[0] / 3)
        i_w = int(size[1] / 3)
        img = resize(_unnormalize(goal_img.detach().cpu()), [i_h, i_w])
        self.videos[-1][..., -i_h:, :i_w] = img

    def write_to_tmp(self):
        """
        In case of logging with WandB, save the videos as GIF in tmp directory,
        then log them at the end of the validation epoch from rank 0 process.
        """
        if isinstance(self.logger, WandbLogger) and not self.log_to_file:
            for video, task_name in zip(self.videos, self.task_names):
                video = np.clip(video.numpy() * 255, 0, 255).astype(np.uint8)
                for task in task_name:
                    wandb_vid = wandb.Video(video, fps=10, format="gif")
                    self.video_paths[task] = wandb_vid._path
            self.videos = []
            self.task_names = []

    @staticmethod
    def _empty_cache():
        """
        Clear GPU reserved memory. Do not call this unnecessarily.
        """
        mem1 = torch.cuda.memory_reserved(dist.get_rank())
        torch.cuda.empty_cache()
        mem2 = torch.cuda.memory_reserved(dist.get_rank())
        log.info(f"GPU: {dist.get_rank()} freed {(mem1 - mem2) / 10**9:.1f}GB of reserved memory")

    def log(self, global_step: int) -> None:
        """
        Call this method at the end of a validation epoch to log videos to tensorboard, wandb or filesystem.
        Args:
            global_step: global step of the training
        """
        if self.log_to_file:
            self._log_videos_to_file(global_step)
        elif isinstance(self.logger, WandbLogger):
            self._log_videos_to_wandb()
        elif isinstance(self.logger, TensorBoardLogger):
            self._log_videos_to_tb(global_step)
        else:
            raise NotImplementedError
        self.videos = []
        self.task_names = []
        self.video_paths = {}

    def _log_videos_to_tb(self, global_step):
        if dist.is_available() and dist.is_initialized():
            if self.empty_cache:
                self._empty_cache()

            all_videos = [None for _ in range(torch.distributed.get_world_size())]
            all_task_names = [None for _ in range(torch.distributed.get_world_size())]
            try:
                torch.distributed.all_gather_object(all_videos, self.videos)
                torch.distributed.all_gather_object(all_task_names, self.task_names)
            except RuntimeError as e:
                log.warning(e)
                return
            # only log videos from rank 0 process
            if dist.get_rank() != 0:
                return
            videos = flatten(all_videos)
            task_names = flatten(all_task_names)

            for video, task_name in zip(videos, task_names):
                self._plot_video_tb(video, task_name, global_step)
        else:
            for video, task_name in zip(self.videos, self.task_names):
                self._plot_video_tb(video, task_name, global_step)

    def _plot_video_tb(self, video, task_name, global_step):
        video = video.unsqueeze(0)
        for task in task_name:
            self.logger.experiment.add_video(f"rollout_{task}", video, global_step=global_step, fps=10)

    def _log_videos_to_wandb(self):
        if dist.is_available() and dist.is_initialized():
            all_video_paths = [None for _ in range(torch.distributed.get_world_size())]
            try:
                torch.distributed.all_gather_object(all_video_paths, self.video_paths)
            except RuntimeError as e:
                log.warning(e)
                return
            # only log videos from rank 0 process
            if dist.get_rank() != 0:
                return
            video_paths = flatten_list_of_dicts(all_video_paths)
        else:
            video_paths = self.video_paths
        for task, path in video_paths.items():
            self.logger.experiment.log({f"rollout_{task}": wandb.Video(path, fps=10, format="gif")})
            delete_tmp_video(path)

    def _log_videos_to_file(self, global_step):
        """
        Mostly taken from WandB
        """
        for video, task_name in zip(self.videos, self.task_names):
            video = video.unsqueeze(0)
            video = np.clip(video.numpy() * 255, 0, 255).astype(np.uint8)

            mpy = wandb.util.get_module(
                "moviepy.editor",
                required='wandb.Video requires moviepy and imageio when passing raw data.  Install with "pip install moviepy imageio"',
            )
            tensor = self._prepare_video(video)
            _, _height, _width, _channels = tensor.shape

            # encode sequence of images into gif string
            clip = mpy.ImageSequenceClip(list(tensor), fps=10)

            for task in task_name:
                task = task.replace("/", "_")
                filename = self.save_dir / f"{task}_{global_step}.gif"
                clip.write_gif(filename, logger=None)

    @staticmethod
    def _prepare_video(video):
        """This logic was mostly taken from tensorboardX"""
        if video.ndim < 4:
            raise ValueError("Video must be atleast 4 dimensions: time, channels, height, width")
        if video.ndim == 4:
            video = video.reshape(1, *video.shape)
        b, t, c, h, w = video.shape

        if video.dtype != np.uint8:
            logging.warning("Converting video data to uint8")
            video = video.astype(np.uint8)

        def is_power2(num):
            return num != 0 and ((num & (num - 1)) == 0)

        # pad to nearest power of 2, all at once
        if not is_power2(video.shape[0]):
            len_addition = int(2 ** video.shape[0].bit_length() - video.shape[0])
            video = np.concatenate((video, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)

        n_rows = 2 ** ((b.bit_length() - 1) // 2)
        n_cols = video.shape[0] // n_rows

        video = np.reshape(video, newshape=(n_rows, n_cols, t, c, h, w))
        video = np.transpose(video, axes=(2, 0, 4, 1, 5, 3))
        video = np.reshape(video, newshape=(t, n_rows * h, n_cols * w, c))
        return video
