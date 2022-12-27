from collections import Counter
from itertools import chain
import logging
import multiprocessing
import os
from typing import Any

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition, join_vis_lang
from calvin_agent.rollout.rollout_video import RolloutVideo
import hydra
import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
from termcolor import colored
import torch
import torch.distributed as dist

log_print = logging.getLogger(__name__)


def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    log_print.info(*args, **kwargs)


def divide_across_ranks(elements, world_size, rank):
    """
    Divide a number across subprocesses in multiprocessing.
    Example: distribute 4 elements in a world of size 3
    rank 0->2, rank 1->1, rank 2->1
    """
    assert rank < world_size
    rest = lambda n, w, i: 1 if n % w > i else 0
    return elements // world_size + rest(elements, world_size, rank)


def sequences_for_rank(num_sequences):
    """
    When using ddp, determine how many sequences every process should evaluate.
    """
    rank = dist.get_rank()
    ws = dist.get_world_size()
    num_seq_per_gpu = divide_across_ranks(num_sequences, ws, rank)
    num_workers = multiprocessing.cpu_count() // ws
    return [
        seq.tolist()
        for seq in np.array_split(get_sequences(num_sequences, num_workers=num_workers), ws)[rank][:num_seq_per_gpu]
    ]


def gather_results(local_results):
    """
    Collect eval results from all processes.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return local_results
    results = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(results, local_results)
    return list(chain(*results))


def get_video_tag(i):
    if dist.is_available() and dist.is_initialized():
        i = i * dist.get_world_size() + dist.get_rank()
    return f"_long_horizon/sequence_{i}"


class RolloutLongHorizon(Callback):
    """
    A class for performing rollouts during validation step.
    """

    def __init__(
        self,
        env_cfg,
        skip_epochs,
        rollout_freq,
        num_videos,
        num_sequences,
        replan_freq,
        ep_len,
        tasks,
        log_video_to_file,
        save_dir,
        lang_folder,
        empty_cache,
        val_annotations,
        debug,
    ):
        self.env = None  # type: Any
        self.env_cfg = env_cfg
        self.task_checker = hydra.utils.instantiate(tasks)
        self.skip_epochs = skip_epochs
        self.rollout_freq = rollout_freq
        self.num_videos = num_videos
        self.num_sequences = num_sequences
        self.replan_freq = replan_freq
        self.ep_len = ep_len
        self.log_video_to_file = log_video_to_file
        self.save_dir = save_dir
        self.rollout_video = None  # type: Any
        self.empty_cache = empty_cache
        self.device = None  # type: Any
        self.lang_embeddings = None
        self.lang_folder = lang_folder
        self.eval_sequences = None
        self.val_annotations = val_annotations
        self.debug = debug

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the validation loop begins."""
        if self.env is None:
            self.device = pl_module.device
            dataset = trainer.val_dataloaders[0].dataset.datasets["lang"]  # type: ignore
            from calvin_agent.rollout.rollout import Rollout

            for callback in trainer.callbacks:
                if isinstance(callback, Rollout) and callback.env is not None:
                    self.env = callback.env
                    break
            else:
                self.env = hydra.utils.instantiate(self.env_cfg, dataset, pl_module.device)
            if self.num_videos > 0:
                if dist.is_available() and dist.is_initialized():
                    self.num_videos = divide_across_ranks(self.num_videos, dist.get_world_size(), dist.get_rank())
                self.rollout_video = RolloutVideo(
                    logger=pl_module.logger,
                    empty_cache=self.empty_cache,
                    log_to_file=self.log_video_to_file,
                    save_dir=self.save_dir,
                )
            pl_module.load_lang_embeddings(dataset.abs_datasets_dir / dataset.lang_folder / "embeddings.npy")  # type: ignore
            if dist.is_available() and dist.is_initialized():
                self.eval_sequences = sequences_for_rank(self.num_sequences)
            else:
                self.eval_sequences = get_sequences(self.num_sequences)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args) -> None:  # type: ignore
        if pl_module.current_epoch == 0 and self.skip_epochs > 0:
            for i in range(1, 6):
                pl_module.log(f"eval_lh/sr_chain_{i}", torch.tensor(0.0), on_step=False, sync_dist=True)
            pl_module.log("eval_lh/avg_seq_len", torch.tensor(0.0), on_step=False, sync_dist=True)
        elif pl_module.current_epoch >= self.skip_epochs and pl_module.current_epoch % self.rollout_freq == 0:
            results = self.evaluate_policy(pl_module)

            if self.num_videos > 0:
                # log rollout videos
                self.rollout_video.log(pl_module.global_step)

            results = gather_results(results)
            count = Counter(results)  # type: ignore
            print()
            for i in range(1, 6):
                n_success = sum(count[j] for j in reversed(range(i, 6)))
                sr = n_success / len(results)
                pl_module.log(f"eval_lh/sr_chain_{i}", torch.tensor(sr), on_step=False, sync_dist=True)
                log_rank_0(f"{i} / 5 subtasks: {n_success} / {len(results)} sequences, SR: {sr * 100:.1f}%")
            avg_seq_len = np.mean(results)
            pl_module.log("eval_lh/avg_seq_len", torch.tensor(avg_seq_len), on_step=False, sync_dist=True)
            log_rank_0(f"Average successful sequence length: {avg_seq_len:.1f}")
            print()

    def evaluate_policy(self, model):
        results = []
        for i, (initial_state, eval_sequence) in enumerate(self.eval_sequences):
            record = i < self.num_videos
            result = self.evaluate_sequence(model, initial_state, eval_sequence, record, i)
            results.append(result)
            if record:
                self.rollout_video.write_to_tmp()
        return results

    def evaluate_sequence(self, model, initial_state, eval_sequence, record, i):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        if record:
            caption = " | ".join(eval_sequence)
            self.rollout_video.new_video(tag=get_video_tag(i), caption=caption)
        success_counter = 0
        if self.debug:
            print()
            print()
            print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
            print("Subtask: ", end="")
        for subtask in eval_sequence:
            if record:
                self.rollout_video.new_subtask()
            success = self.rollout(model, subtask, record)
            if record:
                self.rollout_video.draw_outcome(success)
            if success:
                success_counter += 1
            else:
                return success_counter
        return success_counter

    def rollout(self, model, subtask, record):
        if self.debug:
            print(f"{subtask} ", end="")
        obs = self.env.get_obs()
        # get lang annotation for subtask
        lang_annotation = self.val_annotations[subtask][0]
        model.reset()
        start_info = self.env.get_info()
        success = False
        for step in range(self.ep_len):
            action = model.step(obs, lang_annotation)
            obs, _, _, current_info = self.env.step(action)
            if self.debug and os.environ.get("DISPLAY") is not None:
                img = self.env.render(mode="rgb_array")
                join_vis_lang(img, lang_annotation)
            if record:
                # update video
                self.rollout_video.update(obs["rgb_obs"]["rgb_static"])
            # check if current step solves a task
            current_task_info = self.task_checker.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                success = True
                break
        if self.debug:
            if success:
                print(colored("success", "green"), end=" ")
            else:
                print(colored("fail", "red"), end=" ")
        if record:
            self.rollout_video.add_language_instruction(lang_annotation)
        return success
