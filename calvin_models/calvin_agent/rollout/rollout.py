from collections import defaultdict
from functools import partial, reduce
import logging
from operator import add
from typing import Any, Dict, List, Tuple

from calvin_agent.datasets.base_dataset import get_validation_window_size
from calvin_agent.rollout.rollout_video import RolloutVideo
from calvin_agent.utils.utils import get_portion_of_batch_ids
import hydra
import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
import torch
import torch.distributed as dist

log_print = logging.getLogger(__name__)


def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    log_print.info(*args, **kwargs)


def select_first(all_task_ids, num, *args, **kwargs):
    """
    Select the first num indices
    """
    return all_task_ids[:num]


def select_balanced(all_task_ids, num, *args, **kwargs):
    """
    Select the indices equally balanced validation
    """
    split_ids = np.array_split(sorted(all_task_ids), num)[: len(all_task_ids)]
    return [ids[0] for ids in split_ids]


def select_longest(all_task_ids, num, min_window_size, max_window_size):
    """
    Select the indices with the longest sequence window
    """
    sorted_ids = sorted(
        all_task_ids,
        key=partial(get_validation_window_size, min_window_size=min_window_size, max_window_size=max_window_size),
        reverse=True,
    )
    return sorted_ids[:num]


def get_video_tag(task, mod):
    return f"_{mod}/{list(task)[0]}"


class Rollout(Callback):
    """
    A class for performing rollouts during validation step.
    """

    def __init__(
        self,
        env_cfg,
        skip_epochs,
        rollout_freq,
        video,
        num_rollouts_per_task,
        check_percentage_of_batch,
        ep_len,
        tasks,
        empty_cache,
        log_video_to_file,
        save_dir,
        add_goal_thumbnail,
        min_window_size,
        max_window_size,
        lang_folder,
        val_annotations,
        id_selection_strategy="select_first",
    ):
        self.env = None  # type: Any
        self.env_cfg = env_cfg
        self.tasks = hydra.utils.instantiate(tasks)
        self.skip_epochs = skip_epochs
        self.rollout_freq = rollout_freq
        self.video = video
        self.num_rollouts_per_task = num_rollouts_per_task
        self.check_percentage_of_batch = check_percentage_of_batch
        self.ep_len = ep_len
        self.empty_cache = empty_cache
        self.log_video_to_file = log_video_to_file
        self.save_dir = save_dir
        self.task_to_id_dict = None  # type: Any
        self.id_to_task_dict = None  # type: Any
        self.full_task_to_id_dict = None  # type: Any
        self.groundtruth_task_counter = None  # type: Any
        self.rollout_video = None  # type: Any
        self.device = None  # type: Any
        self.outputs = []
        self.modalities = []  # ["vis", "lang"] if self.lang else ["vis"]
        self.embeddings = None
        self.add_goal_thumbnail = add_goal_thumbnail
        self.val_annotations = val_annotations
        self.lang_folder = lang_folder
        self.pick_task_ids = partial(
            eval(id_selection_strategy), min_window_size=min_window_size, max_window_size=max_window_size
        )

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the validation loop begins."""
        if self.env is None:
            self.modalities = trainer.datamodule.modalities  # type: ignore
            self.device = pl_module.device
            dataset = trainer.val_dataloaders[0].dataset.datasets["vis"]  # type: ignore
            from calvin_agent.rollout.rollout_long_horizon import RolloutLongHorizon

            for callback in trainer.callbacks:
                if isinstance(callback, RolloutLongHorizon) and callback.env is not None:
                    self.env = callback.env
                    break
            else:
                self.env = hydra.utils.instantiate(self.env_cfg, dataset, pl_module.device)
            if self.video:
                self.rollout_video = RolloutVideo(
                    logger=pl_module.logger,
                    empty_cache=self.empty_cache,
                    log_to_file=self.log_video_to_file,
                    save_dir=self.save_dir,
                )
            if "lang" in self.modalities:
                pl_module.load_lang_embeddings(dataset.abs_datasets_dir / dataset.lang_folder / "embeddings.npy")  # type: ignore

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        batch = batch["vis"]
        if pl_module.current_epoch >= self.skip_epochs and (pl_module.current_epoch + 1) % self.rollout_freq == 0:
            # in first validation epoch collect groundtruth task information of current validation batch
            if self.task_to_id_dict is None:
                outputs["task_ids"], outputs["batch_seq_ids"] = self.get_task_info_of_batch(batch)
            else:
                # do rollout for batch
                outputs["rollout_task_counter"] = self.env_rollouts(batch, pl_module)
            self.outputs.append(outputs)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args) -> None:  # type: ignore
        # TODO: remove lightning fixes callback hook
        outputs = [self.outputs]

        if pl_module.current_epoch == 0:
            pl_module.log("tasks/average_sr", torch.tensor(0.0), on_step=False, sync_dist=True)
        elif pl_module.current_epoch >= self.skip_epochs and (pl_module.current_epoch + 1) % self.rollout_freq == 0:
            # after first validation epoch, create task lookup dictionaries
            if self.task_to_id_dict is None:
                self.build_task_dict(outputs, pl_module)
            else:
                if self.video:
                    # log rollout videos
                    self.rollout_video.log(pl_module.global_step)
                # collect the task rollout counters of all validation batches and sum across tasks
                acc_score = torch.tensor(0.0, device=pl_module.device)
                for mod in self.modalities:
                    rollout_task_counter = reduce(add, [x["rollout_task_counter"][mod] for x in outputs[0]])
                    if dist.is_available() and dist.is_initialized():
                        rollout_task_counter = torch.sum(
                            pl_module.all_gather(rollout_task_counter), dim=0
                        )  # shape: (num_tasks,)
                    score = (
                        torch.sum(rollout_task_counter) / torch.sum(self.groundtruth_task_counter)
                        if torch.sum(self.groundtruth_task_counter) > 0
                        else torch.tensor(0.0)
                    )
                    pl_module.log(
                        f"tasks/average_sr_{mod}",
                        score,
                        on_step=False,
                        sync_dist=True,
                    )
                    acc_score += score
                    print()
                    log_rank_0(f"Evaluating {mod} task success rates:")
                    for i in range(rollout_task_counter.shape[0]):
                        if self.groundtruth_task_counter[i] > 0:
                            # log the ratio of successful task executions per task
                            # log to tensorboard
                            pl_module.log(
                                f"tasks/{self.tasks.id_to_task[i]}_{mod}",
                                rollout_task_counter[i] / self.groundtruth_task_counter[i],
                                on_step=False,
                                sync_dist=True,
                            )
                            # log to cmd line
                            log_rank_0(
                                f"{self.tasks.id_to_task[i]}: "
                                + f"{rollout_task_counter[i] / self.groundtruth_task_counter[i] * 100:.0f}%"
                                + f" ({rollout_task_counter[i]} / {self.groundtruth_task_counter[i]})"
                            )
                    print()
                pl_module.log(
                    "tasks/average_sr",
                    acc_score / len(self.modalities),
                    on_step=False,
                    sync_dist=True,
                )
        self.outputs = []

    def build_task_dict(self, validation_step_outputs, pl_module):
        """
        Called once after the first validation epoch.
        It creates:
            self.task_to_id_dict: maps from task name to indices of sequences in the validation dataset, in which this
                                  task was solved. To be reused in later validation epochs.
                                  Contains maximum self.num_rollouts_per_task sequence ids
            self.id_to_task_dict: reverse map of self.task_to_id_dict
                                  Values are sets since more than one task may be solved in one sequence.
            self.groundtruth_task_counter: Tensor of shape (n_tasks,) that counts the number of successful groundtruth
                                           tasks per task.
        """
        batch_seq_ids = torch.LongTensor(reduce(add, [x["batch_seq_ids"] for x in validation_step_outputs[0]])).to(
            self.device
        )
        task_ids = torch.LongTensor(reduce(add, [x["task_ids"] for x in validation_step_outputs[0]])).to(self.device)

        if dist.is_available() and dist.is_initialized():
            # since task may be distributed unevenly across the validation splits on different gpus we have to truncate
            # task_ids and batch_seq_ids to the min length before calling self.all_gather
            len_b = torch.LongTensor([len(batch_seq_ids)]).to(self.device)
            len_t = torch.LongTensor([len(task_ids)]).to(self.device)
            len_b = int(torch.min(pl_module.all_gather(len_b)))
            len_t = int(torch.min(pl_module.all_gather(len_t)))
            batch_seq_ids = batch_seq_ids[:len_b]
            task_ids = task_ids[:len_t]
            batch_seq_ids = pl_module.all_gather(batch_seq_ids)  # shape: (world_size, validation_sequence_ids)
            task_ids = pl_module.all_gather(task_ids)
        # transpose and flatten is used to later distribute tasks evenly among gpus when using ddp
        batch_seq_ids = batch_seq_ids.cpu().numpy().T.flatten()
        task_ids = task_ids.cpu().numpy().T.flatten()
        self.task_to_id_dict = defaultdict(list)
        self.full_task_to_id_dict = defaultdict(list)
        unique_task_ids = np.unique(task_ids)
        # how many rollouts we want to test per task each validation epoch
        n_tasks = self.num_rollouts_per_task
        for task_id in unique_task_ids:
            all_task_ids = batch_seq_ids[np.where(task_ids == task_id)[0]]
            self.task_to_id_dict[self.tasks.id_to_task[task_id]] = self.pick_task_ids(all_task_ids, n_tasks)
            self.full_task_to_id_dict[self.tasks.id_to_task[task_id]] = all_task_ids
        self.id_to_task_dict = defaultdict(set)
        self.groundtruth_task_counter = torch.LongTensor([0] * self.tasks.num_tasks)  # .to(self.device)
        for k, v in self.task_to_id_dict.items():
            for i in v:
                self.id_to_task_dict[i] |= {k}
            self.groundtruth_task_counter[self.tasks.task_to_id[k]] = len(v)

    def env_rollouts(
        self,
        batch: Dict[
            str,
            Dict,
        ],
        pl_module: LightningModule,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: tuple(
               val_obs: Tensor,
               val_rgbs: tuple(Tensor, ),
               val_depths: tuple(Tensor, ),
               val_acts: Tensor,
               val_lang: Tensor,
               info: Dict,
               idx: int
            pl_module: LightningModule
        Returns:
            rollout_task_counter: tensor counting the number of successful tasks rollouts in this batch
        """
        state_obs = batch["robot_obs"]
        rgb_obs = batch["rgb_obs"]
        depth_obs = batch["depth_obs"]
        reset_info = batch["state_info"]
        idx = batch["idx"]
        # create tensor of zeros to count number of successful tasks in
        counter = {}

        for mod in self.modalities:
            rollout_task_counter = torch.LongTensor([0] * self.tasks.num_tasks).to(self.device)
            for i, global_idx in enumerate(idx):
                # check if sequence should be evaluated with rollout
                if int(global_idx) in self.id_to_task_dict:
                    # get set of task(s) that where originally performed. Use set because theoretically
                    # there can be more than one task solved in one sequence
                    groundtruth_task = self.id_to_task_dict[int(global_idx)]
                    # reset env to state of first step in the episode
                    obs = self.env.reset(reset_info, i, 0)
                    start_info = self.env.get_info()

                    if mod == "lang":
                        # language goal
                        _task = np.random.choice(list(groundtruth_task))
                        goal = self.val_annotations[_task][0]
                    else:
                        # goal image is last step of the episode
                        goal = {
                            "rgb_obs": {k: v[i, -1].unsqueeze(0).unsqueeze(0) for k, v in rgb_obs.items()},  # type: ignore
                            "depth_obs": {k: v[i, -1].unsqueeze(0).unsqueeze(0) for k, v in depth_obs.items()},  # type: ignore
                            "robot_obs": state_obs[i, -1].unsqueeze(0).unsqueeze(0),
                        }

                    # only save video of first task execution per rollout
                    record_video = self.video and np.any(
                        np.asarray([int(global_idx) == self.task_to_id_dict[task][0] for task in groundtruth_task])
                    )
                    if record_video:
                        self.rollout_video.new_video(tag=get_video_tag(groundtruth_task, mod))
                    pl_module.reset()  # type: ignore
                    success = False
                    for step in range(self.ep_len):
                        action = pl_module.step(obs, goal)  # type: ignore
                        obs, _, _, current_info = self.env.step(action)
                        if record_video:
                            # update video
                            self.rollout_video.update(obs["rgb_obs"]["rgb_static"])
                        # check if current step solves a task
                        current_task_info = self.tasks.get_task_info_for_set(start_info, current_info, groundtruth_task)
                        # check if a task was achieved and if that task is a subset of the original tasks
                        # we do not just want to solve any task, we want to solve the task that was proposed
                        if len(current_task_info) > 0:
                            for task in current_task_info:
                                task_id = self.tasks.task_to_id[task]
                                # count successful task rollouts
                                rollout_task_counter[task_id] += 1
                            # skip current sequence if task was achieved
                            success = True
                            break
                    if record_video:
                        if self.add_goal_thumbnail:
                            if mod == "lang":
                                self.rollout_video.add_language_instruction(goal)
                            else:
                                self.rollout_video.add_goal_thumbnail(rgb_obs["rgb_static"][i, -1])
                        self.rollout_video.draw_outcome(success)
                        self.rollout_video.write_to_tmp()

            counter[mod] = rollout_task_counter  # type: ignore
        # return counter of successful tasks for this batch
        return counter

    def get_task_info_of_batch(
        self,
        batch: Dict[
            str,
            Any,
        ],
    ) -> Tuple[List, List]:
        """
        Called in the first validation epoch for every batch. This method checks which tasks where successfully
        performed in batch by resetting env to first and last state of the sequence.
        Args:
            batch: tuple(
               val_obs: Tensor,
               val_rgbs: tuple(Tensor, ),
               val_depths: tuple(Tensor, ),
               val_acts: Tensor,
               val_lang: Tensor,
               info: Dict,
               idx: int
        Returns:
            task_ids: list of task ids of successful tasks in this batch
            batch_seq_ids: list sequence indices of successful tasks in this batch
        """
        task_ids = []
        batch_seq_ids = []
        reset_info = batch["state_info"]
        state_obs = batch["robot_obs"]
        idx = batch["idx"]
        batch_size = state_obs.shape[0]
        for i in get_portion_of_batch_ids(self.check_percentage_of_batch, batch_size):
            # reset env to state of last step in the episode (goal state)
            self.env.reset(reset_info, i, -1)
            goal_info = self.env.get_info()
            # reset env to state of first step in the episode
            self.env.reset(reset_info, i, 0)
            start_info = self.env.get_info()

            # check if task was achieved in sequence
            task_info = self.tasks.get_task_info(start_info, goal_info)
            if len(task_info) != 1:
                continue
            for task in task_info:
                task_ids.append(self.tasks.task_to_id[task])
                batch_seq_ids.append(idx.cpu().numpy()[i])
        return task_ids, batch_seq_ids

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:  # type: ignore
        checkpoint["task_to_id_dict"] = self.task_to_id_dict
        checkpoint["id_to_task_dict"] = self.id_to_task_dict
        checkpoint["groundtruth_task_counter"] = self.groundtruth_task_counter

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        self.task_to_id_dict = checkpoint.get("task_to_id_dict", None)
        self.id_to_task_dict = checkpoint.get("id_to_task_dict", None)
        self.groundtruth_task_counter = checkpoint.get("groundtruth_task_counter", None)
