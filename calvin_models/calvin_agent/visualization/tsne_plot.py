import io
import logging
from typing import Any, Optional

from calvin_agent.rollout.rollout import Rollout
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


def plotly_fig2array(fig):
    """convert Plotly fig to  an array"""
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


class TSNEPlot(Callback):
    def __init__(self, perplexity, n_jobs, plot_percentage, opacity, marker_size):
        self.perplexity = perplexity
        self.n_jobs = n_jobs
        self.plot_percentage = plot_percentage
        self.opacity = opacity
        self.marker_size = marker_size
        self.task_labels = None
        self.sampled_plans = []
        self.all_idx = []

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.sampled_plans.append(outputs["sampled_plan_pp_vis"])  # type: ignore
        self.all_idx.append(outputs["idx_vis"])  # type: ignore

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if pl_module.global_step > 0:
            if self.task_labels is None:
                self._create_task_labels(trainer)

            sampled_plans = torch.cat(self.sampled_plans)
            all_idx = torch.cat(self.all_idx)
            self.sampled_plans = []
            self.all_idx = []
            if dist.is_available() and dist.is_initialized():
                sampled_plans = pl_module.all_gather(sampled_plans)
                all_idx = pl_module.all_gather(all_idx)

                if dist.get_rank() != 0:
                    return

            x_tsne = self._get_tsne(sampled_plans.view(-1, pl_module.action_decoder.plan_features))  # type: ignore
            if self.task_labels is not None:
                self._create_tsne_figure(
                    label_list=self.task_labels,
                    x_tsne=x_tsne,
                    all_idx=all_idx,
                    step=pl_module.global_step,
                    logger=pl_module.logger,
                    name="task_consistency",
                )

    def _get_tsne(self, sampled_plans):
        x_tsne = TSNE(perplexity=self.perplexity, n_jobs=self.n_jobs).fit_transform(sampled_plans.cpu())
        return x_tsne

    def _create_tsne_figure(self, label_list, x_tsne, all_idx, step, logger, name):
        """compute t-SNE plot of embeddings os a task to visualize temporal consistency"""
        # because with ddp, data doesn't come ordered anymore
        labels = label_list[torch.flatten(all_idx).cpu()]
        non_task_ids = np.random.choice(
            n := np.where(labels == -1)[0], replace=False, size=int(len(n) * self.plot_percentage)
        )
        task_ids = np.random.choice(
            n := np.where(labels != -1)[0], replace=False, size=int(len(n) * self.plot_percentage)
        )
        tasks = [self.id_to_task[i] for i in labels[task_ids]]
        symbol_seq = ["circle", "square", "diamond", "cross"]
        assert x_tsne.shape[0] == len(labels), "plt X shape {}, label len {}".format(x_tsne.shape[0], len(labels))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=x_tsne[[non_task_ids], 0].flatten(),
                y=x_tsne[[non_task_ids], 1].flatten(),
                opacity=self.opacity,
                marker={"color": "black", "size": self.marker_size},
                showlegend=True,
                name="no task",
            )
        )
        task_scatter = px.scatter(
            x=x_tsne[[task_ids], 0].flatten(),
            y=x_tsne[[task_ids], 1].flatten(),
            color=tasks,
            color_discrete_sequence=px.colors.qualitative.Alphabet,
            symbol=labels[task_ids],
            symbol_sequence=symbol_seq,
            labels={"color": "Tasks"},
        )
        for scatter in task_scatter.data:
            fig.add_trace(scatter)
        self._log_figure(fig, logger, step, name)

    @staticmethod
    def _log_figure(fig, logger, step, name):
        if isinstance(logger, WandbLogger):
            logger.experiment.log({name: fig})
        else:
            logger.experiment.add_image(name, plotly_fig2array(fig), global_step=step)

    def _create_task_labels(self, trainer):
        for callback in trainer.callbacks:
            if isinstance(callback, Rollout) and callback.full_task_to_id_dict is not None:
                self.task_to_id = callback.tasks.task_to_id
                self.id_to_task = callback.tasks.id_to_task
                self.task_labels = np.zeros(len(trainer.datamodule.val_datasets["vis"])) - 1
                if not len(callback.full_task_to_id_dict):
                    self.task_labels = None
                    log.warning("No tasks found for tsne plot.")
                    return
                for task, ids in callback.full_task_to_id_dict.items():
                    self.task_labels[ids] = self.task_to_id[task]
                return
