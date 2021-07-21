# Force matplotlib to not use any Xwindows backend.
from typing import Any, List, Optional

import matplotlib
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.distributed as dist
import wandb

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class PlanPlot(Callback):
    def __init__(self, max_num_samples, perplexity=40, n_jobs=8):
        self.pr_plans_vis = []
        self.pp_plans_vis = []
        self.pr_plans_lang = []
        self.pp_plans_lang = []
        self.max_num_samples = max_num_samples
        self.perplexity = perplexity
        self.n_jobs = n_jobs
        self.mods = []

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if pl.__version__ < "1.3.0":
            encoders_dict = outputs[0][0]["extra"]["encoders_dict"]
        else:
            encoders_dict = outputs["encoders_dict"]
        self.mods = list(encoders_dict.keys())  # todo: solve this in a better way
        for mod in encoders_dict:

            pp_dist = encoders_dict[mod][0]
            pr_dist = encoders_dict[mod][1]
            pr, pp = (
                [self.pr_plans_vis, self.pp_plans_vis] if "vis" in mod else [self.pr_plans_lang, self.pp_plans_lang]
            )
            if len(pr) < self.max_num_samples:
                pr.append(pr_dist.sample().detach().cpu())
                pp.append(pp_dist.sample().detach().cpu())

    @rank_zero_only
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional = None  # type: ignore
    ) -> None:
        for mod in self.mods:
            pr, pp = (
                [self.pr_plans_vis, self.pp_plans_vis] if "vis" in mod else [self.pr_plans_lang, self.pp_plans_lang]
            )
            pr_plans = torch.cat(pr)
            pp_plans = torch.cat(pp)
            all_plans = torch.cat([pr_plans, pp_plans])
            tsne = self._get_tsne(all_plans)
            labels = np.concatenate([np.zeros(pr_plans.shape[0]), np.ones(pp_plans.shape[0])])
            # shuffle tsne plot for not having one color being plotted on top of the other one
            shuffle_ids = np.random.permutation(labels.shape[0])
            tsne = tsne[shuffle_ids]
            labels = labels[shuffle_ids]
            color_fn = lambda x: (1, 0, 0) if x == 0 else (0, 0, 1)
            legend = (
                mpatches.Patch(color=color_fn(0), label="pr_plan_" + mod),
                mpatches.Patch(color=color_fn(1), label="pp_plan_" + mod),
            )
            self._create_tsne_figure(
                labels=labels,
                color_fn=color_fn,
                x_tsne=tsne,
                step=pl_module.global_step,
                logger=pl_module.logger,
                name="plan_embeddings_" + mod,
                legend=legend,
            )
        self.pr_plans_vis = []
        self.pp_plans_vis = []
        self.pr_plans_lang = []
        self.pp_plans_lang = []

    def _get_tsne(self, sampled_plans):
        x_tsne = TSNE(perplexity=self.perplexity, n_jobs=self.n_jobs).fit_transform(sampled_plans.view(-1, 256).cpu())
        return x_tsne

    def _create_tsne_figure(self, labels, color_fn, x_tsne, step, logger, name, legend=None, sort=False):
        """compute t-SNE plot of embeddings os a task to visualize temporal consistency"""
        # because with ddp, data doesn't come ordered anymore
        colors = np.array([color_fn(y_i) for y_i in labels])
        assert x_tsne.shape[0] == len(labels), "plt X shape {}, label len {}".format(x_tsne.shape[0], len(labels))
        plt.close("all")
        fig, ax = plt.subplots()
        if sort:
            sort_ids = np.argsort(colors, axis=0)[:, 0]
            _ = ax.scatter(x_tsne[[sort_ids], 0], x_tsne[[sort_ids], 1], c=colors[sort_ids])
        else:
            _ = ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=colors)
        if legend:
            ax.legend(handles=legend)
        fig.suptitle(name)
        ax.axis("off")
        self._log_figure(fig, logger, step, name)

    @staticmethod
    def _log_figure(fig, logger, step, name):
        if isinstance(logger, WandbLogger):
            logger.experiment.log({name: wandb.Image(fig)})
        else:
            logger.experiment.add_figure(name, fig, global_step=step)


class NoPlanCallback(Callback):
    pass
