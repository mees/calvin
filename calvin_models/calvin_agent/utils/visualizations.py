# Force matplotlib to not use any Xwindows backend.
import matplotlib
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def visualize_temporal_consistency(max_batched_length_per_demo, gpus, sampled_plans, all_idx, step, logger, prefix=""):
    """compute t-SNE plot of embeddings os a task to visualize temporal consistency"""
    labels = []
    for demo in max_batched_length_per_demo:
        labels = np.concatenate((labels, np.arange(demo) / float(demo)), axis=0)
    # because with ddp, data doesn't come ordered anymore
    labels = labels[torch.flatten(all_idx).cpu()]
    colors = [plt.cm.Spectral(y_i) for y_i in labels]
    assert sampled_plans.shape[0] == len(labels), "plt X shape {}, label len {}".format(
        sampled_plans.shape[0], len(labels)
    )

    from MulticoreTSNE import MulticoreTSNE as TSNE

    x_tsne = TSNE(perplexity=40, n_jobs=8).fit_transform(sampled_plans.cpu())

    plt.close("all")
    fig, ax = plt.subplots()
    _ = ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=colors, cmap=plt.cm.Spectral)
    fig.suptitle("Temporal Consistency of Latent space")
    ax.axis("off")
    if isinstance(logger, WandbLogger):
        logger.experiment.log({prefix + "latent_embedding": wandb.Image(fig)})
    else:
        logger.experiment.add_figure(prefix + "latent_embedding", fig, global_step=step)
