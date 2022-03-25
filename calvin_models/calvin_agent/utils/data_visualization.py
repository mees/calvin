import logging

import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

logger = logging.getLogger(__name__)

from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
import numpy as np


def visualize(data):
    seq_img = data[1][0][0].numpy()
    title = data[4][0]
    s, c, h, w = seq_img.shape
    seq_img = np.transpose(seq_img, (0, 2, 3, 1))
    imgs = []
    fig = plt.figure()
    for j in range(s):
        # imgRGB = seq_img[j].astype(int)
        imgRGB = seq_img[j]
        imgRGB = (imgRGB - imgRGB.min()) / (imgRGB.max() - imgRGB.min())
        img = plt.imshow(imgRGB, animated=True)
        imgs.append([img])
    ArtistAnimation(fig, imgs, interval=50)
    plt.title(title)
    plt.show()


@hydra.main(config_path="../../conf", config_name="default.yaml")
def train(cfg: DictConfig) -> None:
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(cfg.seed)
    data_module = hydra.utils.instantiate(cfg.dataset, num_workers=0)
    data_module.setup()
    train = data_module.train_dataloader()
    dataset = train["lang"]
    logger.info(f"Dataset Size: {len(dataset)}")
    for i, lang in enumerate(dataset):
        logger.info(f"Element : {i}")
        visualize(lang)


if __name__ == "__main__":
    train()
