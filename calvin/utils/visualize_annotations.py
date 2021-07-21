import logging
import os.path

import hydra
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

"""This script will collect data snt store it with a fixed window size"""

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="lang_ann.yaml")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    data_module = hydra.utils.instantiate(cfg.datamodule)
    data_module.setup()
    dataset = data_module.train_datasets["vis"]

    file_name = os.path.join(dataset.abs_datasets_dir, "auto_lang_ann.npy")
    if os.path.isfile(file_name):
        collected_data = np.load(file_name, allow_pickle=True).reshape(-1)[0]
    else:
        print("File Not Found")

    for i, idx in enumerate(collected_data["info"]["indx"]):
        seq_length = idx[1] - idx[0]
        start = dataset.episode_lookup.index(idx[0])
        imgs = []
        seq_img = dataset[start][1][0].numpy()
        s, c, h, w = seq_img.shape
        seq_img = np.transpose(seq_img, (0, 2, 3, 1))
        print("Seq length: {}".format(s))
        print("From: {} To: {}".format(idx[0], idx[1]))
        fig = plt.figure()
        for j in range(seq_length):
            imgRGB = seq_img[j]
            imgRGB = imgRGB / (imgRGB.max() - imgRGB.min())
            img = plt.imshow(imgRGB, animated=True)
            text = plt.text(
                200, 200, f"t = {j}", ha="center", va="center", size=15, bbox=dict(boxstyle="round", ec="b", lw=2)
            )

            imgs.append([img, text])
        anim = ArtistAnimation(fig, imgs, interval=50)
        plt.axis("off")
        plt.title(collected_data["language"]["ann"][i][0])
        plt.show()


if __name__ == "__main__":
    main()
