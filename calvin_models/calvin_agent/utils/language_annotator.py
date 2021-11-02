import logging
import os.path

import hydra
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

"""This script will collect data snt store it with a fixed window size"""

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="lang_ann.yaml")
def main(cfg: DictConfig) -> None:
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    data_module = hydra.utils.instantiate(cfg.datamodule)
    bert = hydra.utils.instantiate(cfg.model)
    data_module.setup()
    if cfg.training:
        dataset = data_module.train_datasets
    else:
        dataset = data_module.val_datasets  # Tupla(obs [32,9], img tuple([32, 3, 300, 300]), tuple(), act [32,9])

    # To make sure that we dont overwrite previous annotations and always keep adding
    file_name = os.path.join(dataset.dataset_loader.abs_datasets_dir, "lang_ann.npy")
    if os.path.isfile(file_name):
        collected_data = np.load(file_name, allow_pickle=True).reshape(-1)[0]
        # start = collected_data['indx'][-1][0] + collected_data['indx'][-1][1]
        start = len(collected_data["indx"])
        logger.info("Join the language annotation number {}".format(len(collected_data["indx"])))
    else:
        collected_data = {"language": [], "indx": []}
        start = 0

    length = len(dataset)
    print(length, len(dataset.dataset_loader.episode_lookup))
    steps = int((length - start) // (length * 0.01))
    total = int(1 // 0.01)
    logger.info("Progress --> {} / {}".format(total - steps, total))
    for i in range(start, length, steps):
        imgs = []
        seq_img = dataset[i][1][0].numpy()
        s, c, h, w = seq_img.shape
        seq_img = np.transpose(seq_img, (0, 2, 3, 1))
        print("Seq length: {}".format(s))
        print("From: {} To: {}".format(i, i + s))
        fig = plt.figure()
        for j in range(s):
            imgRGB = seq_img[j].astype(int)
            img = plt.imshow(imgRGB, animated=True)
            imgs.append([img])
        anim = ArtistAnimation(fig, imgs, interval=50)
        plt.show(block=False)
        lang_ann = [input("Which instructions would you give to the robot to do: (press q to quit)\n")]
        plt.close()

        if lang_ann[0] == "q":
            break
        logger.info(
            " Added indexes: {}".format(
                (
                    dataset.dataset_loader.episode_lookup[i],
                    dataset.dataset_loader.episode_lookup[i] + dataset.window_size,
                )
            )
        )
        collected_data["language"].append(lang_ann)
        collected_data["indx"].append(
            (dataset.dataset_loader.episode_lookup[i], dataset.dataset_loader.episode_lookup[i] + dataset.window_size)
        )
    file_name = "lang_ann"
    np.save(file_name, collected_data)

    if cfg.postprocessing:
        language = [item for sublist in collected_data["language"] for item in sublist]
        language_embedding = bert(language)
        collected_data["language"] = language_embedding.unsqueeze(1)
        file_name = "lang_emb_ann"
        np.save(file_name, collected_data)
        logger.info("Done extracting language embeddings !")


if __name__ == "__main__":
    main()
