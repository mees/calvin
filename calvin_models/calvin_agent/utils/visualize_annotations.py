import logging

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from sklearn.manifold import TSNE
from tqdm import tqdm

"""This script will collect data snt store it with a fixed window size"""

logger = logging.getLogger(__name__)


def generate_single_seq_gif(seq_img, seq_length, imgs, idx, i, data):
    s, c, h, w = seq_img.shape
    seq_img = np.transpose(seq_img, (0, 2, 3, 1))
    print("Seq length: {}".format(s))
    print("From: {} To: {}".format(idx[0], idx[1]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for j in range(seq_length):
        imgRGB = seq_img[j]
        imgRGB = cv2.resize(
            ((imgRGB - imgRGB.min()) / (imgRGB.max() - imgRGB.min()) * 255).astype(np.uint8), (500, 500)
        )
        # img = plt.imshow(imgRGB, animated=True)
        # text1 = plt.text(
        #     200, 200, f"t = {j}", ha="center", va="center", size=10, bbox=dict(boxstyle="round", ec="b", lw=2)
        # )
        img = cv2.putText(imgRGB, f"t = {j}", (350, 450), font, color=(0, 0, 0), fontScale=1, thickness=2)
        img = cv2.putText(
            img, f"{i}. {data['language']['ann'][i]}", (100, 20), font, color=(0, 0, 0), fontScale=0.5, thickness=1
        )[:, :, ::-1]
        # text = plt.text(
        #     100,
        #     20,
        #     f"{i}. {data['language']['ann'][i]}",
        #     ha="center",
        #     va="center",
        #     size=10,
        #     bbox=dict(boxstyle="round", ec="b", lw=2),
        # )
        if j == 0:
            for _ in range(25):
                imgs.append(img)
        imgs.append(img)
    return imgs


def generate_all_seq_gifs(data, dataset):
    imgs = []
    # fig = plt.figure()
    for i, idx in enumerate(tqdm(data["info"]["indx"][:100])):
        seq_length = idx[1] - idx[0]
        dataset.max_window_size, dataset.min_window_size = seq_length, seq_length
        start = dataset.episode_lookup.tolist().index(idx[0])
        seq_img = dataset[start]["rgb_obs"]["rgb_static"].numpy()
        # if 'lift' in data['language']['task'][i]:
        imgs = generate_single_seq_gif(seq_img, seq_length, imgs, idx, i, data)
    return imgs


def load_data(cfg):
    seed_everything(cfg.seed)
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=4)
    data_module.prepare_data()
    data_module.setup()
    dataset = data_module.train_dataloader()["vis"].dataset

    file_name = dataset.abs_datasets_dir / cfg.lang_folder / "auto_lang_ann.npy"
    return np.load(file_name, allow_pickle=True).reshape(-1)[0], dataset


def plot_and_save_gifs(imgs):
    # anim = ArtistAnimation(fig, imgs, interval=75)
    # plt.axis("off")
    # plt.title("Annotated Sequences")
    # plt.show()
    # anim.save("/tmp/summary_lang_anns.mp4", writer="ffmpeg", fps=15)
    video = cv2.VideoWriter("/tmp/summary_lang_anns.avi", cv2.VideoWriter_fourcc(*"XVID"), 15, (500, 500))
    for img in imgs:
        video.write(img)
    video.release()


def generate_task_id(tasks):
    labels = list(sorted(set(tasks)))
    task_ids = [labels.index(task) for task in tasks]
    return task_ids


def visualize_embeddings(data, with_text=True):
    emb = data["language"]["emb"].squeeze()
    tsne_emb = TSNE(n_components=2, random_state=40, perplexity=20.0).fit_transform(emb)

    emb_2d = tsne_emb

    task_ids = generate_task_id(data["language"]["task"])

    cmap = ["orange", "blue", "green", "pink", "brown", "black", "purple", "yellow", "cyan", "red", "grey", "olive"]
    ids_in_legend = []
    for i, task_id in enumerate(task_ids):
        if task_id not in ids_in_legend:
            ids_in_legend.append(task_id)
            plt.scatter(emb_2d[i, 0], emb_2d[i, 1], color=cmap[task_id], label=data["language"]["task"][i])
            if with_text:
                plt.text(emb_2d[i, 0], emb_2d[i, 1], data["language"]["ann"][i])
        else:
            plt.scatter(emb_2d[i, 0], emb_2d[i, 1], color=cmap[task_id])
            if with_text:
                plt.text(emb_2d[i, 0], emb_2d[i, 1], data["language"]["ann"][i])
    plt.legend()
    plt.title("Language Embeddings")
    plt.show()


@hydra.main(config_path="../../conf", config_name="lang_ann.yaml")
def main(cfg: DictConfig) -> None:
    data, dataset_obj = load_data(cfg)
    # visualize_embeddings(data)
    imgs = generate_all_seq_gifs(data, dataset_obj)
    plot_and_save_gifs(imgs)


if __name__ == "__main__":
    main()
