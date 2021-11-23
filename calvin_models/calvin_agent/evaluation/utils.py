import logging
from pathlib import Path

from calvin_agent.models.play_lmp import PlayLMP
from calvin_agent.utils.utils import get_last_checkpoint
import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pytorch_lightning import seed_everything
import torch

logger = logging.getLogger(__name__)


def get_default_model_and_env(train_folder, dataset_path, checkpoint=None):
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
    cfg = OmegaConf.load(train_cfg_path)
    cfg = OmegaConf.create(OmegaConf.to_yaml(cfg).replace("calvin_models.", ""))
    hydra.initialize(".")
    seed_everything(cfg.seed, workers=True)  # type: ignore
    # since we don't use the trainer during inference, manually set up data_module
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["lang"]
    device = torch.device("cuda:0")
    env = hydra.utils.instantiate(cfg.callbacks.rollout.env_cfg, dataset, device, show_gui=False)

    if checkpoint is None:
        checkpoint = get_last_checkpoint(Path(train_folder))

    print(f"Loading model from {checkpoint}")
    model = PlayLMP.load_from_checkpoint(checkpoint)
    model.freeze()
    if cfg.model.decoder.get("load_action_bounds", False):
        model.action_decoder._setup_action_bounds(cfg.datamodule.root_data_dir, None, None, True)
    model = model.cuda(device)
    print("Successfully loaded model.")

    return model, env, data_module


def join_vis_lang(img, lang_text):
    """Takes as input an image and a language instruction and visualizes them with cv2"""
    img = img[:, :, ::-1].copy()
    img = cv2.resize(img, (500, 500))
    if lang_text != "":
        coord = (1, 490)
        font_scale = 0.7
        thickness = 1
        cv2.putText(
            img,
            text=lang_text,
            org=coord,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(0, 0, 0),
            thickness=thickness * 3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text=lang_text,
            org=coord,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    cv2.imshow("simulation cam", img)
    cv2.waitKey(1)


class DefaultLangEmbeddings:
    def __init__(self, dataset_path):
        embeddings = np.load(
            Path(dataset_path) / "validation/lang_annotations/embeddings.npy", allow_pickle=True
        ).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}
        self.device = torch.device("cuda:0")

    def get_lang_goal(self, task):
        return {"lang": torch.from_numpy(self.lang_embeddings[task]).to(self.device).squeeze(0)}


def get_eval_env_state():
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    scene_obs = np.array(
        [
            -5.39181361e-08,
            0.00000000e00,
            0.00000000e00,
            2.89120579e-20,
            0.00000000e00,
            0.00000000e00,
            -3.02401299e-08,
            -1.20000018e-01,
            4.59990006e-01,
            -1.53907893e-06,
            -9.26957745e-07,
            1.57000003e00,
            1.99994053e-01,
            -1.20005201e-01,
            4.59989301e-01,
            2.60083197e-04,
            -2.97388397e-04,
            -3.88975496e-08,
            9.99703015e-02,
            8.07779584e-02,
            4.60145309e-01,
            -2.46010770e-03,
            -1.02184019e-03,
            1.57194293e00,
        ]
    )
    return robot_obs, scene_obs


def imshow_tensor(window, img_tensor, wait=0, resize=True, keypoints=None):
    img_tensor = img_tensor.squeeze()
    img = np.transpose(img_tensor.cpu().numpy(), (1, 2, 0))
    img = np.clip(((img / 2) + 0.5) * 255, 0, 255).astype(np.uint8)

    if keypoints is not None:
        key_coords = np.clip(keypoints * 200 + 100, 0, 200)
        key_coords = key_coords.reshape(-1, 2)
        cv_kp1 = [cv2.KeyPoint(x=pt[1], y=pt[0], _size=1) for pt in key_coords]
        img = cv2.drawKeypoints(img, cv_kp1, None, color=(255, 0, 0))

    if resize:
        cv2.imshow(window, cv2.resize(img[:, :, ::-1], (500, 500)))
    else:
        cv2.imshow(window, img[:, :, ::-1])
    cv2.waitKey(wait)


def print_task_log(demo_task_counter, live_task_counter, mod):
    print()
    logger.info(f"Modality: {mod}")
    for task in demo_task_counter:
        logger.info(
            f"{task}: SR = {(live_task_counter[task] / demo_task_counter[task]) * 100:.0f}%"
            + f" |  {live_task_counter[task]} of {demo_task_counter[task]}"
        )
    logger.info(
        f"Average Success Rate {mod} = "
        + f"{(sum(live_task_counter.values()) / s if (s := sum(demo_task_counter.values())) > 0 else 0) * 100:.0f}% "
    )
    logger.info(
        f"Success Rates averaged throughout classes = {np.mean([live_task_counter[task] / demo_task_counter[task] for task in demo_task_counter]) * 100:.0f}%"
    )


def get_checkpoint(cfg):
    try:
        checkpoint = cfg.load_checkpoint
    except MissingMandatoryValue:
        checkpoint = get_last_checkpoint(Path(cfg.train_folder))
    return checkpoint


def format_sftp_path(cfg):
    """
    When using network mount from nautilus, format path
    """
    if cfg.train_folder.startswith("sftp"):
        cfg.train_folder = "/run/user/9984/gvfs/sftp:host=" + cfg.train_folder[7:]
