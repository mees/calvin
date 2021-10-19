import logging
from pathlib import Path

import numpy as np
from omegaconf.errors import MissingMandatoryValue
import cv2

from calvin.utils.utils import get_last_checkpoint

logger = logging.getLogger(__name__)


def get_eval_env_state():
    robot_obs = np.array([ -0.07633776, -0.01318682,  0.54709562,  3.02227836, -0.41871135,  1.69439876,
                          0.07999909, -0.79498086 , 1.00384044 , 1.93246088 ,-2.42119664 ,-1.07050417,
                          2.19612673,  0.83544771 , 1.         ])
    scene_obs = np.array([ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                           0.00000000e+00,  0.00000000e+00,  9.99843551e-02, -1.20015679e-01,
                           4.59984516e-01,  7.87400988e-04, -7.82234531e-04,  3.66303263e-07,
                          -1.00032898e-01, -8.00142771e-02,  4.59975160e-01, -1.64390920e-03,
                          -7.15795502e-04,  1.57000020e+00,  2.29965221e-01,  7.99551135e-02,
                           4.59998809e-01,  4.59888857e-04, -5.50255798e-03,  1.56999367e+00])
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
