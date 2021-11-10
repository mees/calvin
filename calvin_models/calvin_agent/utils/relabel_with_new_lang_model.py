import argparse
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

"""This script allows for re-annotating video sequences of PlayData.
   Parameters:
        · +path=/path/to/current/auto_lang_ann.npy
        · +name_folder=name_to_new_annotations
   NLP model selection:
        · model.nlp_model=mini -> 'paraphrase-MiniLM-L6-v2'
        · model.nlp_model=multi -> 'paraphrase-multilingual-mpnet-base-v2'
        · model.nlp_model=mpnet -> 'paraphrase-mpnet-base-v2'
"""


@hydra.main(config_path="../../conf", config_name="lang_ann.yaml")
def main(cfg: DictConfig) -> None:
    print("Loading data")
    folder = ["training", "validation"]
    for i in folder:
        path = Path(cfg.path) / i
        split = path / "lang_annotations/auto_lang_ann.npy"
        data = np.load(split, allow_pickle=True).reshape(-1)[0]
        task_ann = cfg.val_instructions if "val" in i else cfg.train_instructions
        if cfg.reannotate and "train" in i:
            print("Re-annotating sequences...")
            data["language"]["ann"] = [
                task_ann[task][np.random.randint(len(task_ann[task]))] for task in data["language"]["task"]
            ]
        print("Loading Language Model")
        model = hydra.utils.instantiate(cfg.model)
        print(f"Computing Embeddings with Model --> {cfg.model}")
        data["language"]["emb"] = model(data["language"]["ann"]).cpu().numpy()
        print("Saving data")
        save_path = path / cfg.name_folder
        save_path.mkdir(exist_ok=True)
        np.save(save_path / "auto_lang_ann.npy", data)


if __name__ == "__main__":
    main()
