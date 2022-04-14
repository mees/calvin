from pathlib import Path
from typing import Dict

import hydra
import numpy as np
from omegaconf import DictConfig

"""This script allows for re-annotating video sequences of PlayData.
   Parameters:
        · +path=/path/to/current/auto_lang_ann.npy
        · +name_folder=name_to_new_annotations
   New annotations sampling from 'annotations=expert' defined in expert.yaml
   NLP model selection:
        · model.nlp_model=mini -> 'paraphrase-MiniLM-L6-v2'
        · model.nlp_model=multi -> 'paraphrase-multilingual-mpnet-base-v2'
        · model.nlp_model=mpnet -> 'paraphrase-mpnet-base-v2'
"""


@hydra.main(config_path="../../conf", config_name="lang_ann.yaml")
def main(cfg: DictConfig) -> None:
    print("Loading data")
    path = Path(cfg.path)
    data = np.load(path, allow_pickle=True).reshape(-1)[0]
    if "training" in cfg.path:
        print("using training instructions...")
        task_ann = cfg.train_instructions
    else:
        print("using validation instructions...")
        task_ann = cfg.val_instructions
    if cfg.reannotate:
        print("Re-annotating sequences...")
        data["language"]["ann"] = [
            task_ann[task][np.random.randint(len(task_ann[task]))] for task in data["language"]["task"]
        ]
    print("Loading Language Model")
    model = hydra.utils.instantiate(cfg.model)
    print(f"Computing Embeddings with Model --> {cfg.model}")
    data["language"]["emb"] = model(data["language"]["ann"]).cpu().numpy()
    print("Saving data")
    save_path = path.parent / ".." / cfg.name_folder
    save_path.mkdir(exist_ok=True)
    np.save(save_path / "auto_lang_ann.npy", data)

    if "validation" in cfg.path:
        embeddings: Dict = {}
        for task, ann in cfg.val_instructions.items():
            embeddings[task] = {}
            language_embedding = model(list(ann))
            embeddings[task]["emb"] = language_embedding.cpu().numpy()
            embeddings[task]["ann"] = ann
        np.save(save_path / "embeddings", embeddings)  # type:ignore
        print("Done saving val language embeddings for Rollouts !")


if __name__ == "__main__":
    main()
