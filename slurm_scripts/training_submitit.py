import logging
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
import submitit

from lfp.utils.utils import get_git_commit_hash, get_last_checkpoint

logger = logging.getLogger(__name__)

"""
for starting a training with submitit on a slurm cluster with the hydra submitit program.
Not tested and currently not worked on.
"""


@hydra.main(config_path="../lfp/conf", config_name="config.yaml")
def train(cfg: DictConfig) -> None:
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(cfg.seed)
    data_module = hydra.utils.instantiate(cfg.dataset)
    model = hydra.utils.instantiate(cfg.model)
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info("Repo commit hash: {}".format(get_git_commit_hash(Path(__file__))))
    train_logger = hydra.utils.instantiate(cfg.logger) or False
    chk = get_last_checkpoint(Path.cwd())
    checkpoint_callback = hydra.utils.instantiate(cfg.callback) or False
    trainer_args = {
        **cfg.trainer,
        "logger": train_logger,
        "checkpoint_callback": checkpoint_callback,
        "resume_from_checkpoint": chk,
        "benchmark": False,
    }
    gpus = trainer_args["gpus"]
    if (
        isinstance(gpus, int)
        and (gpus > 1 or gpus == -1)
        or isinstance(gpus, str)
        and len(gpus) > 1
        or isinstance(gpus, ListConfig)
        and len(gpus) > 1
    ):
        trainer_args["accelerator"] = "ddp"
        # trainer_args['plugins'] = 'ddp_sharded'

        cwd = str(Path.cwd())
        cwd = f'"{cwd}"'
        sys.argv = sys.argv[:1]
        sys.argv.extend(
            [
                f"hydra.run.dir={cwd}",
                "hydra/hydra_logging=disabled",
                "hydra/job_logging=disabled",
            ]
        )
        overrides = OmegaConf.load(".hydra/overrides.yaml")
        for o in overrides:
            if "hydra/sweeper" in o:  # type: ignore
                continue

            if "hydra/launcher" in o:  # type: ignore
                continue

            sys.argv.append(o)  # type: ignore

    trainer = Trainer(**trainer_args)
    trainer.fit(model, datamodule=data_module)


@hydra.main(config_path="../lfp/conf", config_name="config.yaml")
def main(cfg):
    # Cleanup log folder.
    # This folder may grow rapidly especially if you have large checkpoints,
    # or submit lot of jobs. You should think about an automated way of cleaning it.
    folder = "."
    ex = submitit.AutoExecutor(folder)
    if ex.cluster == "slurm":
        print("Executor will schedule jobs on Slurm.")
    else:
        print(f"!!! Slurm executable `srun` not found. Will execute jobs on '{ex.cluster}'")

    # Specify the job requirements.
    # Reserving only as much resource as you need ensure the cluster resource are
    # efficiently allocated.
    ex.update_parameters(
        mem_gb=64,
        tasks_per_node=2,
        cpus_per_task=8,
        timeout_min=5,
        slurm_partition="alldlc_gpu-rtx2080",
        gpus_per_node=2,
    )
    job = ex.submit(train, cfg)

    print(f"Scheduled {job}.")

    # # Wait for the job to be running.
    # while job.state != "RUNNING":
    #     time.sleep(1)
    #
    # print("Run the following command to see what's happening")
    # print(f"  less +F {job.paths.stdout}")
    #
    # # Simulate preemption.
    # # Tries to stop the job after the first stage.
    # # If the job is preempted before the end of the first stage, try to increase it.
    # # If the job is not preempted, try to decrease it.
    # time.sleep(25)
    # print(f"preempting {job} after {time.time() - t0:.0f}s")
    # job._interrupt()
    #
    # score = job.result()
    # print(f"Finished training. Final score: {score}.")
    # print(f"---------------- Job output ---------------------")
    # print(job.stdout())
    # print(f"-------------------------------------------------")
    #
    # assert model_path.exists()
    # with open(model_path, "rb") as f:
    #     (scaler, clf) = pickle.load(f)
    # sparsity = np.mean(clf.coef_ == 0) * 100
    # print(f"Sparsity with L1 penalty: {sparsity / 100:.2%}")


if __name__ == "__main__":
    main()
