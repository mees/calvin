import logging
from typing import Dict, Optional, Tuple, Union

from calvin_agent.models.decoders.action_decoder import ActionDecoder
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch import Tensor
import torch.distributions as D
from torch.nn.functional import mse_loss

logger = logging.getLogger(__name__)


class PlayLMP(pl.LightningModule):
    def __init__(
        self,
        perceptual_encoder: DictConfig,
        plan_proposal: DictConfig,
        plan_recognition: DictConfig,
        visual_goal: DictConfig,
        language_goal: DictConfig,
        decoder: DictConfig,
        kl_beta: float,
        optimizer: DictConfig,
        replan_freq: int = 30,
    ):
        super(PlayLMP, self).__init__()
        self.perceptual_encoder = hydra.utils.instantiate(perceptual_encoder)
        self.setup_input_sizes(
            self.perceptual_encoder,
            plan_proposal,
            plan_recognition,
            visual_goal,
            decoder,
        )
        self.plan_proposal = hydra.utils.instantiate(plan_proposal)
        self.plan_recognition = hydra.utils.instantiate(plan_recognition)
        self.visual_goal = hydra.utils.instantiate(visual_goal)
        self.language_goal = hydra.utils.instantiate(language_goal) if language_goal else None
        self.action_decoder: ActionDecoder = hydra.utils.instantiate(decoder)
        self.kl_beta = kl_beta
        self.modality_scope = "vis"
        self.optimizer_config = optimizer
        # workaround to resolve hydra config file before calling save_hyperparams  until they fix this issue upstream
        # without this, there is conflict between lightning and hydra
        decoder.out_features = decoder.out_features

        self.optimizer_config["lr"] = self.optimizer_config["lr"]
        self.save_hyperparameters()

        # for inference
        self.rollout_step_counter = 0
        self.replan_freq = replan_freq
        self.latent_goal = None
        self.plan = None

    @staticmethod
    def setup_input_sizes(
        perceptual_encoder,
        plan_proposal,
        plan_recognition,
        visual_goal,
        decoder,
    ):
        plan_proposal.perceptual_features = perceptual_encoder.latent_size
        plan_recognition.in_features = perceptual_encoder.latent_size
        visual_goal.in_features = perceptual_encoder.latent_size
        decoder.perceptual_features = perceptual_encoder.latent_size

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        return optimizer

    def lmp_train(
        self, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, train_acts: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.distributions.Distribution, torch.distributions.Distribution
    ]:
        # ------------Plan Proposal------------ #
        pp_dist = self.plan_proposal(perceptual_emb[:, 0], latent_goal)  # (batch, 256) each

        # ------------Plan Recognition------------ #
        pr_dist = self.plan_recognition(perceptual_emb)  # (batch, 256) each

        sampled_plan = pr_dist.rsample()  # sample from recognition net
        action_loss = self.action_decoder.loss(sampled_plan, perceptual_emb, latent_goal, train_acts)
        kl_loss = self.compute_kl_loss(pr_dist, pp_dist)
        total_loss = action_loss + kl_loss

        return kl_loss, action_loss, total_loss, pp_dist, pr_dist

    def lmp_val(
        self, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # ------------Plan Proposal------------ #
        pp_dist = self.plan_proposal(perceptual_emb[:, 0], latent_goal)  # (batch, 256) each

        # ------------ Policy network ------------ #
        sampled_plan_pp = pp_dist.sample()  # sample from proposal net
        action_loss_pp, sample_act_pp = self.action_decoder.loss_and_act(
            sampled_plan_pp, perceptual_emb, latent_goal, actions
        )

        mae_pp = torch.nn.functional.l1_loss(
            sample_act_pp[..., :-1], actions[..., :-1], reduction="none"
        )  # (batch, seq, 6)
        mae_pp = torch.mean(mae_pp, 1)  # (batch, 6)
        # gripper action
        gripper_discrete_pp = sample_act_pp[..., -1]
        gt_gripper_act = actions[..., -1]
        m = gripper_discrete_pp > 0
        gripper_discrete_pp[m] = 1
        gripper_discrete_pp[~m] = -1
        gripper_sr_pp = torch.mean((gt_gripper_act == gripper_discrete_pp).float())

        # ------------Plan Recognition------------ #
        pr_dist = self.plan_recognition(perceptual_emb)  # (batch, 256) each

        sampled_plan_pr = pr_dist.sample()  # sample from recognition net
        action_loss_pr, sample_act_pr = self.action_decoder.loss_and_act(
            sampled_plan_pr, perceptual_emb, latent_goal, actions
        )
        mae_pr = torch.nn.functional.l1_loss(
            sample_act_pr[..., :-1], actions[..., :-1], reduction="none"
        )  # (batch, seq, 6)
        mae_pr = torch.mean(mae_pr, 1)  # (batch, 6)
        kl_loss = self.compute_kl_loss(pr_dist, pp_dist)
        # gripper action
        gripper_discrete_pr = sample_act_pr[..., -1]
        m = gripper_discrete_pr > 0
        gripper_discrete_pr[m] = 1
        gripper_discrete_pr[~m] = -1
        gripper_sr_pr = torch.mean((gt_gripper_act == gripper_discrete_pr).float())

        return (
            sampled_plan_pp,
            action_loss_pp,
            sampled_plan_pr,
            action_loss_pr,
            kl_loss,
            mae_pp,
            mae_pr,
            gripper_sr_pp,
            gripper_sr_pr,
        )

    def training_step(  # type: ignore
        self,
        batch: Dict[
            str,
            Dict,
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        batch: list( batch_dataset_vision, batch_dataset_lang, ..., batch_dataset_differentModalities)
            - batch_dataset_vision: tuple( train_obs: Tensor,
                                           train_rgbs: tuple(Tensor, ),
                                           train_depths: tuple(Tensor, ),
                                           train_acts: Tensor ),
                                           info: Dict,
                                           idx: int
            - batch_dataset_lang: tuple( train_obs: Tensor,
                                         train_rgbs: tuple(Tensor, ),
                                         train_depths: tuple(Tensor, ),
                                         train_acts: Tensor,
                                         train_lang: Tensor   ),
                                         info: Dict,
                                         idx: int
        """
        kl_loss, action_loss, total_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

        for self.modality_scope, dataset_batch in batch.items():
            perceptual_emb = self.perceptual_encoder(
                dataset_batch["rgb_obs"], dataset_batch["depth_obs"], dataset_batch["robot_obs"]
            )
            latent_goal = (
                self.visual_goal(perceptual_emb[:, -1])
                if "vis" in self.modality_scope
                else self.language_goal(dataset_batch["lang"])
            )
            kl, act_loss, mod_loss, pp_dist, pr_dist = self.lmp_train(
                perceptual_emb, latent_goal, dataset_batch["actions"]
            )
            kl_loss += kl
            action_loss += act_loss
            total_loss += mod_loss
            self.log(f"train/action_loss_{self.modality_scope}", act_loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"train/total_loss_{self.modality_scope}", mod_loss, on_step=False, on_epoch=True, sync_dist=True)
        total_loss = total_loss / len(batch)  # divide accumulated gradients by number of datasets
        kl_loss = kl_loss / len(batch)
        action_loss = action_loss / len(batch)
        self.log("train/kl_loss", kl_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def compute_kl_loss(
        self, pr_dist: torch.distributions.Distribution, pp_dist: torch.distributions.Distribution
    ) -> torch.Tensor:
        kl_loss = D.kl_divergence(pr_dist, pp_dist).mean()
        kl_loss_scaled = kl_loss * self.kl_beta
        self.log(f"train/kl_loss_{self.modality_scope}", kl_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            f"train/kl_loss_scaled_{self.modality_scope}", kl_loss_scaled, on_step=False, on_epoch=True, sync_dist=True
        )
        return kl_loss_scaled

    def set_kl_beta(self, kl_beta):
        """Set kl_beta from Callback"""
        self.kl_beta = kl_beta

    def validation_step(  # type: ignore
        self,
        batch: Dict[
            str,
            Dict,
        ],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        batch: list( batch_dataset_vision, batch_dataset_lang, ..., batch_dataset_differentModalities)
            - batch_dataset_vision: tuple( train_obs: Tensor,
                                           train_rgbs: tuple(Tensor, ),
                                           train_depths: tuple(Tensor, ),
                                           train_acts: Tensor ),
                                           info: Dict,
                                           idx: int
            - batch_dataset_lang: tuple( train_obs: Tensor,
                                         train_rgbs: tuple(Tensor, ),
                                         train_depths: tuple(Tensor, ),
                                         train_acts: Tensor,
                                         train_lang: Tensor   ),
                                         info: Dict,
                                         idx: int
        """
        output = {}
        for self.modality_scope, dataset_batch in batch.items():
            perceptual_emb = self.perceptual_encoder(
                dataset_batch["rgb_obs"], dataset_batch["depth_obs"], dataset_batch["robot_obs"]
            )
            latent_goal = (
                self.visual_goal(perceptual_emb[:, -1])
                if "vis" in self.modality_scope
                else self.language_goal(dataset_batch["lang"])
            )
            (
                sampled_plan_pp,
                action_loss_pp,
                sampled_plan_pr,
                action_loss_pr,
                kl_loss,
                mae_pp,
                mae_pr,
                gripper_sr_pp,
                gripper_sr_pr,
            ) = self.lmp_val(perceptual_emb, latent_goal, dataset_batch["actions"])
            output[f"val_action_loss_pp_{self.modality_scope}"] = action_loss_pp
            output[f"sampled_plan_pp_{self.modality_scope}"] = sampled_plan_pp
            output[f"val_action_loss_pr_{self.modality_scope}"] = action_loss_pr
            output[f"sampled_plan_pr_{self.modality_scope}"] = sampled_plan_pr
            output[f"kl_loss_{self.modality_scope}"] = kl_loss
            output[f"mae_pp_{self.modality_scope}"] = mae_pp
            output[f"mae_pr_{self.modality_scope}"] = mae_pr
            output[f"gripper_sr_pp{self.modality_scope}"] = gripper_sr_pp
            output[f"gripper_sr_pr{self.modality_scope}"] = gripper_sr_pr
            output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]

        return output

    def validation_epoch_end(self, validation_step_outputs):
        val_total_act_loss_pr = torch.tensor(0.0).to(self.device)
        val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
        val_kl_loss = torch.tensor(0.0).to(self.device)
        val_total_mae_pr = torch.tensor(0.0).to(self.device)
        val_total_mae_pp = torch.tensor(0.0).to(self.device)
        val_pos_mae_pp = torch.tensor(0.0).to(self.device)
        val_pos_mae_pr = torch.tensor(0.0).to(self.device)
        val_orn_mae_pp = torch.tensor(0.0).to(self.device)
        val_orn_mae_pr = torch.tensor(0.0).to(self.device)
        val_grip_sr_pr = torch.tensor(0.0).to(self.device)
        val_grip_sr_pp = torch.tensor(0.0).to(self.device)
        for mod in self.trainer.datamodule.modalities:
            act_loss_pp = torch.stack([x[f"val_action_loss_pp_{mod}"] for x in validation_step_outputs]).mean()
            act_loss_pr = torch.stack([x[f"val_action_loss_pr_{mod}"] for x in validation_step_outputs]).mean()
            kl_loss = torch.stack([x[f"kl_loss_{mod}"] for x in validation_step_outputs]).mean()
            mae_pp = torch.cat([x[f"mae_pp_{mod}"] for x in validation_step_outputs])
            mae_pr = torch.cat([x[f"mae_pr_{mod}"] for x in validation_step_outputs])
            pr_mae_mean = mae_pr.mean()
            pp_mae_mean = mae_pp.mean()
            pos_mae_pp = mae_pp[..., :3].mean()
            pos_mae_pr = mae_pr[..., :3].mean()
            orn_mae_pp = mae_pp[..., 3:6].mean()
            orn_mae_pr = mae_pr[..., 3:6].mean()
            grip_sr_pp = torch.stack([x[f"gripper_sr_pp{mod}"] for x in validation_step_outputs]).mean()
            grip_sr_pr = torch.stack([x[f"gripper_sr_pr{mod}"] for x in validation_step_outputs]).mean()
            val_total_mae_pr += pr_mae_mean
            val_total_mae_pp += pp_mae_mean
            val_pos_mae_pp += pos_mae_pp
            val_pos_mae_pr += pos_mae_pr
            val_orn_mae_pp += orn_mae_pp
            val_orn_mae_pr += orn_mae_pr
            val_grip_sr_pp += grip_sr_pp
            val_grip_sr_pr += grip_sr_pr
            val_total_act_loss_pp += act_loss_pp
            val_total_act_loss_pr += act_loss_pr
            val_kl_loss += kl_loss

            self.log(f"val_act/{mod}_act_loss_pp", act_loss_pp, sync_dist=True)
            self.log(f"val_act/{mod}_act_loss_pr", act_loss_pr, sync_dist=True)
            self.log(f"val_total_mae/{mod}_total_mae_pr", pr_mae_mean, sync_dist=True)
            self.log(f"val_total_mae/{mod}_total_mae_pp", pp_mae_mean, sync_dist=True)
            self.log(f"val_pos_mae/{mod}_pos_mae_pr", pos_mae_pr, sync_dist=True)
            self.log(f"val_pos_mae/{mod}_pos_mae_pp", pos_mae_pp, sync_dist=True)
            self.log(f"val_orn_mae/{mod}_orn_mae_pr", orn_mae_pr, sync_dist=True)
            self.log(f"val_orn_mae/{mod}_orn_mae_pp", orn_mae_pp, sync_dist=True)
            self.log(f"val_grip/{mod}_grip_sr_pr", grip_sr_pr, sync_dist=True)
            self.log(f"val_grip/{mod}_grip_sr_pp", grip_sr_pp, sync_dist=True)
            self.log(f"val_kl/{mod}_kl_loss", kl_loss, sync_dist=True)
        self.log(
            "val_act/action_loss_pp", val_total_act_loss_pp / len(self.trainer.datamodule.modalities), sync_dist=True
        )
        self.log(
            "val_act/action_loss_pr", val_total_act_loss_pr / len(self.trainer.datamodule.modalities), sync_dist=True
        )
        self.log("val_kl/kl_loss", val_kl_loss / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log(
            "val_total_mae/total_mae_pr", val_total_mae_pr / len(self.trainer.datamodule.modalities), sync_dist=True
        )
        self.log(
            "val_total_mae/total_mae_pp", val_total_mae_pp / len(self.trainer.datamodule.modalities), sync_dist=True
        )
        self.log("val_pos_mae/pos_mae_pr", val_pos_mae_pr / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_pos_mae/pos_mae_pp", val_pos_mae_pp / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_orn_mae/orn_mae_pr", val_orn_mae_pr / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_orn_mae/orn_mae_pp", val_orn_mae_pp / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_grip/grip_sr_pr", val_grip_sr_pr / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_grip/grip_sr_pp", val_grip_sr_pp / len(self.trainer.datamodule.modalities), sync_dist=True)

    def reset(self):
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0

    def step(self, obs, goal):
        # replan every replan_freq steps (default 30 i.e every second)
        if self.rollout_step_counter % self.replan_freq == 0:
            if "lang" in goal:
                self.plan, self.latent_goal = self.get_pp_plan_lang(obs, goal)
            else:
                self.plan, self.latent_goal = self.get_pp_plan_vision(obs, goal)
        # use plan to predict actions with current observations
        action = self.predict_with_plan(obs, self.latent_goal, self.plan)
        self.rollout_step_counter += 1
        return action

    def predict_with_plan(
        self,
        obs: Dict[str, Dict],
        latent_goal: torch.Tensor,
        sampled_plan: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(obs["rgb_obs"], obs["depth_obs"], obs["robot_obs"])
            action = self.action_decoder.act(sampled_plan, perceptual_emb, latent_goal)

        return action

    def get_pp_plan_vision(self, obs: dict, goal: dict) -> Tuple[Tensor, Tensor]:
        assert len(obs["rgb_obs"]) == len(goal["rgb_obs"])
        assert len(obs["depth_obs"]) == len(goal["depth_obs"])
        imgs = {k: torch.cat([v, goal["rgb_obs"][k]], dim=1) for k, v in obs["rgb_obs"].items()}  # (1, 2, C, H, W)
        depth_imgs = {k: torch.cat([v, goal["depth_obs"][k]], dim=1) for k, v in obs["depth_obs"].items()}
        state = torch.cat([obs["robot_obs"], goal["robot_obs"]], dim=1)
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(imgs, depth_imgs, state)
            latent_goal = self.visual_goal(perceptual_emb[:, -1])
            # ------------Plan Proposal------------ #
            pp_dist = self.plan_proposal(perceptual_emb[:, 0], latent_goal)
            sampled_plan = pp_dist.sample()  # sample from proposal net
        # self.action_decoder.clear_hidden_state()
        return sampled_plan, latent_goal

    def get_pp_plan_lang(self, obs: dict, goal: dict) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(obs["rgb_obs"], obs["depth_obs"], obs["robot_obs"])
            latent_goal = self.language_goal(goal["lang"])
            # ------------Plan Proposal------------ #
            pp_dist = self.plan_proposal(perceptual_emb[:, 0], latent_goal)
            sampled_plan = pp_dist.sample()  # sample from proposal net

        return sampled_plan, latent_goal

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_start(self) -> None:
        logger.info(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")
