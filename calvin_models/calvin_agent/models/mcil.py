import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from calvin_agent.models.calvin_base_model import CalvinBaseModel
from calvin_agent.models.decoders.action_decoder import ActionDecoder
import hydra
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.distributions as D

logger = logging.getLogger(__name__)


class MCIL(pl.LightningModule, CalvinBaseModel):
    """
    The lightning module used for training.

    Args:
        perceptual_encoder: DictConfig for perceptual_encoder.
        plan_proposal: DictConfig for plan_proposal network.
        plan_recognition: DictConfig for plan_recognition network.
        language_goal: DictConfig for language_goal encoder.
        visual_goal: DictConfig for visual_goal encoder.
        action_decoder: DictConfig for action_decoder.
        kl_beta: Weight for KL loss term.
        optimizer: DictConfig for optimizer.
    """

    def __init__(
        self,
        perceptual_encoder: DictConfig,
        plan_proposal: DictConfig,
        plan_recognition: DictConfig,
        visual_goal: DictConfig,
        language_goal: DictConfig,
        action_decoder: DictConfig,
        kl_beta: float,
        optimizer: DictConfig,
        replan_freq: int = 30,
    ):
        super(MCIL, self).__init__()
        self.perceptual_encoder = hydra.utils.instantiate(perceptual_encoder)
        self.setup_input_sizes(
            self.perceptual_encoder,
            plan_proposal,
            plan_recognition,
            visual_goal,
            action_decoder,
        )
        # plan networks
        self.plan_proposal = hydra.utils.instantiate(plan_proposal)
        self.plan_recognition = hydra.utils.instantiate(plan_recognition)

        # goal encoders
        self.visual_goal = hydra.utils.instantiate(visual_goal)
        self.language_goal = hydra.utils.instantiate(language_goal) if language_goal else None

        # policy network
        self.action_decoder: ActionDecoder = hydra.utils.instantiate(action_decoder)

        self.kl_beta = kl_beta
        self.modality_scope = "vis"
        self.optimizer_config = optimizer
        # workaround to resolve hydra config file before calling save_hyperparams  until they fix this issue upstream
        # without this, there is conflict between lightning and hydra
        action_decoder.out_features = action_decoder.out_features

        self.optimizer_config["lr"] = self.optimizer_config["lr"]
        self.save_hyperparameters()

        # for inference
        self.rollout_step_counter = 0
        self.replan_freq = replan_freq
        self.latent_goal = None
        self.plan = None
        self.lang_embeddings = None

    @staticmethod
    def setup_input_sizes(
        perceptual_encoder,
        plan_proposal,
        plan_recognition,
        visual_goal,
        action_decoder,
    ):
        """
        Configure the input feature sizes of the respective parts of the network.

        Args:
            perceptual_encoder: DictConfig for perceptual encoder.
            plan_proposal: DictConfig for plan proposal network.
            plan_recognition: DictConfig for plan recognition network.
            visual_goal: DictConfig for visual goal encoder.
            action_decoder: DictConfig for action decoder network.
        """
        plan_proposal.perceptual_features = perceptual_encoder.latent_size
        plan_recognition.in_features = perceptual_encoder.latent_size
        visual_goal.in_features = perceptual_encoder.latent_size
        action_decoder.perceptual_features = perceptual_encoder.latent_size

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        return optimizer

    def lmp_train(
        self, perceptual_emb: torch.Tensor, latent_goal: torch.Tensor, train_acts: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.distributions.Distribution, torch.distributions.Distribution
    ]:
        """
        Main forward pass for training step after encoding raw inputs.

        Args:
            perceptual_emb: Encoded input modalities.
            latent_goal: Goal embedding (visual or language goal).
            train_acts: Ground truth actions.

        Returns:
            kl_loss: KL loss
            action_loss: Behavior cloning action loss.
            total_loss: Sum of kl_loss and action_loss.
            pp_dist: Plan proposal distribution.
            pr_dist: Plan recognition distribution
        """
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
        """
        Main forward pass for validation step after encoding raw inputs.

        Args:
            perceptual_emb: Encoded input modalities.
            latent_goal: Goal embedding (visual or language goal).
            actions: Groundtruth actions.

        Returns:
            sampled_plan_pp: Plan sampled from plan proposal network.
            action_loss_pp: Behavior cloning action loss computed with plan proposal network.
            sampled_plan_pr: Plan sampled from plan recognition network.
            action_loss_pr: Behavior cloning action loss computed with plan recognition network.
            kl_loss: KL loss
            mae_pp: Mean absolute error (L1) of action sampled with input from plan proposal network w.r.t ground truth.
            mae_pr: Mean absolute error of action sampled with input from plan recognition network w.r.t ground truth.
            gripper_sr_pp: Success rate of binary gripper action sampled with input from plan proposal network.
            gripper_sr_pr: Success rate of binary gripper action sampled with input from plan recognition network.
        """
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

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.


        Returns:
            loss tensor
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
            if "lang" in self.modality_scope:
                latent_goal = self.language_goal(dataset_batch["lang"])
            else:
                latent_goal = self.visual_goal(perceptual_emb[:, -1])
            kl, act_loss, mod_loss, pp_dist, pr_dist = self.lmp_train(
                perceptual_emb, latent_goal, dataset_batch["actions"]
            )
            kl_loss += kl
            action_loss += act_loss
            total_loss += mod_loss
            self.log(f"train/kl_loss_scaled_{self.modality_scope}", kl, on_step=False, on_epoch=True)
            self.log(f"train/action_loss_{self.modality_scope}", act_loss, on_step=False, on_epoch=True)
            self.log(f"train/total_loss_{self.modality_scope}", mod_loss, on_step=False, on_epoch=True)
        total_loss = total_loss / len(batch)  # divide accumulated gradients by number of datasets
        kl_loss = kl_loss / len(batch)
        action_loss = action_loss / len(batch)
        self.log("train/kl_loss", kl_loss, on_step=False, on_epoch=True)
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True)
        return total_loss

    def compute_kl_loss(
        self, pr_dist: torch.distributions.Distribution, pp_dist: torch.distributions.Distribution
    ) -> torch.Tensor:
        """
        Compute the KL divergence loss between the distributions of the plan recognition and plan proposal network.

        Args:
            pr_dist: Distribution produced by plan recognition network.
            pp_dist: Distribution produced by plan proposal network.

        Returns:
            Scaled KL loss.
        """
        kl_loss = D.kl_divergence(pr_dist, pp_dist).mean()
        kl_loss_scaled = kl_loss * self.kl_beta
        return kl_loss_scaled

    def set_kl_beta(self, kl_beta):
        """Set kl_beta from Callback"""
        self.kl_beta = kl_beta

    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.

        Returns:
            Dictionary containing losses and the sampled plans of plan recognition and plan proposal networks.
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
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0

    def step(self, obs, goal):
        """
        Do one step of inference with the model.

        Args:
            obs (dict): Observation from environment.
            goal (str or dict): The goal as a natural language instruction or dictionary with goal images.

        Returns:
            Predicted action.
        """
        # replan every replan_freq steps (default 30 i.e every second)
        if self.rollout_step_counter % self.replan_freq == 0:
            if isinstance(goal, str):
                embedded_lang = torch.from_numpy(self.lang_embeddings[goal]).to(self.device).squeeze(0).float()
                self.plan, self.latent_goal = self.get_pp_plan_lang(obs, embedded_lang)
            else:
                self.plan, self.latent_goal = self.get_pp_plan_vision(obs, goal)
        # use plan to predict actions with current observations
        action = self.predict_with_plan(obs, self.latent_goal, self.plan)
        self.rollout_step_counter += 1
        return action

    def load_lang_embeddings(self, embeddings_path):
        """
        This has to be called before inference. Loads the lang embeddings from the dataset.

        Args:
            embeddings_path: Path to <dataset>/validation/embeddings.npy
        """
        embeddings = np.load(embeddings_path, allow_pickle=True).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}

    def predict_with_plan(
        self,
        obs: Dict[str, Any],
        latent_goal: torch.Tensor,
        sampled_plan: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pass observation, goal and plan through decoder to get predicted action.

        Args:
            obs: Observation from environment.
            latent_goal: Encoded goal.
            sampled_plan: Sampled plan proposal plan.

        Returns:
            Predicted action.
        """
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(obs["rgb_obs"], obs["depth_obs"], obs["robot_obs"])
            action = self.action_decoder.act(sampled_plan, perceptual_emb, latent_goal)

        return action

    def get_pp_plan_vision(self, obs: dict, goal: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use plan proposal network to sample new plan using a visual goal embedding.

        Args:
            obs: Observation from environment.
            goal: Goal observation (vision & proprioception).

        Returns:
            sampled_plan: Sampled plan.
            latent_goal: Encoded visual goal.
        """
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
        self.action_decoder.clear_hidden_state()
        return sampled_plan, latent_goal

    def get_pp_plan_lang(self, obs: dict, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use plan proposal network to sample new plan using a visual goal embedding.

        Args:
            obs: Observation from environment.
            goal: Embedded language instruction.

        Returns:
            sampled_plan: Sampled plan.
            latent_goal: Encoded language goal.
        """
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(obs["rgb_obs"], obs["depth_obs"], obs["robot_obs"])
            latent_goal = self.language_goal(goal)
            # ------------Plan Proposal------------ #
            pp_dist = self.plan_proposal(perceptual_emb[:, 0], latent_goal)
            sampled_plan = pp_dist.sample()  # sample from proposal net
        self.action_decoder.clear_hidden_state()
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
