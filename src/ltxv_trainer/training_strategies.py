"""Training strategies for different conditioning modes.

This module implements the Strategy Pattern to handle different training modes:
- Standard training (no conditioning)
- Reference video training (IC-LoRA mode)

Each strategy encapsulates the specific logic for preparing batches, model inputs, and loss computation.
"""

import random
from abc import ABC, abstractmethod
from typing import Any

import torch
import wandb
from pydantic import BaseModel, computed_field
from torch import Tensor

from ltxv_trainer import logger
from ltxv_trainer.config import ConditioningConfig
from ltxv_trainer.ltxv_utils import get_rope_scale_factors, prepare_video_coordinates
from ltxv_trainer.timestep_samplers import TimestepSampler

from .ring_zipper_flow import FlowMatchingBase, create_flow_matching



DEFAULT_FPS = 24  # Default frames per second for video missing in the FPS metadata


class TrainingBatch(BaseModel):
    """Container for prepared training data.

    This model holds all the prepared data needed for a training step,
    organized in a way that's agnostic to the specific training strategy.
    """

    # Core latent data
    latents: Tensor  # The main latent input to the transformer
    targets: Tensor  # The target values for loss computation

    # Text conditioning
    prompt_embeds: Tensor  # Text embeddings
    prompt_attention_mask: Tensor  # Attention mask for text

    # Timestep information
    timesteps: Tensor  # Timestep values for the transformer
    sigmas: Tensor  # Noise schedule values

    # Conditioning information
    conditioning_mask: Tensor  # Boolean mask: True = conditioning token, False = target token

    # Video metadata
    num_frames: int  # Number of frames in the video
    height: int  # Height of the video latents
    width: int  # Width of the video latents
    fps: float  # Frames per second

    # Model input parameters
    rope_interpolation_scale: list[float]  # Scaling factors for positional embeddings
    video_coords: Tensor | None = None  # Optional explicit video coordinates

    @computed_field
    @property
    def batch_size(self) -> int:
        """Compute batch size from latents tensor."""
        return self.latents.shape[0]

    @computed_field
    @property
    def sequence_length(self) -> int:
        """Compute sequence length from latents tensor."""
        return self.latents.shape[1]

    model_config = {"arbitrary_types_allowed": True}  # Allow torch.Tensor type


class TrainingStrategy(ABC):
    """Abstract base class for training strategies.

    Each strategy encapsulates the logic for a specific training mode,
    handling batch preparation, model input preparation, and loss computation.
    """

    def __init__(self, conditioning_config: ConditioningConfig):
        """Initialize strategy with conditioning configuration.

        Args:
            conditioning_config: Configuration for conditioning behavior
        """
        self.conditioning_config = conditioning_config

    @abstractmethod
    def get_data_sources(self) -> list[str] | dict[str, str]:
        """Get the required data sources for this training strategy.

        Returns:
            Dictionary mapping data directory names to output keys for the dataset
        """

    @abstractmethod
    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        """Prepare a raw data batch for training.

        Args:
            batch: Raw batch data from the dataset
            timestep_sampler: Sampler for generating timesteps and noise

        Returns:
            Prepared training batch with all necessary data
        """

    def _create_timesteps_from_conditioning_mask(
        self, conditioning_mask: Tensor, sampled_timestep_values: Tensor
    ) -> Tensor:
        """Create timesteps based on conditioning mask.

        Args:
            conditioning_mask: Boolean mask of shape (batch_size, sequence_length),
            where True = conditioning, False = target.
            sampled_timestep_values: Sampled timestep values for target tokens of shape (batch_size,)

        Returns:
            Timesteps tensor with 0 for conditioning tokens, sampled values for target tokens
        """
        # Expand sampled values to match conditioning mask shape
        expanded_timesteps = sampled_timestep_values.unsqueeze(1).expand_as(conditioning_mask)

        # Use conditioning mask to select between 0 (conditioning) and sampled values (target)
        return torch.where(conditioning_mask, 0, expanded_timesteps)

    def _create_first_frame_conditioning_mask(
        self, batch_size: int, sequence_length: int, height: int, width: int, device: torch.device
    ) -> Tensor:
        """Create conditioning mask for first frame conditioning.

        Returns:
            Boolean mask where True indicates first frame tokens (if conditioning is enabled)
        """
        conditioning_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)

        if (
            self.conditioning_config.first_frame_conditioning_p > 0
            and random.random() < self.conditioning_config.first_frame_conditioning_p
        ):
            first_frame_end_idx = height * width
            if first_frame_end_idx < sequence_length:
                conditioning_mask[:, :first_frame_end_idx] = True

        return conditioning_mask

    @staticmethod
    def prepare_model_inputs(batch: TrainingBatch) -> dict[str, Any]:
        """Prepare inputs for the transformer model.

        Args:
            batch: Prepared training data

        Returns:
            Dictionary of keyword arguments for the transformer forward call
        """

        return {
            "hidden_states": batch.latents,
            "encoder_hidden_states": batch.prompt_embeds,
            "timestep": batch.timesteps,
            "encoder_attention_mask": batch.prompt_attention_mask,
            "num_frames": batch.num_frames,
            "height": batch.height,
            "width": batch.width,
            "rope_interpolation_scale": batch.rope_interpolation_scale,
            "video_coords": batch.video_coords,
            "return_dict": False,
        }

    @abstractmethod
    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """Compute the training loss.

        Args:
            model_pred: Output from the transformer model
            batch: The prepared training data containing targets

        Returns:
            Scalar loss tensor
        """


class RingZipperTrainingStrategy(TrainingStrategy):
    """
    Identity-Anchored 2-Stage Flow Matching 전략.
    1단계(Global): Identity 수렴 학습 (t < 0.8)
    2단계(Local): Residual 디테일 학습 (t >= 0.8, 2배 가중치)
    """
    def __init__(self, conditioning_config: ConditioningConfig, t_star: float = 0.8):
        super().__init__(conditioning_config)
        self.t_star = t_star
        self.flow_matching = create_flow_matching(
            method="ring_fm",
            t_star=t_star,
            latent_dim=128
        )
        self.accelerator = None  # Will be set by trainer
        self.global_step = 0  # Will be updated by trainer

    def get_data_sources(self) -> list[str]:
        return ["latents", "conditions"]

    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        # 1. 데이터 추출 및 Video Shape 변환 (B, C, F, H, W)
        latents_info = batch["latents"]
        x0_seq = latents_info["latents"] 
        B, S, C = x0_seq.shape
        F, H, W = latents_info["num_frames"][0].item(), latents_info["height"][0].item(), latents_info["width"][0].item()
        
        x0 = x0_seq.view(B, F, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        device = x0.device

        # 2. 타임스텝 샘플링 (B,) with Motion-Centric Importance Sampling
        t = timestep_sampler.sample_for(x0_seq).to(device)
        
        # MCIS Importance Weight 계산 (샘플링 편향 보정)
        if hasattr(timestep_sampler, 'get_importance_weight'):
            importance_weights = timestep_sampler.get_importance_weight(t)
        else:
            importance_weights = torch.ones_like(t)  # Uniform sampling - no bias correction needed

        # [디버깅] Timestep 분포 및 경계값 정밀 로깅
        t_mean_precise = t.mean().item()
        # std() 계산 시 batch_size=1일 때 경고 방지
        t_std_precise = t.std().item() if t.numel() > 1 else 0.0
        t_scaled = (t * 1000.0).cpu()  # 0~1000 스케일
        t_scaled_mean = t_scaled.mean().item()
        t_scaled_std = t_scaled.std().item() if t_scaled.numel() > 1 else 0.0

        # t_boundary (0.2) 근처 샘플 탐지
        t_boundary = self.t_star  # 0.2
        boundary_tolerance = 0.01  # ±0.01 범위
        near_boundary_mask = (t >= t_boundary - boundary_tolerance) & (t <= t_boundary + boundary_tolerance)
        near_boundary_samples = t[near_boundary_mask]

        # WandB 로깅 (정밀 timestep 분포, step 자동 관리)
        timestep_log = {
            "timestep/t_mean": t_mean_precise,
            "timestep/t_std": t_std_precise,
            "timestep/t_scaled_mean": t_scaled_mean,
            "timestep/t_scaled_std": t_scaled_std,
            "timestep/near_boundary_count": near_boundary_mask.sum().item(),
        }
        if wandb.run is not None:
            wandb.log(timestep_log)

        # 터미널 로깅 (매 50스텝마다 + 경계값 근처 샘플 발견 시)
        if self.global_step % 50 == 0 or near_boundary_mask.any():
            boundary_info = ""
            if near_boundary_mask.any():
                boundary_samples_str = ", ".join([f"{s.item():.8f}" for s in near_boundary_samples[:3]])
                boundary_info = f"\n  [BOUNDARY DETECTED] t near {t_boundary}: [{boundary_samples_str}], Scaled: {(near_boundary_samples[0] * 1000).item():.2f}"

            logger.info(
                f"[TRAIN DEBUG] Step {self.global_step} - Timestep Distribution:\n"
                f"  t (0-1 scale): mean={t_mean_precise:.8f}, std={t_std_precise:.8f}\n"
                f"  t (0-1000 scale): mean={t_scaled_mean:.6f}, std={t_scaled_std:.6f}\n"
                f"  Global/Local Boundary t: {t_boundary:.8f} (Scaled: {t_boundary * 1000:.2f})"
                f"{boundary_info}"
            )

        # 3. [RAB] Sparse Anchor 생성 (blur + noise sparsification)
        # compute_anchor 내부에서 Gaussian blur와 sparse noise 추가
        anchor, anchor_mu, anchor_log_sigma2 = self.flow_matching.compute_anchor(x0)

        # 4. Noise 및 RAB Path 연산
        noise = self.flow_matching.sample_noise(x0.shape).to(device)

        # [RAB] Unified Bridge Path: x_t = α_t·noise + β_t·anchor + γ_t·data
        z_t_vid = self.flow_matching.compute_forward_path(noise, x0, t, anchor=anchor)

        # [RAB] Residual Velocity: u_t = dα/dt·noise + dβ/dt·anchor + dγ/dt·data
        u_t_vid = self.flow_matching.compute_teacher_velocity(noise, x0, t, anchor=anchor)

        # 5. Sequence 형태로 복구
        z_t_seq = z_t_vid.permute(0, 2, 3, 4, 1).reshape(B, S, C)
        u_t_seq = u_t_vid.permute(0, 2, 3, 4, 1).reshape(B, S, C)

        # [RAB] Conservative velocity clamp for VAE latent manifold
        # VAE latents have std ≈ 0.18, so velocity should be in similar range
        u_t_seq = torch.clamp(u_t_seq, min=-1.5, max=1.5)

        # ===========================
        # 📊 RAB 진단 로깅
        # ===========================

        # RAB 메트릭 가져오기
        rab_metrics = self.flow_matching.get_rab_metrics()

        # 표준편차 계산
        x0_std = x0.std().item()
        anchor_std = anchor.std().item()
        z_t_vid_std = z_t_vid.std().item()
        noise_std = noise.std().item()

        # Target norm 계산
        u_t_flat = u_t_vid.reshape(B, -1)
        target_norm = torch.linalg.vector_norm(u_t_flat, ord=2, dim=1).mean().item()

        # NaN/Inf 체크
        anchor_has_nan = torch.isnan(anchor).any().item()
        anchor_has_inf = torch.isinf(anchor).any().item()
        u_t_has_nan = torch.isnan(u_t_vid).any().item()
        u_t_has_inf = torch.isinf(u_t_vid).any().item()

        # 로그 메트릭 수집
        log_dict = {
            "data/x0_std": x0_std,
            "data/anchor_std": anchor_std,
            "data/z_t_std": z_t_vid_std,
            "data/noise_std": noise_std,
            "data/target_norm": target_norm,
            "data/anchor_has_nan": float(anchor_has_nan),
            "data/anchor_has_inf": float(anchor_has_inf),
            "data/u_t_has_nan": float(u_t_has_nan),
            "data/u_t_has_inf": float(u_t_has_inf),
            # RAB-specific metrics
            "rab/residual_norm": rab_metrics.get('residual_norm', 0.0),
            "rab/alpha_t": rab_metrics.get('alpha_t', 0.0),
            "rab/beta_t": rab_metrics.get('beta_t', 0.0),
            "rab/gamma_t": rab_metrics.get('gamma_t', 0.0),
            "rab/u_noise_contrib": rab_metrics.get('u_noise_contrib', 0.0),
            "rab/u_anchor_contrib": rab_metrics.get('u_anchor_contrib', 0.0),
            "rab/u_data_contrib": rab_metrics.get('u_data_contrib', 0.0),
        }

        # WandB 로깅 (wandb.log 직접 사용, step 자동 관리)
        if wandb.run is not None:
            wandb.log(log_dict)

        # 터미널 로깅 (매 50스텝마다)
        if self.global_step % 50 == 0:
            logger.info(
                f"[Step {self.global_step}] RAB Data Stats - "
                f"x0_std: {x0_std:.4f}, anchor_std: {anchor_std:.4f}, "
                f"z_t_std: {z_t_vid_std:.4f}, noise_std: {noise_std:.4f}, "
                f"target_norm: {target_norm:.4f}, "
                f"residual_norm: {rab_metrics.get('residual_norm', 0.0):.4f}"
            )
            logger.info(
                f"[Step {self.global_step}] RAB Bridge Weights - "
                f"α_t: {rab_metrics.get('alpha_t', 0.0):.4f}, "
                f"β_t: {rab_metrics.get('beta_t', 0.0):.4f}, "
                f"γ_t: {rab_metrics.get('gamma_t', 0.0):.4f}"
            )

        # 6. 컨디셔닝 준비
        conditions = batch["conditions"]
        prompt_embeds = conditions["prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        sampled_timestep_values = torch.round(t * 1000.0).long()
        timesteps = sampled_timestep_values.unsqueeze(1).expand(B, S)

        return TrainingBatch(
            latents=z_t_seq,
            targets=u_t_seq, # Stage 2에서는 자동으로 (GroundTruth - Anchor) 방향의 속도가 계산됨
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=t.view(-1, 1, 1), # Loss 계산을 위해 t값 보관
            conditioning_mask=torch.zeros(B, S, dtype=torch.bool, device=device),
            num_frames=F, height=H, width=W,
            fps=latents_info.get("fps", [24])[0].item(),
            rope_interpolation_scale=get_rope_scale_factors(24),
            anchor=anchor,
            importance_weights=importance_weights  # MCIS 편향 보정 가중치
        )

    def compute_loss(self, model_pred: torch.Tensor, batch: TrainingBatch) -> torch.Tensor:
        """
        [RAB] Combined MSE + Cosine Similarity Loss

        Loss = 0.8 * MSE + 0.2 * (1 - CosineSim)
        - MSE ensures magnitude accuracy
        - CosineSim ensures directional alignment
        """
        B, S, C = model_pred.shape
        t = batch.sigmas
        t_flat = t.squeeze()  # (B,)

        # Flatten for per-sample operations
        pred_flat = model_pred.reshape(B, -1)  # (B, S*C)
        target_flat = batch.targets.reshape(B, -1)  # (B, S*C)

        # Video dimensions for temporal analysis
        F_frames, H, W = batch.num_frames, batch.height, batch.width

        # 1. MSE Loss (per-sample)
        mse_per_sample = torch.mean((pred_flat - target_flat) ** 2, dim=1)  # (B,)

        # 2. Cosine Similarity Loss (per-sample)
        import torch.nn.functional as F
        cosine_sim_per_sample = F.cosine_similarity(pred_flat, target_flat, dim=1)  # (B,)
        cosine_loss_per_sample = 1.0 - cosine_sim_per_sample  # (B,) - Convert to loss

        # 3. [ANTI-AVERAGING] Temporal Diversity Loss - Forces Inter-Frame Variance
        # Reshape to video: (B, S, C) -> (B, F, H*W*C)
        pred_video = model_pred.reshape(B, F_frames, H * W * C)  # (B, F, D) where D = H*W*C
        target_video = batch.targets.reshape(B, F_frames, H * W * C)  # (B, F, D)

        # Compute inter-frame variance for predictions
        # If model outputs same value for all frames, std→0, penalty→∞
        pred_frame_std = pred_video.std(dim=1)  # (B, D) - std across frames
        pred_temporal_variance = pred_frame_std.mean(dim=1)  # (B,) - mean variance

        # Compute target inter-frame variance (ground truth motion)
        target_frame_std = target_video.std(dim=1)  # (B, D)
        target_temporal_variance = target_frame_std.mean(dim=1)  # (B,)

        # Temporal Diversity Penalty: penalize pred_variance < target_variance
        # variance_ratio = 0 (all frames identical) → penalty = 1.0
        # variance_ratio = 1 (matches target) → penalty = 0.0
        variance_ratio = pred_temporal_variance / (target_temporal_variance.clamp(min=1e-6))
        temporal_diversity_loss = torch.clamp(1.0 - variance_ratio, min=0.0) ** 2  # (B,)

        # 4. Combined Loss (per-sample)
        # 60% MSE, 15% cosine direction, 25% temporal diversity (increased for anti-averaging)
        combined_loss_per_sample = (
            0.60 * mse_per_sample +
            0.15 * cosine_loss_per_sample +
            0.25 * temporal_diversity_loss
        )  # (B,)

        # 4. MCIS Importance Weight 적용
        if hasattr(batch, 'importance_weights'):
            importance_weights = batch.importance_weights  # (B,)
        else:
            importance_weights = torch.ones_like(t_flat)  # (B,)

        # 5. Apply importance weights
        weighted_loss_per_sample = combined_loss_per_sample * importance_weights  # (B,)

        # ===========================
        # 📊 RAB 메트릭 로깅
        # ===========================

        # Compute metrics for logging
        cosine_sim_mean = cosine_sim_per_sample.mean().item()
        mse_mean = mse_per_sample.mean().item()
        temporal_div_mean = temporal_diversity_loss.mean().item()
        variance_ratio_mean = variance_ratio.mean().item()
        combined_loss_mean = combined_loss_per_sample.mean().item()

        target_norm = torch.linalg.vector_norm(target_flat, ord=2, dim=1).mean().item()
        pred_norm = torch.linalg.vector_norm(pred_flat, ord=2, dim=1).mean().item()

        # Temporal variance statistics
        pred_variance_mean = pred_temporal_variance.mean().item()
        target_variance_mean = target_temporal_variance.mean().item()

        # MCIS statistics
        avg_importance_weight = importance_weights.mean().item()

        # Log metrics
        log_dict = {
            "loss/cosine_similarity": cosine_sim_mean,
            "loss/mse": mse_mean,
            "loss/temporal_diversity": temporal_div_mean,
            "loss/variance_ratio": variance_ratio_mean,
            "loss/combined": combined_loss_mean,
            "loss/target_norm": target_norm,
            "loss/pred_norm": pred_norm,
            "loss/pred_temporal_variance": pred_variance_mean,
            "loss/target_temporal_variance": target_variance_mean,
            "loss/t_mean": t_flat.mean().item(),
            "mcis/avg_importance_weight": avg_importance_weight,
        }

        if wandb.run is not None:
            wandb.log(log_dict)

        # Terminal logging
        if self.global_step % 50 == 0:
            logger.info(
                f"[Step {self.global_step}] RAB Loss Metrics\n"
                f"  Cosine: {cosine_sim_mean:.4f}, MSE: {mse_mean:.6f}, "
                f"Temporal Div: {temporal_div_mean:.6f}, Combined: {combined_loss_mean:.6f}\n"
                f"  Variance Ratio: {variance_ratio_mean:.4f} (pred: {pred_variance_mean:.4f}, "
                f"target: {target_variance_mean:.4f})\n"
                f"  Norms - Target: {target_norm:.4f}, Pred: {pred_norm:.4f}"
            )

        # Return final loss (mean across batch)
        final_loss = weighted_loss_per_sample.mean()
        return final_loss



class StandardTrainingStrategy(TrainingStrategy):
    """Standard training strategy without conditioning.

    This strategy implements regular video generation training where:
    - Only target latents are used (no reference videos)
    - Standard noise application and loss computation
    - Single video sequence length
    - Supports first frame conditioning
    """

    def __init__(self, conditioning_config: ConditioningConfig):
        """Initialize standard training strategy.

        Args:
            conditioning_config: Configuration for conditioning behavior
        """
        super().__init__(conditioning_config)

    def get_data_sources(self) -> list[str]:
        """Standard training requires latents and text conditions."""
        return ["latents", "conditions"]

    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        """Prepare batch for standard training."""
        # Get pre-encoded latents
        latents = batch["latents"]
        target_latents = latents["latents"]

        # Note: Batch sizes > 1 are partially supported, assuming
        # num_frames, height, width, fps are the same for all batch elements.
        latent_frames = latents["num_frames"][0].item()
        latent_height = latents["height"][0].item()
        latent_width = latents["width"][0].item()

        # Handle FPS with backward compatibility for old preprocessed datasets
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get pre-encoded text conditions
        conditions = batch["conditions"]
        prompt_embeds = conditions["prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        # Create conditioning mask (only first frame conditioning for standard training)
        conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=target_latents.shape[0],
            sequence_length=target_latents.shape[1],
            height=latent_height,
            width=latent_width,
            device=target_latents.device,
        )

        # Create noise for the target latents
        sigmas = timestep_sampler.sample_for(target_latents)
        noise = torch.randn_like(target_latents, device=target_latents.device)

        # Apply noise only to non-conditioning tokens
        sigmas = sigmas.view(-1, 1, 1)
        noisy_latents = (1 - sigmas) * target_latents + sigmas * noise

        # For conditioning tokens, use clean latents instead of noisy ones
        conditioning_mask_expanded = conditioning_mask.unsqueeze(-1)  # (B, seq_len, 1)
        noisy_latents = torch.where(conditioning_mask_expanded, target_latents, noisy_latents)

        targets = noise - target_latents

        # Create timesteps based on conditioning mask
        sampled_timestep_values = torch.round(sigmas.squeeze(-1).squeeze(-1) * 1000.0).long()
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_timestep_values)

        # Use existing utility function for ROPE scale factors
        rope_interpolation_scale_factors = get_rope_scale_factors(fps)

        return TrainingBatch(
            latents=noisy_latents,
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            fps=fps,
            rope_interpolation_scale=rope_interpolation_scale_factors,
            video_coords=None,
        )

    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """Compute masked MSE loss using conditioning mask."""
        loss = (model_pred - batch.targets).pow(2)

        # Create loss mask: exclude conditioning tokens
        loss_mask = (~batch.conditioning_mask.unsqueeze(-1)).float()

        # Apply original loss computation pattern
        loss = loss.mul(loss_mask).div(loss_mask.mean())
        return loss.mean()


class ReferenceVideoTrainingStrategy(TrainingStrategy):
    """Reference video training strategy for IC-LoRA.

    This strategy implements training with reference video conditioning where:
    - Reference latents (clean) are concatenated with target latents (noised)
    - Video coordinates are doubled to handle concatenated sequence
    - Loss is computed only on the target portion (masked loss)
    - Supports first frame conditioning on the target sequence
    """

    def __init__(self, conditioning_config: ConditioningConfig):
        """Initialize with configurable reference latents directory.

        Args:
            conditioning_config: Configuration for conditioning behavior
        """
        super().__init__(conditioning_config)

    def get_data_sources(self) -> dict[str, str]:
        """IC-LoRA training requires latents, conditions, and reference latents."""
        return {
            "latents": "latents",
            "conditions": "conditions",
            self.conditioning_config.reference_latents_dir: "ref_latents",
        }

    def prepare_batch(self, batch: dict[str, dict[str, Tensor]], timestep_sampler: TimestepSampler) -> TrainingBatch:
        """Prepare batch for IC-LoRA training with reference videos."""
        # Get pre-encoded latents
        latents = batch["latents"]
        target_latents = latents["latents"]
        ref_latents = batch["ref_latents"]["latents"]

        # Note: Batch sizes > 1 are partially supported, assuming
        # num_frames, height, width, fps are the same for all batch elements.
        latent_frames = latents["num_frames"][0].item()
        latent_height = latents["height"][0].item()
        latent_width = latents["width"][0].item()

        # Handle FPS with backward compatibility for old preprocessed datasets
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get pre-encoded text conditions
        conditions = batch["conditions"]
        prompt_embeds = conditions["prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        # Create noise only for the target part
        sigmas = timestep_sampler.sample_for(target_latents)
        noise = torch.randn_like(target_latents, device=target_latents.device)
        sigmas = sigmas.view(-1, 1, 1)

        # Create conditioning mask
        batch_size = target_latents.shape[0]
        ref_seq_len = ref_latents.shape[1]
        target_seq_len = target_latents.shape[1]

        # Reference tokens are always conditioning
        ref_conditioning_mask = torch.ones(batch_size, ref_seq_len, dtype=torch.bool, device=target_latents.device)

        # Target tokens: check for first frame conditioning
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            height=latent_height,
            width=latent_width,
            device=target_latents.device,
        )

        # Combine reference and target conditioning masks
        conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)

        # Create timesteps based on conditioning mask
        sampled_timestep_values = torch.round(sigmas.squeeze(-1).squeeze(-1) * 1000.0).long()
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_timestep_values)

        # Apply noise only to target part
        noisy_target = (1 - sigmas) * target_latents + sigmas * noise

        # For first frame conditioning in target, use clean latents instead of noisy ones
        target_conditioning_mask_expanded = target_conditioning_mask.unsqueeze(-1)  # (B, target_seq_len, 1)
        noisy_target = torch.where(target_conditioning_mask_expanded, target_latents, noisy_target)

        targets = noise - target_latents

        # Concatenate reference and noisy target in the sequence dimension
        # Shape [batch, sequence_length * 2, channels]  # noqa: ERA001
        combined_latents = torch.cat([ref_latents, noisy_target], dim=1)

        # Use existing utility function for ROPE scale factors
        rope_scale_factors = get_rope_scale_factors(fps)

        # Prepare video coordinates (doubled sequence for concatenation)
        batch_size = combined_latents.shape[0]
        raw_video_coords = prepare_video_coordinates(
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            batch_size=batch_size,
            sequence_multiplier=2,  # IC-LoRA uses doubled sequence (reference + target)
            device=target_latents.device,
        )

        # Apply pre-scaling to raw coordinates.
        # The LTXVideoRotaryPosEmbed expects video_coords to be (B, 3, SeqLen) if provided.
        # It then divides video_coords[:, 0] by base_num_frames, etc.
        # So, the video_coords we pass should be: raw_coord * rope_interpolation_factor
        # (B, 2 * F * H * W)  # noqa: ERA001
        prescaled_f = raw_video_coords[..., 0] * rope_scale_factors[0]
        prescaled_h = raw_video_coords[..., 1] * rope_scale_factors[1]
        prescaled_w = raw_video_coords[..., 2] * rope_scale_factors[2]

        # Stack to (B, 3, 2*F*H*W) for the transformer's video_coords argument
        video_coords = torch.stack([prescaled_f, prescaled_h, prescaled_w], dim=1)

        return TrainingBatch(
            latents=combined_latents,
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            fps=fps,
            rope_interpolation_scale=rope_scale_factors,
            video_coords=video_coords,
        )

    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """Compute masked loss only on target portion, excluding conditioning tokens."""
        # Extract target portion from model prediction and conditioning mask
        target_seq_len = batch.targets.shape[1]
        target_pred = model_pred[:, -target_seq_len:]
        target_conditioning_mask = batch.conditioning_mask[:, -target_seq_len:]

        loss = (target_pred - batch.targets).pow(2)

        # Create loss mask: exclude conditioning tokens
        loss_mask = (~target_conditioning_mask.unsqueeze(-1)).float()

        # Apply original loss computation pattern
        loss = loss.mul(loss_mask).div(loss_mask.mean())
        return loss.mean()



def get_training_strategy(conditioning_config: ConditioningConfig, 
                         current_step: int = 0, 
                         auto_transition_step: int = 10000,
                         use_phase_separation: bool = False) -> TrainingStrategy:
    """
    Advanced Training Strategy Factory with AFM Phase Separation Support
    
    Args:
        conditioning_config: 조건부 학습 설정
        current_step: 현재 스텝 (체크포인트 로딩 시)
        auto_transition_step: AFM Stage 자동 전환 스텝
        use_phase_separation: AFM Phase Separation 활성화 여부
    """
    conditioning_mode = conditioning_config.mode

    if conditioning_mode == "ring_fm":
        if use_phase_separation:
            # AFM Phase Separation 모드
            from .afm_training_strategy import create_afm_training_strategy
            strategy = create_afm_training_strategy(
                conditioning_config, 
                current_step=current_step,
                auto_transition_step=auto_transition_step
            )
            logger.info(f"🎯 Using AFM Phase Separation Strategy")
        else:
            # 기존 통합 Ring FM 모드
            t_star = getattr(conditioning_config, "t_star", 0.2)
            strategy = RingZipperTrainingStrategy(conditioning_config, t_star=t_star)
            logger.info(f"🎯 Using Unified Ring/Zipper FM Strategy")
        
    elif conditioning_mode == "none":
        strategy = StandardTrainingStrategy(conditioning_config)
    elif conditioning_mode == "reference_video":
        strategy = ReferenceVideoTrainingStrategy(conditioning_config)
    else:
        raise ValueError(f"Unknown conditioning mode: {conditioning_mode}")

    logger.debug(f"🎯 Strategy: {strategy.__class__.__name__}")
    return strategy
