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
    사용자의 Ring/Zipper 알고리즘을 LTX-Video 트레이너에 통합한 핵심 전략.
    """
    def __init__(self, conditioning_config: ConditioningConfig, t_star: float = 0.8):
        super().__init__(conditioning_config)
        self.t_star = t_star
        # Ring FM 엔진 초기화 (내부에 AnchorNetwork 포함)
        self.flow_matching = create_flow_matching(
            method="ring_fm",
            t_star=t_star,
            latent_dim=128 # LTX-Video 기본 latent dim, prepare_batch에서 동적 확인
        )

    def get_data_sources(self) -> list[str]:
        return ["latents", "conditions"]

    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        # 1. 원본 데이터 추출 (B, Seq, C)
        latents_info = batch["latents"]
        x0_seq = latents_info["latents"] 
        B, S, C = x0_seq.shape
        
        F = latents_info["num_frames"][0].item()
        H = latents_info["height"][0].item()
        W = latents_info["width"][0].item()
        fps = latents_info.get("fps", [DEFAULT_FPS])[0].item()

        # 2. Ring FM 연산을 위해 Video Shape로 변환 (B, C, F, H, W)
        # LTX-Video는 (B, S, C) 형태이므로 텐서 재배열이 필요함
        x0 = x0_seq.view(B, F, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        device = x0.device

        # 3. Ring/Zipper 핵심 파라미터 생성
        t = timestep_sampler.sample_for(x0_seq).to(device) # (B,)
        
        # [원칙 2, 3] Shared Noise 생성 ε ∈ R[B, C, 1, H, W]
        noise = self.flow_matching.sample_noise(x0.shape).to(device)
        
        # [원칙 5] Anchor 계산 (Stochastic Anchor A)
        anchor, mu, log_sigma2 = self.flow_matching.compute_anchor(x0)
        
        # [원칙 6, 7] Piecewise Path 및 Teacher Velocity 계산
        z_t_vid = self.flow_matching.compute_forward_path(noise, x0, t, anchor=anchor)
        u_t_vid = self.flow_matching.compute_teacher_velocity(noise, x0, t, anchor=anchor)
        
        # 4. 모델 입력을 위해 다시 Sequence 형태로 복구 (B, S, C)
        z_t_seq = z_t_vid.permute(0, 2, 3, 4, 1).reshape(B, S, C)
        u_t_seq = u_t_vid.permute(0, 2, 3, 4, 1).reshape(B, S, C)

        # 5. 기타 컨디셔닝 준비 (기존 로직 유지)
        conditions = batch["conditions"]
        prompt_embeds = conditions["prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]
        
        # First frame conditioning 지원 (선택 사항)
        conditioning_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        # Ring FM에서는 t_star 이전 단계가 Global이므로 별도의 mask 없이도 일관성이 잡히나, 
        # 기존 코드와의 호환을 위해 0으로 채운 mask를 넘김
        
        sampled_timestep_values = torch.round(t * 1000.0).long()
        timesteps = sampled_timestep_values.unsqueeze(1).expand(B, S)

        return TrainingBatch(
            latents=z_t_seq,
            targets=u_t_seq,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=t.view(-1, 1, 1),
            conditioning_mask=conditioning_mask,
            num_frames=F,
            height=H,
            width=W,
            fps=fps,
            rope_interpolation_scale=get_rope_scale_factors(fps),
            anchor=anchor,
            junction_error=self.flow_matching.verify_junction_constraint(noise, x0, anchor)
        )

    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """
        [원칙 8] Piecewise MSE Loss.
        u_t가 이미 Ring FM 수식에 의해 t < t* (Global)와 t > t* (Local)로 구분되어 계산됨.
        """
        loss = (model_pred - batch.targets).pow(2)
        # Ring FM은 모든 프레임 학습이 중요하므로 mask 없이 전체 평균
        return loss.mean()





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



def get_training_strategy(conditioning_config: ConditioningConfig) -> TrainingStrategy:
    """
    기존 팩토리 함수에 Ring FM 모드를 추가.
    config.mode가 "ring_fm"일 때 작동하도록 설정.
    """
    conditioning_mode = conditioning_config.mode

    if conditioning_mode == "ring_fm":
        # t_star는 config에서 받아오도록 확장 가능
        t_star = getattr(conditioning_config, "t_star", 0.8)
        strategy = RingZipperTrainingStrategy(conditioning_config, t_star=t_star)
        
    elif conditioning_mode == "none":
        from .training_strategy import StandardTrainingStrategy
        strategy = StandardTrainingStrategy(conditioning_config)
    elif conditioning_mode == "reference_video":
        from .training_strategy import ReferenceVideoTrainingStrategy
        strategy = ReferenceVideoTrainingStrategy(conditioning_config)
    else:
        raise ValueError(f"Unknown conditioning mode: {conditioning_mode}")

    logger.debug(f"🎯 Using {strategy.__class__.__name__} for Ring/Zipper FM")
    return strategy
