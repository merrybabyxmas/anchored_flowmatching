"""
AFM Phase-Aware Training Strategy
통합된 Phase Separation 학습 전략 구현
"""

import random
import torch
import wandb
from typing import Any
from pydantic import BaseModel

from ltxv_trainer import logger
from ltxv_trainer.config import ConditioningConfig
from ltxv_trainer.ltxv_utils import get_rope_scale_factors
from ltxv_trainer.timestep_samplers import TimestepSampler
from ltxv_trainer.training_strategies import TrainingBatch, TrainingStrategy

from .afm_phase_manager import AFMPhaseManager, AFMTrainingStage
from .afm_timestep_samplers import AFMPhaseSampler
from .ring_zipper_flow import FlowMatchingBase, create_flow_matching


class AFMTrainingBatch(TrainingBatch):
    """AFM 전용 Training Batch - Phase 정보 추가"""
    
    # 기존 TrainingBatch 필드들 상속
    afm_stage: AFMTrainingStage
    motion_gain: float
    stage_loss_weights: tuple[float, float]  # (local_weight, global_weight)
    
    model_config = {"arbitrary_types_allowed": True}


class AFMPhaseTrainingStrategy(TrainingStrategy):
    """
    AFM Phase-Aware Training Strategy
    
    Phase Manager와 연동하여 자동으로 Stage별 학습 전략을 적용
    - Stage 1: Global Identity Formation (t ∈ [0.2, 1.0])
    - Stage 2: Local Motion Refinement (t ∈ [0.0, 0.2])
    """
    
    def __init__(self, conditioning_config: ConditioningConfig, phase_manager: AFMPhaseManager):
        super().__init__(conditioning_config)
        self.phase_manager = phase_manager
        self.t_star = getattr(conditioning_config, 't_star', 0.2)
        
        # Flow Matching 엔진 초기화
        self.flow_matching = create_flow_matching(
            method="ring_fm",
            t_star=self.t_star,
            latent_dim=128
        )
        
        # Phase-Aware Timestep Sampler
        self.phase_sampler = AFMPhaseSampler(self.phase_manager)
        
        self.accelerator = None
        self.global_step = 0
        
        logger.info(f"🎯 AFM Phase Training Strategy initialized with t_star={self.t_star}")
    
    def update_step(self, step: int):
        """스텝 업데이트 및 Phase Manager 동기화"""
        self.global_step = step
        stage_changed = self.phase_manager.update_step(step)
        
        if stage_changed:
            # Stage 전환 시 추가 로직 (필요하다면)
            current_stage = self.phase_manager.current_stage
            logger.info(f"🔄 Training strategy adapted to {current_stage.value}")
    
    def get_data_sources(self) -> list[str]:
        return ["latents", "conditions"]
    
    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> AFMTrainingBatch:
        """Phase-Aware Batch 준비"""
        
        # 1. 기본 데이터 추출 및 변환
        latents_info = batch["latents"]
        x0_seq = latents_info["latents"]
        B, S, C = x0_seq.shape
        F, H, W = latents_info["num_frames"][0].item(), latents_info["height"][0].item(), latents_info["width"][0].item()
        
        x0 = x0_seq.view(B, F, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        device = x0.device
        
        # 2. Phase-Aware 타임스텝 샘플링
        t = self.phase_sampler.sample_for(x0_seq).to(device)
        
        # 3. 현재 Stage 정보 가져오기
        current_stage = self.phase_manager.current_stage
        motion_gain = self.phase_manager.get_motion_gain()
        loss_weights = self.phase_manager.get_loss_weights()
        
        # 4. Stage별 로깅
        t_mean = t.mean().item()
        t_min, t_max = self.phase_manager.get_timestep_range()
        
        stage_log = {
            "afm/current_stage": current_stage.value,
            "afm/t_mean": t_mean,
            "afm/t_range_min": t_min,
            "afm/t_range_max": t_max,
            "afm/motion_gain": motion_gain,
            "afm/samples_in_range": ((t >= t_min) & (t <= t_max)).float().mean().item()
        }
        
        if wandb.run is not None:
            wandb.log(stage_log)
        
        # 터미널 로깅 (매 50스텝마다)
        if self.global_step % 50 == 0:
            logger.info(
                f"[AFM Step {self.global_step}] Stage: {current_stage.value}, "
                f"t_range: [{t_min:.2f}, {t_max:.2f}], t_mean: {t_mean:.4f}, "
                f"motion_gain: {motion_gain:.1f}x"
            )
        
        # 5. [RAB] Sparse Anchor 생성
        anchor, anchor_mu, anchor_log_sigma2 = self.flow_matching.compute_anchor(x0)

        # 6. [RAB] Flow Matching 연산 (Unified Bridge Path)
        noise = self.flow_matching.sample_noise(x0.shape).to(device)
        z_t_vid = self.flow_matching.compute_forward_path(noise, x0, t, anchor=anchor)

        # [RAB] Residual Velocity (simplified interface)
        u_t_vid = self.flow_matching.compute_teacher_velocity(noise, x0, t, anchor=anchor)

        # [RAB] Get diagnostic metrics
        rab_metrics = self.flow_matching.get_rab_metrics()

        # Legacy compatibility: store reference norm if needed
        if current_stage == AFMTrainingStage.GLOBAL_IDENTITY:
            u_flat = u_t_vid.reshape(B, -1)
            current_norm = torch.linalg.vector_norm(u_flat, ord=2, dim=1).mean().item()
            alpha = 0.99
            if hasattr(self, '_stage1_reference_norm'):
                self._stage1_reference_norm = alpha * self._stage1_reference_norm + (1 - alpha) * current_norm
            else:
                self._stage1_reference_norm = current_norm
        
        # 7. Sequence 형태로 변환 + [TEMPORAL CONTRAST INJECTION]
        z_t_seq = z_t_vid.permute(0, 2, 3, 4, 1).reshape(B, S, C)
        u_t_seq = u_t_vid.permute(0, 2, 3, 4, 1).reshape(B, S, C)
        
        # [TEMPORAL CONTRAST] Frame-wise Divergence Boost (5x Temporal Signal Amplification)
        # 프레임별로 서로 다른 temporal signal을 강제로 주입하여 평균화 방지
        frame_positions = torch.arange(F, device=device, dtype=torch.float32)  # [0, 1, 2, ..., F-1]
        frame_contrast = 5.0 * torch.sin(frame_positions * 3.14159 / (F-1))  # 5x amplified sinusoidal contrast
        
        # Apply frame-wise contrast to velocity targets (B, F, H, W, C) -> (B, S, C)
        u_t_contrast = u_t_vid.clone()
        for f in range(F):
            frame_scale = 1.0 + 0.3 * frame_contrast[f]  # ±30% frame-wise variation
            u_t_contrast[:, :, f, :, :] = u_t_contrast[:, :, f, :, :] * frame_scale
        
        u_t_seq = u_t_contrast.permute(0, 2, 3, 4, 1).reshape(B, S, C)
        u_t_seq = torch.clamp(u_t_seq, min=-2.0, max=2.0)
        
        # 8. 텍스트 조건 준비
        conditions = batch["conditions"]
        prompt_embeds = conditions["prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]
        
        sampled_timestep_values = torch.round(t * 1000.0).long()
        timesteps = sampled_timestep_values.unsqueeze(1).expand(B, S)
        
        return AFMTrainingBatch(
            latents=z_t_seq,
            targets=u_t_seq,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=t.view(-1, 1, 1),
            conditioning_mask=torch.zeros(B, S, dtype=torch.bool, device=device),
            num_frames=F, height=H, width=W,
            fps=latents_info.get("fps", [24])[0].item(),
            rope_interpolation_scale=get_rope_scale_factors(24),
            
            # AFM 전용 필드
            afm_stage=current_stage,
            motion_gain=motion_gain,
            stage_loss_weights=loss_weights
        )
    
    def compute_loss(self, model_pred: torch.Tensor, batch: AFMTrainingBatch) -> torch.Tensor:
        """
        Phase-Aware Loss Computation
        Stage별 Loss 가중치로 Identity/Motion 학습 제어
        """
        
        # 1. 기본 MSE Loss
        loss = (model_pred - batch.targets).pow(2)
        
        # 2. 현재 Stage의 Loss 가중치 적용
        local_weight, global_weight = batch.stage_loss_weights
        t_flat = batch.sigmas.squeeze()
        t_threshold = self.t_star
        
        # Stage별 Loss 마스킹
        mask_local = t_flat <= t_threshold
        mask_global = t_flat > t_threshold
        
        # Phase-Specific Loss Weighting
        if batch.afm_stage == AFMTrainingStage.GLOBAL_IDENTITY:
            # Stage 1: Global Loss만 활성화, Local Loss 차단
            stage_weights = torch.where(mask_local, 0.0, global_weight)
            focus_description = "Identity Formation"
            
        elif batch.afm_stage == AFMTrainingStage.LOCAL_MOTION:
            # Stage 2: Local Loss만 활성화, Global Loss 차단
            stage_weights = torch.where(mask_local, local_weight, 0.0)
            focus_description = "Motion Refinement"
            
        else:
            # Fallback: 기본 가중치
            stage_weights = torch.where(mask_local, local_weight, global_weight)
            focus_description = "Mixed Training"
        
        # 3. Dynamic Loss Scaling based on target_norm
        # Stage별로 target norm이 다를 수 있으므로 정규화하여 균형 맞춤
        B = model_pred.shape[0]
        target_flat = batch.targets.reshape(B, -1)
        target_norm_per_sample = torch.linalg.vector_norm(target_flat, ord=2, dim=1)  # (B,)
        
        # 각 Stage별 target norm 통계
        local_mask = mask_local
        global_mask = mask_global
        
        # Stage별 평균 target norm 계산
        if local_mask.any():
            local_target_norm_mean = target_norm_per_sample[local_mask].mean().item()
        else:
            local_target_norm_mean = 1.0
            
        if global_mask.any():
            global_target_norm_mean = target_norm_per_sample[global_mask].mean().item()
        else:
            global_target_norm_mean = 1.0
        
        # Dynamic Loss Scaling: target norm에 역수를 취해 정규화
        # 목표: 모든 Stage의 loss가 비슷한 크기를 갖도록 조정
        reference_norm = max(local_target_norm_mean, global_target_norm_mean, 0.1)  # 안정성을 위한 최소값
        
        dynamic_local_scale = reference_norm / max(local_target_norm_mean, 0.1)
        dynamic_global_scale = reference_norm / max(global_target_norm_mean, 0.1)
        
        # Stage별 Dynamic Scaling 적용
        dynamic_scale = torch.where(local_mask, dynamic_local_scale, dynamic_global_scale)
        
        # 최종 가중치 = Stage Weight × Dynamic Scale
        final_weights = stage_weights.unsqueeze(-1).unsqueeze(-1) * dynamic_scale.unsqueeze(-1).unsqueeze(-1)
        
        # 3. Loss 적용 (Dynamic Scaling 포함)
        weighted_loss = loss * final_weights
        base_loss = weighted_loss.mean()
        
        # [TEMPORAL DIVERSITY LOSS] Frame-wise Variance Penalty to Prevent Averaging
        temporal_diversity_penalty = 0.0
        if F > 1:  # Only apply if we have multiple frames
            # Reshape predictions per frame: (B, S, C) -> (B, F, spatial_tokens, C)
            spatial_tokens = S // F
            pred_per_frame = model_pred.reshape(B, F, spatial_tokens, model_pred.shape[-1])  # (B, F, spatial, C)
            
            # Compute frame-wise predictions (average across spatial)
            frame_preds = pred_per_frame.mean(dim=2)  # (B, F, C) - averaged prediction per frame
            
            # Frame-wise variance within each batch
            frame_variance = frame_preds.var(dim=1, unbiased=False).mean()  # Mean variance across frames and channels
            
            # Exponential penalty: force diversity, heavily penalize low variance
            # λ = 2.0 (aggressive penalty), minimum variance = 0.01
            min_variance_threshold = 0.01
            diversity_lambda = 2.0
            temporal_diversity_penalty = diversity_lambda * torch.exp(-10.0 * torch.clamp(frame_variance, min=1e-8))
            
            # Add to loss
            final_loss = base_loss + temporal_diversity_penalty
        else:
            final_loss = base_loss
        
        # 4. 상세 메트릭 계산
        pred_flat = model_pred.reshape(B, -1)
        target_flat = batch.targets.reshape(B, -1)
        
        # 코사인 유사도
        import torch.nn.functional as F
        cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean().item()
        
        # L2 Norm with Predictive Norm Regularization
        target_norm_per_sample = torch.linalg.vector_norm(target_flat, ord=2, dim=1)
        pred_norm_per_sample = torch.linalg.vector_norm(pred_flat, ord=2, dim=1)
        
        target_norm = target_norm_per_sample.mean().item()
        pred_norm = pred_norm_per_sample.mean().item()
        
        # Advanced Pred-Norm Variance Regularization (Stage 2 only)
        pred_norm_variance = 0.0
        norm_variance_penalty = 0.0
        scale_regularization_penalty = 0.0
        current_stage = batch.afm_stage  # Get current stage from batch
        
        if current_stage == AFMTrainingStage.LOCAL_MOTION:
            # Debug variance calculation - 실제 pred_norm이 요동치는 상황을 정확히 포착
            if B > 1:
                pred_norm_variance = pred_norm_per_sample.var().item()
            else:
                # Single sample일 때도 이전 배치들의 norm 값과 비교하여 variance 추정
                # Maintain a rolling buffer of recent pred_norm values
                if not hasattr(self, '_norm_history_buffer'):
                    self._norm_history_buffer = []
                
                self._norm_history_buffer.append(pred_norm)
                # Keep last 10 values for variance estimation
                if len(self._norm_history_buffer) > 10:
                    self._norm_history_buffer.pop(0)
                
                # Calculate variance from historical buffer
                if len(self._norm_history_buffer) >= 3:
                    import statistics
                    pred_norm_variance = statistics.variance(self._norm_history_buffer)
                else:
                    # Not enough history - use magnitude-based variance proxy
                    expected_norm_center = 450.0  # Target stable norm
                    deviation = abs(pred_norm - expected_norm_center)
                    pred_norm_variance = deviation ** 2  # Squared deviation as variance proxy
            
            # Aggressive Scale Regularization: Manifold Forced Anchoring
            batch_mean_norm = pred_norm_per_sample.mean().item()
            if batch_mean_norm > 450.0:  # Aggressive threshold for texture preservation
                # Exponential penalty for strong enforcement
                overshoot = batch_mean_norm - 450.0
                scale_penalty_factor = 0.05  # Strong penalty factor
                scale_regularization_penalty = scale_penalty_factor * (overshoot ** 1.2)  # Superlinear growth
                final_loss = final_loss + scale_regularization_penalty
            
            # Variance penalty for high inter-sample variance
            if pred_norm_variance > 10000.0:  # Threshold for dangerous variance
                variance_penalty_factor = 0.002  # Conservative penalty
                norm_variance_penalty = variance_penalty_factor * pred_norm_variance
                final_loss = final_loss + norm_variance_penalty
        
        # Stage별 Loss 분석
        loss_per_sample = loss.reshape(B, -1).mean(dim=1)
        
        local_loss_raw = loss_per_sample[mask_local].mean().item() if mask_local.any() else 0.0
        global_loss_raw = loss_per_sample[mask_global].mean().item() if mask_global.any() else 0.0
        
        # 활성 Loss (가중치 적용 후)
        weighted_loss_per_sample = weighted_loss.reshape(B, -1).mean(dim=1)
        active_loss = weighted_loss_per_sample[weighted_loss_per_sample > 0].mean().item() if (weighted_loss_per_sample > 0).any() else 0.0
        
        # 5. AFM 전용 메트릭 로깅 (SNR + Dynamic Scaling 포함)
        
        # SNR 메트릭 수집
        snr_metrics = self.flow_matching.get_snr_metrics()
        
        # Hybrid Sampling 통계 (만약 사용 중이라면)
        if hasattr(self.phase_sampler, 'get_sampling_stats'):
            sampling_stats = self.phase_sampler.get_sampling_stats(t_flat)
        else:
            sampling_stats = {}
        
        afm_metrics = {
            # 기본 메트릭
            "loss/cosine_similarity": cosine_sim,
            "loss/target_norm": target_norm,
            "loss/pred_norm": pred_norm,
            
            # Stage-Specific Metrics
            f"afm_loss/local_raw": local_loss_raw,
            f"afm_loss/global_raw": global_loss_raw,
            f"afm_loss/active_loss": active_loss,
            f"afm_loss/stage_focus": focus_description,
            
            # Phase Progress
            "afm_loss/local_samples_ratio": mask_local.float().mean().item(),
            "afm_loss/motion_gain_applied": batch.motion_gain,
            
            # Temporal Diversity Metrics (Averaging Collapse Prevention)
            "afm_temporal/diversity_penalty": temporal_diversity_penalty.item() if isinstance(temporal_diversity_penalty, torch.Tensor) else temporal_diversity_penalty,
            "afm_temporal/frame_variance": frame_variance.item() if F > 1 and 'frame_variance' in locals() else 0.0,
            
            # Dynamic Loss Scaling Metrics
            "afm_snr/local_target_norm": local_target_norm_mean,
            "afm_snr/global_target_norm": global_target_norm_mean,
            "afm_snr/dynamic_local_scale": dynamic_local_scale,
            "afm_snr/dynamic_global_scale": dynamic_global_scale,
            "afm_snr/reference_norm": reference_norm,
            
            # Stage 1 기준 norm (SNR Rescaling)
            "afm_snr/stage1_reference_norm": getattr(self, '_stage1_reference_norm', 1.0),
            
            # Advanced Predictive Norm Regularization
            "afm_stability/pred_norm_variance": pred_norm_variance,
            "afm_stability/norm_variance_penalty": norm_variance_penalty,
            "afm_stability/scale_regularization_penalty": scale_regularization_penalty,
            "afm_stability/batch_mean_norm": pred_norm_per_sample.mean().item() if current_stage == AFMTrainingStage.LOCAL_MOTION else 0.0,
        }
        
        # SNR 메트릭 추가
        for key, value in snr_metrics.items():
            afm_metrics[f"afm_snr/{key}"] = value
            
        # Sampling 통계 추가
        for key, value in sampling_stats.items():
            afm_metrics[f"afm_sampling/{key}"] = value
        
        # Phase Manager 메트릭 결합
        afm_metrics.update(self.phase_manager.get_stage_metrics())
        
        if wandb.run is not None:
            wandb.log(afm_metrics)
        
        # 터미널 로깅 (매 50스텝마다)
        if self.global_step % 50 == 0:
            stability_info = ""
            if current_stage == AFMTrainingStage.LOCAL_MOTION:
                stability_info = f", Pred_Var: {pred_norm_variance:.1f}"
                if scale_regularization_penalty > 0:
                    stability_info += f", Scale_Penalty: {scale_regularization_penalty:.4f}"
                if norm_variance_penalty > 0:
                    stability_info += f", Var_Penalty: {norm_variance_penalty:.4f}"
            
            logger.info(
                f"[AFM Loss Step {self.global_step}] Stage: {batch.afm_stage.value} ({focus_description})\n"
                f"  Similarity: cosine={cosine_sim:.4f}, target_norm={target_norm:.4f}, pred_norm={pred_norm:.4f}{stability_info}\n"
                f"  Raw Loss - Local: {local_loss_raw:.4f}, Global: {global_loss_raw:.4f}\n"
                f"  Active Loss: {active_loss:.4f}, Motion Gain: {batch.motion_gain:.1f}x"
            )
        
        return final_loss


def create_afm_training_strategy(conditioning_config: ConditioningConfig, 
                                current_step: int = 0, 
                                auto_transition_step: int = 10000) -> AFMPhaseTrainingStrategy:
    """
    AFM Training Strategy Factory
    
    Args:
        conditioning_config: 조건부 학습 설정
        current_step: 현재 스텝 (체크포인트 로딩 시)
        auto_transition_step: 자동 Stage 전환 스텝
    """
    
    # Phase Manager 초기화
    phase_manager = AFMPhaseManager(
        current_step=current_step,
        auto_transition_step=auto_transition_step
    )
    
    # Training Strategy 생성
    strategy = AFMPhaseTrainingStrategy(conditioning_config, phase_manager)
    strategy.global_step = current_step
    
    logger.info(
        f"✅ AFM Phase Training Strategy created - "
        f"Current Stage: {phase_manager.current_stage.value}, Step: {current_step}"
    )
    
    return strategy