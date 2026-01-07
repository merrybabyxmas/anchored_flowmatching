"""
AFM Hybrid Phase Sampler with Identity Retention
Stage 2에서 Identity 보존을 위한 80/20 Hybrid Sampling 구현
"""

import torch
from typing import Tuple

from ltxv_trainer.timestep_samplers import TimestepSampler
from ltxv_trainer.afm_phase_manager import AFMPhaseManager, AFMTrainingStage


class AFMHybridPhaseSampler(TimestepSampler):
    """
    AFM Hybrid Phase Sampler with Identity Retention
    
    Stage 2에서 Identity 망각을 방지하기 위한 Hybrid Sampling:
    - Stage 1: 100% Global Phase [0.2, 1.0]
    - Stage 2: 80% Local Phase [0.0, 0.2] + 20% Global Phase [0.2, 1.0]
    """
    
    def __init__(self, phase_manager: AFMPhaseManager, identity_retention_ratio: float = 0.2):
        """
        Args:
            phase_manager: AFM Phase Manager 인스턴스
            identity_retention_ratio: Stage 2에서 Stage 1 샘플 유지 비율 (기본 20%)
        """
        self.phase_manager = phase_manager
        self.identity_retention_ratio = identity_retention_ratio
        self.local_focus_ratio = 1.0 - identity_retention_ratio
    
    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:
        """Hybrid Phase Sampling with Identity Retention"""
        current_stage = self.phase_manager.current_stage
        
        if current_stage == AFMTrainingStage.GLOBAL_IDENTITY:
            # Stage 1: 100% Global Phase [0.2, 1.0]
            t_min, t_max = 0.2, 1.0
            return torch.rand(batch_size, device=device) * (t_max - t_min) + t_min
            
        elif current_stage == AFMTrainingStage.LOCAL_MOTION:
            # Stage 2: Identity-Reinforced Hybrid Sampling (IRHS)
            # 40% Identity retention for texture/face feature preservation
            num_local_samples = int(batch_size * self.local_focus_ratio)    # 60%
            num_identity_samples = batch_size - num_local_samples           # 40%
            
            # Local Phase 샘플링: [0.0, 0.2] (Motion Focus)
            local_samples = torch.rand(num_local_samples, device=device) * 0.2
            
            # Identity Phase 샘플링: [0.2, 1.0] (Identity Retention)
            identity_samples = torch.rand(num_identity_samples, device=device) * 0.8 + 0.2
            
            # 결합 및 섞기
            all_samples = torch.cat([local_samples, identity_samples])
            shuffled_indices = torch.randperm(batch_size, device=device)
            
            return all_samples[shuffled_indices]
        
        else:
            # Fallback: 균등 샘플링
            return torch.rand(batch_size, device=device)
    
    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """배치 텐서에 대한 Hybrid Sampling"""
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")
        
        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, device=batch.device)
    
    def get_sampling_stats(self, t: torch.Tensor) -> dict:
        """
        샘플링 통계 반환 (디버깅용)
        """
        current_stage = self.phase_manager.current_stage
        
        if current_stage == AFMTrainingStage.LOCAL_MOTION:
            local_mask = t <= 0.2
            identity_mask = t > 0.2
            
            local_ratio = local_mask.float().mean().item()
            identity_ratio = identity_mask.float().mean().item()
            
            return {
                'stage': current_stage.value,
                'local_samples_ratio': local_ratio,
                'identity_samples_ratio': identity_ratio,
                'target_local_ratio': self.local_focus_ratio,
                'target_identity_ratio': self.identity_retention_ratio,
                'hybrid_sampling_active': True
            }
        else:
            return {
                'stage': current_stage.value,
                'local_samples_ratio': 0.0,
                'identity_samples_ratio': 1.0,
                'hybrid_sampling_active': False
            }