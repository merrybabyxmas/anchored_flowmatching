"""
AFM Phase-Aware Timestep Samplers
Phase Separation에 특화된 타임스텝 샘플링 시스템
"""

import torch
from typing import Optional

from ltxv_trainer.timestep_samplers import TimestepSampler
from ltxv_trainer.afm_phase_manager import AFMPhaseManager, AFMTrainingStage


class AFMPhaseSampler(TimestepSampler):
    """
    AFM Phase-Aware Timestep Sampler
    
    현재 학습 Stage에 따라 자동으로 타임스텝 샘플링 범위를 조정
    - Stage 1 (Global Identity): t ∈ [0.2, 1.0] 
    - Stage 2 (Local Motion): t ∈ [0.0, 0.2]
    """
    
    def __init__(self, phase_manager: AFMPhaseManager, noise_factor: float = 0.01):
        """
        Args:
            phase_manager: AFM Phase Manager 인스턴스
            noise_factor: 경계 근처에서의 미세한 노이즈 추가 (수치 안정성)
        """
        self.phase_manager = phase_manager
        self.noise_factor = noise_factor
    
    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:
        """현재 Stage에 맞는 타임스텝 샘플링"""
        t_min, t_max = self.phase_manager.get_timestep_range()
        
        # 균등 분포 샘플링
        samples = torch.rand(batch_size, device=device) * (t_max - t_min) + t_min
        
        # 경계 근처 수치 안정성을 위한 미세 조정
        if self.noise_factor > 0:
            noise = torch.randn(batch_size, device=device) * self.noise_factor
            samples = torch.clamp(samples + noise, min=t_min, max=t_max)
            
        return samples
    
    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """배치 텐서에 대한 타임스텝 샘플링"""
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")
        
        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, device=batch.device)


class AFMHybridSampler(TimestepSampler):
    """
    AFM Hybrid Sampler
    
    Motion-Centric Importance Sampling + Phase-Aware Sampling의 결합
    - 현재 Stage의 범위 내에서 70/30 Local/Global 비율 유지
    """
    
    def __init__(self, phase_manager: AFMPhaseManager, local_ratio: float = 0.7, t_boundary: float = 0.2):
        """
        Args:
            phase_manager: AFM Phase Manager 인스턴스  
            local_ratio: Local Phase 샘플 비율 (Stage 2에서만 의미 있음)
            t_boundary: Local/Global 경계값
        """
        self.phase_manager = phase_manager
        self.local_ratio = local_ratio
        self.global_ratio = 1.0 - local_ratio
        self.t_boundary = t_boundary
    
    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:
        """Hybrid 샘플링"""
        current_stage = self.phase_manager.current_stage
        t_min, t_max = self.phase_manager.get_timestep_range()
        
        if current_stage == AFMTrainingStage.GLOBAL_IDENTITY:
            # Stage 1: t ∈ [0.2, 1.0] 균등 샘플링
            return torch.rand(batch_size, device=device) * (t_max - t_min) + t_min
            
        elif current_stage == AFMTrainingStage.LOCAL_MOTION:
            # Stage 2: t ∈ [0.0, 0.2] 100% 집중 샘플링
            return torch.rand(batch_size, device=device) * (t_max - t_min) + t_min
        
        else:
            # Fallback: 전체 범위 MCIS 적용
            return self._motion_centric_sampling(batch_size, device)
    
    def _motion_centric_sampling(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """원래 MCIS 로직 (Stage 구분 없을 때 사용)"""
        num_local_samples = int(batch_size * self.local_ratio)
        num_global_samples = batch_size - num_local_samples
        
        # Local Phase 샘플링: [0, t_boundary]
        local_samples = torch.rand(num_local_samples, device=device) * self.t_boundary
        
        # Global Phase 샘플링: (t_boundary, 1.0]  
        global_samples = torch.rand(num_global_samples, device=device) * (1.0 - self.t_boundary) + self.t_boundary
        
        # 결합 및 섞기
        all_samples = torch.cat([local_samples, global_samples])
        shuffled_indices = torch.randperm(batch_size, device=device)
        
        return all_samples[shuffled_indices]
    
    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """배치 텐서에 대한 샘플링"""
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")
        
        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, device=batch.device)
    
    def get_importance_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Phase-Aware Importance Weight 계산
        
        현재 Stage에 맞게 가중치 조정
        """
        current_stage = self.phase_manager.current_stage
        
        if current_stage == AFMTrainingStage.GLOBAL_IDENTITY:
            # Stage 1: 균등 샘플링이므로 균등 가중치
            return torch.ones_like(t)
            
        elif current_stage == AFMTrainingStage.LOCAL_MOTION:
            # Stage 2: Local 범위 내 균등 샘플링이므로 균등 가중치
            return torch.ones_like(t) 
        
        else:
            # Fallback: 원래 MCIS 가중치
            is_local = t <= self.t_boundary
            local_weight = self.t_boundary / self.local_ratio
            global_weight = (1.0 - self.t_boundary) / self.global_ratio
            return torch.where(is_local, local_weight, global_weight)


# Import Hybrid Phase Sampler
from .afm_hybrid_sampler import AFMHybridPhaseSampler

# AFM 전용 샘플러 등록
AFM_SAMPLERS = {
    "afm_phase": AFMPhaseSampler,
    "afm_hybrid": AFMHybridSampler,
    "afm_hybrid_identity": AFMHybridPhaseSampler,  # Identity 보존용 Hybrid 샘플러
}