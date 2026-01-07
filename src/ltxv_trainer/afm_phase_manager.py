"""
Anchored Flow Matching (AFM) Phase Manager
통합된 Phase Separation 학습 시스템

철학: 비디오 생성 난이도를 낮추기 위해 Identity와 Motion을 계층적으로 분리
- Stage 1 (Global Phase): t ∈ [0.2, 1.0] - Identity 형성
- Stage 2 (Local Phase): t ∈ [0.0, 0.2] - Motion 분화
"""

import torch
from enum import Enum
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import wandb
from ltxv_trainer import logger


class AFMTrainingStage(Enum):
    """AFM Training Stage 정의"""
    GLOBAL_IDENTITY = "global_identity"    # Stage 1: t ∈ [0.2, 1.0] Identity 형성
    LOCAL_MOTION = "local_motion"          # Stage 2: t ∈ [0.0, 0.2] Motion 분화
    

@dataclass
class AFMStageConfig:
    """각 Stage별 설정"""
    stage: AFMTrainingStage
    t_min: float
    t_max: float
    motion_gain: float
    loss_weight_local: float
    loss_weight_global: float
    description: str


class AFMPhaseManager:
    """
    AFM Phase Separation 관리자
    
    자동 Stage 전환, 체크포인트 관리, Phase별 학습 로직 제어
    """
    
    # Stage 설정 정의
    STAGE_CONFIGS = {
        AFMTrainingStage.GLOBAL_IDENTITY: AFMStageConfig(
            stage=AFMTrainingStage.GLOBAL_IDENTITY,
            t_min=0.2, t_max=1.0,
            motion_gain=1.0,                # Global에서는 Motion Gain 비활성화
            loss_weight_local=0.0,          # Local Loss 차단
            loss_weight_global=1.0,         # Global Loss만 활성화
            description="Identity Formation - 모든 프레임이 공통 앵커로 수렴"
        ),
        AFMTrainingStage.LOCAL_MOTION: AFMStageConfig(
            stage=AFMTrainingStage.LOCAL_MOTION,
            t_min=0.0, t_max=0.2,
            motion_gain=2.0,                # 2.0x Motion Gain 활성화
            loss_weight_local=10.0,         # Local Loss 극대화
            loss_weight_global=0.0,         # Global Loss 차단
            description="Motion Refinement - 앵커에서 프레임별 움직임 분화"
        )
    }
    
    def __init__(self, current_step: int = 0, auto_transition_step: Optional[int] = 10000):
        """
        AFM Phase Manager 초기화
        
        Args:
            current_step: 현재 학습 스텝
            auto_transition_step: Stage 1 → Stage 2 자동 전환 스텝 (None이면 수동 전환)
        """
        self.auto_transition_step = auto_transition_step
        self.current_step = current_step
        
        # 현재 Stage 결정
        if auto_transition_step and current_step >= auto_transition_step:
            self._current_stage = AFMTrainingStage.LOCAL_MOTION
            logger.info(f"🎯 AFM Phase Manager: Starting at Stage 2 (Local Motion) - Step {current_step}")
        else:
            self._current_stage = AFMTrainingStage.GLOBAL_IDENTITY
            logger.info(f"🎯 AFM Phase Manager: Starting at Stage 1 (Global Identity) - Step {current_step}")
            
        self.stage_transition_logged = False
    
    @property
    def current_stage(self) -> AFMTrainingStage:
        """현재 학습 Stage 반환"""
        return self._current_stage
    
    @property 
    def current_config(self) -> AFMStageConfig:
        """현재 Stage의 설정 반환"""
        return self.STAGE_CONFIGS[self._current_stage]
    
    def update_step(self, step: int) -> bool:
        """
        스텝 업데이트 및 자동 Stage 전환 확인
        
        Args:
            step: 현재 학습 스텝
            
        Returns:
            bool: Stage가 전환되었는지 여부
        """
        self.current_step = step
        
        # 자동 전환 조건 확인
        if (self.auto_transition_step and 
            step >= self.auto_transition_step and 
            self._current_stage == AFMTrainingStage.GLOBAL_IDENTITY):
            
            self._transition_to_local_motion()
            return True
            
        return False
    
    def _transition_to_local_motion(self):
        """Stage 1 → Stage 2 전환 로직"""
        old_stage = self._current_stage
        self._current_stage = AFMTrainingStage.LOCAL_MOTION
        
        if not self.stage_transition_logged:
            logger.info(
                f"\n{'='*80}\n"
                f"🔄 AFM STAGE TRANSITION - Step {self.current_step}\n"
                f"📍 From: {old_stage.value} → {self._current_stage.value}\n"
                f"🎯 Focus: Identity Formation → Motion Refinement\n"
                f"⚙️  Config: t ∈ [{self.current_config.t_min}, {self.current_config.t_max}], "
                f"Motion Gain: {self.current_config.motion_gain}x\n"
                f"{'='*80}"
            )
            
            # WandB 이벤트 로깅
            if wandb.run:
                wandb.log({
                    "afm/stage_transition": 1,
                    "afm/current_stage": self._current_stage.value,
                    "afm/transition_step": self.current_step
                }, step=self.current_step)
                
            self.stage_transition_logged = True
    
    def force_transition(self, target_stage: AFMTrainingStage):
        """수동 Stage 전환"""
        old_stage = self._current_stage
        self._current_stage = target_stage
        
        logger.info(
            f"🔧 Manual AFM Stage Transition: {old_stage.value} → {target_stage.value} "
            f"at Step {self.current_step}"
        )
        
        if wandb.run:
            wandb.log({
                "afm/manual_stage_transition": 1,
                "afm/current_stage": target_stage.value
            }, step=self.current_step)
    
    def get_timestep_range(self) -> Tuple[float, float]:
        """현재 Stage의 타임스텝 샘플링 범위 반환"""
        config = self.current_config
        return (config.t_min, config.t_max)
    
    def get_motion_gain(self, warmup_steps: int = 2000) -> float:
        """
        현재 Stage의 Motion Gain 반환 (Warm-up 포함)
        
        Stage 2에서 Motion Gain을 서서히 증가시켜 수치적 안정성 확보
        """
        base_gain = self.current_config.motion_gain
        
        if self._current_stage == AFMTrainingStage.LOCAL_MOTION and base_gain > 1.0:
            # Stage 2에서만 warm-up 적용
            stage2_start_step = self.auto_transition_step or 10000
            steps_in_stage2 = max(0, self.current_step - stage2_start_step)
            
            # Warm-up progress (0.0 ~ 1.0)
            warmup_progress = min(1.0, steps_in_stage2 / warmup_steps)
            
            # 1.0에서 target_gain까지 서서히 증가
            final_gain = 1.0 + (base_gain - 1.0) * warmup_progress
            
            return final_gain
        else:
            return base_gain
    
    def get_loss_weights(self) -> Tuple[float, float]:
        """현재 Stage의 Loss 가중치 반환 (local_weight, global_weight)"""
        config = self.current_config
        return (config.loss_weight_local, config.loss_weight_global)
    
    def should_apply_linear_alpha(self) -> bool:
        """Linear Alpha Decoupling 적용 여부"""
        return self._current_stage == AFMTrainingStage.LOCAL_MOTION
    
    def get_stage_metrics(self) -> Dict[str, Any]:
        """현재 Stage 관련 메트릭 반환"""
        config = self.current_config
        return {
            "afm/current_stage": self._current_stage.value,
            "afm/t_min": config.t_min,
            "afm/t_max": config.t_max,
            "afm/motion_gain": config.motion_gain,
            "afm/loss_weight_local": config.loss_weight_local,
            "afm/loss_weight_global": config.loss_weight_global,
            "afm/is_local_motion_stage": self._current_stage == AFMTrainingStage.LOCAL_MOTION,
            "afm/progress_to_transition": min(1.0, self.current_step / (self.auto_transition_step or float('inf')))
        }
    
    def create_checkpoint_metadata(self) -> Dict[str, Any]:
        """체크포인트 저장용 메타데이터 생성"""
        return {
            "afm_stage": self._current_stage.value,
            "afm_step": self.current_step,
            "afm_auto_transition_step": self.auto_transition_step,
            "afm_stage_config": {
                "t_min": self.current_config.t_min,
                "t_max": self.current_config.t_max,
                "motion_gain": self.current_config.motion_gain,
                "loss_weight_local": self.current_config.loss_weight_local,
                "loss_weight_global": self.current_config.loss_weight_global
            }
        }
    
    @classmethod
    def from_checkpoint(cls, checkpoint_metadata: Dict[str, Any], current_step: int) -> "AFMPhaseManager":
        """체크포인트에서 Phase Manager 복원"""
        saved_stage = AFMTrainingStage(checkpoint_metadata.get("afm_stage", "global_identity"))
        auto_transition_step = checkpoint_metadata.get("afm_auto_transition_step")
        
        manager = cls(current_step=current_step, auto_transition_step=auto_transition_step)
        manager._current_stage = saved_stage
        
        logger.info(
            f"✅ AFM Phase Manager restored from checkpoint: "
            f"Stage = {saved_stage.value}, Step = {current_step}"
        )
        
        return manager