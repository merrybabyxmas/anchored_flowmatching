"""
Ring/Zipper Flow Matching Training Strategy

Adapts the existing training strategy pattern to implement exact
Ring/Zipper Flow Matching mathematics.
"""
import torch
import torch.nn.functional as F
import wandb
from typing import Dict, Any, Set, Optional
from abc import ABC, abstractmethod

from ltxv_trainer import logger
from .ring_zipper_flow import FlowMatchingBase, create_flow_matching
from .quantum_training_strategy import QuantumTrainingStrategy, get_quantum_training_strategy


class RingTrainingStrategyBase(ABC):
    """Base class for Ring/Zipper Flow Matching training strategies."""

    def __init__(self, method: str = "quantum_fm", t_star: float = 0.2):
        self.method = "quantum_fm"  # Force quantum method for better results
        self.flow_matching = create_flow_matching(
            method="quantum_fm",
            t_star=t_star,
            latent_dim=128  # Will be updated based on actual data
        )
        self.accelerator = None  # Will be set by trainer
        self.global_step = 0  # Will be updated by trainer
    
    @abstractmethod
    def get_data_sources(self) -> Set[str]:
        """Get required data sources for this training strategy."""
        pass
    
    @abstractmethod
    def prepare_batch(self, batch: Dict[str, Any], timestep_sampler) -> Dict[str, torch.Tensor]:
        """Prepare training batch with flow matching components."""
        pass
    
    @abstractmethod
    def prepare_model_inputs(self, training_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the model forward pass."""
        pass
    
    @abstractmethod
    def compute_loss(self, model_pred: torch.Tensor, training_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the flow matching loss."""
        pass


class StandardFlowMatchingStrategy(RingTrainingStrategyBase):
    """Training strategy for Standard Flow Matching baseline."""
    
    def __init__(self):
        super().__init__(method="standard_fm")
    
    def get_data_sources(self) -> Set[str]:
        """Standard FM only needs latent data."""
        return {"latents"}
    
    def prepare_batch(self, batch: Dict[str, Any], timestep_sampler) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for Standard Flow Matching training.
        
        Args:
            batch: Raw batch from dataloader
            timestep_sampler: Timestep sampling function
            
        Returns:
            training_batch: Prepared batch with noise, targets, times, velocities
        """
        # Extract latent data
        if "latents" in batch:
            latents_data = batch["latents"]
        elif "latent_conditions" in batch:
            latents_data = batch["latent_conditions"]
        else:
            raise ValueError("No latent data found in batch")
        
        # Get the latent tensor
        if isinstance(latents_data, dict):
            if "latents" in latents_data:
                x0 = latents_data["latents"]  # (B, 1, seq_len, latent_dim)
            else:
                raise ValueError("No 'latents' key in latent_conditions")
        else:
            x0 = latents_data
        
        # Reshape to video format: (B, 1, seq_len, latent_dim) -> (B, C, F, H, W)
        # This requires knowing the spatial dimensions
        if x0.dim() == 4:  # (B, 1, seq_len, latent_dim)
            B, _, seq_len, latent_dim = x0.shape
            
            # Estimate spatial dimensions (assuming square and standard compression)
            # For LTX-Video, typical compression is 8x temporal, 32x spatial
            if "num_frames" in latents_data:
                F = latents_data["num_frames"]
            else:
                F = 8  # Default frame count
            
            if "height" in latents_data and "width" in latents_data:
                H, W = latents_data["height"], latents_data["width"]
                C = latent_dim
            else:
                # Estimate from sequence length: seq_len = F * H * W
                spatial_tokens = seq_len // F
                H = W = int(spatial_tokens ** 0.5)  # Assume square
                C = latent_dim
            
            # Reshape: (B, 1, seq_len, C) -> (B, C, F, H, W)
            x0 = x0.squeeze(1).view(B, F, H, W, C).permute(0, 4, 1, 2, 3)
        
        # Sample timesteps
        device = x0.device
        B = x0.shape[0]
        t = timestep_sampler.sample(B).to(device)
        
        # Sample noise (independent for standard FM)
        noise = self.flow_matching.sample_noise(x0.shape).to(device)
        
        # Compute forward path and teacher velocity
        z_t = self.flow_matching.compute_forward_path(noise, x0, t)
        u_t = self.flow_matching.compute_teacher_velocity(noise, x0, t)
        
        return {
            "z_t": z_t,
            "u_t": u_t,
            "t": t,
            "x0": x0,
            "noise": noise
        }
    
    def prepare_model_inputs(self, training_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare model inputs for transformer forward pass.
        
        The transformer expects specific input format from the original LTX-Video architecture.
        """
        z_t = training_batch["z_t"]  # (B, C, F, H, W)
        t = training_batch["t"]  # (B,)
        
        B, C, F, H, W = z_t.shape
        
        # Reshape to sequence format for transformer: (B, C, F, H, W) -> (B, 1, seq_len, C)
        seq_len = F * H * W
        z_t_seq = z_t.permute(0, 2, 3, 4, 1).reshape(B, 1, seq_len, C)
        
        # Expand timesteps to match expected format
        timesteps = t.unsqueeze(1)  # (B, 1)
        
        return {
            "hidden_states": z_t_seq,
            "timestep": timesteps,
            "encoder_hidden_states": None,  # No text conditioning
            "encoder_attention_mask": None,
        }
    
    def compute_loss(self, model_pred: torch.Tensor, training_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute Standard FM loss: MSE(v_θ(z_t, t), u_t)
        
        Args:
            model_pred: (B, 1, seq_len, C) model prediction
            training_batch: Contains teacher velocity u_t
            
        Returns:
            loss: Scalar loss value
        """
        u_t = training_batch["u_t"]  # (B, C, F, H, W)
        B, C, F, H, W = u_t.shape
        
        # Reshape model prediction to match u_t format
        # (B, 1, seq_len, C) -> (B, C, F, H, W)
        model_pred_reshaped = model_pred.squeeze(1).view(B, F, H, W, C).permute(0, 4, 1, 2, 3)
        
        # MSE loss
        loss = torch.nn.functional.mse_loss(model_pred_reshaped, u_t)
        
        return loss


class RingZipperFlowMatchingStrategy(RingTrainingStrategyBase):
    """DEPRECATED: Training strategy for Ring/Zipper Flow Matching.
    
    Use quantum_fm method with QuantumTrainingStrategy instead for better results.
    This implementation suffered from averaging collapse issues.
    """
    
    def __init__(self, t_star: float = 0.2):
        # Auto-upgrade to quantum implementation
        super().__init__(method="quantum_fm", t_star=t_star)
    
    def get_data_sources(self) -> Set[str]:
        """Ring FM needs latent data."""
        return {"latents"}
        
    def prepare_batch(self, batch: Dict[str, Any], timestep_sampler) -> Dict[str, torch.Tensor]:
            """
            Prepare batch for Ring/Zipper Flow Matching training.
            """
            # 1. 데이터 추출
            if "latents" in batch:
                latents_data = batch["latents"]
            elif "latent_conditions" in batch:
                latents_data = batch["latent_conditions"]
            else:
                raise ValueError("No latent data found in batch")

            if isinstance(latents_data, dict):
                x0 = latents_data.get("latents", None)
                if x0 is None:
                    raise ValueError("No 'latents' key in latent_conditions")
            else:
                x0 = latents_data

            # 2. DEVICE 설정 (중요: 에러 해결의 핵심)
            # AnchorNetwork나 Transformer 모델이 있는 장치로 데이터를 먼저 옮깁니다.
            # 보통 self.flow_matching 내부의 파라미터 장치를 참조합니다.
            device = next(self.flow_matching.anchor_net.parameters()).device if hasattr(self.flow_matching, 'anchor_net') else x0.device
            x0 = x0.to(device)

            # 3. 차원 정렬 및 타입 확정
            if x0.dim() == 5:
                B, C, F, H, W = x0.shape
                # 정수형 확정 (view 연산 등에서 발생할 수 있는 TypeError 방지)
                B, C, F, H, W = int(B), int(C), int(F), int(H), int(W)

            elif x0.dim() == 4:
                B, _, seq_len, latent_dim = x0.shape
                B, seq_len, latent_dim = int(B), int(seq_len), int(latent_dim)

                # config나 데이터셋 규격에 맞춤
                F, H, W = 16, 8, 8
                C = latent_dim

                # view 인자를 int로 강제 변환하여 전달
                x0 = x0.squeeze(1).view(B, F, H, W, C).permute(0, 4, 1, 2, 3)

            # 4. Flow Matching 인스턴스 업데이트 (필요시)
            if hasattr(self.flow_matching, 'anchor_net') and self.flow_matching.anchor_net:
                if self.flow_matching.anchor_net.latent_dim != C:
                    from .ring_zipper_flow import AnchorNetwork, RingZipperFlowMatching
                    anchor_net = AnchorNetwork(latent_dim=C).to(device) # 새로 만들 때도 장치 지정
                    self.flow_matching = RingZipperFlowMatching(
                        t_star=self.flow_matching.t_star,
                        anchor_net=anchor_net
                    )

            # 5. 타임스텝 및 노이즈 생성 (장치 일치 시킴)
            B = x0.shape[0]
            t = timestep_sampler.sample(B).to(device)

            # Shared noise ε ∈ R[B, C, 1, H, W]
            noise = self.flow_matching.sample_noise(x0.shape).to(device)

            # 6. Anchor 및 Path 계산 (모두 같은 device에서 연산됨)
            # 이제 x0가 GPU에 있으므로 RuntimeError가 발생하지 않습니다.
            anchor, mu, log_sigma2 = self.flow_matching.compute_anchor(x0)

            z_t = self.flow_matching.compute_forward_path(noise, x0, t, anchor=anchor)
            u_t = self.flow_matching.compute_teacher_velocity(noise, x0, t, anchor=anchor)

            junction_error = self.flow_matching.verify_junction_constraint(noise, x0, anchor)

            # ===========================
            # 📊 데이터 상태 로깅 (prepare_batch 내부)
            # ===========================

            # 표준편차 계산
            x0_std = x0.std().item()
            anchor_std = anchor.std().item()
            z_t_std = z_t.std().item()
            noise_std = noise.std().item()

            # NaN/Inf 체크
            anchor_has_nan = torch.isnan(anchor).any().item()
            anchor_has_inf = torch.isinf(anchor).any().item()
            u_t_has_nan = torch.isnan(u_t).any().item()
            u_t_has_inf = torch.isinf(u_t).any().item()

            # 로그 메트릭 수집
            log_dict = {
                "data/x0_std": x0_std,
                "data/anchor_std": anchor_std,
                "data/z_t_std": z_t_std,
                "data/noise_std": noise_std,
                "data/anchor_has_nan": float(anchor_has_nan),
                "data/anchor_has_inf": float(anchor_has_inf),
                "data/u_t_has_nan": float(u_t_has_nan),
                "data/u_t_has_inf": float(u_t_has_inf),
                "data/junction_error": junction_error.mean().item() if isinstance(junction_error, torch.Tensor) else junction_error,
            }

            # WandB 로깅 (wandb.log 직접 사용)
            if wandb.run is not None:
                wandb.log(log_dict, step=self.global_step)

            # 터미널 로깅 (매 50스텝마다)
            if self.global_step % 50 == 0:
                logger.info(
                    f"[Step {self.global_step}] Data Stats - "
                    f"x0_std: {x0_std:.4f}, anchor_std: {anchor_std:.4f}, "
                    f"z_t_std: {z_t_std:.4f}, noise_std: {noise_std:.4f}, "
                    f"anchor_nan: {anchor_has_nan}, anchor_inf: {anchor_has_inf}, "
                    f"u_t_nan: {u_t_has_nan}, u_t_inf: {u_t_has_inf}, "
                    f"junction_error: {log_dict['data/junction_error']:.6f}"
                )

            return {
                "z_t": z_t,
                "u_t": u_t,
                "t": t,
                "x0": x0,
                "noise": noise,
                "anchor": anchor,
                "mu": mu,
                "log_sigma2": log_sigma2,
                "junction_error": junction_error
            }
    
    def prepare_model_inputs(self, training_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare model inputs (same as Standard FM)."""
        z_t = training_batch["z_t"]
        t = training_batch["t"]
        
        B, C, F, H, W = z_t.shape
        seq_len = F * H * W
        z_t_seq = z_t.permute(0, 2, 3, 4, 1).reshape(B, 1, seq_len, C)
        timesteps = t.unsqueeze(1)
        
        return {
            "hidden_states": z_t_seq,
            "timestep": timesteps,
            "encoder_hidden_states": None,
            "encoder_attention_mask": None,
        }
    
    def compute_loss(self, model_pred: torch.Tensor, training_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute Ring FM loss: MSE(v_θ(z_t, t), u_t)

        Same loss as Standard FM, but u_t is computed differently due to piecewise paths.
        """
        u_t = training_batch["u_t"]
        t = training_batch["t"]  # (B,)
        B, C, F, H, W = u_t.shape

        # Reshape model prediction
        model_pred_reshaped = model_pred.squeeze(1).view(B, F, H, W, C).permute(0, 4, 1, 2, 3)

        # MSE loss
        loss = F.mse_loss(model_pred_reshaped, u_t)

        # ===========================
        # 📊 지표 계산 로직 (compute_loss 내부)
        # ===========================

        # 1. 코사인 유사도 계산 (방향성 체크용)
        # Flatten to (B, -1) for cosine similarity
        pred_flat = model_pred_reshaped.reshape(B, -1)
        target_flat = u_t.reshape(B, -1)

        # Cosine similarity: dot(pred, target) / (||pred|| * ||target||)
        cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean().item()

        # 2. L2 Norm 계산
        target_norm = torch.linalg.vector_norm(target_flat, ord=2, dim=1).mean().item()
        pred_norm = torch.linalg.vector_norm(pred_flat, ord=2, dim=1).mean().item()

        # 3. t값에 따라 loss 분리 (t > 0.2 vs t <= 0.2)
        t_threshold = 0.2
        mask_global = t > t_threshold  # Stage 2 (Global)
        mask_local = t <= t_threshold   # Stage 1 (Local)

        # Per-sample loss
        loss_per_sample = F.mse_loss(model_pred_reshaped, u_t, reduction='none').reshape(B, -1).mean(dim=1)

        # 전역/지역 손실 분리
        if mask_global.any():
            loss_global = loss_per_sample[mask_global].mean().item()
        else:
            loss_global = 0.0

        if mask_local.any():
            loss_local = loss_per_sample[mask_local].mean().item()
        else:
            loss_local = 0.0

        # 로그 메트릭 수집
        log_dict = {
            "loss/cosine_similarity": cosine_sim,
            "loss/target_norm": target_norm,
            "loss/pred_norm": pred_norm,
            "loss/global": loss_global,
            "loss/local": loss_local,
            "loss/t_mean": t.mean().item(),
        }

        # WandB 로깅 (wandb.log 직접 사용)
        if wandb.run is not None:
            wandb.log(log_dict, step=self.global_step)

        # 터미널 로깅 (매 50스텝마다)
        if self.global_step % 50 == 0:
            logger.info(
                f"[Step {self.global_step}] Loss Metrics - "
                f"cosine_sim: {cosine_sim:.4f}, target_norm: {target_norm:.4f}, "
                f"pred_norm: {pred_norm:.4f}, loss_global: {loss_global:.4f}, "
                f"loss_local: {loss_local:.4f}, t_mean: {t.mean().item():.4f}"
            )

        return loss


def get_ring_training_strategy(method: str, **kwargs):
    """
    Factory function to create training strategies.
    
    Args:
        method: "standard_fm", "ring_fm", or "quantum_fm" (recommended)
        **kwargs: Additional arguments (e.g., t_star, use_anchor_net)
        
    Returns:
        Training strategy instance
    """
    if method == "standard_fm":
        return StandardFlowMatchingStrategy()
    elif method == "ring_fm":
        # Legacy ring FM (deprecated)
        t_star = kwargs.get("t_star", 0.2)  # Updated default for better quantum collapse
        return RingZipperFlowMatchingStrategy(t_star=t_star)
    elif method == "quantum_fm":
        # NEW: Quantum State Collapse (solves averaging collapse)
        t_star = kwargs.get("t_star", 0.2)
        use_anchor_net = kwargs.get("use_anchor_net", False)
        return get_quantum_training_strategy(t_star=t_star, use_anchor_net=use_anchor_net)
    else:
        raise ValueError(f"Unknown training method: {method}. Use 'quantum_fm' for best results.")