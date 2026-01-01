"""
Ring/Zipper Flow Matching Training Strategy

Adapts the existing training strategy pattern to implement exact
Ring/Zipper Flow Matching mathematics.
"""
import torch
from typing import Dict, Any, Set
from abc import ABC, abstractmethod

from .ring_zipper_flow import FlowMatchingBase, create_flow_matching


class RingTrainingStrategyBase(ABC):
    """Base class for Ring/Zipper Flow Matching training strategies."""
    
    def __init__(self, method: str = "ring_fm", t_star: float = 0.8):
        self.method = method
        self.flow_matching = create_flow_matching(
            method=method,
            t_star=t_star,
            latent_dim=128  # Will be updated based on actual data
        )
    
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
    """Training strategy for Ring/Zipper Flow Matching."""
    
    def __init__(self, t_star: float = 0.8):
        super().__init__(method="ring_fm", t_star=t_star)
    
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
        B, C, F, H, W = u_t.shape
        
        # Reshape model prediction
        model_pred_reshaped = model_pred.squeeze(1).view(B, F, H, W, C).permute(0, 4, 1, 2, 3)
        
        # MSE loss
        loss = torch.nn.functional.mse_loss(model_pred_reshaped, u_t)
        
        return loss


def get_ring_training_strategy(method: str, **kwargs) -> RingTrainingStrategyBase:
    """
    Factory function to create training strategies.
    
    Args:
        method: "standard_fm" or "ring_fm"
        **kwargs: Additional arguments (e.g., t_star)
        
    Returns:
        RingTrainingStrategyBase instance
    """
    if method == "standard_fm":
        return StandardFlowMatchingStrategy()
    elif method == "ring_fm":
        t_star = kwargs.get("t_star", 0.8)
        return RingZipperFlowMatchingStrategy(t_star=t_star)
    else:
        raise ValueError(f"Unknown training method: {method}")