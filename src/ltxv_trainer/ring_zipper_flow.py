"""
Ring/Zipper Flow Matching Implementation

Mathematically exact implementation following the specification:
- Shared noise topology: ε ∈ R[B, C, 1, H, W]
- Gaussian anchors: A = μ + σ ⊙ ξ
- Junction constraint: Z(t⋆) = A (numerical)
- Piecewise linear paths with exact velocities
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from abc import ABC, abstractmethod

class FlowMatchingBase(ABC):
    """Base class for flow matching implementations."""
    
    @abstractmethod
    def sample_noise(self, x0_shape: Tuple[int, ...]) -> torch.Tensor:
        """Sample noise for the flow matching process."""
        pass
    
    @abstractmethod
    def compute_forward_path(self, noise: torch.Tensor, target: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the forward path Z(t)."""
        pass
    
    @abstractmethod
    def compute_teacher_velocity(self, noise: torch.Tensor, target: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the exact teacher velocity u(t)."""
        pass


class StandardFlowMatching(FlowMatchingBase):
    """
    Standard Flow Matching (Baseline)
    
    Independent linear paths: noise → target
    """
    
    def sample_noise(self, x0_shape: Tuple[int, ...]) -> torch.Tensor:
        """Sample independent noise per frame: ε ∈ R[B, C, F, H, W]"""
        return torch.randn(x0_shape)
    
    def compute_forward_path(self, noise: torch.Tensor, target: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Standard linear path: Z(t) = (1-t) * noise + t * target
        
        Args:
            noise: (B, C, F, H, W) independent noise
            target: (B, C, F, H, W) target frames  
            t: (B,) time values
            
        Returns:
            z_t: (B, C, F, H, W) path at time t
        """
        B = target.shape[0]
        t_broad = t.reshape(B, 1, 1, 1, 1)  # Broadcast to match target shape
        
        return (1 - t_broad) * noise + t_broad * target
    
    def compute_teacher_velocity(self, noise: torch.Tensor, target: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Standard velocity: u = target - noise (constant)
        
        Args:
            noise: (B, C, F, H, W)
            target: (B, C, F, H, W)
            t: (B,) - not used (velocity is constant)
            
        Returns:
            u: (B, C, F, H, W) constant velocity
        """
        return target - noise


class AnchorNetwork(nn.Module):
    """
    Gaussian anchor network: (μ, log_σ²) = g_ψ(x0)
    
    Input: x0 ∈ R[B, C, F, H, W] (all frames)
    Output: μ, σ ∈ R[B, C, 1, H, W] (one anchor per video)
    """
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Frame encoder: processes individual frames
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim // 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(8),  # Fixed spatial size for processing
        )
        
        # Temporal aggregation across frames
        feature_dim = hidden_dim // 2 * 8 * 8
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # Anchor parameter prediction heads
        self.mu_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 8 * 8),  # Match encoder spatial size
        )
        
        self.log_sigma2_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 8 * 8),
        )
    
    def forward(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anchor parameters.
        
        Args:
            x0: (B, C, F, H, W) video latents
            
        Returns:
            mu: (B, C, 1, H, W) anchor means
            log_sigma2: (B, C, 1, H, W) log variances
        """
        B, C, F, H, W = x0.shape
        
        # Process each frame: (B, C, F, H, W) -> (B*F, C, H, W)
        frames_flat = x0.reshape(B * F, C, H, W)
        frame_features = self.frame_encoder(frames_flat)  # (B*F, C', 8, 8)
        
        # Flatten spatial: (B*F, C', 8, 8) -> (B*F, feature_dim)
        frame_features_flat = frame_features.reshape(B * F, -1)
        
        # Reshape for temporal processing: (B*F, feature_dim) -> (B, F, feature_dim)
        frame_features = frame_features_flat.reshape(B, F, -1)
        
        # Temporal attention aggregation
        video_features, _ = self.temporal_attention(
            frame_features, frame_features, frame_features
        )  # (B, F, feature_dim)
        
        # Pool across frames for video-level representation
        video_repr = video_features.mean(dim=1)  # (B, feature_dim)
        
        # Predict anchor parameters
        mu_flat = self.mu_head(video_repr)  # (B, C*8*8)
        log_sigma2_flat = self.log_sigma2_head(video_repr)  # (B, C*8*8)
        
        # Reshape to anchor format: (B, C*8*8) -> (B, C, 1, 8, 8)
        mu = mu_flat.reshape(B, C, 1, 8, 8)
        log_sigma2 = log_sigma2_flat.reshape(B, C, 1, 8, 8)
        
        # Clamp log_sigma2 for numerical stability
        log_sigma2 = torch.clamp(log_sigma2, min=-10, max=5)
        
        # Upsample to match input resolution if needed
        if 8 != H or 8 != W:
            mu = nn.functional.interpolate(
                mu.squeeze(2), size=(H, W), mode='bilinear', align_corners=False
            ).unsqueeze(2)
            log_sigma2 = nn.functional.interpolate(
                log_sigma2.squeeze(2), size=(H, W), mode='bilinear', align_corners=False
            ).unsqueeze(2)
        
        return mu, log_sigma2
    
    def sample_anchor(self, mu: torch.Tensor, log_sigma2: torch.Tensor) -> torch.Tensor:
        """
        Sample anchor: A = μ + σ ⊙ ξ
        
        Args:
            mu: (B, C, 1, H, W) means
            log_sigma2: (B, C, 1, H, W) log variances
            
        Returns:
            anchor: (B, C, 1, H, W) sampled anchors
        """
        sigma = torch.exp(0.5 * log_sigma2)
        xi = torch.randn_like(mu)
        return mu + sigma * xi


class RingZipperFlowMatching(FlowMatchingBase):
    """
    Ring/Zipper Flow Matching Implementation
    
    Key properties:
    - Shared noise: ε ∈ R[B, C, 1, H, W] (broadcast across frames)
    - Gaussian anchors: A = μ + σ ⊙ ξ
    - Junction constraint: Z(t⋆) = A ∀ frames
    - Piecewise linear paths: Global (t < t⋆), Local (t ≥ t⋆)
    """
    
    def __init__(self, t_star: float = 0.8, anchor_net: Optional[AnchorNetwork] = None):
        self.t_star = t_star
        self.anchor_net = anchor_net
    
    def sample_noise(self, x0_shape: Tuple[int, ...]) -> torch.Tensor:
            """원칙 2: 비디오당 단 하나의 노이즈 샘플링 ε ∈ R[B, C, 1, H, W]"""
            B, C, F, H, W = x0_shape
            # 프레임 차원(F)을 1로 설정하여 공통 노이즈 생성
            return torch.randn(B, C, 1, H, W)
    
    def broadcast_noise(self, noise: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Broadcast shared noise across frames: ε_{b,i} = ε_b
        
        Args:
            noise: (B, C, 1, H, W) shared noise
            num_frames: F
            
        Returns:
            broadcasted: (B, C, F, H, W) 
        """
        return noise.expand(-1, -1, num_frames, -1, -1)
    
    def compute_anchor(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """원칙 3: 비디오당 단 하나의 스토캐스틱 앵커 A 생성"""
            if self.anchor_net is None:
                mu = x0.mean(dim=2, keepdim=True) # Fallback: Temporal average
                log_sigma2 = torch.full_like(mu, -10.0) # Low variance
            else:
                self.anchor_net.to(device=x0.device, dtype=x0.dtype)
                mu, log_sigma2 = self.anchor_net(x0)            
            # Stochastic sampling: A = μ + σ ⊙ ξ
            sigma = torch.exp(0.5 * log_sigma2)
            xi = torch.randn_like(mu)
            anchor = mu + sigma * xi
            return anchor, mu, log_sigma2
    
    def compute_forward_path(self, noise: torch.Tensor, target: torch.Tensor, t: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """원칙 5: Piecewise Linear Forward Path (수치적 정확성 보장)"""
        B, C, F, H, W = target.shape
        t_broad = t.view(B, 1, 1, 1, 1)
        
        # 앵커와 노이즈를 프레임 차원으로 브로드캐스팅
        A_up = anchor.expand(-1, -1, F, -1, -1)
        eps_up = noise.expand(-1, -1, F, -1, -1)
        
        # 구간별 마스크
        mask_g = (t < self.t_star).view(B, 1, 1, 1, 1)
        
        # Global: (1 - t/t*)ε + (t/t*)A
        z_g = (1 - t_broad / self.t_star) * eps_up + (t_broad / self.t_star) * A_up
        
        # Local: ((1-t)/(1-t*))A + ((t-t*)/(1-t*))x0
        t_local = (t_broad - self.t_star) / (1 - self.t_star)
        z_l = (1 - t_local) * A_up + t_local * target
        
        return torch.where(mask_g, z_g, z_l)
    
    def compute_teacher_velocity(self, noise: torch.Tensor, target: torch.Tensor, t: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """원칙 7: Exact Constant Teacher Velocity (시간 t에 의존하지 않음)"""
        B, C, F, H, W = target.shape
        
        # A와 ε를 타겟 모양에 맞게 확장
        A_up = anchor.expand(-1, -1, F, -1, -1)
        eps_up = noise.expand(-1, -1, F, -1, -1)
        
        mask_g = (t < self.t_star).view(B, 1, 1, 1, 1)
        
        # Global velocity: uG = (A - ε) / t* (프레임 간 동일)
        u_g = (A_up - eps_up) / self.t_star
        
        # Local velocity: uL = (x0 - A) / (1 - t*) (프레임별 다름)
        u_l = (target - A_up) / (1 - self.t_star)
        
        return torch.where(mask_g, u_g, u_l)
    
    def verify_junction_constraint(self, noise: torch.Tensor, target: torch.Tensor, anchor: torch.Tensor) -> float:
            """원칙 6: Junction Constraint Z(t*) = A 검증"""
            # t = t* 일 때의 경로 계산
            t_star_batch = torch.full((target.shape[0],), self.t_star, device=target.device)
            z_at_star = self.compute_forward_path(noise, target, t_star_batch, anchor)
            
            # A와 Z(t*) 사이의 최대 오차
            error = torch.max(torch.abs(z_at_star - anchor.expand_as(z_at_star)))
            return error.item()
    
    def _get_device_from_shape(self, x0_shape: Tuple[int, ...]) -> str:
        """Helper to determine device."""
        return "cuda" if torch.cuda.is_available() else "cpu"


def create_flow_matching(method: str, t_star: float = 0.8, latent_dim: int = 128) -> FlowMatchingBase:
    """
    Factory function to create flow matching instances.
    
    Args:
        method: "standard_fm" or "ring_fm"
        t_star: Junction time for ring FM
        latent_dim: Latent dimension for anchor network
        
    Returns:
        FlowMatchingBase instance
    """
    if method == "standard_fm":
        return StandardFlowMatching()
    elif method == "ring_fm":
        anchor_net = AnchorNetwork(latent_dim=latent_dim)
        return RingZipperFlowMatching(t_star=t_star, anchor_net=anchor_net)
    else:
        raise ValueError(f"Unknown method: {method}")