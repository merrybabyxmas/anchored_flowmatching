"""
Quantum Spherical Geodesic Flow Matching Implementation

Replaces Euclidean linear interpolation with Spherical Geodesic paths.
- Manifold: Hypersphere (Norm=1).
- Dynamics: Rotation (Slerp) + Projected Velocity.
- Two-Phase Generation: Image Phase (Noise->Anchor) -> Video Phase (Anchor->Video).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from abc import ABC, abstractmethod


class FlowMatchingBase(ABC):
    """Base class for flow matching implementations."""
    
    @abstractmethod
    def sample_noise(self, x0_shape: Tuple[int, ...]) -> torch.Tensor:
        """Sample noise for the flow matching process."""
        pass
    
    @abstractmethod
    def compute_forward_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the forward path Z(t)."""
        pass
    
    @abstractmethod
    def compute_teacher_velocity(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, z_t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the exact teacher velocity u(t)."""
        pass


class QuantumAnchorNetwork(nn.Module):
    """
    Quantum Anchor Network: Extracts the Identity (Anchor) from the Video.
    
    Theory: The anchor |A> represents the "Identity" of the video, stripped of time.
    It sits on the Hypersphere (Norm=1).
    
    Input: x0 ∈ R[B, C, F, H, W] (Video)
    Output: A ∈ R[B, C, 1, H, W] (Anchor, Norm=1)
    """
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Simple encoder to extract identity from frames
        # Input: (B, C, F, H, W) -> Collapse Time -> (B, C, 1, H, W)
        
        # We process the temporal dimension.
        # Ideally, we want to find the "center" of the frame cluster on the sphere.
        # A simple approximation is the normalized mean.
        # But we can add a small learnable adapter to refine it.
        
        self.adapter = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, latent_dim, 3, padding=1)
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Extract Anchor |A> from Video |V>.
        
        Args:
            x0: (B, C, F, H, W) video latents
            
        Returns:
            anchor: (B, C, 1, H, W) normalized anchor
        """
        # 1. Temporal Collapse (Mean)
        # x0 is expected to be normalized? No, x0 is raw latents from VAE.
        # We first extract the mean.
        # (B, C, F, H, W) -> (B, C, 1, H, W)
        mean_frame = x0.mean(dim=2, keepdim=True)
        
        # 2. Refine Identity
        B, C, _, H, W = mean_frame.shape
        mean_flat = mean_frame.squeeze(2) # (B, C, H, W)
        refined = mean_flat + self.adapter(mean_flat)
        anchor = refined.unsqueeze(2) # (B, C, 1, H, W)
        
        # 3. Spherical Projection (Manifold Constraint)
        anchor = F.normalize(anchor, p=2, dim=1)
        
        return anchor


class QuantumStateCollapseFlowMatching(FlowMatchingBase):
    """
    Spherical Geodesic Flow Matching.

    Implements:
    1. Slerp for forward process z_t.
    2. Tangent Projection for velocity field v_theta.

    Manifold: S^{d-1} (Hypersphere)
    Path: Geodesic (Great Circle)
    """
    
    def __init__(self, t_star: float = 0.2, anchor_net: Optional[QuantumAnchorNetwork] = None):
        self.t_star = t_star # Not strictly used in pure Slerp, but kept for compatibility
        self.anchor_net = anchor_net

    def sample_noise(self, x0_shape: Tuple[int, ...]) -> torch.Tensor:
        """Sample noise on the hypersphere."""
        noise = torch.randn(x0_shape)
        return F.normalize(noise, p=2, dim=1)

    def slerp(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Spherical Linear Interpolation.
        
        z_t = [sin((1-t)Ω) / sin(Ω)] * x0 + [sin(tΩ) / sin(Ω)] * x1
        where Ω = arccos(<x0, x1>)
        """
        # Ensure inputs are normalized
        x0_norm = F.normalize(x0, p=2, dim=1)
        x1_norm = F.normalize(x1, p=2, dim=1)
        
        # Compute Omega (angle)
        # Dot product along channel dimension
        cos_omega = (x0_norm * x1_norm).sum(dim=1, keepdim=True)
        
        # Clamp for numerical stability
        cos_omega = torch.clamp(cos_omega, -0.9999, 0.9999)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega) + 1e-8 # Avoid division by zero
        
        # Broadcast t to compatible shape
        # t: (B,) -> (B, 1, 1, 1, 1)
        t_bc = t.view(-1, 1, 1, 1, 1)
        
        # Slerp coefficients
        coeff0 = torch.sin((1.0 - t_bc) * omega) / sin_omega
        coeff1 = torch.sin(t_bc * omega) / sin_omega
        
        z_t = coeff0 * x0_norm + coeff1 * x1_norm
        
        # Re-normalize to ensure numerical precision keeps it on sphere
        return F.normalize(z_t, p=2, dim=1)

    def compute_forward_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute z_t on the geodesic between x0 and x1.
        
        Args:
            x0: Source (e.g., Noise or Anchor)
            x1: Target (e.g., Anchor or Video)
            t: Time [0, 1]
        """
        return self.slerp(x0, x1, t)

    def compute_teacher_velocity(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, z_t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute Projected Flow Velocity.
        
        u_linear = x1 - x0
        u_tangent = u_linear - <u_linear, z_t> * z_t
        
        This removes the radial component, leaving only the rotational component.
        """
        # Ensure inputs are normalized for correct calculation
        x0_norm = F.normalize(x0, p=2, dim=1)
        x1_norm = F.normalize(x1, p=2, dim=1)
        z_t_norm = F.normalize(z_t, p=2, dim=1)
        
        # 1. Linear Velocity (Secant vector)
        u_linear = x1_norm - x0_norm
        
        # 2. Tangent Projection
        # Project u_linear onto the tangent space at z_t
        # Formula: v = u - (u . z) * z
        dot_prod = (u_linear * z_t_norm).sum(dim=1, keepdim=True)
        radial_component = dot_prod * z_t_norm
        
        u_tangent = u_linear - radial_component
        
        return u_tangent

    # Compatibility methods
    def compute_quantum_anchor(self, x0: torch.Tensor) -> torch.Tensor:
        """Computes anchor using the network."""
        if self.anchor_net:
            return self.anchor_net(x0)
        else:
            # Fallback: Temporal Mean + Normalize
            return F.normalize(x0.mean(dim=2, keepdim=True), p=2, dim=1)

    def get_quantum_metrics(self):
        return {}


def create_quantum_flow_matching(t_star: float = 0.2, latent_dim: int = 128, use_anchor_net: bool = True) -> QuantumStateCollapseFlowMatching:
    if use_anchor_net:
        anchor_net = QuantumAnchorNetwork(latent_dim=latent_dim)
    else:
        anchor_net = None
    
    return QuantumStateCollapseFlowMatching(t_star=t_star, anchor_net=anchor_net)
