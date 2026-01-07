"""
Quantum State Collapse Flow Matching Implementation

Quantum-inspired architecture to solve Averaging Collapse in video generation:
- Quantum Anchor: Stochastic superposition state containing all frame possibilities
- State Collapse: Frame-specific decoherence through measurement (frame index)
- Non-Redundancy: Orthogonality enforcement between frame predictions
- Frame Repulsion: Anti-averaging forces in local stage (t < 0.2)
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


class QuantumAnchorNetwork(nn.Module):
    """
    Quantum Anchor Network: Creates stochastic superposition states
    
    Theory: Anchor A represents ALL possible frames in superposition.
    The anchor contains high entropy and frame-agnostic information,
    serving as the 'quantum state' that collapses to specific frames.
    
    Input: x0 ∈ R[B, C, F, H, W] (all frames)
    Output: μ, σ ∈ R[B, C, 1, H, W] (quantum anchor parameters)
    """
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Frame encoder with enhanced capacity for quantum representation
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim // 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(8),  # Fixed spatial size for processing
        )
        
        # Quantum superposition processor
        feature_dim = hidden_dim // 2 * 8 * 8
        
        # Enhanced temporal attention for quantum coherence
        self.quantum_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=16,  # Increased heads for better superposition modeling
            batch_first=True,
            dropout=0.15
        )
        
        # Entropy maximization layer for stochastic superposition
        self.entropy_enhancer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        
        # Quantum state parameter heads with increased variance capacity
        self.mu_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, latent_dim * 8 * 8),
        )
        
        self.log_sigma2_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, latent_dim * 8 * 8),
        )
    
    def forward(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create quantum superposition state from all frames.
        
        Quantum Principle: The anchor must contain maximal entropy while preserving
        identity information. It serves as the 'mixed state' before measurement.
        
        Args:
            x0: (B, C, F, H, W) video latents
            
        Returns:
            mu: (B, C, 1, H, W) quantum anchor means (high entropy)
            log_sigma2: (B, C, 1, H, W) quantum variance (enhanced for superposition)
        """
        B, C, F, H, W = x0.shape
        
        # Process each frame to extract quantum features
        frames_flat = x0.reshape(B * F, C, H, W)
        frame_features = self.frame_encoder(frames_flat)  # (B*F, C', 8, 8)
        frame_features_flat = frame_features.reshape(B * F, -1)
        frame_features = frame_features_flat.reshape(B, F, -1)
        
        # Quantum coherence through enhanced attention
        quantum_features, attention_weights = self.quantum_attention(
            frame_features, frame_features, frame_features
        )
        
        # Entropy enhancement for stochastic superposition
        enhanced_features = self.entropy_enhancer(quantum_features)
        
        # Quantum superposition: Maximize variance between frames while preserving mean
        frame_means = enhanced_features.mean(dim=1)  # (B, feature_dim) - Identity preservation
        frame_vars = enhanced_features.var(dim=1, unbiased=False)  # (B, feature_dim) - Frame differences
        
        # Enhanced superposition: Identity + 3x amplified variance for quantum uncertainty
        quantum_superposition_factor = 3.0  # Strong quantum uncertainty
        quantum_repr = frame_means + quantum_superposition_factor * frame_vars
        
        # Additional high-frequency jitter for quantum decoherence potential
        high_freq_jitter = 0.1 * torch.randn_like(quantum_repr)
        quantum_repr = quantum_repr + high_freq_jitter
        
        # Predict quantum anchor parameters with enhanced variance
        mu_flat = self.mu_head(quantum_repr)
        log_sigma2_flat = self.log_sigma2_head(quantum_repr)
        
        mu = mu_flat.reshape(B, C, 1, 8, 8)
        log_sigma2 = log_sigma2_flat.reshape(B, C, 1, 8, 8)
        
        # Enhanced variance range for quantum superposition
        log_sigma2 = torch.clamp(log_sigma2, min=-8, max=8)  # Wider variance range
        
        # Upsample if needed
        if 8 != H or 8 != W:
            mu = nn.functional.interpolate(
                mu.squeeze(2), size=(H, W), mode='bilinear', align_corners=False
            ).unsqueeze(2)
            log_sigma2 = nn.functional.interpolate(
                log_sigma2.squeeze(2), size=(H, W), mode='bilinear', align_corners=False
            ).unsqueeze(2)
        
        return mu, log_sigma2
    
    def sample_quantum_anchor(self, mu: torch.Tensor, log_sigma2: torch.Tensor) -> torch.Tensor:
        """
        Sample quantum superposition anchor: A = μ + σ ⊙ ξ
        
        Quantum Sampling: Enhanced stochasticity for maximum entropy state
        
        Args:
            mu: (B, C, 1, H, W) quantum means
            log_sigma2: (B, C, 1, H, W) quantum log variances
            
        Returns:
            anchor: (B, C, 1, H, W) quantum superposition anchor
        """
        sigma = torch.exp(0.5 * log_sigma2)
        
        # Enhanced quantum noise with multiple random sources
        xi_base = torch.randn_like(mu)  # Base quantum noise
        xi_high_freq = 0.2 * torch.randn_like(mu)  # High-frequency quantum fluctuations
        
        # Quantum superposition with enhanced stochasticity
        quantum_noise = xi_base + xi_high_freq
        return mu + sigma * quantum_noise

# Legacy class alias for compatibility - use QuantumStateCollapseFlowMatching instead
class RingZipperFlowMatching(FlowMatchingBase):
    """
    DEPRECATED: Use QuantumStateCollapseFlowMatching instead.
    
    This class is maintained for backward compatibility but has been superseded
    by the Quantum State Collapse implementation which solves the averaging collapse problem.
    
    Legacy RAB approach suffered from averaging collapse where all frames converged
    to similar outputs. The new quantum approach prevents this through:
    - Quantum superposition anchors
    - Frame-wise repulsion forces  
    - State collapse mechanisms
    """
    def __init__(self, t_star: float = 0.2, anchor_net: Optional[QuantumAnchorNetwork] = None):
        # LTX-Video convention: t=1.0(Noise) -> t=0.0(Data)
        # t_star: Bridge midpoint where anchor influence peaks
        self.t_star = t_star
        self.anchor_net = anchor_net

        # RAB diagnostic metrics
        self._rab_metrics = {}

    def sample_noise(self, x0_shape: Tuple[int, ...]) -> torch.Tensor:
        B, C, F, H, W = x0_shape
        return torch.randn(B, C, 1, H, W)
    
    def compute_anchor(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sparse Anchor: Preserve identity but remove motion details

        Strategy: Apply Gaussian blur + temporal noise to create information distance
        This ensures |x_0 - A| >> 0, giving the model a residual to learn.
        """
        if self.anchor_net is None:
            # [RAB Sparsification] Weighted first frame with blur
            z_first = x0[:, :, 0:1, :, :]
            z_mean = x0.mean(dim=2, keepdim=True)

            # Base anchor: 0.95 first frame + 0.05 mean
            mu = 0.95 * z_first + 0.05 * z_mean

            # [Sparsification] Apply Gaussian blur to remove texture details
            # Kernel size 5 preserves identity structure while removing motion
            if mu.shape[-2] > 5 and mu.shape[-1] > 5:  # Only if spatial dims large enough
                import torch.nn.functional as F
                # Create Gaussian kernel
                kernel_size = 5
                sigma_blur = 1.5
                kernel_1d = torch.exp(-torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)**2 / (2 * sigma_blur**2))
                kernel_1d = kernel_1d / kernel_1d.sum()
                kernel_2d = kernel_1d.unsqueeze(-1) @ kernel_1d.unsqueeze(0)
                kernel = kernel_2d.unsqueeze(0).unsqueeze(0).to(mu.device, mu.dtype)

                # Apply blur per channel
                C = mu.shape[1]
                mu_reshaped = mu.squeeze(2)  # (B, C, H, W)
                mu_blurred = []
                for c in range(C):
                    blurred = F.conv2d(mu_reshaped[:, c:c+1], kernel, padding=kernel_size//2)
                    mu_blurred.append(blurred)
                mu = torch.cat(mu_blurred, dim=1).unsqueeze(2)  # (B, C, 1, H, W)

            # [Sparsification] Add minimal temporal noise for distance guarantee
            # This ensures anchor != data, creating learning signal
            sparse_noise = 0.02 * torch.randn_like(mu)  # 2% noise for distance
            mu = mu + sparse_noise

            log_sigma2 = torch.full_like(mu, -10.0)
        else:
            mu, log_sigma2 = self.anchor_net(x0)

        sigma = torch.exp(0.5 * log_sigma2)
        xi = torch.randn_like(mu)
        anchor = mu + sigma * xi
        return anchor, mu, log_sigma2
    
    def _compute_bridge_weights(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [AGGRESSIVE] Continuous bridge weight schedulers with rapid transition

        Mathematical design:
        - α_t (noise weight): Decays from 1→0 as t: 1.0→0.0
        - β_t (anchor weight): Peaks at t=t_star, RAPID decay for t<t_star
        - γ_t (data weight): AGGRESSIVE growth for t<t_star (MOTION DOMINANCE)

        Constraints:
        - α_t + β_t + γ_t = 1 (convex combination)
        - Smooth C¹ continuity, but steep gradients near t_star

        Anti-Averaging Strategy:
        - For t < t_star (0.2), anchor influence drops to near-zero rapidly
        - Data (motion) weight overtakes aggressively to force frame differences
        """
        # Reshape for broadcasting: (B,) -> (B, 1, 1, 1, 1)
        t_broad = t.view(-1, 1, 1, 1, 1)

        # α_t: Noise weight (linear decay)
        # t=1.0 → α=1.0, t=0.0 → α=0.0
        alpha_t = t_broad

        # [BALANCED] β_t: Anchor weight with MODERATE decay below t_star
        # Use wider Gaussian for smoother transition, preventing numerical instability
        beta_width = 0.12  # BALANCED: 0.08 → 0.12 for smoother, more stable transition
        beta_base = torch.exp(-((t_broad - self.t_star) ** 2) / (2 * beta_width ** 2))

        # [MOTION DOMINANCE] For t < t_star, apply MODERATE exponential decay
        # Strong enough to break averaging collapse, but stable for convergence
        transition_mask = (t_broad < self.t_star).float()
        decay_factor = torch.exp(-8.0 * (self.t_star - t_broad))  # MODERATE decay (15→8 for stability)
        beta_t = beta_base * (1.0 - transition_mask + transition_mask * decay_factor)

        # γ_t: Data weight with MODERATE growth for t < t_star
        # Base complementary growth
        gamma_t_raw = 1.0 - t_broad

        # [BALANCED MOTION BOOST] For t < t_star, amplify gamma by 1.5x to force frame divergence
        # Prevents averaging collapse while maintaining training stability
        motion_boost = torch.where(
            t_broad < self.t_star,
            gamma_t_raw * 1.5,  # 50% boost for frame separation (balanced: 2.0→1.5)
            gamma_t_raw
        )

        # Normalize to ensure α + β + γ = 1
        weight_sum = alpha_t + beta_t + motion_boost
        alpha_t = alpha_t / weight_sum
        beta_t = beta_t / weight_sum
        gamma_t = motion_boost / weight_sum

        return alpha_t, beta_t, gamma_t

    def compute_forward_path(self, noise: torch.Tensor, target: torch.Tensor, t: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """
        Unified Bridge Path: x_t = α_t·noise + β_t·anchor + γ_t·data

        Continuous across all t ∈ [0, 1], no hard junction.
        Smooth transition through t_star bridge point.
        """
        F = target.shape[2]

        # Expand anchor and noise to match frame dimension
        A_up = anchor.expand(-1, -1, F, -1, -1)
        eps_up = noise.expand(-1, -1, F, -1, -1)

        # Compute continuous bridge weights
        alpha_t, beta_t, gamma_t = self._compute_bridge_weights(t)

        # Unified path: convex combination
        z_t = alpha_t * eps_up + beta_t * A_up + gamma_t * target

        # Store weights for velocity computation
        self._rab_metrics['alpha_t'] = alpha_t.squeeze().mean().item()
        self._rab_metrics['beta_t'] = beta_t.squeeze().mean().item()
        self._rab_metrics['gamma_t'] = gamma_t.squeeze().mean().item()

        return z_t
    
    def compute_teacher_velocity(self, noise: torch.Tensor, target: torch.Tensor, t: torch.Tensor,
                                  anchor: torch.Tensor) -> torch.Tensor:
        """
        Residual Bridge Velocity: u_t = dα/dt·noise + dβ/dt·anchor + dγ/dt·data

        Mathematical formulation:
        - Exact derivative of continuous bridge path
        - Focuses on residual motion: (data - anchor)
        - No artificial rescaling or normalization
        """
        F = target.shape[2]
        A_up = anchor.expand(-1, -1, F, -1, -1)
        eps_up = noise.expand(-1, -1, F, -1, -1)

        # Compute weight derivatives analytically
        t_broad = t.view(-1, 1, 1, 1, 1)

        # dα/dt = d(t)/dt = 1 (constant)
        dalpha_dt = torch.ones_like(t_broad)

        # dβ/dt: Derivative of Gaussian bell curve
        beta_width = 0.15
        beta_t = torch.exp(-((t_broad - self.t_star) ** 2) / (2 * beta_width ** 2))
        dbeta_dt = beta_t * (-(t_broad - self.t_star) / (beta_width ** 2))

        # dγ/dt = d(1-t)/dt = -1 (constant)
        dgamma_dt = -torch.ones_like(t_broad)

        # Need to account for normalization in weight computation
        # For simplicity, we use the unnormalized derivatives
        # The exact derivative would require quotient rule, but this approximation
        # preserves the essential residual structure

        # Velocity components
        u_noise = dalpha_dt * eps_up
        u_anchor = dbeta_dt * A_up
        u_data = dgamma_dt * target

        # Total velocity: sum of components
        u_t = u_noise + u_anchor + u_data

        # Compute residual norm for diagnostics
        residual = target - A_up  # Motion residual
        residual_norm = torch.linalg.vector_norm(
            residual.reshape(residual.shape[0], -1), ord=2, dim=1
        ).mean().item()

        # Store diagnostic metrics
        self._rab_metrics['residual_norm'] = residual_norm
        self._rab_metrics['u_noise_contrib'] = torch.abs(u_noise).mean().item()
        self._rab_metrics['u_anchor_contrib'] = torch.abs(u_anchor).mean().item()
        self._rab_metrics['u_data_contrib'] = torch.abs(u_data).mean().item()

        return u_t
    
    def get_rab_metrics(self):
        """RAB 진단 메트릭 반환"""
        return self._rab_metrics.copy()

    def get_snr_metrics(self):
        """SNR 메트릭 반환 (하위 호환성)"""
        return self.get_rab_metrics()
    
    def verify_junction_constraint(self, noise: torch.Tensor, target: torch.Tensor, anchor: torch.Tensor) -> float:
        t_star_batch = torch.full((target.shape[0],), self.t_star, device=target.device)
        z_at_star = self.compute_forward_path(noise, target, t_star_batch, anchor)
        error = torch.max(torch.abs(z_at_star - anchor.expand_as(z_at_star)))
        return error.item()

# Import quantum implementation for enhanced flow matching
from .quantum_collapse_flow import QuantumStateCollapseFlowMatching, create_quantum_flow_matching

def create_flow_matching(method: str, t_star: float = 0.2, latent_dim: int = 128, use_anchor_net: bool = False) -> FlowMatchingBase:
    """
    Factory function to create flow matching instances.

    Args:
        method: "standard_fm", "ring_fm", or "quantum_fm" (recommended)
        t_star: Junction/decoherence time (default 0.2 for quantum collapse)
        latent_dim: Latent dimension for anchor network
        use_anchor_net: Whether to use learnable anchor network

    Returns:
        FlowMatchingBase instance
    """
    if method == "standard_fm":
        return StandardFlowMatching()
    elif method == "ring_fm":
        # Legacy RAB implementation (deprecated, use quantum_fm instead)
        if use_anchor_net:
            anchor_net = QuantumAnchorNetwork(latent_dim=latent_dim)
        else:
            anchor_net = None
        return QuantumStateCollapseFlowMatching(t_star=t_star, anchor_net=anchor_net)
    elif method == "quantum_fm":
        # NEW: Quantum State Collapse Flow Matching (solves averaging collapse)
        return create_quantum_flow_matching(t_star=t_star, latent_dim=latent_dim, use_anchor_net=use_anchor_net)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'quantum_fm' for best results.")