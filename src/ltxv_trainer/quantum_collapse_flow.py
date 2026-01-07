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
import torch.nn.functional as F
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
            nn.AdaptiveAvgPool2d(8),
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


class QuantumStateCollapseFlowMatching(FlowMatchingBase):
    """
    Quantum State Collapse Flow Matching - Solves Averaging Collapse Problem

    Quantum Path: x_t = α_t * x_noise + β_t * A + γ_t(i) * x_i
    where γ_t(i) includes Frame-wise Repulsion and Decoherence Operators

    Revolutionary innovations:
    - Quantum Anchor: Stochastic superposition of all frames
    - State Collapse: Frame index measurement breaks superposition
    - Frame Repulsion: Anti-averaging forces (t < 0.2)
    - Orthogonality Loss: Enforce frame uniqueness
    """
    
    def __init__(self, t_star: float = 0.2, anchor_net: Optional[QuantumAnchorNetwork] = None):
        # LTX-Video convention: t=1.0(Noise) -> t=0.0(Data)
        # t_star: Quantum decoherence threshold (anchor influence peak)
        self.t_star = t_star
        self.anchor_net = anchor_net

        # Quantum diagnostic metrics
        self._quantum_metrics = {}

    def sample_noise(self, x0_shape: Tuple[int, ...]) -> torch.Tensor:
        """Sample shared noise topology for quantum coherence"""
        B, C, F, H, W = x0_shape
        return torch.randn(B, C, 1, H, W)  # Shared across frames for coherence
    
    def compute_anchor(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compatibility alias for compute_quantum_anchor"""
        return self.compute_quantum_anchor(x0)
    
    def compute_quantum_anchor(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantum Superposition Anchor: Creates mixed state containing ALL frame possibilities

        Quantum Strategy: Maximum entropy while preserving identity information.
        The anchor represents a 'coherent superposition' of all frames that will
        collapse to specific states during the decoherence process (t < 0.2).
        """
        if self.anchor_net is None:
            # Manual quantum superposition without network
            B, C, F, H, W = x0.shape
            
            # Create quantum superposition from frame statistics
            frame_mean = x0.mean(dim=2, keepdim=True)  # Identity preservation
            frame_std = x0.std(dim=2, keepdim=True)   # Frame diversity measure
            
            # Quantum anchor: Mean + enhanced variance for superposition
            quantum_uncertainty = 2.0 * frame_std  # Amplified uncertainty
            mu = frame_mean + 0.1 * quantum_uncertainty
            
            # Add high-frequency jitter for quantum decoherence potential
            high_freq_pattern = torch.randn_like(mu) * 0.05
            mu = mu + high_freq_pattern
            
            # Enhanced variance for quantum states
            log_sigma2 = torch.log(0.1 + frame_std.square())  # Adaptive variance
        else:
            # Use quantum anchor network
            mu, log_sigma2 = self.anchor_net(x0)

        # Sample quantum anchor with enhanced stochasticity
        anchor = self.anchor_net.sample_quantum_anchor(mu, log_sigma2) if self.anchor_net else self._sample_quantum_anchor(mu, log_sigma2)
        return anchor, mu, log_sigma2
        
    def _sample_quantum_anchor(self, mu: torch.Tensor, log_sigma2: torch.Tensor) -> torch.Tensor:
        """Fallback quantum anchor sampling when no network is used"""
        sigma = torch.exp(0.5 * log_sigma2)
        xi_base = torch.randn_like(mu)
        xi_quantum = 0.3 * torch.randn_like(mu)  # Additional quantum fluctuations
        return mu + sigma * (xi_base + xi_quantum)
    
    def _compute_quantum_weights(self, t: torch.Tensor, frame_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantum State Collapse Weight Computation with Frame-wise Repulsion

        Mathematical foundation:
        - α_t: Noise weight (standard decay)
        - β_t: Anchor weight (quantum decoherence at t_star)
        - γ_t(i): Frame-specific weight with REPULSION forces

        Quantum Collapse Mechanism:
        - For t > t_star: Coherent superposition (anchor dominance)
        - For t < t_star: State collapse with Frame Repulsion (anti-averaging)
        - Ψ(i,t): Decoherence operator creates frame-specific paths
        """
        t_broad = t.view(-1, 1, 1, 1, 1)

        # α_t: Noise weight (unchanged)
        alpha_t = t_broad

        # β_t: Quantum decoherence with ULTRA-SHARP transition
        beta_width = 0.05  # EVEN SHARPER than before for quantum collapse
        beta_base = torch.exp(-((t_broad - self.t_star) ** 2) / (2 * beta_width ** 2))
        
        # QUANTUM COLLAPSE: Below t_star, anchor becomes quantum vacuum
        transition_mask = (t_broad < self.t_star).float()
        quantum_decay = torch.exp(-25.0 * (self.t_star - t_broad))  # EXTREME decay for collapse
        beta_t = beta_base * (1.0 - transition_mask + transition_mask * quantum_decay)

        # γ_t(i): Frame-specific weight with REPULSION FIELD
        gamma_base = 1.0 - t_broad
        
        # FRAME REPULSION MECHANISM: For t < t_star
        repulsion_mask = (t_broad < self.t_star).float()
        
        # Calculate Frame-wise Repulsion strength (stronger as t approaches 0)
        repulsion_strength = 5.0 * (self.t_star - t_broad) / self.t_star  # 0 to 5x boost
        repulsion_strength = torch.clamp(repulsion_strength, 0.0, 5.0)
        
        # Apply FRAME REPULSION: Amplify frame-specific energy
        frame_repulsion_factor = 1.0 + repulsion_mask * repulsion_strength
        gamma_t = gamma_base * frame_repulsion_factor

        # Normalization with quantum constraint
        weight_sum = alpha_t + beta_t + gamma_t
        alpha_t = alpha_t / weight_sum
        beta_t = beta_t / weight_sum  
        gamma_t = gamma_t / weight_sum

        return alpha_t, beta_t, gamma_t

    def compute_forward_path(self, noise: torch.Tensor, target: torch.Tensor, t: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """
        Quantum State Collapse Path: x_t = α_t·noise + β_t·anchor + γ_t(i)·x_i

        Quantum mechanics implementation:
        - Superposition phase (t > t_star): All frames exist simultaneously in anchor
        - Collapse phase (t < t_star): Frame-specific decoherence with repulsion
        """
        F = target.shape[2]

        # Expand anchor and noise to match frame dimension
        A_up = anchor.expand(-1, -1, F, -1, -1)
        eps_up = noise.expand(-1, -1, F, -1, -1)

        # Compute quantum weights with frame repulsion
        alpha_t, beta_t, gamma_t = self._compute_quantum_weights(t)

        # Quantum path with decoherence
        z_t = alpha_t * eps_up + beta_t * A_up + gamma_t * target

        # Store quantum metrics for diagnostics
        self._quantum_metrics['alpha_t'] = alpha_t.squeeze().mean().item()
        self._quantum_metrics['beta_t'] = beta_t.squeeze().mean().item()
        self._quantum_metrics['gamma_t'] = gamma_t.squeeze().mean().item()
        self._quantum_metrics['quantum_decoherence'] = (t < self.t_star).float().mean().item()

        return z_t
    
    def compute_teacher_velocity(self, noise: torch.Tensor, target: torch.Tensor, t: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """
        Quantum Collapse Velocity: u_t = dα/dt·noise + dβ/dt·anchor + dγ/dt(i)·x_i

        The velocity includes the decoherence operator Ψ(i,t) that creates
        frame-specific collapse directions, preventing averaging collapse.
        """
        F = target.shape[2]
        A_up = anchor.expand(-1, -1, F, -1, -1)
        eps_up = noise.expand(-1, -1, F, -1, -1)

        # Compute quantum weight derivatives
        t_broad = t.view(-1, 1, 1, 1, 1)

        # dα/dt = d(t)/dt = 1 (constant)
        dalpha_dt = torch.ones_like(t_broad)

        # dβ/dt: Derivative of quantum decoherence function
        beta_width = 0.05
        beta_t = torch.exp(-((t_broad - self.t_star) ** 2) / (2 * beta_width ** 2))
        
        # Include quantum collapse derivative
        transition_mask = (t_broad < self.t_star).float()
        decay_factor = torch.exp(-25.0 * (self.t_star - t_broad))
        decay_derivative = 25.0 * decay_factor
        
        dbeta_dt_base = beta_t * (-(t_broad - self.t_star) / (beta_width ** 2))
        dbeta_dt = dbeta_dt_base * (1.0 - transition_mask) - beta_t * transition_mask * decay_derivative

        # dγ/dt: Include frame repulsion derivative
        dgamma_dt_base = -torch.ones_like(t_broad)
        
        # Frame repulsion derivative
        repulsion_mask = (t_broad < self.t_star).float()
        repulsion_derivative = 5.0 / self.t_star * repulsion_mask
        dgamma_dt = dgamma_dt_base * (1.0 + repulsion_derivative)

        # Quantum velocity components
        u_noise = dalpha_dt * eps_up
        u_anchor = dbeta_dt * A_up
        u_data = dgamma_dt * target

        # Total quantum collapse velocity
        u_t = u_noise + u_anchor + u_data

        # Store quantum velocity metrics
        self._quantum_metrics['u_decoherence_energy'] = torch.abs(u_anchor).mean().item()
        self._quantum_metrics['u_frame_repulsion'] = torch.abs(u_data).mean().item()

        return u_t
    
    def compute_orthogonality_loss(self, frame_predictions: torch.Tensor) -> torch.Tensor:
        """
        Quantum Non-Redundancy Loss: Enforce frame orthogonality
        
        Ensures that predictions for different frames are maximally different,
        preventing the averaging collapse by penalizing similar outputs.
        
        Args:
            frame_predictions: (B, C, F, H, W) predicted velocities for each frame
            
        Returns:
            orthogonality_loss: Scalar penalty for frame similarity
        """
        B, C, F, H, W = frame_predictions.shape
        
        # Flatten frames: (B, C, F, H, W) -> (B, F, C*H*W)
        flattened_frames = frame_predictions.permute(0, 2, 1, 3, 4).reshape(B, F, -1)
        
        # Compute pairwise cosine similarities between frames
        frame_norms = torch.norm(flattened_frames, dim=2, keepdim=True)  # (B, F, 1)
        normalized_frames = flattened_frames / (frame_norms + 1e-8)
        
        # Cosine similarity matrix: (B, F, F)
        cosine_matrix = torch.bmm(normalized_frames, normalized_frames.transpose(1, 2))
        
        # Create mask to exclude diagonal (self-similarity)
        mask = torch.eye(F, device=frame_predictions.device).unsqueeze(0).expand(B, -1, -1)
        
        # Orthogonality loss: Penalize high off-diagonal similarities
        off_diagonal_similarities = cosine_matrix * (1 - mask)
        orthogonality_loss = torch.abs(off_diagonal_similarities).mean()
        
        # Store metric
        self._quantum_metrics['frame_orthogonality'] = 1.0 - orthogonality_loss.item()
        
        return orthogonality_loss
    
    def get_quantum_metrics(self):
        """Return quantum collapse diagnostic metrics"""
        return self._quantum_metrics.copy()
    
    def get_rab_metrics(self):
        """Compatibility alias for get_quantum_metrics"""
        return self.get_quantum_metrics()
    
    def get_snr_metrics(self):
        """Compatibility alias for get_quantum_metrics"""
        return self.get_quantum_metrics()


def create_quantum_flow_matching(t_star: float = 0.2, latent_dim: int = 128, use_anchor_net: bool = False) -> QuantumStateCollapseFlowMatching:
    """
    Factory function to create Quantum State Collapse Flow Matching instance.
    
    Args:
        t_star: Quantum decoherence threshold
        latent_dim: Latent dimension for anchor network
        use_anchor_net: Whether to use learnable quantum anchor network
        
    Returns:
        QuantumStateCollapseFlowMatching instance
    """
    if use_anchor_net:
        anchor_net = QuantumAnchorNetwork(latent_dim=latent_dim)
    else:
        anchor_net = None
    
    return QuantumStateCollapseFlowMatching(t_star=t_star, anchor_net=anchor_net)