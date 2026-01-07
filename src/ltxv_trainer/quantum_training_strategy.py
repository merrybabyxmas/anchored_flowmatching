"""
Quantum State Collapse Training Strategy

Implements the complete quantum architecture with:
- Quantum Flow Matching
- Temporal Embedding Overdrive (5x amplification)
- Orthogonality Loss for frame uniqueness
- Quantum metrics logging
"""
import torch
import torch.nn.functional as F
import wandb
from typing import Dict, Any, Set, Optional

from ltxv_trainer import logger
from .quantum_collapse_flow import QuantumStateCollapseFlowMatching, create_quantum_flow_matching


class QuantumTrainingStrategy:
    """Quantum State Collapse Training Strategy Implementation."""

    def __init__(self, t_star: float = 0.2, use_anchor_net: bool = False):
        self.flow_matching = create_quantum_flow_matching(
            t_star=t_star,
            latent_dim=128,  # Will be updated based on actual data
            use_anchor_net=use_anchor_net
        )
        self.accelerator = None  # Will be set by trainer
        self.global_step = 0  # Will be updated by trainer
        
        # Quantum-specific hyperparameters
        self.temporal_embedding_amplification = 5.0  # 5x overdrive
        self.orthogonality_loss_weight = 0.1  # Weight for frame uniqueness loss
        self.quantum_metrics_log_interval = 50  # Log quantum metrics every N steps
    
    def get_data_sources(self) -> Set[str]:
        """Quantum FM needs latent data."""
        return {"latents"}
        
    def prepare_batch(self, batch: Dict[str, Any], timestep_sampler) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for Quantum State Collapse Flow Matching training.
        
        Includes:
        - Quantum anchor computation
        - Temporal embedding overdrive preparation
        - Frame index extraction for state collapse
        """
        # 1. Extract and prepare data
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

        # 2. Device alignment for quantum computations
        device = x0.device
        x0 = x0.to(device)

        # 3. Dimension handling and validation
        if x0.dim() == 5:
            B, C, F, H, W = x0.shape
            B, C, F, H, W = int(B), int(C), int(F), int(H), int(W)
        elif x0.dim() == 4:
            B, _, seq_len, latent_dim = x0.shape
            B, seq_len, latent_dim = int(B), int(seq_len), int(latent_dim)

            # Default configuration for LTX-Video
            F, H, W = 16, 8, 8
            C = latent_dim
            x0 = x0.squeeze(1).view(B, F, H, W, C).permute(0, 4, 1, 2, 3)

        # 4. Update flow matching latent dimension if needed
        if hasattr(self.flow_matching, 'anchor_net') and self.flow_matching.anchor_net:
            if self.flow_matching.anchor_net.latent_dim != C:
                from .quantum_collapse_flow import QuantumAnchorNetwork, QuantumStateCollapseFlowMatching
                anchor_net = QuantumAnchorNetwork(latent_dim=C).to(device)
                self.flow_matching = QuantumStateCollapseFlowMatching(
                    t_star=self.flow_matching.t_star,
                    anchor_net=anchor_net
                )

        # 5. Quantum flow matching computations
        B = x0.shape[0]
        t = timestep_sampler.sample(B).to(device)

        # Quantum noise (shared across frames for coherence)
        noise = self.flow_matching.sample_noise(x0.shape).to(device)

        # Quantum anchor computation (superposition state)
        anchor, mu, log_sigma2 = self.flow_matching.compute_quantum_anchor(x0)

        # Quantum path and velocity
        z_t = self.flow_matching.compute_forward_path(noise, x0, t, anchor=anchor)
        u_t = self.flow_matching.compute_teacher_velocity(noise, x0, t, anchor=anchor)

        # 6. Frame index generation for temporal embedding overdrive
        frame_indices = torch.arange(F, device=device).unsqueeze(0).expand(B, -1)  # (B, F)
        
        # 7. Quantum state analysis
        quantum_metrics = self.flow_matching.get_quantum_metrics()
        
        # Compute quantum-specific diagnostics
        state_collapse_energy = self._compute_state_collapse_energy(x0, anchor)
        frame_divergence = self._compute_frame_divergence(x0)
        
        # 8. Enhanced logging with quantum metrics
        if self.global_step % self.quantum_metrics_log_interval == 0:
            log_dict = {
                "quantum/state_collapse_energy": state_collapse_energy,
                "quantum/frame_divergence": frame_divergence,
                "quantum/anchor_entropy": torch.std(anchor).item(),
                "quantum/superposition_strength": torch.var(mu).item(),
                **quantum_metrics
            }
            
            if wandb.run is not None:
                wandb.log(log_dict, step=self.global_step)
            
            logger.info(
                f"[Quantum Step {self.global_step}] "
                f"Collapse Energy: {state_collapse_energy:.4f}, "
                f"Frame Divergence: {frame_divergence:.4f}, "
                f"Anchor Entropy: {log_dict['quantum/anchor_entropy']:.4f}"
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
            "frame_indices": frame_indices,
            "quantum_metrics": quantum_metrics
        }
    
    def prepare_model_inputs(self, training_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare model inputs with Temporal Embedding Overdrive.
        
        Key innovation: Amplify frame index embeddings by 5x to force the model
        to strongly differentiate between frames during the state collapse process.
        """
        z_t = training_batch["z_t"]
        t = training_batch["t"]
        frame_indices = training_batch["frame_indices"]
        
        B, C, F, H, W = z_t.shape
        seq_len = F * H * W
        
        # Reshape to sequence format for transformer
        z_t_seq = z_t.permute(0, 2, 3, 4, 1).reshape(B, 1, seq_len, C)
        timesteps = t.unsqueeze(1)  # (B, 1)
        
        # TEMPORAL EMBEDDING OVERDRIVE: 5x amplification
        # Create amplified frame position embeddings
        frame_pos_flat = frame_indices.unsqueeze(-1).repeat(1, 1, H*W).reshape(B, seq_len)  # (B, seq_len)
        
        # Apply 5x amplification to frame indices for stronger temporal signal
        amplified_frame_pos = frame_pos_flat.float() * self.temporal_embedding_amplification
        
        return {
            "hidden_states": z_t_seq,
            "timestep": timesteps,
            "encoder_hidden_states": None,  # No text conditioning
            "encoder_attention_mask": None,
            "frame_positions": amplified_frame_pos.long(),  # Amplified temporal signal
            "quantum_overdrive": True,  # Flag for model to use overdrive
        }
    
    def compute_loss(self, model_pred: torch.Tensor, training_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute Quantum State Collapse Loss.
        
        Combines:
        1. Standard flow matching loss (MSE)
        2. Orthogonality loss (frame uniqueness enforcement)
        3. Quantum decoherence penalties
        """
        u_t = training_batch["u_t"]
        t = training_batch["t"]
        B, C, F, H, W = u_t.shape

        # Reshape model prediction to match target format
        model_pred_reshaped = model_pred.squeeze(1).view(B, F, H, W, C).permute(0, 4, 1, 2, 3)

        # 1. Primary Flow Matching Loss (MSE)
        flow_loss = F.mse_loss(model_pred_reshaped, u_t)

        # 2. Orthogonality Loss for Frame Uniqueness (Quantum Non-Redundancy)
        orthogonality_loss = self.flow_matching.compute_orthogonality_loss(model_pred_reshaped)
        
        # 3. Quantum Decoherence Loss (stronger penalty during collapse phase)
        collapse_mask = (t < self.flow_matching.t_star).float()
        decoherence_penalty = collapse_mask.mean() * orthogonality_loss
        
        # Total Quantum Loss
        total_loss = flow_loss + self.orthogonality_loss_weight * (orthogonality_loss + decoherence_penalty)

        # 4. Quantum diagnostics and logging
        if self.global_step % self.quantum_metrics_log_interval == 0:
            # Frame-wise analysis
            pred_flat = model_pred_reshaped.reshape(B, -1)
            target_flat = u_t.reshape(B, -1)
            
            cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean().item()
            target_norm = torch.linalg.vector_norm(target_flat, ord=2, dim=1).mean().item()
            pred_norm = torch.linalg.vector_norm(pred_flat, ord=2, dim=1).mean().item()
            
            # Quantum-specific metrics
            quantum_metrics = training_batch.get("quantum_metrics", {})
            
            log_dict = {
                "loss/flow_matching": flow_loss.item(),
                "loss/orthogonality": orthogonality_loss.item(),
                "loss/decoherence_penalty": decoherence_penalty.item(),
                "loss/total_quantum": total_loss.item(),
                "loss/cosine_similarity": cosine_sim,
                "loss/target_norm": target_norm,
                "loss/pred_norm": pred_norm,
                "quantum/collapse_phase_ratio": collapse_mask.mean().item(),
                **quantum_metrics
            }
            
            if wandb.run is not None:
                wandb.log(log_dict, step=self.global_step)
            
            logger.info(
                f"[Quantum Loss {self.global_step}] "
                f"Flow: {flow_loss.item():.4f}, "
                f"Orthogonal: {orthogonality_loss.item():.4f}, "
                f"Total: {total_loss.item():.4f}"
            )

        return total_loss
    
    def _compute_state_collapse_energy(self, x0: torch.Tensor, anchor: torch.Tensor) -> float:
        """
        Compute quantum state collapse energy.
        
        Measures how much the frames deviate from the anchor (superposition breakdown).
        Higher values indicate stronger frame individuality.
        """
        B, C, F, H, W = x0.shape
        anchor_expanded = anchor.expand(-1, -1, F, -1, -1)
        
        # Frame-wise deviation from anchor
        frame_deviations = torch.norm(x0 - anchor_expanded, dim=(1, 3, 4))  # (B, F)
        collapse_energy = frame_deviations.mean().item()
        
        return collapse_energy
    
    def _compute_frame_divergence(self, x0: torch.Tensor) -> float:
        """
        Compute inter-frame divergence.
        
        Measures how different frames are from each other.
        Higher values indicate better frame uniqueness (anti-averaging).
        """
        B, C, F, H, W = x0.shape
        
        # Compute pairwise frame differences
        frames_flat = x0.reshape(B, C, F, -1)  # (B, C, F, H*W)
        frame_means = frames_flat.mean(dim=-1)  # (B, C, F)
        
        # Pairwise distances between frame means
        frame_diffs = []
        for i in range(F):
            for j in range(i+1, F):
                diff = torch.norm(frame_means[:, :, i] - frame_means[:, :, j], dim=1)
                frame_diffs.append(diff)
        
        if frame_diffs:
            avg_divergence = torch.stack(frame_diffs).mean().item()
        else:
            avg_divergence = 0.0
            
        return avg_divergence


def get_quantum_training_strategy(t_star: float = 0.2, use_anchor_net: bool = False) -> QuantumTrainingStrategy:
    """
    Factory function to create Quantum Training Strategy.
    
    Args:
        t_star: Quantum decoherence threshold
        use_anchor_net: Whether to use learnable quantum anchor network
        
    Returns:
        QuantumTrainingStrategy instance
    """
    return QuantumTrainingStrategy(t_star=t_star, use_anchor_net=use_anchor_net)