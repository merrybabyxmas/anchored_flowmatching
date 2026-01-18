"""
Quantum State Collapse Training Strategy

Implements the complete quantum architecture with:
- Spherical Geodesic Flow Matching (Slerp + Projected Velocity)
- Two-Phase Training (Image Phase & Video Phase)
- Probabilistic Phase Switching
- Quantum metrics logging
"""
import torch
import torch.nn.functional as F
import wandb
import random
from typing import Dict, Any, Set, Optional

from ltxv_trainer import logger
from .quantum_collapse_flow import QuantumStateCollapseFlowMatching, create_quantum_flow_matching


class QuantumTrainingStrategy:
    """Quantum State Collapse Training Strategy Implementation."""

    def __init__(self, t_star: float = 0.2, use_anchor_net: bool = True):
        self.flow_matching = create_quantum_flow_matching(
            t_star=t_star,
            latent_dim=128,  # Will be updated based on actual data
            use_anchor_net=use_anchor_net
        )
        self.accelerator = None  # Will be set by trainer
        self.global_step = 0  # Will be updated by trainer
        
        # Phase probabilities
        self.image_phase_prob = 0.2  # 20% Image Phase (Noise -> Anchor)
        # Remaining 80% is Video Phase (Anchor -> Video)

        self.quantum_metrics_log_interval = 50  # Log quantum metrics every N steps
    
    def get_data_sources(self) -> Set[str]:
        """Quantum FM needs latent data."""
        return {"latents"}
        
    def prepare_batch(self, batch: Dict[str, Any], timestep_sampler) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for Quantum State Collapse Flow Matching training.
        
        Implements Probabilistic Phase Switching:
        - Image Phase: Learn Noise -> Anchor (Identity Formation)
        - Video Phase: Learn Anchor -> Video (Temporal Collapse)
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

        # 2. Device alignment
        device = x0.device
        x0 = x0.to(device)

        # 3. Dimension handling and validation
        if x0.dim() == 5:
            B, C, num_frames, H, W = x0.shape
            B, C, num_frames, H, W = int(B), int(C), int(num_frames), int(H), int(W)
        elif x0.dim() == 4:
            B, _, seq_len, latent_dim = x0.shape
            B, seq_len, latent_dim = int(B), int(seq_len), int(latent_dim)
            # Default configuration for LTX-Video
            num_frames, H, W = 16, 8, 8
            C = latent_dim
            x0 = x0.squeeze(1).view(B, num_frames, H, W, C).permute(0, 4, 1, 2, 3)

        # 4. Update flow matching latent dimension if needed
        if hasattr(self.flow_matching, 'anchor_net') and self.flow_matching.anchor_net:
            if self.flow_matching.anchor_net.latent_dim != C:
                from .quantum_collapse_flow import QuantumAnchorNetwork, QuantumStateCollapseFlowMatching
                anchor_net = QuantumAnchorNetwork(latent_dim=C).to(device)
                self.flow_matching = QuantumStateCollapseFlowMatching(
                    t_star=self.flow_matching.t_star,
                    anchor_net=anchor_net
                )

        # 5. Extract Anchor (Identity)
        # Returns (B, C, 1, H, W), normalized
        anchor = self.flow_matching.compute_quantum_anchor(x0)
        
        # 6. Probabilistic Phase Switching
        # We decide the phase for the entire batch to keep tensor shapes consistent
        is_image_phase = random.random() < self.image_phase_prob
        
        t = timestep_sampler.sample(B).to(device)
        
        if is_image_phase:
            # --- Image Phase: Noise -> Anchor ---
            # Source: Noise (Spherical)
            # Target: Anchor
            # Time: 1 (Noise) -> 0 (Anchor) (Standard Flow Matching convention is usually 0->1 or 1->0 depending on formulation)
            # LTX-Video convention (based on previous code): t=1.0(Noise) -> t=0.0(Data)

            source = self.flow_matching.sample_noise(anchor.shape).to(device) # (B, C, 1, H, W)
            target = anchor # (B, C, 1, H, W)

            # Since target is Anchor (Data), and source is Noise.
            # Forward Path z_t interpolates between Source and Target.
            # If t=1 is Noise, t=0 is Target.

            # z_t calculation using Slerp
            # Note: slerp(source, target, t) usually means t=0 -> source, t=1 -> target.
            # But if we want t=1 to be Noise, we should map t correctly.
            # Let's assume standard FM: x1 (data), x0 (noise). z_t = t*x1 + (1-t)*x0.
            # Then t=0 is noise, t=1 is data.
            # But the previous code comment said: "t=1.0(Noise) -> t=0.0(Data)".
            # Let's stick to the previous convention if possible, OR switch to standard FM.
            # Let's look at `ltxv_pipeline.py`. It calls `scheduler.step`.
            # Usually diffusers schedulers go from T to 0.
            # So t=1.0 (High Noise) -> t=0.0 (Clean Data).

            # If t=1 is Noise (x0) and t=0 is Data (x1).
            # Then z_t should be close to x0 when t=1.
            # slerp(x1, x0, t) -> t=0 => x1(Data), t=1 => x0(Noise).
            # This matches "t=1(Noise) -> t=0(Data)".

            z_t = self.flow_matching.compute_forward_path(target, source, t)
            u_t = self.flow_matching.compute_teacher_velocity(target, source, t, z_t)

            # Broadcast to match F if necessary?
            # Ideally, we train on F=1 to save compute.
            # But trainer might expect consistent F.
            # If we return F=1 tensor, prepare_model_inputs needs to handle it.
            # Let's return F=1 tensor.

            phase_name = "image"

        else:
            # --- Video Phase: Anchor -> Video ---
            # Source: Anchor (broadcasted)
            # Target: Video
            # t=1 (Anchor) -> t=0 (Video)

            # Note: Here "Noise" is the Anchor.
            # So t=1 => Anchor, t=0 => Video.

            # Use shape from x0 to determine F
            num_frames = x0.shape[2]
            source = anchor.expand(-1, -1, num_frames, -1, -1) # (B, C, F, H, W)
            target = x0 # (B, C, F, H, W)

            # Ensure target is normalized (Manifold Constraint)
            target = F.normalize(target, p=2, dim=1)

            z_t = self.flow_matching.compute_forward_path(target, source, t)
            u_t = self.flow_matching.compute_teacher_velocity(target, source, t, z_t)

            phase_name = "video"

        # 7. Quantum metrics logging
        if self.global_step % self.quantum_metrics_log_interval == 0:
            log_dict = {
                "quantum/phase": 1.0 if is_image_phase else 0.0,
                "quantum/anchor_norm": torch.norm(anchor, p=2, dim=1).mean().item(),
                "quantum/z_t_norm": torch.norm(z_t, p=2, dim=1).mean().item(),
            }
            if wandb.run is not None:
                wandb.log(log_dict, step=self.global_step)
            
            logger.info(
                f"[Quantum Step {self.global_step}] Phase: {phase_name}, "
                f"Anchor Norm: {log_dict['quantum/anchor_norm']:.4f}"
            )

        return {
            "z_t": z_t,
            "u_t": u_t,
            "t": t,
            "phase": phase_name,
            "frame_indices": torch.arange(z_t.shape[2], device=device).unsqueeze(0).expand(B, -1)
        }
    
    def prepare_model_inputs(self, training_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare model inputs.
        Handles both Image Phase (F=1) and Video Phase (F=16).
        """
        z_t = training_batch["z_t"]
        t = training_batch["t"]
        frame_indices = training_batch["frame_indices"]
        
        B, C, num_frames, H, W = z_t.shape
        seq_len = num_frames * H * W
        
        # Reshape to sequence format for transformer
        # (B, C, F, H, W) -> (B, F, H, W, C) -> (B, 1, S, C) ?
        # LTXVideoTransformer usually takes (B, C, F, H, W) or (B, S, C)?
        # Checking trainer.py... it just passes **model_inputs.
        # Let's check `ltxv_pipeline.py`. It packs latents.
        # But `trainer.py` uses `self._transformer`.
        # `src/ltxv_trainer/quantum_training_strategy.py` previously did:
        # z_t_seq = z_t.permute(0, 2, 3, 4, 1).reshape(B, 1, seq_len, C)
        # return {"hidden_states": z_t_seq, ...}

        z_t_seq = z_t.permute(0, 2, 3, 4, 1).reshape(B, 1, seq_len, C)
        timesteps = t.unsqueeze(1)  # (B, 1)
        
        # Create frame position embeddings
        frame_pos_flat = frame_indices.unsqueeze(-1).repeat(1, 1, H*W).reshape(B, seq_len)  # (B, seq_len)
        
        # Manifold Constraint: Ensure input to model is normalized?
        # z_t is already normalized from prepare_batch.
        
        return {
            "hidden_states": z_t_seq,
            "timestep": timesteps,
            "encoder_hidden_states": None,  # No text conditioning
            "encoder_attention_mask": None,
            "frame_positions": frame_pos_flat.long(),
            "quantum_overdrive": False,
        }
    
    def compute_loss(self, model_pred: torch.Tensor, training_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute Projected Flow Matching Loss.
        
        Loss = || v_theta - u_t_proj ||^2
        """
        u_t = training_batch["u_t"]
        # model_pred comes from transformer.
        # Shape of model_pred: (B, 1, SeqLen, C) based on input?
        # Or (B, SeqLen, C)?
        # Previous code: model_pred_reshaped = model_pred.squeeze(1).view(B, num_frames, H, W, C).permute(0, 4, 1, 2, 3)

        B, C, num_frames, H, W = u_t.shape

        # Reshape model prediction to match target format
        # Assuming model_pred is (B, 1, S, C) or (B, S, C)
        if model_pred.dim() == 4 and model_pred.shape[1] == 1:
            model_pred = model_pred.squeeze(1)

        model_pred_reshaped = model_pred.view(B, num_frames, H, W, C).permute(0, 4, 1, 2, 3)

        # 1. Projected Flow Matching Loss (MSE)
        # Both u_t and model_pred are in tangent space (ideally).
        # u_t is guaranteed by construction.
        # model_pred is learned.
        flow_loss = F.mse_loss(model_pred_reshaped, u_t)
        
        # 2. Logging
        if self.global_step % self.quantum_metrics_log_interval == 0:
            pred_norm = torch.norm(model_pred_reshaped, p=2, dim=1).mean().item()
            target_norm = torch.norm(u_t, p=2, dim=1).mean().item()
            
            log_dict = {
                "loss/flow_matching": flow_loss.item(),
                "loss/pred_velocity_norm": pred_norm,
                "loss/target_velocity_norm": target_norm,
            }
            if wandb.run is not None:
                wandb.log(log_dict, step=self.global_step)
            
            logger.info(
                f"[Quantum Loss {self.global_step}] Loss: {flow_loss.item():.4f}, "
                f"PredNorm: {pred_norm:.4f}, TargetNorm: {target_norm:.4f}"
            )

        return flow_loss

def get_quantum_training_strategy(t_star: float = 0.2, use_anchor_net: bool = True) -> QuantumTrainingStrategy:
    """
    Factory function to create Quantum Training Strategy.
    """
    return QuantumTrainingStrategy(t_star=t_star, use_anchor_net=use_anchor_net)
