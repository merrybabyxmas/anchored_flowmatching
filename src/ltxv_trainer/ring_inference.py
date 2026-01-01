"""
Ring/Zipper Flow Matching Inference Pipeline

Implements the two-stage inference with zipper property:
- Stage 1: Global ODE (t=0 → t⋆) with state [B,C,1,H,W]
- Stage 2: Local ODE (t⋆ → 1) with state [B,C,F,H,W]
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict, Any
from pathlib import Path
import numpy as np

from .ring_zipper_flow import FlowMatchingBase, create_flow_matching


class RingZipperSampler:
    """
    Two-stage sampler for Ring/Zipper Flow Matching.
    
    Implements the zipper property:
    1. Global phase: Single ODE from noise to anchor
    2. Local phase: Frame-specific ODE from anchor to targets
    """
    
    def __init__(self, 
                 model: nn.Module,
                 method: str = "ring_fm",
                 t_star: float = 0.8,
                 num_inference_steps: int = 50,
                 latent_dim: int = 128):
        
        self.model = model
        self.method = method
        self.t_star = t_star
        self.num_inference_steps = num_inference_steps
        
        # Create flow matching instance
        self.flow_matching = create_flow_matching(
            method=method,
            t_star=t_star,
            latent_dim=latent_dim
        )
        
        # Move anchor network to same device as model
        if hasattr(self.flow_matching, 'anchor_net') and self.flow_matching.anchor_net:
            device = next(model.parameters()).device
            self.flow_matching.anchor_net.to(device)
    
    @torch.no_grad()
    def sample(self,
              batch_size: int = 1,
              num_frames: int = 16,
              height: int = 8,  # Latent space height
              width: int = 8,   # Latent space width
              latent_dim: int = 128,
              device: Optional[torch.device] = None,
              generator: Optional[torch.Generator] = None,
              reference_video: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample videos using Ring/Zipper or Standard Flow Matching.
        
        Args:
            batch_size: Number of videos to generate
            num_frames: Number of frames per video
            height: Latent space height
            width: Latent space width
            latent_dim: Latent space dimension
            device: Device to use
            generator: Random number generator
            reference_video: Optional reference for anchor computation
            
        Returns:
            videos: (B, C, F, H, W) generated video latents
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        self.model.eval()
        
        if self.method == "ring_fm":
            return self._sample_ring_fm(
                batch_size, num_frames, height, width, latent_dim,
                device, generator, reference_video
            )
        else:
            return self._sample_standard_fm(
                batch_size, num_frames, height, width, latent_dim,
                device, generator
            )
    
    def _sample_ring_fm(self,
                       batch_size: int,
                       num_frames: int,
                       height: int,
                       width: int,
                       latent_dim: int,
                       device: torch.device,
                       generator: Optional[torch.Generator],
                       reference_video: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Ring/Zipper Flow Matching sampling with two-stage ODE.
        
        Stage 1: Global ODE (noise → anchor) with shape [B,C,1,H,W]
        Stage 2: Local ODE (anchor → frames) with shape [B,C,F,H,W]
        """
        print(f"Ring/Zipper FM sampling: {batch_size} videos, {num_frames} frames")
        print(f"Two-stage ODE: Global (0→{self.t_star}) → Local ({self.t_star}→1)")
        
        # Get target anchor (if reference provided)
        if reference_video is not None and hasattr(self.flow_matching, 'anchor_net'):
            # Use reference video for anchor computation
            ref_device = reference_video.device
            if ref_device != device:
                reference_video = reference_video.to(device)
            
            # Ensure reference has correct batch size
            if reference_video.shape[0] < batch_size:
                reference_video = reference_video.repeat(batch_size, 1, 1, 1, 1)[:batch_size]
            
            anchor, _, _ = self.flow_matching.compute_anchor(reference_video)
        else:
            # Generate random anchor for unconditional sampling
            if generator is not None:
                anchor = torch.randn(
                    batch_size, latent_dim, 1, height, width,
                    device=device, generator=generator
                ) * 0.5
            else:
                anchor = torch.randn(
                    batch_size, latent_dim, 1, height, width,
                    device=device
                ) * 0.5
        
        # ===== STAGE 1: Global ODE (t=0 → t⋆) =====
        print(f"Stage 1: Global path to anchor...")
        
        # Initial state: shared noise ε ∈ R[B,C,1,H,W]
        if generator is not None:
            z = torch.randn(
                batch_size, latent_dim, 1, height, width,
                device=device, generator=generator
            )
        else:
            z = torch.randn(batch_size, latent_dim, 1, height, width, device=device)
        
        # Global phase integration
        global_steps = int(self.t_star * self.num_inference_steps)
        dt_global = self.t_star / global_steps
        
        for step in range(global_steps):
            t_current = step * dt_global
            t_batch = torch.full((batch_size,), t_current, device=device)
            
            # Expand state to frame dimension for model input
            z_expanded = z.expand(-1, -1, num_frames, -1, -1)  # (B, C, F, H, W)
            
            # Get velocity from model
            v = self._predict_velocity(z_expanded, t_batch)
            
            # Aggregate velocity over frames (global phase property)
            v_global = v.mean(dim=2, keepdim=True)  # (B, C, 1, H, W)
            
            # Euler step
            z = z + v_global * dt_global
        
        # Verify we reached the anchor
        junction_error = torch.abs(z - anchor).max().item()
        print(f"Junction error: {junction_error:.2e}")
        
        # ===== STAGE 2: Local ODE (t⋆ → 1) =====
        print(f"Stage 2: Local paths from anchor...")
        
        # Initialize all frames at anchor: Z(t⋆) = A
        z = anchor.expand(-1, -1, num_frames, -1, -1)  # (B, C, F, H, W)
        
        # Local phase integration
        local_steps = self.num_inference_steps - global_steps
        dt_local = (1 - self.t_star) / local_steps
        
        for step in range(local_steps):
            t_current = self.t_star + step * dt_local
            t_batch = torch.full((batch_size,), t_current, device=device)
            
            # Get frame-specific velocities
            v = self._predict_velocity(z, t_batch)  # (B, C, F, H, W)
            
            # Euler step
            z = z + v * dt_local
        
        print(f"Generated video shape: {z.shape}")
        return z
    
    def _sample_standard_fm(self,
                           batch_size: int,
                           num_frames: int,
                           height: int,
                           width: int,
                           latent_dim: int,
                           device: torch.device,
                           generator: Optional[torch.Generator]) -> torch.Tensor:
        """
        Standard Flow Matching sampling with single-stage ODE.
        """
        print(f"Standard FM sampling: {batch_size} videos, {num_frames} frames")
        print(f"Single-stage ODE: Noise → Frames (0→1)")
        
        # Initial state: independent noise ε ∈ R[B,C,F,H,W]
        if generator is not None:
            z = torch.randn(
                batch_size, latent_dim, num_frames, height, width,
                device=device, generator=generator
            )
        else:
            z = torch.randn(batch_size, latent_dim, num_frames, height, width, device=device)
        
        # Standard integration
        dt = 1.0 / self.num_inference_steps
        
        for step in range(self.num_inference_steps):
            t_current = step * dt
            t_batch = torch.full((batch_size,), t_current, device=device)
            
            # Get velocity
            v = self._predict_velocity(z, t_batch)
            
            # Euler step
            z = z + v * dt
        
        print(f"Generated video shape: {z.shape}")
        return z
    
    def _predict_velocity(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity using the transformer model.
        
        Args:
            z: (B, C, F, H, W) current state
            t: (B,) time values
            
        Returns:
            v: (B, C, F, H, W) predicted velocity
        """
        B, C, F, H, W = z.shape
        
        # Reshape to sequence format for transformer: (B, C, F, H, W) -> (B, 1, seq_len, C)
        seq_len = F * H * W
        z_seq = z.permute(0, 2, 3, 4, 1).reshape(B, 1, seq_len, C)
        
        # Prepare model inputs
        timesteps = t.unsqueeze(1)  # (B, 1)
        
        model_inputs = {
            "hidden_states": z_seq,
            "timestep": timesteps,
            "encoder_hidden_states": None,
            "encoder_attention_mask": None,
        }
        
        # Model prediction
        model_output = self.model(**model_inputs)
        
        if isinstance(model_output, tuple):
            v_pred_seq = model_output[0]  # (B, 1, seq_len, C)
        else:
            v_pred_seq = model_output  # (B, 1, seq_len, C)
        
        # Reshape back to video format: (B, 1, seq_len, C) -> (B, C, F, H, W)
        v_pred = v_pred_seq.squeeze(1).view(B, F, H, W, C).permute(0, 4, 1, 2, 3)
        
        return v_pred
    
    def get_trajectory(self,
                      target_video: torch.Tensor,
                      num_trajectory_points: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get path trajectories for visualization.
        
        Args:
            target_video: (1, C, F, H, W) target video
            num_trajectory_points: Number of points along path
            
        Returns:
            times: (T,) time points
            trajectories: (T, C, F, H, W) path states
        """
        device = target_video.device
        times = torch.linspace(0, 1, num_trajectory_points, device=device)
        trajectories = []
        
        if self.method == "ring_fm":
            # Get anchor for target video
            anchor, _, _ = self.flow_matching.compute_anchor(target_video)
            
            # Sample noise
            noise = self.flow_matching.sample_noise(target_video.shape).to(device)
            
            # Compute path at each time point
            for t in times:
                t_batch = t.unsqueeze(0)  # (1,)
                z_t = self.flow_matching.compute_forward_path(
                    noise, target_video, t_batch, anchor=anchor
                )
                trajectories.append(z_t)
        else:
            # Standard FM
            noise = torch.randn_like(target_video)
            
            for t in times:
                t_batch = t.unsqueeze(0)
                z_t = self.flow_matching.compute_forward_path(noise, target_video, t_batch)
                trajectories.append(z_t)
        
        trajectories = torch.stack(trajectories, dim=0)  # (T, 1, C, F, H, W)
        
        return times.cpu(), trajectories.squeeze(1).cpu()  # Remove batch dim


class RingZipperPipeline:
    """
    High-level pipeline for Ring/Zipper Flow Matching generation.
    
    Handles model loading, VAE decoding, and output formatting.
    """
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 method: str = "ring_fm",
                 t_star: float = 0.8,
                 device: Optional[torch.device] = None):
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.method = method
        
        # Load model (placeholder - would load actual transformer)
        # For now, create dummy model
        self.model = self._create_dummy_model().to(device)
        
        # Create sampler
        self.sampler = RingZipperSampler(
            model=self.model,
            method=method,
            t_star=t_star
        )
    
    def _create_dummy_model(self) -> nn.Module:
        """Create dummy model for demonstration."""
        class DummyTransformer(nn.Module):
            def __init__(self, latent_dim=128):
                super().__init__()
                self.proj = nn.Linear(latent_dim, latent_dim)
            
            def forward(self, hidden_states, timestep, **kwargs):
                # Simple identity-like transformation
                B, seq_len, C = hidden_states.shape[:3]
                out = self.proj(hidden_states.view(-1, C)).view(B, 1, seq_len, C)
                return out * 0.1  # Small output to simulate denoising
        
        return DummyTransformer()
    
    @torch.no_grad()
    def generate(self,
                batch_size: int = 1,
                num_frames: int = 16,
                height: int = 64,
                width: int = 64,
                **kwargs) -> torch.Tensor:
        """
        Generate videos and decode to pixel space.
        
        Args:
            batch_size: Number of videos
            num_frames: Frames per video  
            height: Pixel height
            width: Pixel width
            **kwargs: Additional sampling arguments
            
        Returns:
            videos: (B, 3, F, H, W) RGB videos in [0, 1]
        """
        # Compute latent dimensions (8x downsampling)
        latent_h, latent_w = height // 8, width // 8
        
        # Sample in latent space
        latent_videos = self.sampler.sample(
            batch_size=batch_size,
            num_frames=num_frames,
            height=latent_h,
            width=latent_w,
            device=self.device,
            **kwargs
        )
        
        # Decode to pixel space (placeholder VAE decoder)
        pixel_videos = self._decode_latents(latent_videos, height, width)
        
        return pixel_videos
    
    def _decode_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Decode latents to RGB videos.
        
        For real implementation, would use actual VAE decoder.
        """
        B, C, F, H, W = latents.shape
        
        # Simple upsampling decoder
        decoder = nn.Sequential(
            nn.ConvTranspose2d(C, 64, 4, stride=2, padding=1),  # 2x
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 4x
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),   # 8x
            nn.Tanh()  # Output in [-1, 1]
        ).to(latents.device)
        
        # Apply to each frame
        latents_flat = latents.view(B * F, C, H, W)
        frames_flat = decoder(latents_flat)  # (B*F, 3, H', W')
        
        # Reshape back
        _, C_out, H_out, W_out = frames_flat.shape
        frames = frames_flat.view(B, F, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)
        
        # Resize to target resolution if needed
        if H_out != height or W_out != width:
            frames_flat_resize = frames.view(B * F, C_out, H_out, W_out)
            frames_resized = nn.functional.interpolate(
                frames_flat_resize, size=(height, width), mode='bilinear'
            )
            frames = frames_resized.view(B, C_out, F, height, width)
        
        # Convert from [-1, 1] to [0, 1]
        frames = (frames + 1) * 0.5
        
        return frames