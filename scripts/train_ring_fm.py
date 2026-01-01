#!/usr/bin/env python
"""
Ring/Zipper Flow Matching Training Script

Trains both Standard FM and Ring FM methods on Moving MNIST and UCF101.
Ensures mathematical exactness and audit compliance.

Usage:
    python scripts/train_ring_fm.py --dataset moving_mnist --method ring_fm --output results/moving_mnist/ring_fm
    python scripts/train_ring_fm.py --dataset moving_mnist --method standard_fm --output results/moving_mnist/standard_fm
    python scripts/train_ring_fm.py --dataset ucf101 --method ring_fm --output results/ucf101/ring_fm
    python scripts/train_ring_fm.py --dataset ucf101 --method standard_fm --output results/ucf101/standard_fm
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import json
import time
from typing import Dict, Any
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltxv_trainer.ring_datasets import create_ring_dataloader
from ltxv_trainer.ring_training_strategy import get_ring_training_strategy
from ltxv_trainer.ring_inference import RingZipperSampler
from ltxv_trainer.ring_zipper_flow import create_flow_matching

# Simple timestep sampler
class UniformTimestepSampler:
    def sample(self, batch_size: int) -> torch.Tensor:
        return torch.rand(batch_size)

# Simple transformer model for Ring FM
class SimpleTransformer(torch.nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Time embedding
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Main transformer
        self.input_proj = torch.nn.Linear(latent_dim, hidden_dim)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.output_proj = torch.nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, hidden_states, timestep, **kwargs):
        """
        Args:
            hidden_states: (B, 1, seq_len, latent_dim)
            timestep: (B, 1)
            
        Returns:
            output: (B, 1, seq_len, latent_dim)
        """
        B, _, seq_len, latent_dim = hidden_states.shape
        
        # Flatten sequence dimension
        x = hidden_states.squeeze(1)  # (B, seq_len, latent_dim)
        
        # Time embedding
        t_embed = self.time_embed(timestep.float())  # (B, hidden_dim)
        
        # Project input
        x = self.input_proj(x)  # (B, seq_len, hidden_dim)
        
        # Add time conditioning to each token
        x = x + t_embed.unsqueeze(1)  # Broadcast time to all tokens
        
        # Transformer
        x = self.transformer(x)  # (B, seq_len, hidden_dim)
        
        # Project output
        x = self.output_proj(x)  # (B, seq_len, latent_dim)
        
        # Reshape back
        output = x.unsqueeze(1)  # (B, 1, seq_len, latent_dim)
        
        return output



def save_video_gif(video_tensor: torch.Tensor, save_path: Path, fps: int = 8):
    """
    video_tensor: [C, F, H, W] (RGB, normalized to [-1, 1] or [0, 1])
    """
    # [C, F, H, W] -> [F, H, W, C]
    video = video_tensor.permute(1, 2, 3, 0).cpu().numpy()
    
    # Normalize to [0, 255]
    if video.min() < 0:
        video = (video + 1.0) / 2.0
    video = (video * 255).clip(0, 255).astype(np.uint8)
    
    # Save as GIF
    imageio.mimsave(save_path, video, fps=fps, loop=0)

def train_ring_fm(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train Ring/Zipper Flow Matching model.
    
    Args:
        config: Training configuration
        
    Returns:
        results: Training results and metrics
    """
    print("=" * 80)
    print(f"RING/ZIPPER FLOW MATCHING TRAINING")
    print("=" * 80)
    print(f"Dataset: {config['dataset']}")
    print(f"Method: {config['method']}")
    print(f"Output: {config['output_dir']}")
    if config['method'] == 'ring_fm':
        print(f"Junction time t⋆: {config['t_star']}")
    print("=" * 80)
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(f"{config['output_dir']}/videos", exist_ok=True)
    os.makedirs(f"{config['output_dir']}/gifs", exist_ok=True)
    
    # Create dataloader
    print("\n📊 Creating dataloader...")
    dataloader = create_ring_dataloader(
        dataset_name=config['dataset'],
        batch_size=config['batch_size'],
        num_sequences=config['num_sequences'],
        num_frames=config['num_frames'],
        spatial_size=config['spatial_size'],
        latent_dim=config['latent_dim']
    )
    print(f"Created dataloader with {len(dataloader.dataset)} samples")
    
    # Create model
    print("\n🧠 Creating model...")
    model = SimpleTransformer(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    # Create training strategy
    print(f"\n🎯 Creating {config['method']} training strategy...")
    training_strategy = get_ring_training_strategy(
        method=config['method'],
        t_star=config.get('t_star', 0.8)
    )
    
    # Move anchor network to device if Ring FM
    if hasattr(training_strategy, 'flow_matching') and hasattr(training_strategy.flow_matching, 'anchor_net'):
        if training_strategy.flow_matching.anchor_net is not None:
            training_strategy.flow_matching.anchor_net.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Create timestep sampler
    timestep_sampler = UniformTimestepSampler()
    
    # 샘플러 초기화 (GIF 생성용)
    sampler = RingZipperSampler(
        model=model,
        method=config['method'],
        t_star=config.get('t_star', 0.8),
        num_inference_steps=20, # 빠른 확인을 위해 스텝 감소
        latent_dim=config['latent_dim']
    )

    # Training loop
    print(f"\n🚀 Starting training for {config['num_steps']} steps...")
    model.train()
    
    losses = []
    junction_errors = []
    start_time = time.time()
    
    

    
    for step in range(config['num_steps']):
        # Get batch
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        # Prepare training batch
        training_batch = training_strategy.prepare_batch(batch, timestep_sampler)
        
        # Move to device
        for key in training_batch:
            if isinstance(training_batch[key], torch.Tensor):
                training_batch[key] = training_batch[key].to(device)
        
        # Prepare model inputs
        model_inputs = training_strategy.prepare_model_inputs(training_batch)
        
        # Forward pass
        model_pred = model(
            hidden_states=model_inputs['hidden_states'],
            timestep=model_inputs['timestep']
        )
        
        # Compute loss
        loss = training_strategy.compute_loss(model_pred, training_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        losses.append(loss.item())
        
        if 'junction_error' in training_batch:
            junction_errors.append(training_batch['junction_error'])
        
        # Log progress
        if step % 100 == 0 or step == config['num_steps'] - 1:
            avg_loss = np.mean(losses[-100:])
            print(f"Step {step:4d}: Loss = {avg_loss:.6f}")
            
            # 500스텝마다 샘플 GIF 생성
            if step % 500 == 0 and step > 0:
                model.eval()
                with torch.no_grad():
                    print(f"🎬 Generating intermediate sample GIF at step {step}...")
                    sample_latents = sampler.sample(
                        batch_size=1,
                        num_frames=config['num_frames'],
                        height=config['spatial_size'] // 8,
                        width=config['spatial_size'] // 8,
                        device=device
                    )
                    # _decode_latents를 통해 이미지화
                    sample_video = sampler._decode_latents(
                        sample_latents, 
                        height=config['spatial_size'], 
                        width=config['spatial_size']
                    )
                    save_video_gif(sample_video[0], gifs_dir / f"step_{step}.gif")
                model.train()
            log_str = f"Step {step:4d}/{config['num_steps']}: Loss = {avg_loss:.6f}, Time = {elapsed:.1f}s"
            
            if junction_errors and config['method'] == 'ring_fm':
                avg_junction_error = np.mean(junction_errors[-100:])
                log_str += f", Junction Error = {avg_junction_error:.2e}"
            
            print(log_str)
    
    training_time = time.time() - start_time
    print(f"\n✅ Training complete in {training_time:.1f}s")
    print(f"Final loss: {losses[-1]:.6f}")
    if junction_errors:
        print(f"Final junction error: {junction_errors[-1]:.2e}")
    
    # Save model
    model_path = f"{config['output_dir']}/model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'losses': losses,
        'junction_errors': junction_errors
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Generate samples
    print("\n🎬 Generating samples...")
    model.eval()
    
    sampler = RingZipperSampler(
        model=model,
        method=config['method'],
        t_star=config.get('t_star', 0.8),
        num_inference_steps=config['num_inference_steps'],
        latent_dim=config['latent_dim']
    )
    
    with torch.no_grad():
        # Generate videos
        generated_videos = sampler.sample(
            batch_size=config['num_samples'],
            num_frames=config['num_frames'],
            height=config['spatial_size'] // 8,  # Latent space size
            width=config['spatial_size'] // 8,
            latent_dim=config['latent_dim'],
            device=device
        )
        
        # Save generated videos
        torch.save(generated_videos.cpu(), f"{config['output_dir']}/videos/generated.pt")
        print(f"Generated videos saved: {generated_videos.shape}")
    
    # Compute metrics
    print("\n📈 Computing metrics...")
    
    # Get real videos for comparison
    sample_batch = next(iter(dataloader))
    real_latents = []
    
    for i in range(min(config['num_samples'], len(sample_batch['latents']['latents']))):
        latent_data = sample_batch['latents']['latents'][i]  # (1, seq_len, C)
        
        # Get dimensions
        F = sample_batch['latents']['num_frames']
        H = sample_batch['latents']['height'] 
        W = sample_batch['latents']['width']
        
        # Reshape to video format
        C = latent_data.shape[-1]
        latent_video = latent_data.squeeze(0).view(F, H, W, C).permute(3, 0, 1, 2)  # (C, F, H, W)
        real_latents.append(latent_video)
    
    real_videos = torch.stack(real_latents)  # (B, C, F, H, W)
    
    # Compute simple metrics
    mse = torch.nn.functional.mse_loss(generated_videos.cpu(), real_videos).item()
    
    # Temporal consistency (frame differences)
    gen_diffs = (generated_videos[:, :, 1:] - generated_videos[:, :, :-1]).norm(dim=1).mean().item()
    real_diffs = (real_videos[:, :, 1:] - real_videos[:, :, :-1]).norm(dim=1).mean().item()
    temporal_consistency = abs(gen_diffs - real_diffs)
    
    metrics = {
        'dataset': config['dataset'],
        'method': config['method'],
        'final_loss': losses[-1],
        'training_time': training_time,
        'mse': mse,
        'temporal_consistency': temporal_consistency,
        'num_videos': len(generated_videos),
        'video_shape': list(generated_videos.shape)
    }
    
    if junction_errors:
        metrics['final_junction_error'] = junction_errors[-1]
        metrics['junction_constraint_satisfied'] = junction_errors[-1] < 1e-5
    
    # Save metrics
    with open(f"{config['output_dir']}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics: MSE = {mse:.6f}, Temporal = {temporal_consistency:.6f}")
    
    return metrics


def create_path_visualization(config: Dict[str, Any]) -> None:
    """Create path topology visualization for Ring vs Standard FM."""
    print(f"\n📊 Creating path visualization for {config['dataset']}...")
    
    if config['dataset'] != 'moving_mnist':
        print("Path visualization only available for Moving MNIST")
        return
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Load sample video
    dataloader = create_ring_dataloader(
        dataset_name=config['dataset'],
        batch_size=1,
        num_sequences=10,
        num_frames=config['num_frames'],
        spatial_size=config['spatial_size'],
        latent_dim=config['latent_dim']
    )
    
    sample_batch = next(iter(dataloader))
    latent_data = sample_batch['latents']['latents'][0]  # (1, seq_len, C)
    
    # Convert to video format
    F = sample_batch['latents']['num_frames']
    H = sample_batch['latents']['height']
    W = sample_batch['latents']['width']
    C = latent_data.shape[-1]
    
    target_video = latent_data.squeeze(0).view(F, H, W, C).permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, F, H, W)
    target_video = target_video.to(device)
    
    # Create flow matching instances
    standard_fm = create_flow_matching("standard_fm", latent_dim=C)
    ring_fm = create_flow_matching("ring_fm", t_star=config.get('t_star', 0.8), latent_dim=C)
    
    # Move to device
    if hasattr(ring_fm, 'anchor_net') and ring_fm.anchor_net:
        ring_fm.anchor_net.to(device)
    
    # Get trajectories
    num_points = 50
    times = torch.linspace(0, 1, num_points, device=device)
    
    # Standard FM trajectory
    standard_noise = standard_fm.sample_noise(target_video.shape).to(device)
    standard_trajectory = []
    
    for t in times:
        t_batch = t.unsqueeze(0)
        z_t = standard_fm.compute_forward_path(standard_noise, target_video, t_batch)
        standard_trajectory.append(z_t.cpu())
    
    standard_trajectory = torch.stack(standard_trajectory, dim=0)  # (T, 1, C, F, H, W)
    
    # Ring FM trajectory
    ring_noise = ring_fm.sample_noise(target_video.shape).to(device)
    anchor, _, _ = ring_fm.compute_anchor(target_video)
    ring_trajectory = []
    
    for t in times:
        t_batch = t.unsqueeze(0)
        z_t = ring_fm.compute_forward_path(ring_noise, target_video, t_batch, anchor=anchor)
        ring_trajectory.append(z_t.cpu())
    
    ring_trajectory = torch.stack(ring_trajectory, dim=0)  # (T, 1, C, F, H, W)
    
    # Save trajectories
    os.makedirs(f"{config['output_dir']}/path_visualizations", exist_ok=True)
    
    torch.save({
        'times': times.cpu(),
        'standard_trajectory': standard_trajectory,
        'ring_trajectory': ring_trajectory,
        'target_video': target_video.cpu(),
        't_star': config.get('t_star', 0.8)
    }, f"{config['output_dir']}/path_visualizations/trajectories.pt")
    
    # Create simple visualization using matplotlib
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        # Flatten spatial dimensions for PCA
        def flatten_for_pca(trajectory):
            T, B, C, F, H, W = trajectory.shape
            return trajectory.view(T * F, C * H * W).numpy()
        
        standard_flat = flatten_for_pca(standard_trajectory)
        ring_flat = flatten_for_pca(ring_trajectory)
        
        # Combine for consistent PCA
        all_data = np.vstack([standard_flat, ring_flat])
        pca = PCA(n_components=2)
        all_pca = pca.fit_transform(all_data)
        
        # Split back
        n_std = len(standard_flat)
        std_pca = all_pca[:n_std].reshape(num_points, F, 2)
        ring_pca = all_pca[n_std:].reshape(num_points, F, 2)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Standard FM
        ax1.set_title("Standard Flow Matching")
        for f in range(min(F, 5)):  # Plot first 5 frames
            ax1.plot(std_pca[:, f, 0], std_pca[:, f, 1], alpha=0.7, label=f'Frame {f}')
        ax1.set_xlabel("PCA Component 1")
        ax1.set_ylabel("PCA Component 2")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Ring FM
        ax2.set_title("Ring/Zipper Flow Matching")
        t_star_idx = int(config.get('t_star', 0.8) * num_points)
        
        for f in range(min(F, 5)):
            # Global phase
            ax2.plot(ring_pca[:t_star_idx, f, 0], ring_pca[:t_star_idx, f, 1], 
                    alpha=0.7, linestyle='-', label=f'Frame {f} (Global)' if f < 3 else "")
            # Local phase
            ax2.plot(ring_pca[t_star_idx:, f, 0], ring_pca[t_star_idx:, f, 1],
                    alpha=0.7, linestyle='--', label=f'Frame {f} (Local)' if f < 3 else "")
            # Junction point
            ax2.scatter(ring_pca[t_star_idx, f, 0], ring_pca[t_star_idx, f, 1],
                       color='red', s=50, alpha=0.8)
        
        # Mark shared anchor (average junction position)
        anchor_pos = ring_pca[t_star_idx, :, :].mean(axis=0)
        ax2.scatter(anchor_pos[0], anchor_pos[1], color='red', marker='X', s=200,
                   label=f'Junction (t={config.get("t_star", 0.8)})')
        
        ax2.set_xlabel("PCA Component 1")
        ax2.set_ylabel("PCA Component 2")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{config['output_dir']}/path_visualizations/path_comparison.png", dpi=300)
        plt.close()
        
        print(f"Path visualization saved to {config['output_dir']}/path_visualizations/")
        
    except ImportError:
        print("Warning: matplotlib or sklearn not available, skipping visualization")


def main():
    parser = argparse.ArgumentParser(description="Ring/Zipper Flow Matching Training")
    parser.add_argument("--dataset", choices=["moving_mnist", "ucf101"], required=True,
                       help="Dataset to use")
    parser.add_argument("--method", choices=["standard_fm", "ring_fm"], required=True,
                       help="Flow matching method")
    parser.add_argument("--output", required=True,
                       help="Output directory")
    parser.add_argument("--t_star", type=float, default=0.8,
                       help="Junction time for Ring FM (default: 0.8)")
    parser.add_argument("--num_steps", type=int, default=2000,
                       help="Training steps (default: 2000)")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size (default: 2)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--device", default="cuda",
                       help="Device (default: cuda)")
    parser.add_argument("--create_visualization", action="store_true",
                       help="Create path visualization")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'dataset': args.dataset,
        'method': args.method,
        'output_dir': args.output,
        't_star': args.t_star,
        'num_steps': args.num_steps,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'device': args.device,
        
        # Dataset parameters
        'num_sequences': 500,  # Smaller for faster training
        'num_frames': 16,
        'spatial_size': 64,
        'latent_dim': 128,
        'hidden_dim': 512,
        
        # Inference parameters
        'num_inference_steps': 50,
        'num_samples': 8
    }
    
    # Train model
    results = train_ring_fm(config)
    
    # Create path visualization if requested
    if args.create_visualization:
        create_path_visualization(config)
    
    print(f"\n🎉 Training complete! Results saved to {args.output}")
    print(f"Final metrics: {results}")


if __name__ == "__main__":
    main()