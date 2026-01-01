"""
Dataset loaders for Ring/Zipper Flow Matching experiments.

Implements Moving MNIST and UCF101 datasets with exact tensor shapes:
- x0 ∈ R[B, C, F, H, W] format
- Proper latent encoding simulation
- No dataset mixing
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import requests
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import cv2


class MovingMNISTDataset(Dataset):
    """
    Moving MNIST dataset for Ring/Zipper Flow Matching.
    
    Downloads standard Moving MNIST and converts to latent format.
    Output: x0 ∈ R[C, F, H, W] per sample
    """
    
    def __init__(self, 
                 root: str = "data/moving_mnist",
                 num_sequences: int = 1000,
                 num_frames: int = 16,
                 spatial_size: int = 64,
                 latent_dim: int = 128):
        super().__init__()
        
        self.root = Path(root)
        self.num_sequences = num_sequences
        self.num_frames = num_frames
        self.spatial_size = spatial_size
        self.latent_dim = latent_dim
        
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Download and load data
        self.data_path = self.root / "mnist_test_seq.npy"
        if not self.data_path.exists():
            self._download_data()
        
        self.videos = self._load_and_preprocess()
    
    def _download_data(self) -> None:
        """Download Moving MNIST dataset."""
        url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
        print(f"Downloading Moving MNIST from {url}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(self.data_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded to {self.data_path}")
    
    def _load_and_preprocess(self) -> torch.Tensor:
        """
        Load and preprocess Moving MNIST data.
        
        Returns:
            videos: (N, C, F, H, W) preprocessed video latents
        """
        print(f"Loading Moving MNIST from {self.data_path}...")
        
        # Load: (20, 10000, 64, 64) = (frames, sequences, height, width)
        data = np.load(self.data_path)
        
        # Transpose: (10000, 20, 64, 64) = (sequences, frames, height, width)
        data = data.transpose(1, 0, 2, 3)
        
        # Select subset and frames
        data = data[:self.num_sequences, :self.num_frames]
        
        # Normalize to [-1, 1]
        videos = torch.from_numpy(data).float()
        videos = videos / 127.5 - 1.0
        
        # Add channel dimension: (N, F, H, W) -> (N, 1, F, H, W)
        videos = videos.unsqueeze(1)
        
        # Resize if needed
        if self.spatial_size != 64:
            videos = self._resize_videos(videos)
        
        # Encode to latent space
        videos = self._encode_to_latents(videos)
        
        print(f"Loaded {len(videos)} videos: {videos.shape}")
        print(f"Value range: [{videos.min():.3f}, {videos.max():.3f}]")
        
        return videos
    
    def _resize_videos(self, videos: torch.Tensor) -> torch.Tensor:
        """Resize videos to target spatial size."""
        N, C, F, H, W = videos.shape
        
        # Flatten frames for interpolation: (N, C, F, H, W) -> (N*C*F, 1, H, W)
        videos_flat = videos.reshape(N * C * F, 1, H, W)
        
        # Resize
        videos_resized = nn.functional.interpolate(
            videos_flat,
            size=(self.spatial_size, self.spatial_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Reshape back: (N*C*F, 1, H, W) -> (N, C, F, H, W)
        videos = videos_resized.reshape(N, C, F, self.spatial_size, self.spatial_size)
        
        return videos
        
    def _encode_to_latents(self, videos: torch.Tensor) -> torch.Tensor:
            N, C_in, F, H, W = videos.shape
            
            encoder = nn.Sequential(
                nn.Conv2d(1, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, self.latent_dim, 4, stride=2, padding=1),
            )
            encoder.eval() # 추론 모드 설정

            with torch.no_grad(): # 미분 계산 비활성화
                videos_flat = videos.reshape(N * F, 1, H, W)
                latents_flat = encoder(videos_flat).detach() # 그래프 연결 완전히 끊기
            
            C_out, H_out, W_out = latents_flat.shape[1], latents_flat.shape[2], latents_flat.shape[3]
            latents = latents_flat.reshape(N, F, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)
            latents = latents * 0.3
            
            return latents
    
    def __len__(self) -> int:
        return len(self.videos)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            dict with 'latents' key containing (C, F, H, W) tensor
        """
        video = self.videos[idx]  # (C, F, H, W)
        
        # Convert to expected format for training strategy
        C, F, H, W = video.shape
        seq_len = F * H * W
        
        # Reshape to sequence format: (C, F, H, W) -> (1, seq_len, C)
        video_seq = video.permute(1, 2, 3, 0).reshape(1, seq_len, C)
        
        return {
            "latents": {
                "latents": video_seq,
                "num_frames": F,
                "height": H,
                "width": W
            }
        }


class UCF101Dataset(Dataset):
    """
    UCF101 video dataset for Ring/Zipper Flow Matching.
    
    Loads videos and converts to latent format.
    Falls back to synthetic data if UCF101 not available.
    """
    
    def __init__(self,
                 root: str = "data/ucf101",
                 num_sequences: int = 1000,
                 num_frames: int = 16,
                 spatial_size: int = 64,
                 latent_dim: int = 128):
        super().__init__()
        
        self.root = Path(root)
        self.num_sequences = num_sequences
        self.num_frames = num_frames
        self.spatial_size = spatial_size
        self.latent_dim = latent_dim
        
        # Find video files
        self.video_paths = self._find_video_files()
        
        if len(self.video_paths) == 0:
            print(f"Warning: No UCF101 videos found in {root}")
            print("Using synthetic data...")
            self.videos = self._generate_synthetic_videos()
        else:
            self.videos = self._load_real_videos()
    
    def _find_video_files(self) -> list[Path]:
        """Find video files in UCF101 directory."""
        if not self.root.exists():
            return []
        
        video_extensions = {'.avi', '.mp4', '.mov', '.mkv'}
        video_paths = []
        
        for ext in video_extensions:
            video_paths.extend(self.root.rglob(f'*{ext}'))
        
        return video_paths[:self.num_sequences]
    
    def _load_real_videos(self) -> torch.Tensor:
        """Load and process real UCF101 videos."""
        print(f"Loading {len(self.video_paths)} UCF101 videos...")
        
        processed_videos = []
        
        for video_path in self.video_paths[:self.num_sequences]:
            try:
                video = self._load_single_video(video_path)
                if video is not None:
                    processed_videos.append(video)
            except Exception as e:
                print(f"Error loading {video_path}: {e}")
                continue
        
        if len(processed_videos) == 0:
            print("Failed to load any videos, using synthetic data...")
            return self._generate_synthetic_videos()
        
        # Pad with synthetic if needed
        while len(processed_videos) < self.num_sequences:
            synthetic = self._generate_single_synthetic_video()
            processed_videos.append(synthetic)
        
        videos = torch.stack(processed_videos[:self.num_sequences])
        
        print(f"Loaded {len(videos)} videos: {videos.shape}")
        return videos
    
    def _load_single_video(self, video_path: Path) -> Optional[torch.Tensor]:
        """Load and process a single video file."""
        cap = cv2.VideoCapture(str(video_path))
        
        frames = []
        
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.spatial_size, self.spatial_size))
            
            # Normalize to [-1, 1]
            frame = frame.astype(np.float32) / 127.5 - 1.0
            frames.append(frame)
        
        cap.release()
        
        # Pad if too short
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.spatial_size, self.spatial_size, 3)))
        
        # Convert to tensor: (F, H, W, 3) -> (3, F, H, W)
        video = np.stack(frames)  # (F, H, W, 3)
        video = torch.from_numpy(video).permute(3, 0, 1, 2)  # (3, F, H, W)
        
        # Encode to latents
        video = self._encode_to_latents(video.unsqueeze(0)).squeeze(0)  # (C, F, H, W)
        
        return video
    
    def _generate_synthetic_videos(self) -> torch.Tensor:
        """Generate synthetic videos when real data unavailable."""
        print(f"Generating {self.num_sequences} synthetic videos...")
        
        videos = []
        for i in range(self.num_sequences):
            video = self._generate_single_synthetic_video()
            videos.append(video)
        
        videos = torch.stack(videos)
        
        print(f"Generated {len(videos)} synthetic videos: {videos.shape}")
        return videos
    
    def _generate_single_synthetic_video(self) -> torch.Tensor:
        """Generate a single synthetic video."""
        frames = []
        
        # Random moving object parameters
        start_pos = np.random.randint(10, self.spatial_size - 20, 2)
        end_pos = np.random.randint(10, self.spatial_size - 20, 2)
        color = np.random.rand(3)
        
        for f in range(self.num_frames):
            frame = np.zeros((self.spatial_size, self.spatial_size, 3))
            
            # Linear interpolation for position
            alpha = f / (self.num_frames - 1)
            pos = start_pos + alpha * (end_pos - start_pos)
            pos = pos.astype(int)
            
            # Draw moving object
            size = 8
            y_min, y_max = max(0, pos[0] - size), min(self.spatial_size, pos[0] + size)
            x_min, x_max = max(0, pos[1] - size), min(self.spatial_size, pos[1] + size)
            frame[y_min:y_max, x_min:x_max] = color
            
            # Normalize to [-1, 1]
            frame = frame * 2 - 1
            frames.append(frame)
        
        # Convert to tensor: (F, H, W, 3) -> (3, F, H, W)
        video = np.stack(frames)
        video = torch.from_numpy(video).float().permute(3, 0, 1, 2)
        
        # Encode to latents
        video = self._encode_to_latents(video.unsqueeze(0)).squeeze(0)
        
        return video
        
    def _encode_to_latents(self, videos: torch.Tensor) -> torch.Tensor:
            N, C_in, F, H, W = videos.shape
            
            encoder = nn.Sequential(
                nn.Conv2d(C_in, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, self.latent_dim, 4, stride=2, padding=1),
            )
            encoder.eval() # 추론 모드 설정

            with torch.no_grad(): # 미분 계산 비활성화
                videos_flat = videos.reshape(N * F, C_in, H, W)
                latents_flat = encoder(videos_flat).detach() # 그래프 연결 완전히 끊기
            
            C_out, H_out, W_out = latents_flat.shape[1], latents_flat.shape[2], latents_flat.shape[3]
            latents = latents_flat.reshape(N, F, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)
            latents = latents * 0.2
            
            return latents
    
    def __len__(self) -> int:
        return len(self.videos)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns dict with 'latents' key containing (C, F, H, W) tensor."""
        video = self.videos[idx]  # (C, F, H, W)
        
        C, F, H, W = video.shape
        seq_len = F * H * W
        
        # Reshape to sequence format
        video_seq = video.permute(1, 2, 3, 0).reshape(1, seq_len, C)
        
        return {
            "latents": {
                "latents": video_seq,
                "num_frames": F,
                "height": H,
                "width": W
            }
        }


def create_ring_dataloader(dataset_name: str,
                          batch_size: int = 2,
                          num_sequences: int = 1000,
                          num_frames: int = 16,
                          spatial_size: int = 64,
                          latent_dim: int = 128,
                          **kwargs) -> DataLoader:
    """
    Create dataloader for Ring/Zipper Flow Matching experiments.
    
    Args:
        dataset_name: "moving_mnist" or "ucf101"
        batch_size: Batch size
        num_sequences: Number of video sequences
        num_frames: Number of frames per video
        spatial_size: Spatial resolution
        latent_dim: Latent space dimension
        **kwargs: Additional dataset arguments
        
    Returns:
        DataLoader yielding batches in correct format
    """
    if dataset_name == "moving_mnist":
        dataset = MovingMNISTDataset(
            num_sequences=num_sequences,
            num_frames=num_frames,
            spatial_size=spatial_size,
            latent_dim=latent_dim,
            **kwargs
        )
    elif dataset_name == "ucf101":
        dataset = UCF101Dataset(
            num_sequences=num_sequences,
            num_frames=num_frames,
            spatial_size=spatial_size,
            latent_dim=latent_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader