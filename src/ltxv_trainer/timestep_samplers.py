import torch


class TimestepSampler:
    """Base class for timestep samplers.

    Timestep samplers are used to sample timesteps for diffusion models.
    They should implement both sample() and sample_for() methods.
    """

    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:
        """Sample timesteps for a batch.

        Args:
            batch_size: Number of timesteps to sample
            seq_length: (optional) Length of the sequence being processed
            device: Device to place the samples on

        Returns:
            Tensor of shape (batch_size,) containing timesteps
        """
        raise NotImplementedError

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)

        Returns:
            Tensor of shape (batch_size,) containing timesteps
        """
        raise NotImplementedError


class UniformTimestepSampler(TimestepSampler):
    """Samples timesteps uniformly between min_value and max_value (default 0 and 1)."""

    def __init__(self, min_value: float = 0.0, max_value: float = 1.0):
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:  # noqa: ARG002
        return torch.rand(batch_size, device=device) * (self.max_value - self.min_value) + self.min_value

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, device=batch.device)


class ShiftedLogitNormalTimestepSampler:
    """
    Samples timesteps from a shifted logit-normal distribution,
    where the shift is determined by the sequence length.
    """

    def __init__(self, std: float = 1.0):
        self.std = std

    def sample(self, batch_size: int, seq_length: int, device: torch.device = None) -> torch.Tensor:
        """Sample timesteps for a batch from a shifted logit-normal distribution.

        Args:
            batch_size: Number of timesteps to sample
            seq_length: Length of the sequence being processed, used to determine the shift
            device: Device to place the samples on

        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by seq_length
        """
        shift = self._get_shift_for_sequence_length(seq_length)
        normal_samples = torch.randn((batch_size,), device=device) * self.std + shift
        timesteps = torch.sigmoid(normal_samples)
        return timesteps

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)

        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by the sequence length
            of the input batch

        Raises:
            ValueError: If the input batch does not have 3 dimensions
        """
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, seq_length, device=batch.device)

    @staticmethod
    def _get_shift_for_sequence_length(
        seq_length: int,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
    ) -> float:
        # Calculate the shift value for a given sequence length using linear interpolation
        # between min_shift and max_shift based on sequence length.
        m = (max_shift - min_shift) / (max_tokens - min_tokens)  # Calculate slope
        b = min_shift - m * min_tokens  # Calculate y-intercept
        shift = m * seq_length + b  # Apply linear equation y = mx + b
        return shift


class MotionCentricImportanceSampler(TimestepSampler):
    """
    Motion-Centric Importance Sampling (MCIS) for AFM Temporal Collapse 해결
    
    샘플링 분포: 전체 샘플의 70%를 Local Phase [0, t_boundary]에서, 
                30%를 Global Phase (t_boundary, 1.0]에서 추출
    """
    
    def __init__(self, t_boundary: float = 0.2, local_ratio: float = 0.7):
        self.t_boundary = t_boundary
        self.local_ratio = local_ratio
        self.global_ratio = 1.0 - local_ratio
    
    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:
        """Sample timesteps with Motion-Centric Importance Sampling"""
        # 배치 크기에 따른 Local/Global 샘플 수 결정
        num_local_samples = int(batch_size * self.local_ratio)
        num_global_samples = batch_size - num_local_samples
        
        # Local Phase 샘플링: [0, t_boundary]
        local_samples = torch.rand(num_local_samples, device=device) * self.t_boundary
        
        # Global Phase 샘플링: (t_boundary, 1.0]
        global_samples = torch.rand(num_global_samples, device=device) * (1.0 - self.t_boundary) + self.t_boundary
        
        # 결합 및 섞기
        all_samples = torch.cat([local_samples, global_samples])
        shuffled_indices = torch.randperm(batch_size, device=device)
        
        return all_samples[shuffled_indices]
    
    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor"""
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")
        
        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, device=batch.device)
    
    def get_importance_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Importance Weight 계산 - 샘플링 편향 보정
        
        수학적 기댓값 유지를 위해:
        - Local 구간 (over-sampled): 낮은 가중치 (1/local_ratio)
        - Global 구간 (under-sampled): 높은 가중치 (1/global_ratio)
        """
        is_local = t <= self.t_boundary
        
        # 정확한 Importance Weight 계산
        # p_uniform(t) = 1.0 (uniform distribution over [0,1])
        # p_mcis(t) = local_ratio/t_boundary (local) or global_ratio/(1-t_boundary) (global)
        # weight = p_uniform(t) / p_mcis(t)
        
        local_weight = self.t_boundary / self.local_ratio  # = 0.2 / 0.7 ≈ 0.286
        global_weight = (1.0 - self.t_boundary) / self.global_ratio  # = 0.8 / 0.3 ≈ 2.667
        
        return torch.where(is_local, local_weight, global_weight)


# Import AFM samplers
try:
    from ltxv_trainer.afm_timestep_samplers import AFM_SAMPLERS
    afm_samplers_available = True
except ImportError:
    AFM_SAMPLERS = {}
    afm_samplers_available = False

SAMPLERS = {
    "uniform": UniformTimestepSampler,
    "shifted_logit_normal": ShiftedLogitNormalTimestepSampler,
    "motion_centric": MotionCentricImportanceSampler,
}

# Add AFM samplers if available
if afm_samplers_available:
    SAMPLERS.update(AFM_SAMPLERS)


def example() -> None:
    import matplotlib.pyplot as plt  # type: ignore

    sampler = ShiftedLogitNormalTimestepSampler()
    for seq_length in [1024, 2048, 4096, 8192]:
        samples = sampler.sample(batch_size=1_000_000, seq_length=seq_length)

        # plot the histogram of the samples
        plt.hist(samples.numpy(), bins=100, density=True)
        plt.title(f"Timestep Samples for Sequence Length {seq_length}")
        plt.xlabel("Timestep")
        plt.ylabel("Density")
        plt.show()


if __name__ == "__main__":
    example()
