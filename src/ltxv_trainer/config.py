from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from ltxv_trainer.model_loader import LtxvModelVersion
from ltxv_trainer.quantization import QuantizationOptions


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(ConfigBaseModel):
    """Configuration for the base model and training mode"""

    model_source: str | Path | LtxvModelVersion = Field(
        default=LtxvModelVersion.latest(),
        description="Model source - can be a HuggingFace repo ID, local path, or LtxvModelVersion",
    )
    
    # Add fields from YAML config
    id: str = Field(
        default="Lightricks/LTX-Video",
        description="Model ID from HuggingFace",
    )
    
    variant: str = Field(
        default="2b",
        description="Model variant (e.g., 2b, 13b)",
    )
    
    cache_dir: str | None = Field(
        default=None,
        description="Cache directory for model files",
    )

    training_mode: Literal["lora", "full"] = Field(
        default="lora",
        description="Training mode - either LoRA fine-tuning or full model fine-tuning",
    )

    load_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to a checkpoint file or directory to load from. "
        "If a directory is provided, the latest checkpoint will be used.",
    )

    # noinspection PyNestedDecorators
    @field_validator("model_source", mode="before")
    @classmethod
    def validate_model_source(cls, v):  # noqa: ANN001, ANN206
        """Try to convert model source to LtxvModelVersion if possible."""
        if isinstance(v, (str, LtxvModelVersion)):
            try:
                return LtxvModelVersion(v)
            except ValueError:
                return v
        return v


class LoraConfig(ConfigBaseModel):
    """Configuration for LoRA fine-tuning"""
    
    # Add enabled field from YAML
    enabled: bool = Field(
        default=True,
        description="Whether LoRA is enabled",
    )

    rank: int = Field(
        default=64,
        description="Rank of LoRA adaptation",
        ge=2,
    )

    alpha: int = Field(
        default=64,
        description="Alpha scaling factor for LoRA",
        ge=1,
    )

    dropout: float = Field(
        default=0.0,
        description="Dropout probability for LoRA layers",
        ge=0.0,
        le=1.0,
    )

    target_modules: list[str] = Field(
        default=("to_k", "to_q", "to_v", "to_out.0"),
        description="List of modules to target with LoRA",
    )


class ConditioningConfig(ConfigBaseModel):
    """Configuration for conditioning during training"""

    # 1. 'ring_fm'과 'quantum_fm'을 Literal 목록에 추가합니다.
    mode: Literal["none", "reference_video", "ring_fm", "quantum_fm"] = Field(
        default="none",
        description="Type of conditioning to use during training",
    )

    # 2. ring_fm에서 사용할 t_star 필드를 새로 정의합니다.
    t_star: float = Field(
        default=0.8,
        description="The timestep threshold for Ring/Zipper FM strategy",
        ge=0.0,
        le=1.0,
    )

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    reference_latents_dir: str = Field(
        default="ref_latents",
        description="Directory name for latents of reference videos when using reference_video mode",
    )


class QuantumConfig(ConfigBaseModel): # 새로운 Quantum 전용 설정 클래스 추가
    quantum_superposition_factor: float = 3.0
    high_freq_jitter: float = 0.1
    anchor_entropy_range: list[int] = [-8, 8]
    collapse_threshold: float = 0.2
    frame_repulsion_strength: float = 5.0
    quantum_decay_rate: float = 25.0
    cosine_similarity_penalty: bool = True
    orthogonal_projection: bool = False



class OptimizationConfig(ConfigBaseModel):
    """Configuration for optimization parameters"""

    learning_rate: float = Field(
        default=5e-4,
        description="Learning rate for optimization",
    )

    steps: int = Field(
        default=3000,
        description="Number of training steps",
    )

    batch_size: int = Field(
        default=2,
        description="Batch size for training",
    )

    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of steps to accumulate gradients",
    )

    max_grad_norm: float = Field(
        default=1.0,
        description="Maximum gradient norm for clipping",
    )

    optimizer_type: Literal["adamw", "adamw8bit"] = Field(
        default="adamw",
        description="Type of optimizer to use for training",
    )

    scheduler_type: Literal[
        "constant",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
    ] = Field(
        default="linear",
        description="Type of scheduler to use for training",
    )

    scheduler_params: dict = Field(
        default_factory=dict,
        description="Parameters for the scheduler",
    )

    enable_gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to save memory at the cost of slower training",
    )


class AccelerationConfig(ConfigBaseModel):
    """Configuration for hardware acceleration and compute optimization"""

    mixed_precision_mode: Literal["no", "fp16", "bf16"] | None = Field(
        default="bf16",
        description="Mixed precision training mode",
    )

    quantization: QuantizationOptions | None = Field(
        default=None,
        description="Quantization precision to use",
    )

    load_text_encoder_in_8bit: bool = Field(
        default=False,
        description="Whether to load the text encoder in 8-bit precision to save memory",
    )

    compile_with_inductor: bool = Field(
        default=True,
        description="Compile the model with Torch Inductor",
    )

    compilation_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        default="reduce-overhead",
        description="Compilation mode for Torch Inductor",
    )


class DataConfig(ConfigBaseModel):
    """Configuration for data loading and processing"""

    preprocessed_data_root: str = Field(
        description="Path to folder containing preprocessed training data",
    )
    
    # Add fields from YAML config
    dataset_path: str = Field(
        default="path/to/your/video/dataset",
        description="Path to the video dataset",
    )
    
    resolution: list[int] = Field(
        default=[256, 256],
        description="Spatial resolution [width, height]",
    )
    
    num_frames: int = Field(
        default=16,
        description="Number of frames per video",
    )
    
    frame_stride: int = Field(
        default=1,
        description="Frame stride for sampling",
    )
    
    normalize_frames: bool = Field(
        default=True,
        description="Whether to normalize frames",
    )
    
    temporal_consistency_check: bool = Field(
        default=True,
        description="Whether to check temporal consistency",
    )

    num_dataloader_workers: int = Field(
        default=2,
        description="Number of background processes for data loading (0 means synchronous loading)",
        ge=0,
    )


class ValidationConfig(ConfigBaseModel):
    """Configuration for validation during training"""

    prompts: list[str] = Field(
        default_factory=list,
        description="List of prompts to use for validation",
    )

    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt to use for validation examples",
    )

    images: list[str] | None = Field(
        default=None,
        description="List of image paths to use for validation. "
        "One image path must be provided for each validation prompt",
    )

    reference_videos: list[str] | None = Field(
        default=None,
        description="List of reference video paths to use for validation. "
        "One video path must be provided for each validation prompt",
    )

    video_dims: tuple[int, int, int] = Field(
        default=(704, 480, 161),
        description="Dimensions of validation videos (width, height, frames)",
    )

    seed: int = Field(
        default=42,
        description="Random seed used when sampling validation videos",
    )

    inference_steps: int = Field(
        default=50,
        description="Number of inference steps for validation",
        gt=0,
    )

    interval: int | None = Field(
        default=100,
        description="Number of steps between validation runs. If None, validation is disabled.",
        gt=0,
    )

    videos_per_prompt: int = Field(
        default=1,
        description="Number of videos to generate per validation prompt",
        gt=0,
    )

    guidance_scale: float = Field(
        default=3.5,
        description="Guidance scale to use during validation",
        ge=1.0,
    )

    skip_initial_validation: bool = Field(
        default=False,
        description="Skip validation video sampling at step 0 (beginning of training)",
    )

    @field_validator("images")
    @classmethod
    def validate_num_images(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        """Validate that number of images (if provided) matches number of prompts."""
        num_prompts = len(info.data.get("prompts", []))
        if v is not None and len(v) != num_prompts:
            raise ValueError(f"Number of images ({len(v)}) must match number of prompts ({num_prompts})")
        return v

    @field_validator("reference_videos")
    @classmethod
    def validate_num_reference_videos(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        """Validate that number of reference videos (if provided) matches number of prompts."""
        num_prompts = len(info.data.get("prompts", []))
        if v is not None and len(v) != num_prompts:
            raise ValueError(f"Number of reference videos ({len(v)}) must match number of prompts ({num_prompts})")
        return v


class CheckpointsConfig(ConfigBaseModel):
    """Configuration for model checkpointing during training"""

    interval: int | None = Field(
        default=None,
        description="Number of steps between checkpoint saves. If None, intermediate checkpoints are disabled.",
        gt=0,
    )

    keep_last_n: int = Field(
        default=1,
        description="Number of most recent checkpoints to keep. Set to -1 to keep all checkpoints.",
        ge=-1,
    )


class HubConfig(ConfigBaseModel):
    """Configuration for Hugging Face Hub integration"""

    push_to_hub: bool = Field(default=False, description="Whether to push the model weights to the Hugging Face Hub")
    hub_model_id: str | None = Field(
        default=None, description="Hugging Face Hub repository ID (e.g., 'username/repo-name')"
    )

    @model_validator(mode="after")
    def validate_hub_config(self) -> "HubConfig":
        """Validate that hub_model_id is not None when push_to_hub is True."""
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError("hub_model_id must be specified when push_to_hub is True")
        return self


class WandbConfig(ConfigBaseModel):
    """Configuration for Weights & Biases logging"""

    enabled: bool = Field(
        default=False,
        description="Whether to enable W&B logging",
    )

    project: str = Field(
        default="ltxv-trainer",
        description="W&B project name",
    )

    entity: str | None = Field(
        default=None,
        description="W&B username or team",
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Tags to add to the W&B run",
    )

    log_validation_videos: bool = Field(
        default=True,
        description="Whether to log validation videos to W&B",
    )


class FlowMatchingConfig(ConfigBaseModel):
    """Configuration for flow matching training"""

    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal", "motion_centric", "afm_phase", "afm_hybrid", "afm_hybrid_identity"] = Field(
        default="shifted_logit_normal",
        description="Mode to use for timestep sampling",
    )

    timestep_sampling_params: dict = Field(
        default_factory=dict,
        description="Parameters for timestep sampling",
    )
    
    # Add fields from YAML config
    method: str = Field(
        default="quantum_fm",
        description="Flow matching method",
    )
    
    t_star: float = Field(
        default=0.2,
        description="Quantum decoherence threshold",
    )
    
    use_anchor_net: bool = Field(
        default=True,
        description="Enable learnable quantum anchor network",
    )


class AFMTrainingConfig(ConfigBaseModel):
    """AFM Phase Separation Training Configuration"""
    
    use_phase_separation: bool = Field(
        default=False,
        description="Enable AFM Phase Separation training"
    )
    
    auto_transition_step: int = Field(
        default=10000,
        description="Automatic transition step from Stage 1 to Stage 2"
    )
    
    stage1_config: dict = Field(
        default_factory=lambda: {
            "description": "Identity Formation",
            "timestep_range": [0.2, 1.0],
            "motion_gain": 1.0,
            "loss_weights": {"local": 0.0, "global": 1.0}
        },
        description="Stage 1 (Global Identity) configuration"
    )
    
    stage2_config: dict = Field(
        default_factory=lambda: {
            "description": "Motion Refinement", 
            "timestep_range": [0.0, 0.2],
            "motion_gain": 2.0,
            "loss_weights": {"local": 10.0, "global": 0.0},
            "learning_rate_scale": 0.2,  # Stage 2에서 LR을 1/5로 감소
            "enable_snr_rescaling": True,  # SNR Rescaling 활성화
            "motion_gain_warmup_steps": 2000  # Motion Gain Warm-up 스텝 수
        },
        description="Stage 2 (Local Motion) configuration"
    )


class TrainingConfig(ConfigBaseModel):
    """Training configuration from YAML"""
    strategy: str = Field(default="quantum_fm", description="Training strategy")
    temporal_embedding_amplification: float = Field(default=5.0, description="Temporal signal boost")
    orthogonality_loss_weight: float = Field(default=0.1, description="Frame uniqueness enforcement")
    quantum_metrics_log_interval: int = Field(default=50, description="Log quantum metrics interval")


class TrainingConfigDetailed(ConfigBaseModel):
    """Detailed training configuration from YAML"""
    learning_rate: str = Field(default="1e-4", description="Learning rate")
    batch_size: int = Field(default=1, description="Batch size")
    max_steps: int = Field(default=100000, description="Maximum training steps")
    gradient_clip_norm: float = Field(default=1.0, description="Gradient clip norm")
    mixed_precision: str = Field(default="bf16", description="Mixed precision mode")
    gradient_accumulation_steps: int = Field(default=4, description="Gradient accumulation steps")
    validation_steps: int = Field(default=1500, description="Validation steps")
    num_validation_videos: int = Field(default=4, description="Number of validation videos")
    validation_guidance_scale: float = Field(default=3.0, description="Validation guidance scale")


class OptimizerConfig(ConfigBaseModel):
    """Optimizer configuration from YAML"""
    name: str = Field(default="adamw", description="Optimizer name")
    betas: list[float] = Field(default=[0.9, 0.95], description="Optimizer betas")
    weight_decay: str = Field(default="1e-2", description="Weight decay")
    eps: str = Field(default="1e-8", description="Optimizer epsilon")


class LrSchedulerConfig(ConfigBaseModel):
    """Learning rate scheduler configuration from YAML"""
    type: str = Field(default="cosine_annealing", description="Scheduler type")
    warmup_steps: int = Field(default=1000, description="Warmup steps")
    min_lr: str = Field(default="1e-6", description="Minimum learning rate")


class OutputConfig(ConfigBaseModel):
    """Output configuration from YAML"""
    save_dir: str = Field(default="outputs/quantum_collapse_v1", description="Save directory")
    checkpoint_steps: int = Field(default=5000, description="Checkpoint steps")
    save_full_model: bool = Field(default=False, description="Save full model")
    save_quantum_metrics: bool = Field(default=True, description="Save quantum metrics")
    export_quantum_analysis: bool = Field(default=True, description="Export quantum analysis")


class LoggingConfig(ConfigBaseModel):
    """Logging configuration from YAML"""
    use_wandb: bool = Field(default=True, description="Use Weights & Biases")
    project_name: str = Field(default="quantum-video-generation", description="Project name")
    run_name: str = Field(default="quantum_collapse_experiment", description="Run name")
    log_quantum_metrics: bool = Field(default=True, description="Log quantum metrics")
    log_frame_divergence: bool = Field(default=True, description="Log frame divergence")
    log_state_collapse_energy: bool = Field(default=True, description="Log state collapse energy")
    log_orthogonality_scores: bool = Field(default=True, description="Log orthogonality scores")
    loss_log_steps: int = Field(default=10, description="Loss log steps")
    metric_log_steps: int = Field(default=50, description="Metric log steps")
    sample_log_steps: int = Field(default=1500, description="Sample log steps")


class DebugConfig(ConfigBaseModel):
    """Debug configuration from YAML"""
    enabled: bool = Field(default=False, description="Enable debugging")
    save_intermediate_states: bool = Field(default=False, description="Save intermediate states")
    visualize_state_collapse: bool = Field(default=False, description="Visualize state collapse")
    track_frame_similarity: bool = Field(default=True, description="Track frame similarity")


class LtxvTrainerConfig(ConfigBaseModel):
    """Unified configuration for LTXV training"""

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig | None = Field(default=None)
    conditioning: ConditioningConfig = Field(default_factory=ConditioningConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    afm_training: AFMTrainingConfig = Field(default_factory=AFMTrainingConfig)
    
    # Add missing configurations from YAML
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    quantum: QuantumConfig = Field(default_factory=QuantumConfig)
    training_config: TrainingConfigDetailed = Field(default_factory=TrainingConfigDetailed)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    lr_scheduler: LrSchedulerConfig = Field(default_factory=LrSchedulerConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    # General configuration
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    output_dir: str = Field(
        default="outputs",
        description="Directory to save model outputs",
    )

    # noinspection PyNestedDecorators
    @field_validator("output_dir")
    @classmethod
    def expand_output_path(cls, v: str) -> str:
        """Expand user home directory in output path."""
        return str(Path(v).expanduser().resolve())

    @model_validator(mode="after")
    def validate_conditioning_compatibility(self) -> "LtxvTrainerConfig":
        """Validate that conditioning and validation configurations are compatible."""

        # Check that reference videos are provided when using reference_video conditioning
        if self.conditioning.mode == "reference_video" and self.validation.reference_videos is None:
            raise ValueError(
                "reference_videos must be provided in validation config when conditioning.mode is 'reference_video'"
            )

        # Check that LoRA config is provided when training mode is lora
        if self.model.training_mode == "lora" and self.lora is None:
            raise ValueError("LoRA configuration must be provided when training_mode is 'lora'")

        # Check that LoRA config is provided when using reference_video conditioning with LoRA training mode
        if self.conditioning.mode == "reference_video" and self.model.training_mode != "lora":
            raise ValueError("Training mode must be 'lora' when using reference_video conditioning")

        return self
