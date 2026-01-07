# AFM Phase Separation Training Guide
**Anchored Flow Matching with Hierarchical Identity/Motion Learning**

## 🎯 Philosophy: Phase Separation

AFM Phase Separation은 비디오 생성의 복잡성을 줄이기 위해 **Identity(형체)**와 **Motion(움직임)**을 계층적으로 분리하여 학습하는 혁신적인 접근법입니다.

### Two-Stage Learning Process

**Stage 1: Global Identity Formation** (`t ∈ [0.2, 1.0]`)
- **목표**: 모든 프레임이 공통된 앵커(Identity)로 수렴하는 법을 학습
- **특징**: 프레임 간 차이를 무시하고 오직 형체 형성에만 집중
- **기간**: 0 ~ 10,000 steps

**Stage 2: Local Motion Refinement** (`t ∈ [0.0, 0.2]`)
- **목표**: 형성된 앵커에서 프레임별 움직임을 분화시키는 법을 학습
- **특징**: 2.0x Motion Gain으로 미세한 변위 신호 증폭
- **기간**: 10,000+ steps

## 🚀 Quick Start

### 1. 처음부터 시작 (Stage 1 → Stage 2)

```bash
# AFM Phase Separation 전체 학습 (20k steps)
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python scripts/train_distributed.py configs/afm_phase_config.yaml --num_processes 1
```

### 2. 10k 체크포인트에서 Stage 2만 시작

```yaml
# afm_phase_config.yaml 수정
model:
  load_checkpoint: "outputs/ring_fm_lora_v1/checkpoint-10000"  # 10k 체크포인트 경로

afm_training:
  auto_transition_step: 0  # 즉시 Stage 2 시작
```

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python scripts/train_distributed.py configs/afm_phase_config.yaml --num_processes 1
```

### 3. 기존 Unified 모드와 비교

```yaml
# 기존 방식 (ring_lora_config.yaml)
afm_training:
  use_phase_separation: false  # Phase Separation 비활성화

# Phase Separation 방식 (afm_phase_config.yaml) 
afm_training:
  use_phase_separation: true   # Phase Separation 활성화
```

## 📊 WandB 모니터링 지표

### Core AFM Metrics
- `afm/current_stage`: 현재 학습 Stage (`global_identity` → `local_motion`)
- `afm/motion_gain`: 적용된 Motion Gain (Stage 1: 1.0x, Stage 2: 2.0x)
- `afm/stage_transition`: Stage 전환 이벤트 (1이면 전환 발생)

### Stage-Specific Loss Metrics
- `afm_loss/local_raw`: Local Phase 원시 Loss
- `afm_loss/global_raw`: Global Phase 원시 Loss  
- `afm_loss/active_loss`: 현재 Stage에서 활성화된 Loss
- `afm_loss/stage_focus`: 현재 Stage 설명 (`Identity Formation` / `Motion Refinement`)

### Progress Tracking
- `afm/progress_to_transition`: Stage 1 → Stage 2 전환까지의 진행률 (0.0 ~ 1.0)
- `afm/t_range_min`, `afm/t_range_max`: 현재 Stage의 타임스텝 샘플링 범위
- `afm/samples_in_range`: 샘플링된 타임스텝이 올바른 범위에 있는 비율

## 🔧 Advanced Configuration

### Custom Stage Transition

```yaml
afm_training:
  auto_transition_step: 15000  # 15k에서 Stage 전환
  
  # Stage 1 커스터마이징
  stage1_config:
    motion_gain: 1.2           # 약간의 Motion Gain
    loss_weights:
      local: 0.1               # 소량의 Local Loss 유지
      global: 1.0
  
  # Stage 2 커스터마이징  
  stage2_config:
    motion_gain: 3.0           # 더 강한 Motion Gain
    loss_weights:
      local: 15.0              # 더 강한 Local Loss
      global: 0.0
```

### Phase-Aware Validation

```yaml
validation:
  interval: 1000               # Stage별 더 자주 검증
  prompts:
    - "A person walking in the park"      # Motion이 중요한 프롬프트
    - "A rotating mechanical gear"        # 명확한 움직임 패턴
    - "Facial expression changes"         # 미세한 변화 감지
```

## 🧪 Expected Results

### Stage 1 (Identity Formation)
**성공 지표**:
- `afm_loss/global_raw` 지속적 감소
- `afm_loss/local_raw` ≈ 0 (Local Loss 차단됨)
- 생성된 비디오: 모든 프레임이 유사한 형체 (정지 상태)

### Stage 2 (Motion Refinement)  
**성공 지표**:
- `afm_loss/local_raw` 지속적 감소
- `afm_loss/global_raw` ≈ 0 (Global Loss 차단됨)
- `afm/motion_gain`: 2.0 (Motion 증폭 활성화)
- 생성된 비디오: Identity 유지하며 자연스러운 움직임 생성

### Temporal Collapse 해결
**Before**: `local_vs_global_ratio` ≈ 0 (움직임 없음)
**After**: `local_vs_global_ratio` ≥ 0.5 (활발한 움직임)

## 🎬 Inference with Trained Model

AFM Phase로 학습된 모델은 기존 추론 파이프라인과 완벽 호환됩니다:

```python
from ltxv_trainer.ltxv_pipeline import LtxvPipeline

# Stage 2 완료된 체크포인트 로드
pipeline = LtxvPipeline.from_pretrained("outputs/afm_phase_separation_v1/checkpoint-20000")

# 일반적인 비디오 생성
video = pipeline(
    prompt="A person walking through a bustling city street",
    num_frames=25,
    height=512, width=512,
    num_inference_steps=50
)
```

## 📈 Performance Tips

1. **GPU Memory**: Stage 2는 더 많은 그라디언트 계산으로 인해 약간 더 많은 VRAM 사용
2. **Learning Rate**: Stage 2에서는 Learning Rate를 절반으로 줄이는 것을 권장
3. **Batch Size**: Phase Separation은 작은 배치에서도 안정적으로 작동
4. **Checkpointing**: 각 Stage 완료 후 체크포인트 저장을 권장

## ⚠️ Troubleshooting

### Stage 전환이 안 될 때
```yaml
# 수동 강제 전환 (현재는 config 수정으로 대체)
afm_training:
  auto_transition_step: 1  # 즉시 Stage 2로 강제 전환
```

### Motion이 약할 때
```yaml
# Motion Gain 증가
stage2_config:
  motion_gain: 4.0  # 기본값 2.0에서 4.0으로 증가
```

### Identity가 불안정할 때
```yaml
# Stage 1 기간 연장
afm_training:
  auto_transition_step: 15000  # 10k → 15k로 연장
```

---

**🎯 AFM Phase Separation으로 비디오 생성의 새로운 패러다임을 경험해보세요!**