# Two-Stage VADCLIP Training Pipeline

This code package keeps the proposed two-stage five-crop pipeline used in the
thesis.

## Shared Setting

- Classes: `Normal` and `Accident`.
- Input features: five CLIP crop features per video.
- Feature shape per crop: `[temporal_units, 512]`.
- Spatial aggregation: top-3 crop aggregation by default.
- Temporal top-k:

```text
k = min(length, clamp(length // top_k_divisor + 1,
                      top_k_min,
                      top_k_max))
```

The default temporal top-k parameters are divisor `16`, minimum `2`, and
maximum `5`.

## Stage 1: Five-Crop MIL

Entrypoint:

```text
src/ucf_train_stage1_mil.py
```

Stage 1 trains the model using video-level MIL supervision. The five crops are
grouped as synchronized spatial views of one video instead of independent
training samples. A1 uses crop-aggregated anomaly probabilities, and A2 uses
crop-aggregated two-class alignment logits.

Run:

```powershell
python src\ucf_train_stage1_mil.py `
  --train-list data\finetune_data\list_5crop\train_features.csv `
  --test-list data\finetune_data\list_5crop\val_features.csv `
  --gt-path data\finetune_data\list_5crop\gt_ucf.npy `
  --phase1-pretrained-path models\model_ucf.pth `
  --phase1-output-dir models\phase1_mil `
  --no-use-wandb
```

Output used by Stage 2:

```text
models/phase1_mil/model_best.pth
```

## Stage 2: Timestamp-Guided Intra-Video Gap Loss

Entrypoint:

```text
src/ucf_train_stage2_timestamp.py
```

Stage 2 starts from the best Stage 1 checkpoint and keeps the MIL objective
active. For timestamped accident videos, it additionally applies the
Intra-video Gap Loss. The loss compares high-scoring temporal units near the
accident timestamp with hard-normal temporal units before and after the
accident in the same video.

Run:

```powershell
python src\ucf_train_stage2_timestamp.py `
  --train-list data\finetune_data\list_5crop\train_features.csv `
  --test-list data\finetune_data\list_5crop\val_features.csv `
  --gt-path data\finetune_data\list_5crop\gt_ucf.npy `
  --timestamp-excel data\finetune_data\list_5crop\metadata_train_filename_accident_time.xlsx `
  --phase2-pretrained-path models\phase1_mil\model_best.pth `
  --phase2-output-dir models\phase2_timestamp `
  --gap-weight 0.2 `
  --gap-margin 0.8 `
  --gap-warmup-epochs 3 `
  --no-use-wandb
```

## Evaluation

Frame-level evaluation:

```powershell
python src\ucf_test_5crop.py `
  --test-list data\finetune_data\list_5crop\val_features.csv `
  --gt-path data\finetune_data\list_5crop\gt_ucf.npy `
  --model-path models\phase2_timestamp\model_best.pth
```

Video-level evaluation:

```powershell
python src\ucf_test_video_level_5crop.py `
  --test-list data\finetune_data\list_5crop\val_features.csv `
  --gt-path data\finetune_data\list_5crop\gt_ucf.npy `
  --model-path models\phase2_timestamp\model_best.pth `
  --score-thresh 0.5 `
  --save-dir output\video_level_5crop
```

## Required Files

The two-stage pipeline depends on:

- `model.py`
- `crop.py`
- `clip/`
- `utils/dataset.py`
- `utils/layers.py`
- `utils/lr_warmup.py`
- `utils/tools.py`
- `utils/ucf_detectionMAP.py`
- `extract_clip_features.py`
- `ucf_option_staged_training.py`
- `ucf_staged_training_common.py`
- `ucf_train_stage1_mil.py`
- `ucf_train_stage2_timestamp.py`
- `ucf_test_5crop.py`
- `ucf_test_video_level_5crop.py`
