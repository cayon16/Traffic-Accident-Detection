# Improved VADCLIP for Traffic Accident Detection

This is the cleaned submission code for the thesis. It contains the two-stage
five-crop VADCLIP pipeline and the ONNX monitoring prototype.

Datasets, extracted features, checkpoints, ONNX files, databases, generated
clips, and experiment outputs are not included. Put local data in `data/`,
model files in `models/`, and generated results in `output/`.

The report of this project could be found at:

```text
https://drive.google.com/file/d/1w08GXoqVcFyGAfyAbAN30CujoG10qzSn/view?usp=sharing
```

The training dataset and the three testing datasets used in the thesis can be
downloaded from:

```text
https://drive.google.com/drive/folders/17B1WtEc2M3FFnM6hVNfaHcH3RnHKf7mh?usp=sharing
```

Because the original videos are large, the uploaded dataset files mainly
contain pre-extracted `.npy` CLIP feature files and the required metadata/lists.
If the original videos are needed, please contact:

```text
vuquocdung407061@gmail.com
```

## Structure

```text
final_code/
  src/
    extract_clip_features.py          five-crop CLIP feature extraction
    ucf_train_stage1_mil.py           Stage 1 MIL training
    ucf_train_stage2_timestamp.py     Stage 2 timestamp gap-loss training
    ucf_test_5crop.py                 frame-level evaluation
    ucf_test_video_level_5crop.py     video-level evaluation
    infe_video/                       single-video and ONNX inference
    model.py, crop.py, clip/, utils/  model and shared helpers
  system/                             FastAPI monitoring prototype
  data/                               local data placeholder
  models/                             local model placeholder
  output/                             generated results
```

## Install

```powershell
cd final_code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a CUDA PyTorch build separately if GPU acceleration is needed.
FFmpeg is recommended for the monitoring prototype because it extracts
evidence clips.

## CLIP Weights

Feature extraction uses the CLIP ViT-B/16 image encoder. On the first run, the
code will try to download the CLIP weight file automatically to the default
cache folder:

```text
~/.cache/clip/ViT-B-16.pt
```

If the machine has no internet access, download the file manually:

```text
https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

Then either place it in the default cache folder above, or pass a custom cache
folder when extracting features:

```powershell
python src\extract_clip_features.py `
  --video-root data\videos\train `
  --output-root data\finetune_data\features\train `
  --csv-output data\finetune_data\list_5crop\train_features.csv `
  --preset fivecrop `
  --download-root models\clip_cache
```

## Data Format

After downloading the datasets, place or link them under `data/`. For training
and testing, prepare five-crop CLIP features and CSV files:

```text
data/finetune_data/
  list_5crop/
    train_features.csv
    val_features.csv
    gt_ucf.npy
    metadata_train_filename_accident_time.xlsx
  features/
    train/
    val/
```

Each CSV row contains a feature path and label. Each video must have five
feature files named like `video_name__0.npy` to `video_name__4.npy`.

Only provide the model files needed for the command you run. For example:

- Stage 1 needs an initial/pretrained checkpoint, passed with
  `--phase1-pretrained-path`.
- Stage 2 needs a Stage 1 checkpoint, passed with `--phase2-pretrained-path`.
- Evaluation needs a trained checkpoint, passed with `--model-path`.
- ONNX inference needs a CLIP ONNX file and a VADCLIP ONNX file.

## Feature Extraction

```powershell
python src\extract_clip_features.py `
  --video-root data\videos\train `
  --output-root data\finetune_data\features\train `
  --csv-output data\finetune_data\list_5crop\train_features.csv `
  --preset fivecrop `
  --sample-stride 16
```

Run the same command for the validation/test videos with different output
paths.

## Training

Stage 1:

```powershell
python src\ucf_train_stage1_mil.py `
  --train-list data\finetune_data\list_5crop\train_features.csv `
  --test-list data\finetune_data\list_5crop\val_features.csv `
  --gt-path data\finetune_data\list_5crop\gt_ucf.npy `
  --phase1-pretrained-path models\initial_checkpoint.pth `
  --phase1-output-dir models\phase1_mil `
  --no-use-wandb
```

Stage 2:

```powershell
python src\ucf_train_stage2_timestamp.py `
  --train-list data\finetune_data\list_5crop\train_features.csv `
  --test-list data\finetune_data\list_5crop\val_features.csv `
  --gt-path data\finetune_data\list_5crop\gt_ucf.npy `
  --timestamp-excel data\finetune_data\list_5crop\metadata_train_filename_accident_time.xlsx `
  --phase2-pretrained-path models\phase1_mil\model_best.pth `
  --phase2-output-dir models\phase2_timestamp `
  --no-use-wandb
```

## Evaluation

Frame-level:

```powershell
python src\ucf_test_5crop.py `
  --test-list data\finetune_data\list_5crop\val_features.csv `
  --gt-path data\finetune_data\list_5crop\gt_ucf.npy `
  --model-path models\phase2_timestamp\model_best.pth
```

Video-level:

```powershell
python src\ucf_test_video_level_5crop.py `
  --test-list data\finetune_data\list_5crop\val_features.csv `
  --gt-path data\finetune_data\list_5crop\gt_ucf.npy `
  --model-path models\phase2_timestamp\model_best.pth `
  --score-thresh 0.5
```

## Single-Video Inference

PyTorch inference uses the editable constants near the top of:

```text
src/infe_video/infer_single_video_2class.py
```

Default paths are:

```text
data/sample_videos/demo.mp4
models/model_current.pth
```

ONNX inference:

```powershell
python src\infe_video\infer_onnx_2class.py `
  --input data\sample_videos\demo.mp4 `
  --clip-onnx-model models\clip_vit_b16_image.onnx `
  --onnx-model models\vadclip_2class.onnx `
  --provider auto `
  --score-source a2
```

Export ONNX files:

```powershell
python src\infe_video\convert_vit_b16_to_onnx.py `
  --output-path models\clip_vit_b16_image.onnx

python src\infe_video\convert_VADCLIP_to_onnx.py `
  --model-path models\phase2_timestamp\model_best.pth `
  --output-path models\vadclip_2class.onnx
```

## Monitoring Prototype

Configure:

```text
system/backend/configs/settings.yaml
system/backend/configs/cameras.yaml
```

Run:

```powershell
cd system\backend
python run.py
```

Open `http://localhost:8000`, or upload a segment:

```powershell
curl.exe -X POST "http://localhost:8000/api/segments" `
  -F "camera_id=CAM_TEST_001" `
  -F "file=@data\sample_videos\demo.mp4"
```

Generated evidence is written to `system/backend/storage/evidence/`.

## Notes

- Use `--no-use-wandb` if W&B logging is not needed.
- `gt_ucf.npy` must follow the grouped validation video order.
- Do not submit generated checkpoints, ONNX files, databases, evidence videos,
  or temporary outputs unless they are explicitly requested.

## Acknowledgement

This project is built on top of the original VADCLIP implementation:

```text
https://github.com/nwpu-zxr/VadCLIP
```

Thanks to the VADCLIP authors for releasing their code and pretrained
resources. This submitted code keeps the VADCLIP backbone and adds the
traffic-accident-specific two-stage training, five-crop aggregation, timestamp
gap loss, and monitoring prototype used in this thesis.
