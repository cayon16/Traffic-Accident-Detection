# Five-Crop CLIP Feature Extraction

This guide describes the feature format expected by the two-stage training and
evaluation code.

## Feature Format

- CLIP backbone: ViT-B/16.
- Feature dimension: 512.
- Temporal stride: one feature per 16-frame unit by default.
- Spatial views: five crops per video, crop indices `0,1,2,3,4`.
- Output file naming: `video_name__{crop_id}.npy`.
- Each feature array has shape `[temporal_units, 512]`.

All five crops of the same video must have the same temporal length. The train
and validation CSV files should contain five rows per video.

## Command

The first run downloads the CLIP ViT-B/16 weight file automatically. For
offline use, download `ViT-B-16.pt` from:

```text
https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

Place it in `~/.cache/clip/` or pass `--download-root <folder>` to the command.

```powershell
python src\extract_clip_features.py `
  --video-root data\finetune_data\videos\train `
  --output-root data\finetune_data\features\train `
  --csv-output data\finetune_data\list_5crop\train_features.csv `
  --preset fivecrop `
  --sample-stride 16
```

Use the same command for the validation split after changing `--video-root`,
`--output-root`, and `--csv-output`.

## Notes

- `--label-mode parent` expects the parent folder to be `Accident` or `Normal`.
- `--skip-existing` can be used to resume interrupted extraction.
- `--max-videos N` is useful for a small smoke test.
- The generated `gt_ucf.npy` must follow the grouped validation-video order,
  not the five individual crop rows.
