# Data Placeholder

Place datasets, extracted CLIP features, CSV lists, ground-truth `.npy` files,
and sample videos here.

Suggested layout:

```text
data/
  finetune_data/
    videos/
      train/Accident/
      train/Normal/
      val/Accident/
      val/Normal/
    features/
      train/
      val/
    list_5crop/
      train_features.csv
      val_features.csv
      gt_ucf.npy
      metadata_train_filename_accident_time.xlsx
  sample_videos/
    demo.mp4
```

