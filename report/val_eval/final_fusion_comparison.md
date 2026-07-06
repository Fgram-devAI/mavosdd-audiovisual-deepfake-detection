| Row | roc_auc | EER | F1 | Real specificity | Fake recall | LivePortrait recall | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| audio_only | 0.5478 | 0.4487 | 0.4936 | 0.5520 | 0.5506 | 0.6170 | n=622 excluded_missing=0 |
| video_av_only | 0.9317 | 0.1372 | 0.8337 | 0.8633 | 0.8623 | 0.5957 | n=620 excluded_missing=2 |
| sync_only | 0.2417 | 0.7276 | 0.2291 | 0.2735 | 0.2713 | 0.7872 | n=620 excluded_missing=2 |
| max_audio_video_av | 0.9300 | 0.1419 | 0.8281 | 0.8579 | 0.8583 | 0.5957 | n=620 excluded_missing=2 |
| max_available | 0.8058 | 0.2825 | 0.6692 | 0.7185 | 0.7166 | 0.4681 | n=620 excluded_missing=2 |
| logistic_fusion | 0.9307 | 0.1338 | 0.8376 | 0.8660 | 0.8664 | 0.5532 | n=620 excluded_missing=2; preferred if stable |
| mlp_fusion | 0.9175 | 0.1466 | 0.8226 | 0.8525 | 0.8543 | 0.6170 | n=620 excluded_missing=2; only if it genuinely improves |
| visual_frame_baseline_notebook_only | nan | nan | nan | nan | nan | nan | notebook-only, not part of trainable fusion |

> `visual_frame_baseline_notebook_only`: the EfficientNet-B0 sampled-frame baseline lives in `notebooks/03_visual_frame_baseline_extended_data.ipynb` and has no stable CLI. See `report/val_eval/final_fusion_visual_frame_unavailable.md`.
