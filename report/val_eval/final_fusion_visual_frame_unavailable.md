# `visual_fake_score` unavailable in final fusion

The visual-frame (EfficientNet-B0 over sampled video frames) baseline exists in
this repository only as a Jupyter notebook (`notebooks/03_visual_frame_baseline_extended_data.ipynb`),
without a stable CLI or checkpoint API. The final-fusion score table therefore
leaves `visual_fake_score` blank and the trainable fusion model does not include
that column as a feature.

Val summary of that notebook run (for comparison-report context only, not a
final-fusion baseline):

- ROC-AUC 0.9853, EER 0.0671, F1 0.9167.
- Per-source: `real` specificity 0.9307, `echomimic` / `memo` / `sonic` recall
  1.00, `liveportrait` recall 0.6596.
- Treat as channel-confounded because sampled source folders have disjoint
  resolution / FPS / codec signatures.

Source: `report/visual_frame_baseline/visual_frame_baseline_efficientnet_b0_val.json`.

To include `visual_fake_score` in the fusion input, first extract the notebook
into a stable script (e.g. `src/features/extract_visual_frame_score.py` +
`scripts/score_generated_video_batch.py --visual-frame-ckpt ...`) and re-run
`build_final_fusion_scores`.
