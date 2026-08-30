# EGE-UNet
This is the official code repository for "EGE-UNet: an Efficient Group Enhanced UNet for skin lesion segmentation", which is accpeted by *26th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI2023)* as a regular paper!

**0. Main Environments**
- python >= 3.8
- pytorch >= 1.13, torchvision >= 0.13 (see `requirements.txt`; all dependencies are preinstalled on Google Colab)

```
pip install -r requirements.txt
```

**1. Prepare the dataset.**

- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm) or [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)}.

- The zip extracts to `isic2017/` and `isic2018/`. Place them so the layout is:

- './data/data_isic1718/'
  - isic2017
    - train
      - images
        - .jpg
      - masks
        - .png
    - val
      - images
      - masks
  - isic2018
    - train
      - images
      - masks
    - val
      - images
      - masks

**2. Train the EGE-UNet.**
```
cd EGE-UNet
```
```
python train.py
```

With no flags this reproduces the original behavior (isic17, 300 epochs, a fresh timestamped folder under `./results/`). All settings in `configs/config_setting.py` can be overridden from the command line:

```
python train.py --dataset isic18 --work-dir results/my_experiment --epochs 300 --batch-size 8
```

- `--work-dir DIR` — the experiment folder holding `log/`, `checkpoints/`, `outputs/`, `summary/` (TensorBoard), `metrics.csv` and `test_results.json`. **Re-running with the same `--work-dir` automatically resumes from its `checkpoints/latest.pth`**; pass `--no-resume` to intentionally start over in the same folder.
- `--dataset {isic17,isic18}`, `--data-path DIR`, `--epochs N`, `--batch-size N`, `--num-workers N`, `--val-interval N`, `--seed N`, `--device {cuda,cpu}` (cpu is only for local smoke tests).
- `--hpa-mode {learnable,frozen_ones,none}` — GHPA static-prior ablation (see section 5). Default `learnable` = the original model.

**3. Train on Google Colab (with results persisted to Google Drive).**

Open [colab_train.ipynb](colab_train.ipynb) in Colab, fill in the parameters cell (your repo URL, the Drive path of the dataset zip, a run name), and Run all. All outputs are written to a folder on your Drive, so nothing is lost when the session dies — re-running the notebook with the same run name resumes training from the last saved epoch.

**4. Obtain the outputs.**
- After training, you could obtain the results in the work dir (default: './results/'): per-epoch `metrics.csv`, final `test_results.json`, checkpoints, logs, TensorBoard events and visualization images.

**5. GHPA static-prior analysis (interpretability + ablation).**

In GHPA the feature groups are multiplied by gates `conv_xy(BI(P_xy))`, `conv_zx(BI(P_zx))`, `conv_zy(BI(P_zy))` whose learnable tensors `P` have batch dimension 1 — the gates do **not** depend on the input image, so after training they are static per-channel spatial priors shared by all images (not data-dependent attention). The tools below measure what these priors learn (e.g. the centered-lesion bias of ISIC) and how much they contribute.

*GHPA placement* (`--ghpa-placement {low,mid,high}` or free-form `--ghpa-stages "enc3,enc4,..."`): moves the six GHPA modules to a different resolution band while keeping their count — a controlled experiment for whether the gate's operating resolution decides which lesion scale it helps. `low` = original (enc4/5/6 + dec1/2/3, res 32/16/8 | 8/8/16); `mid` = enc3/4/5 + dec2/3/4 (64/32/16 | 8/16/32); `high` = enc2/3/4 + dec3/4/5 (128/64/32 | 16/32/64). Caveat: parameter counts are NOT matched (plain 3×3 convs at wide low-res stages are expensive: low 53,374 / mid 91,966 / high 111,390 params) — read the per-tertile *pattern shift*, not absolute gains.

*Evaluation extras* in `analysis/eval_per_image.py`: `--thresholds 0.3,0.4,...` sweeps thresholds in one forward pass (per-threshold pooled/mean/per-tertile DSC in the sidecar json — the calibration diagnostic), and the `oracle_dsc` csv column scores a size-matched prediction (exactly k = |GT| highest-probability pixels), removing any area/calibration bias by construction.

*Model variants* (`--hpa-mode`, applied to all six GHPA modules enc4/5/6 + dec1/2/3):
- `learnable` — original model (53,374 params).
- `frozen_ones` — `P` is kept at its init (ones) and not trained; `conv_xy/zx/zy` are still trained. Caveat: the conv of a constant map is a per-channel constant in the interior plus a 1-px zero-padding border band, so this variant keeps a per-channel scaling but has no learned spatial prior (48,414 trainable params).
- `none` — no Hadamard gating at all; the `P`/`conv_*` modules are not created (44,988 params). Checkpoints are not interchangeable across modes (use one `--work-dir` per variant).

*Analysis scripts* (CPU is fine — the model is tiny; run from the repo root, a work_dir or a `.pth` can be given as `--checkpoint`):
```
python analysis/visualize_hpa.py  --checkpoint results/<run> --data-path ./data/data_isic1718/isic2017 --dataset isic17
python analysis/eval_per_image.py --checkpoint results/<run> --data-path ./data/data_isic1718/isic2017 --dataset isic17
python analysis/eval_per_image.py --checkpoint results/<run> --data-path ... --shift 32 0              # test-time translation
python analysis/eval_per_image.py --checkpoint results/<run> --data-path ... --gate-override spatial_mean  # post-hoc knock-out of the gate's spatial structure
python analysis/compare_runs.py --run learnable=results/<run_a> --run frozen_ones=results/<run_b> --run none=results/<run_c>
python analysis/compare_runs.py --group learnable=dirA,dirB,dirC --group frozen_ones=dirD,dirE,dirF   # several seeds per variant
```
- `visualize_hpa.py` → `<work_dir>/analysis/hpa/`: per-module figures of the gates (`fig_<module>_gxy_channels/gxy_mean/pxy_raw/gzx_gzy.png`), summaries vs the dataset lesion prior (`fig_summary_gxy_vs_prior[_interior].png`, `fig_summary_stats.png`), and statistics (`hpa_stats.json`, `hpa_stats_per_module.csv`, `hpa_stats_per_channel.csv`): spatial non-uniformity (`cv_interior`), correlation with the lesion prior / radial distance, center-vs-border ratio — for the trained model and an untrained reference (P = ones).
- `eval_per_image.py` → `<work_dir>/analysis/per_image_metrics[_shift..][_override-..].csv/.json`: per-image DSC/IoU with lesion area, centroid offset from the image center and touches-border flag, plus the pooled metrics computed like `engine.py` (reconciles with `test_results.json`).
- `compare_runs.py` → `<work_dir>/analysis/compare/`: mean metric per lesion-position / lesion-size tertile and group, paired differences vs the first group with bootstrap 95% CIs (and Wilcoxon p when scipy is available), and the **interaction** `diff_T1 − diff_T3` (central minus off-center) — a CI excluding 0 means the gain depends on the lesion position.

*Suggested experiment matrix* (ISIC17; name runs `egeunet_{dataset}_{hpa_mode}_s{seed}`): `learnable`, `frozen_ones`, `none` with seed 42 first (3 runs; time one run on your GPU to budget), then add seeds 43 and 44 for all three if the differences are smaller than ~3× the paper's reported std (0.10 mIoU). `--gate-override spatial_mean` on the learnable checkpoint is a free first look before retraining anything. `colab_analysis.ipynb` runs the whole pipeline on Drive-stored experiments.
