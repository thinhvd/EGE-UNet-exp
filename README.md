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

**3. Train on a rented GPU server.**

```
# on the server
git clone https://github.com/thinhvd/EGE-UNet-exp.git && cd EGE-UNet-exp
git checkout <branch>                 # main, or the experiment branch
bash scripts/setup_server.sh          # venv + deps + CUDA check + dataset check
source .venv/bin/activate
python train.py --work-dir results/egeunet_isic17_learnable_s42 [flags]

# on the local machine, to pull the finished runs back
SERVER=user@host bash scripts/sync_results.sh
```

`--work-dir` is the only thing needed for crash safety: re-running the same command after a disconnect resumes
from `checkpoints/latest.pth`. Every run logs its `code revision: <branch>@<commit>` so results can always be
traced back to the code that produced them.

Google Colab is no longer used (since 09/2026); the old notebooks are archived in [notebooks/](notebooks/).

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

*Suggested experiment matrix* (ISIC17; name runs `egeunet_{dataset}_{hpa_mode}_s{seed}`): `learnable`, `frozen_ones`, `none` with seed 42 first (3 runs; time one run on your GPU to budget), then add seeds 43 and 44 for all three if the differences are smaller than ~3× the paper's reported std (0.10 mIoU). `--gate-override spatial_mean` on the learnable checkpoint is a free first look before retraining anything.

**5b. EXP-4: cross-stage fusion for the decoder (this branch).**

`--fusion {none,sum,concat,csaa}` gives every chosen decoder stage a fused view of ALL five encoder
stages instead of only its own skip connection, following EFCNet's CSAA. `--fusion-stages
{deep3,all5}` picks which decoder stages receive it (deep3 = dec1/2/3, the stages where GHPA
operates) and `--fusion-dim` (default 16) sets the common channel width. `sum` and `concat` are the
controls that carry the multi-scale information without any attention; `csaa` adds a two-step axial
attention as a pure additive term on top of `concat`, so the two differ by exactly the attention and
1,632 parameters. Implementation in [models/fusion.py](models/fusion.py).

Every head is zero-initialized, so at init all six variants are bitwise identical to the baseline —
verified, together with the oracle constants, before any run. Parameter counts: sum-deep3 57,445 /
sum-all5 57,863 / concat-deep3 64,086 / concat-all5 66,030 / csaa-deep3 65,718 / csaa-all5 67,662.
The run matrix is in [scripts/exp04_train.sh](scripts/exp04_train.sh).

`analysis/benchmark_speed.py` reports parameters, MACs and measured latency for any variant (from a
checkpoint or straight from flags). Note the units: the paper's "0.072 GFLOPs" is the GMACs column
here (the thop convention of labelling MACs as FLOPs); the baseline measures 0.0721 GMACs.

**6. Repository layout and experiment workflow.**

```
train.py engine.py utils.py           core training code (author's, kept behavior-identical)
models/ configs/ datasets/            model, config, data loading
models/<feature>.py                   NEW experiment modules live in their own file
analysis/                             evaluation + interpretability tools (post-hoc, never touch training)
scripts/                              GPU-server runbook (setup, result sync, per-experiment train scripts)
notebooks/                            archived Colab notebooks (no longer maintained)
results/                              run outputs (gitignored)
data/                                 dataset (gitignored)
```

*Running the author's original code.* With no flags, `python train.py` reproduces the original training behavior
bit-for-bit. This is not a claim of good faith but a checked invariant: at seed 0 the model's `state_dict` hashes
to `87b3dec8b84590308c5f527402db3e9ebabf0159634d35da5a62c7753ac6737f` (314 keys, 53,374 params), a fixed input
sums to `77392.56237548799` through the forward pass, and one AdamW step gives loss `10.317150115966797`. Any
change that shifts those numbers is a bug in the change, not a new baseline. Known bugs of the original
(fixed-angle random rotation, BCE on probabilities, the no-op normalization) are deliberately preserved.

The author's untouched code is kept on the `author-original` branch (frozen at upstream
[JCruan519/EGE-UNet](https://github.com/JCruan519/EGE-UNet) `f52ba30`, tag `author-original-f52ba30`) for diffing
and auditing — `git diff author-original main -- models/egeunet.py`. It is a reference, not a runnable target:
as published it is missing the package `__init__.py` files and pins dependencies that no longer install cleanly.

*Branches.* `main` holds the faithful baseline plus shared infrastructure (CLI, analysis tools, scripts). Each
experiment gets its own branch, `exp/NN-short-name` (e.g. `exp/04-cross-stage-fusion`), so an old experiment can
be re-run exactly by checking out its branch. Infrastructure that proves generally useful is merged back to
`main` after the experiment concludes; the experimental model code stays on its branch.

*Adding an experiment.* Put new model code in its own module (e.g. `models/fusion.py`), wire it into
`models/egeunet.py` behind a constructor flag that defaults to off, and expose it as a CLI flag in `train.py` +
`configs/config_setting.py` (the `--hpa-mode` / `--ghpa-placement` flags are the pattern to copy). The default
path must stay on the oracle numbers above. Every experiment also gets a Vietnamese write-up in `exp_docs/`
(local only) with its predictions registered before the runs start.
