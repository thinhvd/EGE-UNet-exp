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
# from the local machine: push the code (git stays local; the server needs no clone)
SERVER=<user>@<host> PORT=<port> bash scripts/push_code.sh

# on the server
bash scripts/setup_server.sh          # deps + CUDA check + dataset check
python train.py --work-dir results/egeunet_isic17_learnable_s42 [flags]

# on the local machine, to pull the finished runs back
SERVER=<user>@<host> PORT=<port> LEAN=1 bash scripts/sync_results.sh   # best checkpoint only (project convention)
```

The dataset is pushed once, and only the subset in use: `isic2017` is 29 MB (`isic2018` is 251 MB).
`push_code.sh` also writes a `.code_revision` stamp so a server without `.git` still records in each
run's log which commit produced it.

`--work-dir` is the only thing needed for crash safety: re-running the same command after a disconnect resumes
from `checkpoints/latest.pth`. Every run logs its `code revision: <branch>@<commit>` so results can always be
traced back to the code that produced them. Launch long runs detached (`setsid nohup … &`, as
`PARALLEL=1 scripts/exp04_train.sh` does) — an SSH drop must not take the training with it.

Every training run is a random draw, even on the same machine with the same seed (see *Training is not
reproducible run to run* in section 6), so a gap between two single runs below about 1 mIoU is not evidence of
anything.

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

**5c. EXP-6: loss terms derived from Active Contour Loss (branch `exp/06-contour-loss`).**

The model stays the original EGE-UNet; only the training loss changes. `--extra-term` adds one term,
acting on the final output only, to the unchanged `GT_BceDiceLoss`; `--loss bce_region` instead swaps the
Dice part of all six BceDice terms for the normalized active-contour region term. Implementation and
formulas in [contour_losses.py](contour_losses.py); checks in `python tests/test_contour_losses.py`.

| flag | what the term does |
|---|---|
| `--extra-term tv` | ACL length term (Chen et al. CVPR 2019): rewards a short contour |
| `--extra-term tv_match` | contour length matched to the ground truth, \|TV(u)/TV(g) − 1\| |
| `--extra-term area` | scale-invariant area term, SmoothL1 of log(Σu / Σg) |
| `--extra-term bl` | boundary loss (Kervadec et al. MIDL 2019), distances in pixels |
| `--extra-term snbl` | boundary loss with distances in radii of the lesion itself (`--snbl-tau`, default 3) |
| `--loss bce_region` | Dice → Σ[u(1−g)² + (1−u)g²] / (Σg + 1) in every term |

Every variant needs `--extra-weight`; the weights used are fixed before training by
`analysis/calibrate_loss_weights.py` (gradient-norm matching on the train images of the EXP-5 baseline).
Whatever the training loss, validation, and therefore checkpoint selection, and the final `test_loss` use
the original `GT_BceDiceLoss`, so all variants pick their checkpoint by the same rule. With no loss flags
the run is bit-identical to the previous code (checked on a 2-epoch CPU run: same `metrics.csv`, same
`test_results.json`, same final weights). Run matrix: [scripts/exp06_train.sh](scripts/exp06_train.sh);
evaluation: [scripts/exp06_analyze.sh](scripts/exp06_analyze.sh), which ends in
`analysis/exp06_summary.py` (decision on pooled DSC).

**5d. EXP-7: refining the boundary loss (branch `exp/07-e3a-refine`).**

EXP-6's best variant, the boundary loss (`bl`), removed far spill around small lesions but dropped faint
tissue of large lesions. EXP-7 adds the prediction-side half of Karimi & Salcudean's two-sided distance loss:
`--extra-term fn_dp` penalizes each missed lesion pixel by its distance to the current prediction (capped at
the lesion's inradius), so dropping a whole chunk is expensive and a thin missed rim is cheap. Terms can be
combined with their own weights: `--extra-term bl,fn_dp --extra-weight 0.095,0.51` (each term's per-epoch mean
is logged as `train_extra_<name>`). The `fn_dp` weight comes from a pre-registered total-push balance on the
EXP-6 checkpoint (`analysis/calibrate_loss_weights.py --mass-balance`). `--save-every N` (default off)
also keeps every N-th epoch from `--save-from` (200) on. Runs: [scripts/exp07_train.sh](scripts/exp07_train.sh),
per-box runner [scripts/exp07_box.sh](scripts/exp07_box.sh), evaluation
[scripts/exp07_analyze.sh](scripts/exp07_analyze.sh), mechanism probe `analysis/e3a_mechanism.py`, and
[scripts/finish_box.sh](scripts/finish_box.sh) to pull, verify and destroy a finished box. With no loss flags,
and with `--extra-term bl` alone, runs are bit-identical to the EXP-6 code (checked on a 2-epoch CPU run).

**5e. EXP-9: boundary-guided cross-stage fusion (branch `exp/09-bg-csf`).**

Borrowed from LB-UNet's prediction-map auxiliary module: a boundary head on each of dec3/dec4/dec5 predicts
the lesion contour from the decoder feature (after the GAB skip is added), and that map steers the fusion.
`--fusion bg_stage --fusion-stages shallow3` combines the five projected encoder stages as
sum_j w_j (1 + a_j B) P_j, one learnable `a_j` per source, so a source can weigh differently on the contour
than inside the lesion (one shared `a` would only gate the whole fused output, which is why there is no such
mode). `--boundary-weight W` (required with `bg_stage`) adds W x the contour-band loss on the boundary heads:
the band is dilate3x3 - erode3x3 of the mask, max-pooled to each head's grid, scored with BceDice(0.5, 1) and
weighted 0.1 / 0.2 / 0.3 for dec3 / dec4 / dec5; `--boundary-weight 0` only logs it (a free gate). It combines
with `--extra-term` (e.g. E3a, `--extra-term bl --extra-weight 0.095`); columns `train_extra_bnd_dec3..5` hold
the band terms. Every configuration starts from the baseline function (zero-init heads). Runs:
[scripts/exp09_train.sh](scripts/exp09_train.sh), per-box runner [scripts/exp09_box.sh](scripts/exp09_box.sh),
evaluation [scripts/exp09_analyze.sh](scripts/exp09_analyze.sh) with the mechanism readouts of
`analysis/exp09_mechanism.py` (errors by distance to the contour, source weights, knockouts, boundary-map
quality). Checks: `python tests/test_bg_fusion.py`. The default path, every EXP-4..8 configuration and
`--extra-term bl` are bit-identical to the EXP-8 code (2-epoch CPU runs compared file by file and tensor by
tensor).

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

*Running the author's original code.* With no flags, `python train.py` runs exactly the original computation.
This is not a claim of good faith but a checked invariant: at seed 0 the model's `state_dict` hashes to
`87b3dec8b84590308c5f527402db3e9ebabf0159634d35da5a62c7753ac6737f` (314 keys, 53,374 params) and one AdamW step
gives loss `10.317150115966797`, on every machine tested so far. The forward sum of a fixed input is recorded too,
but it depends on the CPU and PyTorch build (`77392.56237548799` on the original Colab torch and on a local CPU,
`77392.56770953648` and `77392.55116900953` on two rented GPU boxes), so compare it only before and after a
change on the same machine. Any change that shifts these numbers is a bug in the change, not a new baseline.
Known bugs of the original (fixed-angle random rotation, BCE on probabilities, the no-op normalization) are
deliberately preserved.

*Training is not reproducible run to run - in the original code, and therefore here.* The augmentation's
rotation angle is drawn once, when the transforms are built at import time, which is before `set_seed` runs;
every run therefore trains with a different angle, whatever the seed. On top of that, CUDA backward kernels are
nondeterministic. Two runs of the same command on the same machine already differ at the first iteration, and the
same configuration has been measured about 1 mIoU apart between runs. Read single-run gaps below that as noise.
Fixing either source would change the original behavior, so both are left as they are on `main`.

*On this branch (`exp/05-seeded-rotation-isic1718`) the rotation angle is a function of `--seed`:*
`myRandomRotation` draws its single angle from a private `random.Random(seed)`, so every run with seed 42 rotates
by `230.19364744483815` degrees and the angle is written to the log (`rotation angle: ...`). The private generator
touches neither the global python RNG nor torch's, so model initialization and the per-sample flip/rotate coins
are exactly what they are on `main` (checked: identical seed-0 `state_dict` and identical post-`set_seed` draws
before and after building the transforms). CUDA nondeterminism remains, and cannot be removed for this model:
`torch.use_deterministic_algorithms(True)` raises on the backward of bilinear `F.interpolate`, which the network
uses 17 times. Runs on this branch are therefore still single draws, only with one nuisance source fewer.

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
