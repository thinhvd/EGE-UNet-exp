import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.dataset import NPY_datasets, split_dataset
from models.egeunet import EGEUNet

from engine import *
import os
import sys
import json
import argparse
import subprocess

from utils import *
from configs.config_setting import get_config
from contour_losses import LOSS_MODES, EXTRA_TERMS, describe, parse_terms

import warnings
warnings.filterwarnings("ignore")


METRIC_FIELDS = ['epoch', 'train_loss', 'val_loss', 'lr',
                 'miou', 'f1_or_dsc', 'accuracy', 'specificity', 'sensitivity']


def git_revision():
    """Branch + commit of the code, so every run records what produced it.

    A training server gets the code by rsync and has no .git, so fall back to the `.code_revision`
    file that scripts/push_code.sh writes at sync time.
    """
    repo = os.path.dirname(os.path.abspath(__file__))
    def _git(*args):
        try:
            return subprocess.check_output(['git', '-C', repo, *args],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return 'unknown'
    branch, commit = _git('rev-parse', '--abbrev-ref', 'HEAD'), _git('rev-parse', 'HEAD')
    if 'unknown' in (branch, commit):
        stamp = os.path.join(repo, '.code_revision')
        if os.path.isfile(stamp):
            try:
                with open(stamp) as f:
                    return f.read().strip() + ' (from .code_revision)'
            except Exception:
                pass
        return 'unknown'
    dirty = ' (dirty)' if _git('status', '--porcelain') not in ('', 'unknown') else ''
    return f'{branch}@{commit[:10]}{dirty}'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train EGE-UNet. Flags override configs/config_setting.py; '
                    'with no flags the original behavior is kept.')
    parser.add_argument('--work-dir', type=str, default=None,
                        help='experiment directory (holds log/, checkpoints/, outputs/, summary/, metrics.csv). '
                             'Re-running with the same directory auto-resumes from its latest.pth. '
                             'Default: a fresh timestamped dir under ./results/')
    parser.add_argument('--dataset', type=str, default=None, choices=['isic17', 'isic18'])
    parser.add_argument('--data-path', type=str, default=None,
                        help='dataset root containing train/ and val/ (overrides the path derived from --dataset)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--val-interval', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='cpu is only meant for local smoke tests')
    parser.add_argument('--no-resume', action='store_true',
                        help='ignore an existing latest.pth in work-dir and start from scratch')
    parser.add_argument('--hpa-mode', type=str, default=None, choices=['learnable', 'frozen_ones', 'none'],
                        help='GHPA static-prior ablation: learnable (original), frozen_ones (P fixed to ones, '
                             'conv_xy/zx/zy still trained), none (no Hadamard gating). Default: config value (learnable)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed (default 42 from config); use different seeds for multi-seed ablations')
    parser.add_argument('--ghpa-placement', type=str, default=None, choices=['low', 'mid', 'high'],
                        help='which resolution band holds the six GHPA modules (see models.egeunet.GHPA_PLACEMENTS); '
                             'low = original placement')
    parser.add_argument('--ghpa-stages', type=str, default=None,
                        help='free-form comma list of GHPA stages, e.g. "enc3,enc4,enc5,dec2,dec3,dec4" '
                             '(overrides --ghpa-placement)')
    parser.add_argument('--fusion', type=str, default=None,
                        choices=['none', 'sum', 'concat', 'csaa', 'sum_attn', 'bg_stage'],
                        help='EXP-4 cross-stage fusion feeding decoder stages a fused view of all five '
                             'encoder stages: none (original), sum / concat (controls without attention), '
                             'csaa (concat + cross-stage axial attention), sum_attn (sum + the same '
                             'attention). EXP-9: bg_stage (sum whose per-source weights change on the '
                             'contour predicted by a boundary head; needs --fusion-stages shallow3 and '
                             '--boundary-weight). Default: config value (none)')
    parser.add_argument('--fusion-stages', type=str, default=None, choices=['deep3', 'all5', 'shallow3'],
                        help='which decoder stages receive the fused feature (see models.fusion.'
                             'FUSION_STAGE_SETS); deep3 = dec1/2/3, all5 = every decoder stage, '
                             'shallow3 = dec3/4/5')
    parser.add_argument('--fusion-dim', type=int, default=None,
                        help='common channel width the encoder stages are projected to before fusion '
                             '(must be divisible by 4; default 16)')
    parser.add_argument('--loss', type=str, default=None, choices=LOSS_MODES,
                        help='EXP-6 base loss: bcedice (original GT_BceDiceLoss) or bce_region (Dice replaced '
                             'by the normalized active-contour region term in all six terms). Default: bcedice')
    parser.add_argument('--extra-term', type=str, default=None,
                        help='term(s) added to the base loss on the final output, comma-separated (see '
                             'contour_losses.py): tv (ACL length), tv_match (length matched to GT), area '
                             '(log-ratio area), bl (boundary loss, px), snbl (boundary loss in lesion radii), '
                             'fn_dp (misses weighted by distance to the prediction), hp (EXP-11 focal BCE on '
                             'lesion pixels deeper than --hp-radius), ihp / focal / tversky (EXP-14). Default: none. '
                             f'Choices: {", ".join(EXTRA_TERMS[1:])}')
    parser.add_argument('--extra-weight', type=str, default=None,
                        help='weight(s) of --extra-term, comma-separated in the same order (required with it)')
    parser.add_argument('--snbl-tau', type=float, default=3.0,
                        help='snbl: distances beyond this many lesion radii all cost the same (default 3)')
    parser.add_argument('--area-delta', type=float, default=0.05,
                        help='area: SmoothL1 knee on the log area ratio (default 0.05)')
    parser.add_argument('--hp-radius', type=float, default=10.0,
                        help='hp: a lesion pixel counts as deep when its distance to the background exceeds this '
                             'many px at 256 (default 10)')
    parser.add_argument('--ihp-tau', type=float, default=0.2,
                        help='EXP-14 ihp: a lesion pixel is interior when its distance to the background is at least this '
                             'fraction of the largest such distance in the image (default 0.2; fix it on train / val)')
    parser.add_argument('--ihp-gamma', type=float, default=2.0, help='EXP-14 ihp: focal exponent on (1 - u)')
    parser.add_argument('--ihp-plow', type=float, default=0.5, help='EXP-14 ihp: only interior pixels with u below this')
    parser.add_argument('--focal-alpha', type=float, default=0.75, help='EXP-14 focal: weight of class 1 (default 0.75)')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='EXP-14 focal: focusing exponent (default 2)')
    parser.add_argument('--tversky-alpha', type=float, default=0.3, help='EXP-14 tversky: weight of FP (default 0.3)')
    parser.add_argument('--tversky-beta', type=float, default=0.7, help='EXP-14 tversky: weight of FN (default 0.7)')
    parser.add_argument('--hp-gamma', type=float, default=2.0,
                        help='hp: focal exponent on (1 - u) (default 2; 0 = plain BCE on the deep pixels)')
    parser.add_argument('--boundary-weight', type=float, default=None,
                        help='EXP-9: weight of the contour-band loss on the boundary heads of --fusion bg_stage '
                             '(required with it; 0 = the band loss is only logged, the boundary heads then '
                             'learn through the fusion alone). EXP-10: weight of the contour-zone loss on the '
                             'dec5 head of --refine gate (required with it). EXP-11: weight of the contour-ring loss '
                             'on the barrier head of --complete (required with it)')
    parser.add_argument('--refine', type=str, default=None, choices=['none', 'gate', 'plain'],
                        help='EXP-10 dec5 boundary residual refinement on the segmentation logit (needs '
                             '--fusion sum --fusion-stages shallow3): gate = residual times the map of a '
                             'boundary head trained on the contour zone; plain = the same residual, no gate, '
                             'no head. Default: config value (none)')
    parser.add_argument('--refine-radius', type=int, default=10,
                        help='EXP-10: radius in px (at 256) of the contour zone that supervises the dec5 head '
                             '(default 10)')
    parser.add_argument('--complete', type=str, default=None, choices=['none', 'full', 'nobarrier'],
                        help='EXP-11 boundary-constrained region completion at dec5 (needs --fusion sum '
                             '--fusion-stages shallow3): grows the confident core (p >= --complete-seed) over '
                             'an evidence map, stopped by a barrier map trained on the contour ring, and adds '
                             'the grown part to the logit. full = evidence x (1 - barrier); nobarrier = '
                             'evidence only (the barrier head is still trained). Default: config value (none)')
    parser.add_argument('--complete-steps', type=int, default=32,
                        help='EXP-11: growth steps at the 128 grid, one cell each (default 32)')
    parser.add_argument('--complete-seed', type=float, default=0.9,
                        help='EXP-11: probability from which a cell of the base prediction is a seed (default 0.9)')
    parser.add_argument('--ring-radius', type=int, default=3,
                        help='EXP-11: half-width in px (at 256) of the contour ring that supervises the barrier '
                             'head (default 3)')
    parser.add_argument('--propagate', type=str, default=None, choices=['none', 'pair', 'pair_coupled', 'pair_nocap'],
                        help='EXP-13 core-to-region propagation at dec5 (needs --fusion sum --fusion-stages shallow3): '
                             'the confident core spreads along learned edges S between neighbouring cells, capped by '
                             'an image-relative logit level; joined to the output by probabilistic OR. pair = edge '
                             'head trained only by the edge-affinity loss; pair_coupled = + segmentation / completion '
                             'gradients through the propagation; pair_nocap = pair without the relative cap')
    parser.add_argument('--propagate-steps', type=int, default=32)
    parser.add_argument('--propagate-seed', type=float, default=0.9)
    parser.add_argument('--propagate-tau', type=float, default=None,
                        help='EXP-13 cap: U = sigmoid(kappa * (t_z - tau)) (required with --propagate)')
    parser.add_argument('--propagate-kappa', type=float, default=None)
    parser.add_argument('--propagate-beta', type=float, default=None, help='EXP-13: edge weight S ** beta')
    parser.add_argument('--aff-weight', type=float, default=None, help='EXP-13: weight of the edge-affinity loss (required with --propagate)')
    parser.add_argument('--comp-weight', type=float, default=0.0,
                        help='EXP-13: weight of the completion loss on the propagated map (pair_coupled only)')
    parser.add_argument('--hard-frac', type=float, default=0.2,
                        help='EXP-13: share of each image\'s lesion-lesion edges, by contrast, treated as hard "do not cut" edges')
    parser.add_argument('--test-split', type=str, default=None,
                        help='EXP-14 protocol (70:10:20): after training, test the best checkpoint on this split folder of '
                             '--data-path (e.g. "test") instead of the val split; val keeps selecting the checkpoint. '
                             'Default: test on val, as the original code does')
    parser.add_argument('--save-every', type=int, default=None,
                        help='also keep the weights of every N-th epoch from --save-from on '
                             '(checkpoints/epochNNN.pth), to measure how much the chosen epoch matters. '
                             'Default: off (only best and latest, as before)')
    parser.add_argument('--save-from', type=int, default=200)
    args = parser.parse_args()
    parse_terms(args.extra_term or 'none', args.extra_weight)   # fail early on a malformed term list
    # EXP-9: the boundary-guided fusion and its band loss only make sense together, on dec3..dec5
    # (the band-loss weights are defined per stage there). 0 is a valid weight, so test against None.
    if args.fusion == 'bg_stage':
        if args.boundary_weight is None:
            parser.error('--fusion bg_stage needs an explicit --boundary-weight (0 = band loss logged only)')
        if args.fusion_stages != 'shallow3':
            parser.error('--fusion bg_stage needs --fusion-stages shallow3')
    # EXP-10: the residual reads the sum fusion's projections at dec5; only the gated variant has a
    # head to supervise.
    refine = args.refine if args.refine and args.refine != 'none' else None
    if refine is not None:
        if args.fusion != 'sum' or args.fusion_stages != 'shallow3':
            parser.error('--refine needs --fusion sum --fusion-stages shallow3')
        if refine == 'gate' and args.boundary_weight is None:
            parser.error('--refine gate needs an explicit --boundary-weight (0 = zone loss logged only)')
        if refine == 'plain' and args.boundary_weight is not None:
            parser.error('--refine plain has no boundary head; drop --boundary-weight')
        if args.refine_radius <= 0:
            parser.error('--refine-radius must be > 0')
    # EXP-11: the completion reads the sum fusion's projections at dec5 and its barrier head is supervised
    complete = args.complete if args.complete and args.complete != 'none' else None
    if complete is not None:
        if args.fusion != 'sum' or args.fusion_stages != 'shallow3':
            parser.error('--complete needs --fusion sum --fusion-stages shallow3')
        if refine is not None:
            parser.error('--complete and --refine both act on the dec5 logit; use one of them')
        if args.boundary_weight is None:
            parser.error('--complete needs an explicit --boundary-weight (0 = ring loss logged only)')
        if args.complete_steps < 1:
            parser.error('--complete-steps must be >= 1')
        if not 0.5 <= args.complete_seed < 1:
            parser.error('--complete-seed must be in [0.5, 1)')
        if args.ring_radius < 1:
            parser.error('--ring-radius must be >= 1')
    # EXP-13: the propagation reads the sum fusion's projections at dec5 and is trained by the edge-affinity loss
    propagate = args.propagate if args.propagate and args.propagate != 'none' else None
    if propagate is not None:
        if args.fusion != 'sum' or args.fusion_stages != 'shallow3':
            parser.error('--propagate needs --fusion sum --fusion-stages shallow3')
        if refine is not None or complete is not None:
            parser.error('--propagate, --complete and --refine all act on the dec5 output; use one of them')
        if args.boundary_weight is not None:
            parser.error('--propagate does not combine with --boundary-weight')
        if None in (args.propagate_tau, args.propagate_kappa, args.propagate_beta, args.aff_weight):
            parser.error('--propagate needs explicit --propagate-tau, --propagate-kappa, --propagate-beta and --aff-weight')
        if args.propagate_steps < 1 or not 0.5 <= args.propagate_seed < 1 or args.propagate_kappa <= 0 or args.propagate_beta <= 0:
            parser.error('--propagate-steps >= 1, --propagate-seed in [0.5, 1), --propagate-kappa > 0, --propagate-beta > 0')
        if args.comp_weight < 0 or (args.comp_weight > 0 and propagate != 'pair_coupled'):
            parser.error('--comp-weight > 0 only with --propagate pair_coupled (elsewhere the propagation carries no gradient)')
        if not 0 < args.hard_frac < 1:
            parser.error('--hard-frac must be in (0, 1)')
    elif args.aff_weight is not None or args.comp_weight:
        parser.error('--aff-weight / --comp-weight only go with --propagate')
    if args.boundary_weight is not None and args.fusion != 'bg_stage' and refine != 'gate' and complete is None:
        parser.error('--boundary-weight only goes with --fusion bg_stage, --refine gate or --complete')
    if args.boundary_weight is not None and args.boundary_weight < 0:
        parser.error('--boundary-weight must be >= 0')
    return args


def main(config):

    print('#----------Creating logger----------#')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(os.path.join(config.work_dir, 'summary'))

    log_config_info(config, logger)

    log_info = f'code revision: {git_revision()}'
    print(log_info)
    logger.info(log_info)





    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    if config.device == 'cuda':
        torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True)





    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'egeunet':
        model = EGEUNet(num_classes=model_cfg['num_classes'],
                        input_channels=model_cfg['input_channels'],
                        c_list=model_cfg['c_list'],
                        bridge=model_cfg['bridge'],
                        gt_ds=model_cfg['gt_ds'],
                        hpa_mode=model_cfg.get('hpa_mode', 'learnable'),
                        ghpa_stages=model_cfg.get('ghpa_stages'),
                        fusion_mode=model_cfg.get('fusion_mode', 'none'),
                        fusion_stages=model_cfg.get('fusion_stages'),
                        fusion_dim=model_cfg.get('fusion_dim', 16),
                        refine_mode=model_cfg.get('refine_mode', 'none'),
                        complete_mode=model_cfg.get('complete_mode', 'none'),
                        complete_steps=model_cfg.get('complete_steps', 32),
                        complete_seed=model_cfg.get('complete_seed', 0.9),
                        propagate_mode=model_cfg.get('propagate_mode', 'none'),
                        propagate_steps=model_cfg.get('propagate_steps', 32),
                        propagate_seed=model_cfg.get('propagate_seed', 0.9),
                        propagate_tau=model_cfg.get('propagate_tau', 0.3),
                        propagate_kappa=model_cfg.get('propagate_kappa', 10.0),
                        propagate_beta=model_cfg.get('propagate_beta', 1.0),
                        )
    else: raise Exception('network in not right!')
    model = model.to(config.device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info = (f'hpa_mode: {model_cfg.get("hpa_mode", "learnable")}, '
                f'ghpa_stages: {model.ghpa_stages}, '
                f'fusion: {model.fusion_mode}/{model.fusion_stages}/d{model_cfg.get("fusion_dim", 16)}, '
                + (f'refine: {model.refine_mode}, ' if model.refine_mode != 'none' else '')
                + (f'complete: {model.complete_mode} T={model.complete.steps} seed={model.complete.seed_thr:g}, '
                   if model.complete_mode != 'none' else '')
                + (f'propagate: {model.propagate_mode} T={model.propagate.steps} seed={model.propagate.seed_thr:g} '
                   f'tau={model.propagate.tau:g} kappa={model.propagate.kappa:g} beta={model.propagate.beta:g}, '
                   if model.propagate_mode != 'none' else '')
                + f'params: {n_total} total / {n_trainable} trainable')
    print(log_info)
    logger.info(log_info)
    # EXP-5: record the run's rotation angle (reading an attribute consumes no randomness).
    # Goes through logger so it lives in log/train.info.log, which survives the LEAN sync.
    rot = [t for t in config.train_transformer.transforms if isinstance(t, myRandomRotation)]
    log_info = (f'rotation angle: {rot[0].angle!r} (seed {rot[0].seed})' if rot
                else 'rotation angle: n/a')
    print(log_info)
    logger.info(log_info)





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    # EXP-6: validation (checkpoint selection) and the final test loss always use the original
    # GT_BceDiceLoss, so every loss variant picks its checkpoint by the same rule. With no loss flags
    # this is the training criterion itself.
    select_criterion = getattr(config, 'selection_criterion', criterion)
    has_extra = hasattr(criterion, 'epoch_means')
    # one extra term keeps EXP-6's single 'train_extra' column; several get one column each, and so
    # does a lone boundary-head term (EXP-10), whose name says which head it belongs to
    extra_fields = (['train_extra'] if has_extra and len(criterion.names) == 1 and not criterion.names[0].startswith('bnd_') else
                    [f'train_extra_{n}' for n in criterion.names] if has_extra else [])
    log_info = (f'loss: {describe(**getattr(config, "loss_config", {}))}; checkpoint selection: '
                f'{"original GT_BceDiceLoss" if select_criterion is not criterion else "same loss"}')
    print(log_info)
    logger.info(log_info)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1





    if os.path.exists(resume_model) and not config.no_resume:
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'), weights_only=False)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            raise RuntimeError(f'{resume_model} was trained with a different model config '
                               f'(e.g. --hpa-mode / --fusion / c_list). Use a new --work-dir or '
                               f'pass --no-resume.') from e
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)




    metrics_csv = os.path.join(config.work_dir, 'metrics.csv')

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        if config.device == 'cuda':
            torch.cuda.empty_cache()

        # the lr actually used this epoch (scheduler.step() fires at the end of train_one_epoch)
        epoch_lr = optimizer.state_dict()['param_groups'][0]['lr']

        if has_extra:
            criterion.reset_stats()
        step, train_loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        loss, val_metrics = val_one_epoch(
                val_loader,
                model,
                select_criterion,
                epoch,
                logger,
                config
            )

        if loss < min_loss:
            atomic_torch_save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        atomic_torch_save(
            {
                'epoch': epoch,
                'min_loss': float(min_loss),
                'min_epoch': min_epoch,
                'loss': float(loss),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

        # appended after the checkpoint save, so a resumed run can never duplicate rows
        row = {'epoch': epoch, 'train_loss': float(train_loss), 'val_loss': float(loss), 'lr': epoch_lr}
        if val_metrics is not None:
            row.update(val_metrics)
        if has_extra:   # per-epoch mean of each unweighted extra term, for monitoring
            means = criterion.epoch_means()
            if extra_fields == ['train_extra']:
                row['train_extra'] = means[criterion.names[0]]
            else:
                row.update({f'train_extra_{n}': v for n, v in means.items()})
        append_metrics_row(metrics_csv, METRIC_FIELDS + extra_fields, row)
        if getattr(config, 'save_every', None) and epoch >= config.save_from and epoch % config.save_every == 0:
            atomic_torch_save(model.state_dict(), os.path.join(checkpoint_dir, f'epoch{epoch:03d}.pth'))

    test_split = getattr(config, 'test_split', None)
    test_loader = val_loader
    if test_split:   # EXP-14: a held-out test split, never seen during training or checkpoint selection
        test_loader = DataLoader(split_dataset(config.data_path, config, test_split), batch_size=1, shuffle=False,
                                 pin_memory=True, num_workers=config.num_workers, drop_last=True)
        log_info = f'test split: {test_split} ({len(test_loader.dataset)} images); checkpoint selection on val ({len(val_dataset)} images)'
        print(log_info)
        logger.info(log_info)
    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss, test_metrics = test_one_epoch(
                test_loader,
                model,
                select_criterion,
                logger,
                config,
            )
        best_name = f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth'
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, best_name)
        )
        results = {'min_epoch': min_epoch, 'min_loss': float(min_loss),
                   'test_loss': float(loss), 'checkpoint': best_name}
        if test_split:
            results.update({'test_split': test_split, 'n_test_images': len(test_loader.dataset)})
        results.update(test_metrics)
        with open(os.path.join(config.work_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    config = get_config(parse_args())
    main(config)
