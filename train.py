import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.dataset import NPY_datasets
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
                        choices=['none', 'sum', 'concat', 'csaa', 'sum_attn'],
                        help='EXP-4 cross-stage fusion feeding decoder stages a fused view of all five '
                             'encoder stages: none (original), sum / concat (controls without attention), '
                             'csaa (concat + cross-stage axial attention), sum_attn (sum + the same '
                             'attention). Default: config value (none)')
    parser.add_argument('--fusion-stages', type=str, default=None, choices=['deep3', 'all5'],
                        help='which decoder stages receive the fused feature (see models.fusion.'
                             'FUSION_STAGE_SETS); deep3 = dec1/2/3, all5 = every decoder stage')
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
                             'fn_dp (misses weighted by distance to the prediction). Default: none. '
                             f'Choices: {", ".join(EXTRA_TERMS[1:])}')
    parser.add_argument('--extra-weight', type=str, default=None,
                        help='weight(s) of --extra-term, comma-separated in the same order (required with it)')
    parser.add_argument('--snbl-tau', type=float, default=3.0,
                        help='snbl: distances beyond this many lesion radii all cost the same (default 3)')
    parser.add_argument('--area-delta', type=float, default=0.05,
                        help='area: SmoothL1 knee on the log area ratio (default 0.05)')
    parser.add_argument('--save-every', type=int, default=None,
                        help='also keep the weights of every N-th epoch from --save-from on '
                             '(checkpoints/epochNNN.pth), to measure how much the chosen epoch matters. '
                             'Default: off (only best and latest, as before)')
    parser.add_argument('--save-from', type=int, default=200)
    args = parser.parse_args()
    parse_terms(args.extra_term or 'none', args.extra_weight)   # fail early on a malformed term list
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
                        )
    else: raise Exception('network in not right!')
    model = model.to(config.device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info = (f'hpa_mode: {model_cfg.get("hpa_mode", "learnable")}, '
                f'ghpa_stages: {model.ghpa_stages}, '
                f'fusion: {model.fusion_mode}/{model.fusion_stages}/d{model_cfg.get("fusion_dim", 16)}, '
                f'params: {n_total} total / {n_trainable} trainable')
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
    # one extra term keeps EXP-6's single 'train_extra' column; several get one column each
    extra_fields = (['train_extra'] if has_extra and len(criterion.names) == 1 else
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

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss, test_metrics = test_one_epoch(
                val_loader,
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
        results.update(test_metrics)
        with open(os.path.join(config.work_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    config = get_config(parse_args())
    main(config)
