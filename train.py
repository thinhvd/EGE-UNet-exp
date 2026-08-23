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

from utils import *
from configs.config_setting import get_config

import warnings
warnings.filterwarnings("ignore")


METRIC_FIELDS = ['epoch', 'train_loss', 'val_loss', 'lr',
                 'miou', 'f1_or_dsc', 'accuracy', 'specificity', 'sensitivity']


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
    return parser.parse_args()


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
                        )
    else: raise Exception('network in not right!')
    model = model.to(config.device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info = f'hpa_mode: {model_cfg.get("hpa_mode", "learnable")}, params: {n_total} total / {n_trainable} trainable'
    print(log_info)
    logger.info(log_info)





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
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
                               f'(e.g. --hpa-mode / c_list). Use a new --work-dir or pass --no-resume.') from e
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
                criterion,
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
        append_metrics_row(metrics_csv, METRIC_FIELDS, row)

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss, test_metrics = test_one_epoch(
                val_loader,
                model,
                criterion,
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
