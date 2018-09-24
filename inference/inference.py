from model import models
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils.meter import *
from inference.train_val import *
# import ipdb
from model import models
from utils.other import *


def main(options):
    # CUDA
    device = options['device']

    # Dataset
    train_dataset, val_dataset, train_loader, val_loader = get_datasets_and_dataloaders(options, device=options['device'])
    print('\n*** Train set of size {}  -  Val set of size {} ***\n'.format(print_number(len(train_dataset)),
                                                                           print_number(len(val_dataset))))

    # Model
    tmp = models.__dict__[options['arch']]
    model = models.__dict__[options['arch']](num_classes=train_dataset.nb_classes,
                                             size_fm_2nd_head=train_dataset.h_mask,
                                             options=options)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # Trainable params
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # Print number of parameters
    nb_total_params = count_nb_params(model.parameters())
    nb_trainable_params = count_nb_params(filter(lambda p: p.requires_grad, model.parameters()))
    ratio = float(nb_trainable_params / nb_total_params) * 100.
    print("\n* Parameter numbers : {} ({}) - {ratio:.2f}% of the weights are trainable".format(
        print_number(nb_total_params),
        print_number(nb_trainable_params),
        ratio=ratio
    ))

    # Optimizer
    optimizer = torch.optim.Adam(trainable_params, options['lr'])

    # Loss function and Metric
    criterion, metric = get_loss_and_metric(options)

    # Load resume from resume if exists
    model, optimizer, epoch = load_from_dir(model, optimizer, options)

    # My engine
    engine = {'model': model,
              'optimizer': optimizer, 'criterion': criterion, 'metric': metric,
              'train_loader': train_loader, 'val_loader': val_loader}

    # Training/Val or Testing #
    if options['evaluate']:
        # Val
        loss_val, metric_val, per_class_metric_val, df_good, df_failure, df_objects = validate(epoch, engine, options, device=device)
        # Write into log
        log_path='eval_{:s}_gcn{}.log'.format(options['dataset'], options['use_gcn'])
        write_to_log(log_path, val_dataset.dataset, options['resume'], epoch, [loss_val, metric_val], per_class_metric_val)
        # Save good and failures and object presence
        df_good.to_csv(os.path.join(options['resume'], 'df_good_preds.csv'), sep=',', encoding='utf-8')
        df_failure.to_csv(os.path.join(options['resume'], 'df_failure_preds.csv'), sep=',', encoding='utf-8')
        df_objects.to_csv(os.path.join(options['resume'], 'df_objects'), sep=',', encoding='utf-8')

    else:
        # Train (and Val if having access tto the val set)
        is_best = True
        best_metric_val = -0.1

        # experiment dir: ckpt & log
        save_dir = '{}/{}_gcn{}_bt{}_lr{:.0e}_wd{:.0e}'.format(
            options['resume'], options['dataset'], options['use_gcn'],
            options['batch_size'], options['lr'], options['wd'],
            )
        if options['use_gcn']:
            save_dir += '_adj{}_oEmb{}_vEmb{}_nLayer{}_nTopObjs{}'.format(
                options['adj_type'], options['D_obj_embed'], options['D_verb_embed'],
                options['n_layers'], options['n_top_objs']
                )
        if os.path.exists(save_dir):
          proceed = input('WARNING: Dir exists: {}\nWould you like to proceed (may overwrite previous ckpts) [y/N]?'.format(save_dir))
          if 'n' in proceed or 'N' in proceed:
            print('Do not overwrite -- Exiting.')
            exit(0)
        else:
          os.makedirs(save_dir)
        print('ckpt save_dir:', save_dir)

        log_path='train_{:s}_gcn{}.log'.format(options['dataset'], options['use_gcn'])

        for epoch in range(1, options['epochs'] + 1):
            try:
                # train one epoch
                loss_train, metric_train, loss_train_history, metric_train_history = train(epoch, engine, options, device=device)
    
                # write into log
                write_to_log(log_path, train_dataset.dataset, save_dir, epoch, [loss_train, loss_train], None)
    
                # get the val metric
                if options['train_set'] == 'train':
                    # Val
                    loss_val, metric_val, per_class_metric_val, *_ = validate(epoch, engine, options, device=device)
                    # Write into log
                    write_to_log(log_path, val_dataset.dataset, save_dir, epoch, [loss_val, metric_val],
                                 per_class_metric_val)
    
                    # Best compared to previous checkpoint ?
                    is_best = metric_val > best_metric_val
                    best_metric_val = max(metric_val, best_metric_val)
    
                # save checkpoint
                save_checkpoint({
                    'epoch': epoch,
                    'arch': options['arch'],
                    'state_dict': model.state_dict(),
                    'best_metric_val': best_metric_val,
                    'optimizer': optimizer.state_dict(),
                    'loss_history': loss_train_history,
                    'metric_history': metric_train_history,
                }, is_best, save_dir,
                filename='ckpt_epoch{:d}.pth'.format(epoch))
            except Exception as e:
                write_to_log(log_path, train_dataset.dataset, save_dir, epoch, None, None, err_str=str(e))
                print('!!!!!\nException: written to log file {}\n!!!!!'.format(log_path))
                raise e

    return None
