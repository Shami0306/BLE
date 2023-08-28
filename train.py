import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim 
import torch.utils.data
import torch.utils.data.distributed
import logging
import os
import time
import argparse
import init_paths
import math
import numpy as np

from models import get_model
from config import cfg
from config import update_config
from utils import save_checkpoint, get_optimizer, create_logger, ParticleFilter
from pathlib import Path
from datasets import RSSI_Dataset
from torchsummary import summary

#torch.manual_seed(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    # optional

    parser.add_argument('--beforeDir',
                        help='File path of the csv before preprocessing',
                        type=str,
                        default='')

    parser.add_argument('--afterDir',
                        help='File path of the csv after preprocessing',
                        type=str,
                        default='')

    parser.add_argument('--save',
                        help='loss : save by validation loss, acc : save by validation accuracy',
                        type=str,
                        default='loss')
    
    parser.add_argument('--loss2',
                        help='set loss2 weight',
                        type=float,
                        default=1000)
    
    parser.add_argument('--timestamp',
                        help='set timestamp',
                        type=int,
                        default=3)    
        
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir = create_logger(cfg, args.cfg, 'train')

    model = get_model(cfg, is_train=True)
    dataset = RSSI_Dataset(cfg)
    train_len = int(0.85 * len(dataset))
    valid_len = len(dataset) - train_len

    train_data, valid_data = torch.utils.data.random_split(dataset, [train_len, valid_len])

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
    )
    
    optimizer = get_optimizer(cfg, model)
    loss_func = nn.CrossEntropyLoss()
    #loss_func2 = nn.BCELoss()

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    # --- load checkpoint ---
    checkpoint_file = os.path.join(
        final_output_dir, 'model_lstm5.pth'
    )
    if os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['best_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, min_lr= 1e-4, factor=0.5
    )


    best_perf = 10000.0
    best_model = False
    best_acc = 0
    best_inrange_acc = 0
    best_mde = 10000.0
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

        # training
        model.train()
        train_loss = 0
        for i, (x, y1) in enumerate(train_loader):
            y1_pred = model(x)
            loss1 = loss_func(y1_pred, y1)
            #loss2 = loss_func2(y2_pred, y2)
            loss = loss1 #+ loss2*args.loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss

        # validation
        model.eval()
        valid_loss = 0
        # mean distance error
        mde_loss = 0
        # distance of each block 
        distance = 2
        top1_acc = 0
        topk_acc = 0
        total_valid = 0
        k = 3
        pf = ParticleFilter(cfg)
        with torch.no_grad():
            for i, (x, y) in enumerate(valid_loader):
                y_valid = model(x)
                loss = loss_func(y_valid, y)

                _, maxk = torch.topk(y_valid, k, dim=-1)
                #maxk = pf.run(y_valid)
                #print(maxk)
                y = y.view(-1, 1)
                #print(y)
                top1_acc += (y == maxk[:, 0:1]).sum().item()
                topk_acc += (y == maxk[:, 0:1]).sum().item() + (y == (maxk[:, 0:1]+1)).sum().item() + (y == (maxk[:, 0:1]-1)).sum().item()
                total_valid += y_valid.size(0)

                valid_loss += loss.item()
                mde_loss += distance * abs(y-maxk[:,0:1]).sum().item()
            print(f'Epoch {epoch+1} , top-1 valid_acc:{100*top1_acc/total_valid}%, loss : {valid_loss}, mde loss : {mde_loss/total_valid}')  

            # save by loss
            if args.save == "loss":
                lr_scheduler.step(valid_loss)
                # save best
                if valid_loss <= best_perf:
                    best_perf = valid_loss
                    best_model = True
                    best_acc = 100*top1_acc/total_valid
                    best_inrange_acc = 100*topk_acc/total_valid
                    best_mde = mde_loss/total_valid
                else:
                    best_model = False
            # save by acc

            elif args.save == "acc":
                lr_scheduler.step(top1_acc)
                acc = 100*top1_acc/total_valid
                if acc >= best_acc:
                    best_perf = valid_loss
                    best_model = True
                    best_acc = acc
                    best_inrange_acc = 100*topk_acc/total_valid
                    best_mde = mde_loss/total_valid
                else:
                    best_model = False

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            logger.info(f'best validation accuracy : {best_acc}%')
            logger.info(f'best in range accuracy : {best_inrange_acc}%')
            logger.info(f'best mean distance error : {best_mde}')

            save_checkpoint({
                'epoch': epoch + 1,
                'best_state_dict': model.state_dict(),
                'perf': best_perf,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)


if __name__ == '__main__':
    main()