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
import seaborn as sn
import matplotlib.pyplot as plt
import cv2

from models import get_model
from config import cfg
from config import update_config
from utils import save_checkpoint, get_optimizer, create_logger, ParticleFilter
from pathlib import Path
from datasets import RSSI_Dataset, RSSI_DatasetForTest
from sklearn.metrics import confusion_matrix,mean_squared_error
from datetime import datetime
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

    parser.add_argument('--mode',
                        help='Mode of different dataset',
                        type=str,
                        default='valid')

    parser.add_argument('--video',
                        help='Path of the video',
                        type=str,
                        default='valid')

    args = parser.parse_args()

    return args

def test(cfg):

    logger, final_output_dir = create_logger(cfg, args.cfg, 'test')

    model = get_model(cfg, False)
    if args.mode == "valid":
        dataset = RSSI_Dataset(cfg)
        
    elif args.mode == "test":
        dataset = RSSI_DatasetForTest(cfg)

    optimizer = get_optimizer(cfg, model)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False
    )
    checkpoint_file = os.path.join(
        cfg.MODEL.PRETRAINED
    )

    if os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['best_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # testing
    model.eval()
    top1_acc = 0
    # range -1~+1
    topk_acc = 0
    # mean squared error
    mse_loss = 0
    # distance of each block 
    distance = 2
    total_test = 0
    k = 1
    pf = ParticleFilter(cfg)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            y_test = model(x)
            #print(y)
            _, maxk = torch.topk(y_test, k, dim=-1)
            #print(maxk)
            #print(y_test)
            #maxk = pf.update(y_test)
            #print(maxk[:, 0:1])
            y_pred.extend(maxk[:, 0:1].view(-1).detach().numpy())
            y_true.extend(y.view(-1).detach().numpy())
            y = y.view(-1, 1)
            top1_acc += (y == maxk[:, 0:1]).sum().item()
            topk_acc += (y == maxk[:, 0:1]).sum().item() + (y == (maxk[:, 0:1]+1)).sum().item() + (y == (maxk[:, 0:1]-1)).sum().item()
            total_test += y_test.size(0)

            mse_loss += pow(distance * abs(y-maxk[:,0:1]), 2).sum().item()
        print(f'Top-1 test accuracy :{100*top1_acc/total_test}%')  
        print(f'In Range [-1,+1] test accuracy :{100*topk_acc/total_test}%') 
        print(f"Mean Squared Error : {mse_loss/total_test}")
    ConfusionMatrix(y_true, y_pred)

def testOnVideo(cfg):

    logger, final_output_dir = create_logger(cfg, args.cfg, 'test_with_video')

    model = get_model(cfg, False)
    if args.mode == "valid":
        dataset = RSSI_Dataset(cfg)
        
    elif args.mode == "test":
        dataset = RSSI_DatasetForTest(cfg)

    optimizer = get_optimizer(cfg, model)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )
    checkpoint_file = os.path.join(
        cfg.MODEL.PRETRAINED
    )

    if os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['best_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # testing
    model.eval()
    top1_acc = 0
    # range -1~+1
    topk_acc = 0
    # mean distance error
    mde_loss = 0
    # distance of each block 
    distance = 2
    total_test = 0
    k = 1
    pf = ParticleFilter(cfg)
    y_pred = []
    y_true = []
    test_id = 0

    # set width and height
    width = 1920
    height = 1080
    # output dir
    out_name = datetime.now().strftime('%Y%m%d') + '_withline' + '.mp4'
    # set encode type
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # save output to video
    out = cv2.VideoWriter(out_name, fourcc, 30.0, (width,  height)) 

    with torch.no_grad():
        # 開啟影片檔案
        cap = cv2.VideoCapture(cfg.TEST_VIDEO_PATH)
        fps_count = 0
        block_id = 0 
        while(cap.isOpened()):
            # 以迴圈從影片檔案讀取影格，並顯示出來
            # FPS:30 所以每30幀predict一個位置
            if not fps_count%30:
                x = dataset[test_id]
                if cfg.MODEL.TYPE == "DNN":
                    x = x.view(1, -1)
                elif cfg.MODEL.TYPE == "LSTM":
                    x = x.view(1, x.size()[0], x.size()[1])

                y_test = model(x)
                # get top-k result
                _, maxk = torch.topk(y_test, k, dim=-1)
                # run particle filter
                maxk = pf.update(y_test)
                # get value
                block_id = maxk[0][0].item()
                #print(block_id)
                #print(fps_count/30)
                test_id += 1

            ret, frame = cap.read()
            # 寫入新的output
            
            fps_count += 1
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

            overlay = frame.copy()
            # 重疊圖的比例 越高畫線的顏色越深
            alpha = 0.3
            # 導盲磚容許範圍
            pts = np.array([[850, 290], [1185, 1070], [1520, 1070], [940, 290]], dtype=np.int32)
            # 每個Block範圍
            pts_blocks = np.array([
                                [[1113, 894], [1186, 1067], [1513, 1067], [1386, 894]],
                                [[1021, 680], [1113, 894], [1386, 894], [1228, 680]],
                                [[966, 554], [1017, 672], [1219, 672], [1134, 554]],
                                [[927, 463], [966, 554], [1134, 554], [1065, 463]],
                                [[898, 393], [927, 463], [1065, 463], [1013, 393]],
                                [[887, 366], [898, 393], [1013, 393], [993, 366]],
                                [[872, 332], [887, 366], [993, 366], [968, 332]],
                                [[856, 294], [872, 332], [968, 332], [938, 294]],], dtype=np.int32)
            cv2.drawContours(overlay, [pts], -1, (0, 0, 0), thickness=5)
            # # 判斷一個點是否落在多邊形範圍內
            # test_point = ((1000, 500))
            # overlay = cv2.circle(overlay, test_point, 2, (0, 0, 255), 2)
            # # 在範圍內則回傳正值，否則為負
            # print(cv2.pointPolygonTest(pts, test_point, 1))

            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            overlay = frame.copy()
            cv2.drawContours(overlay, [pts_blocks[block_id]], -1, (0, 0, 255), thickness=-1)

            # Perform weighted addition of the input image and the overlay
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()        




def ConfusionMatrix(y_true, y_pred):
    cm = confusion_matrix(y_true , y_pred)
    fit = plt.figure(figsize=(8,6))
    plt.title('confusion matrix')
    sn.heatmap(cm,annot=True,cmap='OrRd',fmt='g')
    plt.xlabel('prediction')
    plt.ylabel('true label')
    plt.show()

if __name__ == '__main__':
    
    args = parse_args()
    update_config(cfg, args)

    if not cfg.TEST_NO_LABEL:
        test(cfg)
    else:
        testOnVideo(cfg)

