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
import csv

from models import get_model
from config import cfg
from config import update_config
from utils import save_checkpoint, get_optimizer, create_logger, ParticleFilter, ParticleFilter_v2
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

    parser.add_argument('--video',
                        help='Path of the video',
                        type=str,
                        default='valid')

    parser.add_argument('--multitest',
                        help='Path of the video',
                        type=bool,
                        default=False)
    
    parser.add_argument('--timestep',
                        help='set time step',
                        type=int,
                        default=3)    
      
    args = parser.parse_args()

    return args

def test(cfg):

    logger, final_output_dir = create_logger(cfg, args.cfg, 'test')

    model = get_model(cfg, False)
    optimizer = get_optimizer(cfg, model)

    test_loaders = []
    if args.multitest:
        # test_list = [
        #              './after/U19e_outdoor0103test1/',
        #              './after/U19e_outdoor0103test2/',
        #              ]
        test_list = [
                     './after/U19e_outdoor0517test1_4/',
                     './after/U19e_outdoor0517test1_5/',
                     './after/U19e_outdoor0517test2_3/',
                     './after/U19e_outdoor0517test2_4/',
                    #  './after/U19e_outdoor0517test2_5/',
                     ]
        # test_list = ['./after/Sharp_outdoor0517test1_1/',
        #              './after/Sharp_outdoor0517test1_2/',
        #              './after/Sharp_outdoor0517test1_3/',
        #              './after/Sharp_outdoor0517test1_4/',
        #              './after/Sharp_outdoor0517test1_5/',
        #              './after/Sharp_outdoor0517test2_1/',
        #              './after/Sharp_outdoor0517test2_2/',
        #              './after/Sharp_outdoor0517test2_3/',
        #              './after/Sharp_outdoor0517test2_4/',
        #              './after/Sharp_outdoor0517test2_5/',
        #             ]  
        for path in test_list:
            cfg.defrost()
            cfg.AFTER_DIR = path
            cfg.freeze()
            dataset = RSSI_DatasetForTest(cfg)
            test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                shuffle=False
            )
            test_loaders.append(test_loader)
    else:
        dataset = RSSI_DatasetForTest(cfg) # 讀test.csv 有label
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False
        )
        test_loaders.append(test_loader)

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
    mde_list = []
    # distance of each block 
    distance = 2
    total_test = 0
    k = 1
    pf = ParticleFilter_v2(cfg)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for test_loader in test_loaders:
            for i, (x, y) in enumerate(test_loader):
                y_test = model(x)
                #print(y)
                _, maxk = torch.topk(y_test, k, dim=-1)
                # run Particle Filter 
                maxk = pf.run(y_test)
                # 紀錄 ground truth y與prediction y
                y_pred.extend(maxk[:, 0:1].view(-1).detach().numpy())
                y_true.extend(y.view(-1).detach().numpy())
                y = y.view(-1, 1)
                # compute accuracy
                top1_acc += (y == maxk[:, 0:1]).sum().item()
                topk_acc += (y == maxk[:, 0:1]).sum().item() + (y == (maxk[:, 0:1]+1)).sum().item() + (y == (maxk[:, 0:1]-1)).sum().item()
                # total data size
                total_test += y_test.size(0)
                # compute mean distance error
                mde_loss += pow(distance * abs(y-maxk[:, 0:1]), 2).sqrt().sum().item()
                for j, pos in enumerate(maxk):
                    mde_list.append(pow(distance * abs(y[j]-pos[0]), 2).sqrt().sum().item())

        print(f'Top-1 test accuracy :{100*top1_acc/total_test}%')  
        print(f'In Range [-1,+1] test accuracy :{100*topk_acc/total_test}%') 
        print(f"Mean Distance Error : {mde_loss/total_test}")

        #ConfusionMatrix(y_true, y_pred)
        GetCDF(mde_list)

def testOnVideo(cfg):

    logger, final_output_dir = create_logger(cfg, args.cfg, 'test_with_video')

    model = get_model(cfg, False)

    dataset = RSSI_DatasetForTest(cfg) # 讀 test.csv 沒label

    optimizer = get_optimizer(cfg, model)
    # test_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False
    # )
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
    k = 1
    # pf = ParticleFilter(cfg)
    pf = ParticleFilter_v2(cfg)
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
                maxk = pf.run(y_test)
                block_id = maxk[0][0].item()
                print(block_id)
                print(fps_count/30)
                test_id += 1

            ret, frame = cap.read()
            if frame is None:
                print(fps_count)
                break
            # 寫入新的output
            
            fps_count += 1
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

            overlay = frame.copy()
            # 重疊圖的比例 越高畫線的顏色越深
            alpha = 0.3
            # 導盲磚容許範圍
            pts = np.array([[850, 290], [1185, 1080], [1520, 1080], [940, 290]], dtype=np.int32)
            # 每個Block範圍
            pts_blocks = np.array([
                                [[1093, 855], [1188, 1080], [1516, 1080], [1350, 855]],
                                [[1005, 650], [1093, 855], [1350, 855], [1202, 650]],
                                [[965, 554], [1005, 650], [1202, 650], [1129, 554]],
                                [[926, 463], [965, 554], [1129, 554], [1062, 463]],
                                [[907, 419], [926, 463], [1062, 463], [1030, 419]],
                                [[888, 373], [907, 419], [1030, 419], [996, 373]],
                                [[874, 339], [888, 373], [996, 373], [971, 339]],
                                [[853, 292], [874, 339], [971, 339], [940, 292]]], dtype=np.int32)
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

def GetCDF(list):
    list = np.array(list)
    print(list)
    #bin_list = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
    bin_list = np.arange(0, 13, 1)
    count, bins_count = np.histogram(list, bins=bin_list)
    print(bins_count)
    pdf = count/np.sum(count)
    #print(pdf)
    cdf = np.cumsum(pdf)
    print(cdf)
    # save cdf
    with open('cdf.csv', 'w', newline='') as cdffile:
        writer = csv.writer(cdffile)

        writer.writerow(bin_list)
        writer.writerow(cdf)

if __name__ == '__main__':
    
    args = parse_args()
    update_config(cfg, args)

    if not cfg.TEST_FOR_VIDEO:
        test(cfg)
    else:
        testOnVideo(cfg)

