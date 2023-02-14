import logging
import os
import torch
import random
import pandas as pd
import numpy as np
import torch.utils.data as data

from collections import Counter

logger = logging.getLogger(__name__)

class RSSI_Dataset(data.Dataset):
    
    def __init__(self, cfg) -> None:
        super().__init__()

        # get total length
        file_path = os.path.join(cfg.AFTER_DIR + 'block_' + str(1) + '.' + cfg.FILE_TYPE)
        df = pd.read_csv(file_path)
        self.total_length = df.shape[0]

        # read each block csv and concat them 

        file_path = os.path.join(cfg.AFTER_DIR + 'block_' + str(1) + '.' + cfg.FILE_TYPE)
        df = pd.read_csv(file_path)
        df = df[:self.total_length]
        self.label_list = df['label']
        self.rssi_list = df.drop(columns=['time', 'label'])

        for i in range(1, len(cfg.START_TIME)):
            file_path = os.path.join(cfg.AFTER_DIR + 'block_' + str(i+1) + '.' + cfg.FILE_TYPE)
            df = pd.read_csv(file_path)
            df = df[:self.total_length]

            self.label_list = pd.concat([self.label_list, df['label']])
            df = df.drop(columns=['time', 'label'])
            self.rssi_list = pd.concat([self.rssi_list, df])
            
        self.total_length = len(self.rssi_list)
        self.label_list = self.label_list.to_numpy()
        self.rssi_list = self.rssi_list.to_numpy()
        self.classes = len(cfg.START_TIME)  

        if cfg.MODEL.TYPE == "DNN":
            pass
        elif cfg.MODEL.TYPE == "1DCNN":
            # change shape (n, features) -> (n/t , t, features)
            # timestamp = 1
            self.T = 1
            # save old lists
            self.old_rssi_list = self.rssi_list.copy()
            self.old_label_list = self.label_list.copy()
            # new lists with shape (n/t, t, features)
            self.rssi_list = []
            self.label_list = []

            reverse = False
            # training direction (1->8) is not reversed, (8->1) is reversed
            self.total_length = self.total_length+1-self.T
            if not reverse :
                for i in range(self.total_length):
                    self.rssi_list.append(self.old_rssi_list[i:i+self.T])
                    data = Counter(self.old_label_list[i:i+self.T])
                    # most frequent value as label e.g. when t=3 [0,0,1] -> [0] as ground truth
                    self.label_list.append(max(self.old_label_list[i:i+self.T], key=data.get))
            else:
                for i in reversed(range(self.total_length)):
                    self.rssi_list.append(self.old_rssi_list[i+self.T-1 : i-1:-1])
                    data = Counter(self.old_label_list[i+self.T-1 : i-1:-1])
                    # most frequent value as label e.g. when t=3 [0,0,1] -> [0] as ground truth
                    self.label_list.append(max(self.old_label_list[i+self.T-1 : i-1:-1], key=data.get))
        elif cfg.MODEL.TYPE == "LSTM":
            # change shape (n, features) -> (n/t , t, features)
            # timestamp = 3
            self.T = 3
            # save old lists
            self.old_rssi_list = self.rssi_list.copy()
            self.old_label_list = self.label_list.copy()
            # new lists with shape (n/t, t, features)
            self.rssi_list = []
            self.label_list = []

            reverse = False
            # training direction (1->8) is not reversed, (8->1) is reversed
            self.total_length = self.total_length+1-self.T
            if not reverse :
                for i in range(self.total_length):
                    self.rssi_list.append(self.old_rssi_list[i:i+self.T])
                    data = Counter(self.old_label_list[i:i+self.T])
                    # most frequent value as label e.g. when t=3 [0,0,1] -> [0] as ground truth
                    self.label_list.append(max(self.old_label_list[i:i+self.T], key=data.get))
            else:
                for i in reversed(range(self.total_length)):
                    self.rssi_list.append(self.old_rssi_list[i+self.T-1 : i-1:-1])
                    data = Counter(self.old_label_list[i+self.T-1 : i-1:-1])
                    # most frequent value as label e.g. when t=3 [0,0,1] -> [0] as ground truth
                    self.label_list.append(max(self.old_label_list[i+self.T-1 : i-1:-1], key=data.get))
                

    def __getitem__(self, index):
        # 加上copy()是因為反方向時stride若為負會報錯，所以必須copy新的一份讓stride是正常。
        x = torch.tensor(self.rssi_list[index].copy(), dtype=torch.float32)
        y = torch.tensor(self.label_list[index].copy(), dtype=torch.long)

        return x, y


    def __len__(self):
        return self.total_length

class RSSI_DatasetForTest(data.Dataset):
    
    def __init__(self, cfg) -> None:
        super().__init__()

        # read each block csv and concat them 

        file_path = os.path.join(cfg.AFTER_DIR + 'test' + '.' + cfg.FILE_TYPE)
        df = pd.read_csv(file_path)
        self.no_label = cfg.TEST_NO_LABEL

        # discrete testing with label (approximately)
        if not self.no_label:
            self.label_list = df['label']
            self.rssi_list = df.drop(columns=['time', 'label'])
            self.total_length = len(self.label_list)
            self.label_list = self.label_list.to_numpy()
            self.rssi_list = self.rssi_list.to_numpy()
        # continuous testing , no label
        else:
            self.rssi_list = df.drop(columns=['time'])
            self.total_length = len(self.rssi_list)
            self.rssi_list = self.rssi_list.to_numpy()

        if cfg.MODEL.TYPE == "DNN":
            pass
        elif cfg.MODEL.TYPE == "1DCNN":
            # change shape (n, features) -> (n+1-t , t, features)
            # timestamp = 1
            self.T = 1
            # save old lists
            self.old_rssi_list = self.rssi_list.copy()
            self.old_label_list = self.label_list.copy()
            # new lists with shape (n+1-t, t, features)
            self.rssi_list = []
            self.label_list = []

            reverse = False
            # training direction (1->8) is not reversed, (8->1) is reversed
            self.total_length = self.total_length+1-self.T
            if not reverse :
                for i in range(self.total_length):
                    self.rssi_list.append(self.old_rssi_list[i:i+self.T])
                    data = Counter(self.old_label_list[i:i+self.T])
                    # most frequent value as label e.g. when t=3 [0,0,1] -> [0] as ground truth
                    self.label_list.append(max(self.old_label_list[i:i+self.T], key=data.get))
            else:
                for i in reversed(range(self.total_length)):
                    self.rssi_list.append(self.old_rssi_list[i+self.T-1 : i-1:-1])
                    data = Counter(self.old_label_list[i+self.T-1 : i-1:-1])
                    # most frequent value as label e.g. when t=3 [0,0,1] -> [0] as ground truth
                    self.label_list.append(max(self.old_label_list[i+self.T-1 : i-1:-1], key=data.get))

        elif cfg.MODEL.TYPE == "LSTM":
            # change shape (n, features) -> (n+1-t , t, features)
            # timestamp = 3
            self.T = 3
            # save old lists
            self.old_rssi_list = self.rssi_list.copy()
            # new lists with shape (n+1-t, t, features)
            self.rssi_list = []
            # testing with label (切割時段的方式)
            if not self.no_label:
                self.old_label_list = self.label_list.copy()
                self.label_list = []

                self.total_length = self.total_length+1-self.T

                for i in range(self.total_length):
                    self.rssi_list.append(self.old_rssi_list[i:i+self.T])
                    data = Counter(self.old_label_list[i:i+self.T])
                    # most frequent value as label e.g. when t=3 [0,0,1] -> [0] as ground truth
                    self.label_list.append(max(self.old_label_list[i:i+self.T], key=data.get))
            
            # testing without label (搭配影片的連續判斷方式)
            # 開頭必須補2秒的rssi list，因為模型input需要3個timestamp的data
            # rssi list為時間點t-2, t-1, t 的 rssi值組合
            else:
                self.rssi_list.append([self.old_rssi_list[0], self.old_rssi_list[0], self.old_rssi_list[0]])
                self.rssi_list.append([self.old_rssi_list[0], self.old_rssi_list[0], self.old_rssi_list[1]])

                #print([self.old_rssi_list[0], self.old_rssi_list[0], self.old_rssi_list[0]])
                for i in range(self.total_length+1-self.T):
                    self.rssi_list.append(self.old_rssi_list[i:i+self.T])
                #print(np.shape(self.rssi_list))
                #print(self.rssi_list)
      
        # remove duplicate label and calculate classes
        self.classes = len(list(dict.fromkeys(cfg.START_TIME)))

    def __getitem__(self, index):

        if not self.no_label:
            # MLP
            x = torch.tensor(self.rssi_list[index], dtype=torch.float32)
            y = torch.tensor(self.label_list[index], dtype=torch.long)
            return x, y
        else:
            # MLP
            x = torch.tensor(self.rssi_list[index], dtype=torch.float32)
            return x

    def __len__(self):
        return self.total_length