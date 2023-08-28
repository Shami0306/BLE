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

        region = [0] # 用來作為分區的索引(LSTM or 1DCNN T=3)
        # read each block csv and concat them 
        file_path = os.path.join(cfg.AFTER_DIR + 'block_' + str(1) + '.' + cfg.FILE_TYPE)
        df = pd.read_csv(file_path)
        self.total_length = df.shape[0] # set each block length same
        #df = df[:self.total_length]
        region.append(df.shape[0]+region[-1])
        self.label_list = df['label']
        self.rssi_list = df.drop(columns=['time', 'label'])

        for i in range(1, len(cfg.START_TIME)):
            file_path = os.path.join(cfg.AFTER_DIR + 'block_' + str(i+1) + '.' + cfg.FILE_TYPE)
            df = pd.read_csv(file_path)
            #df = df[:self.total_length]
            region.append(df.shape[0]+region[-1])

            self.label_list = pd.concat([self.label_list, df['label']], ignore_index=True)
            df = df.drop(columns=['time', 'label'])
            self.rssi_list = pd.concat([self.rssi_list, df], ignore_index=True)
 
        # 隨機將rssi設為0，模擬故障情況
        fault = 0.3
        for i in range(len(self.rssi_list)):
            if np.random.rand() < fault:
                cols = ['sniffer_zero','sniffer_one', 'sniffer_two', 'sniffer_three']
                col = np.random.choice(cols, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
                self.rssi_list.loc[i, col] = 0
 
        self.total_length = len(self.rssi_list)
        self.label_list = self.label_list.to_numpy()
        self.rssi_list = self.rssi_list.to_numpy()
        self.classes = len(cfg.START_TIME) 
        if cfg.MODEL.TYPE == "DNN":
            pass
        elif cfg.MODEL.TYPE == "1DCNN":
            # change shape (n, features) -> (n/t , t, features)
            # timestamp = 1
            # self.T = 1
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

            # reverse = False
            # # training direction (1->8) is not reversed, (8->1) is reversed
            # self.total_length = self.total_length+1-self.T
            # if not reverse :
            #     for i in range(self.total_length):
            #         self.rssi_list.append(self.old_rssi_list[i:i+self.T])
            #         data = Counter(self.old_label_list[i:i+self.T])
            #         # most frequent value as label e.g. when t=3 [0,0,1] -> [0] as ground truth
            #         self.label_list.append(max(self.old_label_list[i:i+self.T], key=data.get))
            # else:
            #     for i in reversed(range(self.total_length)):
            #         self.rssi_list.append(self.old_rssi_list[i+self.T-1 : i-1:-1])
            #         data = Counter(self.old_label_list[i+self.T-1 : i-1:-1])
            #         # most frequent value as label e.g. when t=3 [0,0,1] -> [0] as ground truth
            #         self.label_list.append(max(self.old_label_list[i+self.T-1 : i-1:-1], key=data.get))

            # ----- train lstm version 2-----
            # 兩個區間(left,right)各自挑選rssi形成T秒內的rssi組合
            # 例如T=3，選到0,1區間，則從label為0的data中隨機挑選0~3筆資料，若挑選2筆，則剩餘1筆從label為1的data中再挑選一筆出來
            # 而若label為0的data挑選到3筆，則不挑選label為1的data。透過此方式來讓LSTM模型學習到T秒內位於同個位置或是移動至下個位置的趨勢
            for i in range(self.classes-1): # label有8個，相鄰區間為7種
                for _ in range(region[-1]):
                    left_len = random.choice((0,1,2,3))
                    right_len = self.T - left_len
                    data = []
                    label = []
                    for _ in range(left_len):
                        index = random.randint(region[i], region[i+1]-1)
                        data.append(self.old_rssi_list[index])
                        label.append(self.old_label_list[index])
                    for _ in range(right_len):
                        index = random.randint(region[i+1], region[i+2]-1)
                        data.append(self.old_rssi_list[index])
                        label.append(self.old_label_list[index])
                    
                    self.rssi_list.append(data)
                    label_count = Counter(label)
                    self.label_list.append(max(label, key=label_count.get))

            self.total_length = len(self.rssi_list)
                
    def __getitem__(self, index):
        x = torch.tensor(self.rssi_list[index], dtype=torch.float32).to(0)
        y = torch.tensor(self.label_list[index], dtype=torch.long).to(0)
        return x, y

    def __len__(self):
        return self.total_length

class RSSI_DatasetForTest(data.Dataset):
    
    def __init__(self, cfg) -> None:
        super().__init__()

        # read each block csv and concat them 

        file_path = os.path.join(cfg.AFTER_DIR + 'test' + '.' + cfg.FILE_TYPE)
        df = pd.read_csv(file_path)
        self.for_video = cfg.TEST_FOR_VIDEO

        # discrete testing with label (approximately)
        if not self.for_video:
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
            # self.T = 1
            self.T = 3
            # save old lists
            self.old_rssi_list = self.rssi_list.copy()
            # new lists with shape (n+1-t, t, features)
            self.rssi_list = []
            # testing with label (切割時段的方式)
            if not self.for_video:
                self.old_label_list = self.label_list.copy()
                self.label_list = []

                self.total_length = self.total_length+1-self.T

                for i in range(self.total_length):
                    self.rssi_list.append(self.old_rssi_list[i:i+self.T])
                    data = Counter(self.old_label_list[i:i+self.T])
                    # most frequent value as label e.g. when t=3 [0,0,1] -> [0] as ground truth
                    self.label_list.append(max(self.old_label_list[i:i+self.T], key=data.get))
            
            else:
            # testing without label (搭配影片的連續判斷方式)
            # 開頭補2秒的rssi list，因為模型input需要3個timestamp的data
            # rssi list為時間點t-2, t-1, t 的 rssi值組合
                self.rssi_list.append([self.old_rssi_list[0], self.old_rssi_list[0], self.old_rssi_list[0]])
                self.rssi_list.append([self.old_rssi_list[0], self.old_rssi_list[0], self.old_rssi_list[1]])

                for i in range(self.total_length+1-self.T):
                    self.rssi_list.append(self.old_rssi_list[i:i+self.T])

        elif cfg.MODEL.TYPE == "LSTM":
            # change shape (n, features) -> (n+1-t , t, features)
            # timestamp = 3
            self.T = 3
            # save old lists
            self.old_rssi_list = self.rssi_list.copy()
            # new lists with shape (n+1-t, t, features)
            self.rssi_list = []
            # testing with label (切割時段的方式)
            if not self.for_video:
                self.old_label_list = self.label_list.copy()
                self.label_list = []

                self.total_length = self.total_length+1-self.T

                for i in range(self.total_length):
                    self.rssi_list.append(self.old_rssi_list[i:i+self.T])
                    data = Counter(self.old_label_list[i:i+self.T])
                    # most frequent value as label e.g. when t=3 [0,0,1] -> [0] as ground truth
                    self.label_list.append(max(self.old_label_list[i:i+self.T], key=data.get))
            
            else:
            # testing without label (搭配影片的連續判斷方式)
            # 開頭補2秒的rssi list，因為模型input需要3個timestamp的data
            # rssi list為時間點t-2, t-1, t 的 rssi值組合
                self.rssi_list.append([self.old_rssi_list[0], self.old_rssi_list[0], self.old_rssi_list[0]])
                self.rssi_list.append([self.old_rssi_list[0], self.old_rssi_list[0], self.old_rssi_list[1]])

                for i in range(self.total_length+1-self.T):
                    self.rssi_list.append(self.old_rssi_list[i:i+self.T])
            

      
    def __getitem__(self, index):

        if not self.for_video:
            x = torch.tensor(self.rssi_list[index], dtype=torch.float32)
            y = torch.tensor(self.label_list[index], dtype=torch.long)
            return x, y
        else:
            x = torch.tensor(self.rssi_list[index], dtype=torch.float32)
            return x

    def __len__(self):
        return self.total_length
    
class RSSI_DatasetForMultiTest(data.Dataset): # Only for evaluating mde
    
    def __init__(self, cfg) -> None:
        super().__init__()

        # read each block csv and concat them 
        # test_list = ['./after/U19e_outdoor0517test1_1/',
        #              './after/U19e_outdoor0517test1_2/',
        #              './after/U19e_outdoor0517test1_3/',
        #              './after/U19e_outdoor0517test1_4/',
        #              './after/U19e_outdoor0517test1_5/'
        #              ]
        test_list = ['./after/U19e_outdoor0517test2_1/',
                     './after/U19e_outdoor0517test2_3/',
                     './after/U19e_outdoor0517test2_4/',
                     './after/U19e_outdoor0517test2_5/'
                     ]
        # test_list = ['./after/Sharp_outdoor0517test2_1/',
        #              './after/Sharp_outdoor0517test2_2/',
        #              './after/Sharp_outdoor0517test2_3/',
        #              './after/Sharp_outdoor0517test2_4/',
        #              './after/Sharp_outdoor0517test2_5/',
        #             ]   
        # test_list = ['./after/Sharp_outdoor0517test1_1/',
        #              './after/Sharp_outdoor0517test1_2/',
        #              './after/Sharp_outdoor0517test1_3/',
        #              './after/Sharp_outdoor0517test1_4/',
        #             ]           
        self.for_video = cfg.TEST_FOR_VIDEO
        for test_i,test in enumerate(test_list):
            file_path = os.path.join(test + 'test' + '.' + cfg.FILE_TYPE)
            df = pd.read_csv(file_path)

            # discrete testing with label (approximately)
            self.label_list = df['label']
            self.rssi_list = df.drop(columns=['time', 'label'])
            self.total_length = len(self.label_list)
            self.label_list = self.label_list.to_numpy()
            self.rssi_list = self.rssi_list.to_numpy()

            if cfg.MODEL.TYPE == "DNN":
                pass
            elif cfg.MODEL.TYPE == "1DCNN":
                # change shape (n, features) -> (n+1-t , t, features)
                # timestamp = 1
                # self.T = 1
                self.T = 3
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
                if not self.for_video:
                    self.old_label_list = self.label_list.copy()
                    self.label_list = []

                    self.total_length = self.total_length+1-self.T

                    for i in range(self.total_length):
                        self.rssi_list.append(self.old_rssi_list[i:i+self.T])
                        data = Counter(self.old_label_list[i:i+self.T])
                        # most frequent value as label e.g. when t=3 [0,0,1] -> [0] as ground truth
                        self.label_list.append(max(self.old_label_list[i:i+self.T], key=data.get))
                
                else:
                # testing without label (搭配影片的連續判斷方式)
                # 開頭補2秒的rssi list，因為模型input需要3個timestamp的data
                # rssi list為時間點t-2, t-1, t 的 rssi值組合
                    self.rssi_list.append([self.old_rssi_list[0], self.old_rssi_list[0], self.old_rssi_list[0]])
                    self.rssi_list.append([self.old_rssi_list[0], self.old_rssi_list[0], self.old_rssi_list[1]])

                    for i in range(self.total_length+1-self.T):
                        self.rssi_list.append(self.old_rssi_list[i:i+self.T])
            if test_i==0:
                self.multirssi_list = self.rssi_list
                self.multilabel_list = self.label_list
                self.muilttotal_length = self.total_length
            else:
                self.multirssi_list = np.concatenate((self.multirssi_list, self.rssi_list))
                self.multilabel_list = np.concatenate((self.multilabel_list, self.label_list))
                self.muilttotal_length += self.total_length

        self.rssi_list = self.multirssi_list
        self.label_list = self.multilabel_list
        self.total_length = self.muilttotal_length

    def __getitem__(self, index):

        if not self.for_video:
            x = torch.tensor(self.rssi_list[index], dtype=torch.float32)
            y = torch.tensor(self.label_list[index], dtype=torch.long)
            return x, y
        else:
            x = torch.tensor(self.rssi_list[index], dtype=torch.float32)
            return x

    def __len__(self):
        return self.total_length