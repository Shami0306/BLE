import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DNN_Net(nn.Module):
    
    def __init__(self, cfg):
        super(DNN_Net, self).__init__()

        self.in_dimension = cfg.AP_NUMS
        self.out_dimension = 8

        # self.dnn = nn.Sequential(
        #     nn.LazyLinear(32),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(True),
        #     nn.Linear(32,16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(True),
        #     nn.Linear(16, self.out_dimension),
        # )
        self.dnn = nn.Sequential(
            nn.LazyLinear(32),
            nn.LazyBatchNorm1d(),
            nn.ReLU(True),
            nn.LazyLinear(16),
            nn.LazyBatchNorm1d(),
            nn.ReLU(True),
            nn.LazyLinear(self.out_dimension),
        )
    def forward(self, x):

        x = self.dnn(x)
        x = F.softmax(x, dim=1)
        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info(f'=> loading pretrained model {pretrained}')
            self.load_state_dict(pretrained_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError(f'{pretrained} is not exist!')

class OneDCNN_Net(nn.Module):
    
    def __init__(self, cfg):
        super(OneDCNN_Net, self).__init__()

        self.in_dimension = cfg.AP_NUMS
        self.out_dimension = 8

        self.cnn_1d = nn.Sequential(
            nn.LazyConv1d(32, kernel_size=3),
            nn.LazyBatchNorm1d(),
            nn.Flatten(),
            nn.LazyLinear(32),
            nn.LazyBatchNorm1d(),
            nn.ReLU(True),
            nn.LazyLinear(16),
            nn.LazyBatchNorm1d(),
            nn.ReLU(True),
            nn.LazyLinear(self.out_dimension)
        )

    def forward(self, x):

        x = self.cnn_1d(x)
        x = F.softmax(x, dim=1)
        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info(f'=> loading pretrained model {pretrained}')
            self.load_state_dict(pretrained_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError(f'{pretrained} is not exist!')

class LSTM_Net(nn.Module):
    
    def __init__(self, cfg):
        super(LSTM_Net, self).__init__()
        #  if 6 APs , in_dimension = 6
        self.in_dimension = cfg.AP_NUMS
        # 8 blocks
        self.out_dimension = 8

        self.lstm = nn.LSTM(self.in_dimension, 20)

        # self.classifier = nn.Sequential(
        #             nn.LazyLinear(256),
        #             nn.LazyBatchNorm1d(),
        #             nn.ReLU(),
        #             nn.Dropout(.2),
        #             nn.LazyLinear(64),
        #             nn.LazyBatchNorm1d(),
        #             nn.ReLU(),
        #             nn.Dropout(.2),
        #             nn.LazyLinear(self.out_dimension),  
        # )
        # lstm 1&2
        # self.classifier = nn.Sequential(
        #             nn.Linear(20, self.out_dimension),  
        # )
        # lstm 3
        # self.classifier = nn.Sequential(
        #                   nn.Linear(20, 16),
        #                   nn.BatchNorm1d(16),
        #                   nn.ReLU(),
        #                   nn.Linear(16, self.out_dimension),
        # )    
        # 
        # lstm 4
        # self.lstm = nn.LSTM(self.in_dimension, 50)
        # self.classifier = nn.Sequential(
        #                   nn.Linear(50, 16),
        #                   nn.BatchNorm1d(16),
        #                   nn.ReLU(),
        #                   nn.Linear(16, self.out_dimension),
        # )   

        # lstm 5
        self.classifier = nn.Sequential(
            nn.Linear(20, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, self.out_dimension),  
        )
        # lstm 6      
        # self.lstm2 = nn.LSTM(20, 64)
        # self.classifier = nn.Sequential(
        #     nn.Linear(64, 32),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Linear(32, 16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.Linear(16, self.out_dimension),  
        # )
    def forward(self, x):
        # output and last layer hidden state, cell state
        
        x = x.permute(1,0,2)
        x ,(hn,cn) = self.lstm(x)
        # lstm 6
        #x ,(hn,cn) = self.lstm2(x)
        x = self.classifier(x[-1])
        x = F.softmax(x, dim=1)
        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info(f'=> loading pretrained model {pretrained}')
            self.load_state_dict(pretrained_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError(f'{pretrained} is not exist!')


def get_model(cfg, is_train):
    if cfg.MODEL.TYPE == "DNN":
        model = DNN_Net(cfg)
    elif cfg.MODEL.TYPE == "1DCNN":
        model = OneDCNN_Net(cfg)
    elif cfg.MODEL.TYPE == "LSTM":
        model = LSTM_Net(cfg)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
    return model

