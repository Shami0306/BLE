import os
from yacs.config import CfgNode as CN

_CFG = CN()

#BLE
_CFG.UUID = '0000f22e-0000-1000-8000-00805f9b34fb'
_CFG.DEVICE_NAME = 'MCSLAB'
_CFG.MAC = '4d:cf:4d:c0:11:42'
_CFG.START_TIME = []
_CFG.END_TIME = []
_CFG.LABEL_LIST = []
_CFG.BEFORE_DIR = './before/'
_CFG.AFTER_DIR = './after/'
_CFG.OUTPUT_DIR = 'output'
_CFG.FILE = 'sniffer_'
_CFG.FILE_TYPE = 'csv'
_CFG.OUTPUT_NAME = ''
_CFG.AP_NUMS = 4
_CFG.AP_NAME = ['zero', 'one', 'two', 'three']
_CFG.TEST_NO_LABEL = False
_CFG.TEST_VIDEO_PATH = ''

#BLE
_CFG.MODEL = CN()
_CFG.MODEL.TYPE = 'DNN'
_CFG.MODEL.INIT_WEIGHTS = True
_CFG.MODEL.PRETRAINED = ''
#MEBOW
_CFG.MODEL.USE_FEATUREMAP = True
_CFG.MODEL.NAME = 'pose_hrnet'
_CFG.MODEL.NUM_JOINTS = 17
_CFG.MODEL.MEBOW_PRETRAINED = 'models/pose_hrnet_w32_256x192.pth'
_CFG.MODEL.TARGET_TYPE = 'gaussian'
_CFG.MODEL.IMAGE_SIZE = [192,256]
_CFG.MODEL.HEATMAP_SIZE = []
_CFG.MODEL.SIGMA = 2

_CFG.MODEL.EXTRA = CN()
_CFG.MODEL.EXTRA.PRETRAINED_LAYERS = []
_CFG.MODEL.EXTRA.FINAL_CONV_KERNEL = 1

_CFG.MODEL.EXTRA.STAGE2 = CN()
_CFG.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
_CFG.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
_CFG.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'

_CFG.MODEL.EXTRA.STAGE2.NUM_BLOCKS = []
_CFG.MODEL.EXTRA.STAGE2.NUM_CHANNELS = []
_CFG.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

_CFG.MODEL.EXTRA.STAGE3 = CN()
_CFG.MODEL.EXTRA.STAGE3.NUM_MODULES = 4
_CFG.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
_CFG.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'

_CFG.MODEL.EXTRA.STAGE3.NUM_BLOCKS = []
_CFG.MODEL.EXTRA.STAGE3.NUM_CHANNELS = []
_CFG.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

_CFG.MODEL.EXTRA.STAGE4 = CN()
_CFG.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
_CFG.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
_CFG.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'

_CFG.MODEL.EXTRA.STAGE4.NUM_BLOCKS = []
_CFG.MODEL.EXTRA.STAGE4.NUM_CHANNELS = []
_CFG.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'

#BLE
_CFG.TRAIN = CN()
_CFG.TRAIN.BATCH_SIZE = 32
_CFG.TRAIN.SHUFFLE = True
_CFG.TRAIN.BEGIN_EPOCH = 0
_CFG.TRAIN.END_EPOCH = 20
_CFG.TRAIN.OPTIMIZER = 'adam'
_CFG.TRAIN.LR = 0.001
_CFG.TRAIN.LR_FACTOR = 0.1

_CFG.TEST = CN()
_CFG.TEST.RSSI_LIST = [ [-48, -103], [-52, -95], [-30, -97], [-48, -91]]

def update_config(cfg, args):
    # 變為可更改狀態
    cfg.defrost()
    # 依照對應config檔更新內容
    cfg.merge_from_file(args.cfg)

    if args.beforeDir:
        cfg.BEFORE_DIR = args.beforeDir
    elif os.path.basename(os.getcwd()) != 'BLE':
        cfg.BEFORE_DIR = 'BLE/' + cfg.BEFORE_DIR

    if args.afterDir:
        cfg.AFTER_DIR = args.afterDir
    elif os.path.basename(os.getcwd()) != 'BLE':
        cfg.AFTER_DIR = 'BLE/' + cfg.AFTER_DIR
    # 變為不可更改狀態
    cfg.freeze()

if __name__ == '__main__':
    import sys
    # write file to xxxx.yaml
    with open(sys.argv[1], 'w') as f:
        print(_CFG, file=f)