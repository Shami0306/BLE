import os
import logging
import time
import torch
import numpy as np
from pathlib import Path

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    # get name of the yaml
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)
    

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = f'{cfg_name}_{time_str}_{phase}.log'
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    # get root logger
    logger = logging.getLogger()
    # set level as info to avoid debug level message , default is warning level
    logger.setLevel(logging.INFO)
    # sends logging output to streams such as sys.stdout, sys.stderr or any file-like object
    console = logging.StreamHandler()
    # get root logger and add a handler
    logging.getLogger('').addHandler(console)

    return logger, str(final_output_dir)

def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))

    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))

def get_angle(v1,v2):
    x = np.array(v1)
    y = np.array(v2)

    # 計算兩個向量的模：
    module_x = np.sqrt(x.dot(x))
    module_y = np.sqrt(y.dot(y))

    # 計算兩個向量的點積
    dot_value = x.dot(y)

    # 計算夾角的cos值：
    cos_theta = dot_value/(module_x*module_y)

    # 求弧度：
    angle_radian = np.arccos(cos_theta)

    # 轉換為角度：
    angle_value = angle_radian*180/np.pi
    return angle_value

def check_side(a, b, p):
    side = (p[0]-a[0])*(b[1]-a[1]) - (p[1]-a[1])*(b[0]-a[0])
    return (-1 if side < 0 else 1)

def check_in_range(start, end, angle):
    # 確認目標角度angle是否在start與end之間
    # 角度以逆時針為方向增加
    start = start%360
    end = end%360
    if end > start:
        return (angle > start) and (angle < end)
    if end < start:
        return not ((angle >= end) and (angle <= start))
    return (start == angle)

def check_in_range_eq(start, end, angle):
    # 確認目標角度angle是否在start與end之間
    # 角度以逆時針為方向增加
    start = start%360
    end = end%360
    if end > start:
        return (angle >= start) and (angle <= end)
    if end < start:
        return not ((angle > end) and (angle < start))
    return (start == angle)

def get_acceptable_range(ac_range, side, theta_tp):
    # 修正acceptable range
    ac_range[0] += theta_tp
    ac_range[1] += theta_tp
    # 在右側
    if side > 0:
        return ac_range
    else:
        return ac_range[::-1]

def get_corrected_direction(ac_range, user_angle):
    corrected_angle = 0
    # 若用戶的朝向位於acceptable range中，表示為Situation 3但走的方向正確
    if check_in_range_eq(ac_range[0], ac_range[1], user_angle):
        return 0, corrected_angle
    # 其餘偏離的情況，用戶的朝向位於rejectable range中
    # 取rejectable range的中間值作為區別往左或往右的邊界
    reject_range = ac_range[::-1] # 因為起點到終點的方向是逆時針，所以包含的範圍會不一樣

    if reject_range[1] < reject_range[0]:
        mid = reject_range[0] + (360-(reject_range[0]-reject_range[1]))/2
    else:
        mid = (reject_range[0] + reject_range[1])/2
    mid %= 360
    if check_in_range(reject_range[0], mid, user_angle):
        corrected_angle = min(abs(user_angle - reject_range[0]), 360 - abs(user_angle - reject_range[0]))
        return -1, corrected_angle
    else:
        corrected_angle = min(abs(reject_range[1] - user_angle), 360 - abs(reject_range[1] - user_angle))
        return 1, corrected_angle

