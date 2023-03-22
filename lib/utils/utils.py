import os
import logging
import time
import torch
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