import torch
import datetime
import argparse
from utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default=None, required=True, help='config path')
    # parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distrubuted training')
    return parser.parse_args()

def debug():
    import json
    with open('data/didemo/train_data.json') as fp:
        data = json.load(fp)
        for k, v in data.items():
            print(v.keys())
        exit(0)


def main(args):
    import logging
    import numpy as np
    import random
    import torch
    from runners import MainRunner

    seed = 8
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    # logging.info('base seed {}'.format(seed))
    config = load_json(args.config_path)
    # print(config)

    runner = MainRunner(config)
    runner.train()


if __name__ == '__main__':
    args = parse_args()
    # torch.distributed.init_process_group('nccl', init_method='env://', timeout=datetime.timedelta(seconds=1800))
    # torch.cuda.set_device(args.local_rank)    
    main(args)
