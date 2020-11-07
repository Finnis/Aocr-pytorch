import numpy as np
import torch
from torch.utils.data import DataLoader

from . import dataset


def get_dataloader(config: dict, distributed=False):
    _dataset = getattr(dataset, config['type'])(**config['args'])
    collect_fn = dataset.Collate(**config['img_size'])
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(_dataset)
        config['loader']['shuffle'] = False
        config['loader']['batch_size'] = config['loader']['batch_size'] // torch.cuda.device_count()
    loader = DataLoader(dataset=_dataset, sampler=sampler, collate_fn=collect_fn, **config['loader'])

    return loader, sampler


if __name__ == "__main__":
    cfg = {}
    cfg['type'] = 'LatinDataset'
    cfg['args'] = {'list_file': '../datasets/train_list.txt'}
    cfg['loader'] = {'batch_size': 4, 'pin_memory': False}

    load, _ = get_dataloader(cfg)
    for data in load:
        print(data)
        input()
