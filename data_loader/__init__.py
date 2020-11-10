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
    import yaml
    cfg = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

    load, _ = get_dataloader(cfg)
    for data in load:
        print(data)
        input()
