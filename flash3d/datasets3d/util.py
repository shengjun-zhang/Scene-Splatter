import torch
import logging
from torch.utils.data import DataLoader

from packaging.version import Version
from flash3d.datasets3d.nyu.dataset import NYUv2Dataset
from flash3d.datasets3d.kitti import KITTIDataset

def create_datasets(cfg,dataset):
    # datasets_dict = {
    #     "re10k": Re10KDataset,
    #     "nyuv2": NYUv2Dataset,
    #     "kitti": KITTIDataset,
    # }[cfg.dataset.name]

    # dataset = datasets_dict(cfg, split=split)
    
    shuffle =  False
    data_loader = DataLoader(
        dataset,
        cfg.data_loader.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data_loader.num_workers,
        pin_memory=True,
        drop_last=shuffle,
        collate_fn=custom_collate,
    )

    return data_loader


if Version(torch.__version__) < Version("1.11"):
    from torch.utils.data._utils.collate import default_collate
else:
    from torch.utils.data import default_collate


def custom_collate(batch):
    all_keys = batch[0].keys()
    dense_keys = [k for k in all_keys if "sparse" not in k[0]]
    sparse_keys = [k for k in all_keys if "sparse" in k[0]]
    dense_batch = [{k: b[k] for k in dense_keys} for b in batch]
    sparse_batch = {k: [b[k] for b in batch] for k in sparse_keys}
    dense_batch = default_collate(dense_batch)
    batch = sparse_batch | dense_batch
    return batch