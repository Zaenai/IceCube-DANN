"""
Domain adaptation dataset for IceCube.

Loads MC and real data from separate directories:
  - MC   : config['data']['mc_path']   → label 0
  - Real : config['data']['real_path'] → label 1

Usage
-----
    from iceaggr.data import make_domain_dataloader
    loader, sampler = make_domain_dataloader(config, geometry)

Config example
--------------
    data:
      mc_path:   /path/to/mc/
      real_path: /path/to/real/
      max_events: 5000000
      val_events: 2000
      ...
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader, Dataset

from . import (
    BatchAwareSampler,
    GeometryLoader,
    IceCubeDataset,
    make_collate_flat,
)

# ──────────────────────────────────────────────
# Combined dataset
# ──────────────────────────────────────────────

class CombinedDomainDataset(Dataset):
    """
    Combines two IceCubeDataset instances (MC and real) into one dataset.
    Each item carries a 'domain_label': 0 = MC, 1 = real.

    MC events occupy indices 0..n_mc-1, real events n_mc..n_mc+n_real-1.
    """

    def __init__(
        self,
        mc_dataset:   IceCubeDataset,
        real_dataset: IceCubeDataset,
    ):
        self._mc   = mc_dataset
        self._real = real_dataset
        self._n_mc   = len(mc_dataset)
        self._n_real = len(real_dataset)
        self._metadata = pa.concat_tables([
            mc_dataset.metadata,
            real_dataset.metadata,
        ])
    
    def __len__(self) -> int:
        return self._n_mc + self._n_real

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < self._n_mc:
            item = self._mc[idx]
            item['domain_label'] = torch.tensor(0, dtype=torch.long)
        else:
            item = self._real[idx - self._n_mc]
            item['domain_label'] = torch.tensor(1, dtype=torch.long)
        return item

    @property
    def metadata(self):
        return self._metadata


# ──────────────────────────────────────────────
# Collate with domain labels
# ──────────────────────────────────────────────

def make_collate_flat_domain(
    geometry: GeometryLoader,
    max_pulses_per_dom: int = 84,
    max_doms: int = 128,
):
    """
    Wraps make_collate_flat to additionally stack 'domain_label' tensors.

    Returns a collate_fn producing all standard flat-transformer keys plus:
        domain_labels : (batch_size,) LongTensor  0=MC, 1=real
    """
    base_collate = make_collate_flat(geometry, max_pulses_per_dom, max_doms)

    def collate_fn(batch):
        domain_labels = torch.stack([item['domain_label'] for item in batch])
        clean_batch   = [{k: v for k, v in item.items() if k != 'domain_label'}
                         for item in batch]
        result = base_collate(clean_batch)
        result['domain_labels'] = domain_labels
        return result

    return collate_fn


# ──────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────

def make_domain_dataloader(
    config:      dict,
    geometry:    GeometryLoader,
    split:       str           = 'train',
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, BatchAwareSampler]:
    """
    Build a joint MC + Real DataLoader for domain-adversarial training.

    Expects config['data'] to have:
        mc_path   : path to MC parquet files
        real_path : path to real parquet files
        max_events: total events (split evenly between MC and real)

    Returns
    -------
    loader, sampler
    """
    if num_workers is None:
        num_workers = config['data']['num_workers']

    max_events_cfg = (
        config['data']['max_events'] if split == 'train'
        else config['data'].get('val_events', 50000)
    )
    max_per_domain = max_events_cfg // 2
    min_pulses     = config['data'].get('min_pulses')

    batch_range_key = 'train_batches' if split == 'train' else 'val_batches'
    raw_range       = config['data'].get(batch_range_key)
    batch_range     = tuple(raw_range) if raw_range is not None else None

    mc_range   = config['data'].get('mc_train_batches')
    real_range = config['data'].get('real_train_batches')
    mc_batch_range   = tuple(mc_range)   if mc_range   is not None else None
    real_batch_range = tuple(real_range) if real_range is not None else None

    mc_config_path   = config['data']['mc_config_path']
    real_config_path = config['data']['real_config_path']

    mc_dataset = IceCubeDataset(
        config_path=mc_config_path,
        split=split,
        max_events=max_per_domain,
        cache_size=1,
        batch_range=mc_batch_range,
        min_pulses=min_pulses,
    )
    real_dataset = IceCubeDataset(
        config_path=real_config_path,
        split=split,
        max_events=max_per_domain,
        cache_size=1,
        batch_range=real_batch_range,
        min_pulses=min_pulses,
    )
    combined = CombinedDomainDataset(mc_dataset, real_dataset)
    sampler  = BatchAwareSampler(combined.metadata)

    collate_fn = make_collate_flat_domain(
        geometry,
        max_pulses_per_dom=config['model']['max_pulses_per_dom'],
        max_doms=config['model']['max_doms'],
    )

    loader = DataLoader(
        combined,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return loader, sampler