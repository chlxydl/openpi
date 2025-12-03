# lerobot/agibot_utils/augmented_dataset.py
from functools import partial
from typing import Callable, List, Dict, Optional
import torch

class AugmentedDataset(torch.utils.data.Dataset):
    """
    Wrapper around existing Dataset, supports a config-style augmentations list.

    augmentations: list of dicts, each element:
        {"type": callable, "params": {...}} or {"type": "name", "params": {...}}
    If augmentation items are callables, they will be partialed with params.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        augmentations: Optional[List[Dict]] = None,
        enable_augment: bool = True,
    ):
        self.dataset = dataset
        self.enable_augment = enable_augment
        self._augs: List[Callable] = []

        if augmentations:
            for aug in augmentations:
                if callable(aug):
                    # already a callable, wrap directly (assume signature fn(sample))
                    self._augs.append(aug)
                elif isinstance(aug, dict):
                    fn = aug["type"]
                    params = aug.get("params", {})
                    if not callable(fn):
                        raise ValueError("augmentation 'type' must be callable")
                    self._augs.append(partial(fn, **params))
                else:
                    raise ValueError("augmentation must be callable or dict with 'type' and optional 'params'")

    def __len__(self):
        return len(self.dataset)

    # def __getitem__(self, idx):
    #     sample = self.dataset[idx]
    #     if self.enable_augment:
    #         for fn in self._augs:
    #             sample = fn(sample)
    #             # if sample is None:
    #             #     print(f"[WARN] Augmentation {fn} returned None for idx={idx}")
    #     return sample

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if sample is None:
            print(f"[WARN] Dataset returned None for idx={idx}")
            return {}  # 或者返回空 sample
        if self.enable_augment:
            for fn in self._augs:
                sample = fn(sample)
                if sample is None:
                    print(f"[WARN] Augmentation {fn} returned None for idx={idx}")
        return sample

    def __getattr__(self, name):
        # transparent access to wrapped dataset attributes (meta, episodes, etc.)
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(f"'AugmentedDataset' object has no attribute '{name}'")
