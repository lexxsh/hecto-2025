from abc import abstractmethod
from typing import Callable, Optional, Literal

import lightning as pl
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.utils.logger import print
from src.dataset.balanced_sampler import (
    create_weighted_sampler,
    create_source_balanced_sampler,
    ClassBalancedSampler,
    StratifiedBatchSampler,
)


class BaseDataset(Dataset):
    def __init__(
        self,
        files: list[str],
        labels: list[int],
        preprocess: None | Callable = None,
        augmentations: None | Callable = None,
        shuffle: bool = False,  # Shuffles the dataset once
        dataset2files: Optional[dict[str, list[str]]] = None,
        sources: Optional[list[str]] = None,  # 소스 정보 추가
    ):
        self.files = files
        self.labels = labels
        self.sources = sources  # 각 샘플의 소스 (e.g., "SiT", "DiT", "real")

        self.preprocess = preprocess
        self.augmentations = augmentations

        self.dataset2files = dataset2files

        if shuffle:
            self.shuffle()

    def shuffle(self):
        # create fixed seed for reproducibility
        idx = np.random.RandomState(42).permutation(len(self.files))
        self.files = [self.files[i] for i in idx]
        self.labels = [self.labels[i] for i in idx]
        if self.sources is not None:
            self.sources = [self.sources[i] for i in idx]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path)
        if self.augmentations is not None:
            image = self.augmentations(image)
        if self.preprocess is not None:
            image = self.preprocess(image)
        return {
            "image": image,
            "label": self.labels[idx],
            "path": path,
        }

    def print_statistics(self):
        print(f"Number of samples: {len(self.files)}")
        unique, counts = np.unique(self.labels, return_counts=True)
        print("Class distribution")
        names = self.get_class_names()
        for u, c in zip(unique, counts):
            print(f"Class {u} ({names[u]}): {c}")

    @abstractmethod
    def get_class_names(self) -> dict[int, str]:
        raise NotImplementedError


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        config: Config, 
        preprocess: None | Callable = None,
        use_balanced_sampling: bool = False,
        sampling_strategy: Literal["weighted", "class_balanced", "stratified", "source_balanced"] = "weighted",
    ):
        """
        Args:
            config: Configuration object
            preprocess: Preprocessing function
            use_balanced_sampling: Whether to use balanced sampling for training
            sampling_strategy: Strategy for balanced sampling
                - "weighted": WeightedRandomSampler (Real/Fake 균형)
                - "class_balanced": ClassBalancedSampler (Real/Fake 균형)
                - "stratified": StratifiedBatchSampler (Real/Fake 균형)
                - "source_balanced": WeightedRandomSampler (소스별 균형) ⭐ NEW
        """
        super().__init__()
        self.config = config
        self.preprocess = preprocess
        self.use_balanced_sampling = use_balanced_sampling
        self.sampling_strategy = sampling_strategy

    def train_dataloader(self):
        # Use balanced sampling if enabled
        if self.use_balanced_sampling:
            
            if self.sampling_strategy == "source_balanced":
                # 소스별 균형 샘플링 (NEW)
                if not hasattr(self.train_dataset, 'sources') or self.train_dataset.sources is None:
                    raise ValueError("source_balanced requires dataset with source information")
                
                sources = self.train_dataset.sources
                sampler = create_source_balanced_sampler(
                    sources,
                    strategy="balanced",
                    num_samples=len(sources)
                )
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.config.mini_batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                    sampler=sampler,
                )
            
            # 기존 클래스별 샘플링
            labels = self.train_dataset.labels
            
            if self.sampling_strategy == "weighted":
                sampler = create_weighted_sampler(
                    labels, 
                    strategy="balanced",
                    num_samples=len(labels)  # Same number of samples per epoch
                )
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.config.mini_batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                    sampler=sampler,  # Use sampler instead of shuffle
                )
            
            elif self.sampling_strategy == "class_balanced":
                # Calculate samples per class (use minimum to avoid over-sampling too much)
                from collections import Counter
                label_counts = Counter(labels)
                min_count = min(label_counts.values())
                max_count = max(label_counts.values())
                # Use average to balance between min and max
                samples_per_class = int((min_count + max_count) / 2)
                
                sampler = ClassBalancedSampler(labels, samples_per_class=samples_per_class)
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.config.mini_batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                    sampler=sampler,
                )
            
            elif self.sampling_strategy == "stratified":
                batch_sampler = StratifiedBatchSampler(
                    labels, 
                    batch_size=self.config.mini_batch_size,
                    drop_last=True
                )
                return DataLoader(
                    self.train_dataset,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                    batch_sampler=batch_sampler,
                )
            
            else:
                raise ValueError(f"Unknown sampling_strategy: {self.sampling_strategy}")
        
        # Default: standard random sampling
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.mini_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.mini_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.mini_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
