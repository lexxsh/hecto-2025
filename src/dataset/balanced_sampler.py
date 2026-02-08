"""
Balanced sampling strategies for imbalanced datasets.

Provides various sampling methods to handle class imbalance:
1. WeightedRandomSampler: Sample-level weighting
2. ClassBalancedSampler: Ensures equal samples per class
3. StratifiedBatchSampler: Balanced batches
4. SourceBalancedSampler: Balance by generation source (NEW)
"""

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler
from collections import Counter


def get_source_weights(sources: list[str], strategy: str = "balanced") -> torch.Tensor:
    """
    Calculate source-level weights for balanced sampling.
    
    각 생성 모델(소스)별로 가중치를 계산합니다.
    예: SiT (31,885장) → 낮은 가중치, MidJourney (630장) → 높은 가중치
    
    Args:
        sources: List of source names (e.g., ["SiT", "DiT", "real", ...])
        strategy: 'balanced' or 'inverse_freq'
    
    Returns:
        Tensor of weights for each sample
    """
    source_counts = Counter(sources)
    n_samples = len(sources)
    n_sources = len(source_counts)
    
    if strategy == "balanced":
        # 소스별 균형 가중치
        source_weights = {
            source: n_samples / (n_sources * count) 
            for source, count in source_counts.items()
        }
    elif strategy == "inverse_freq":
        # 역빈도 가중치
        source_weights = {
            source: 1.0 / count 
            for source, count in source_counts.items()
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Map source weights to sample weights
    sample_weights = torch.tensor([source_weights[source] for source in sources])
    
    print(f"\n{'='*60}")
    print(f"Source-Level Weighting Strategy: {strategy}")
    print(f"{'='*60}")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        weight = source_weights[source]
        print(f"{source:20s}: {count:6d} samples | weight: {weight:.4f}")
    print(f"{'='*60}\n")
    
    return sample_weights


def create_source_balanced_sampler(sources: list[str], strategy: str = "balanced", 
                                   num_samples: int = None) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for source-balanced sampling.
    
    각 생성 모델(소스)별로 균형을 맞춥니다.
    
    Args:
        sources: List of source names
        strategy: Weighting strategy ('balanced' or 'inverse_freq')
        num_samples: Number of samples to draw per epoch (default: len(sources))
    
    Returns:
        WeightedRandomSampler
    """
    sample_weights = get_source_weights(sources, strategy)
    
    if num_samples is None:
        num_samples = len(sources)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )
    
    return sampler


def get_class_weights(labels: list[int], strategy: str = "balanced") -> torch.Tensor:
    """
    Calculate class weights for balanced sampling.
    
    Args:
        labels: List of class labels
        strategy: 'balanced' or 'inverse_freq'
            - balanced: weight = n_samples / (n_classes * n_samples_per_class)
            - inverse_freq: weight = 1 / n_samples_per_class
    
    Returns:
        Tensor of weights for each sample
    """
    label_counts = Counter(labels)
    n_samples = len(labels)
    n_classes = len(label_counts)
    
    if strategy == "balanced":
        # Sklearn-style balanced weights
        class_weights = {
            cls: n_samples / (n_classes * count) 
            for cls, count in label_counts.items()
        }
    elif strategy == "inverse_freq":
        # Inverse frequency
        class_weights = {
            cls: 1.0 / count 
            for cls, count in label_counts.items()
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Map class weights to sample weights
    sample_weights = torch.tensor([class_weights[label] for label in labels])
    
    print(f"\n{'='*60}")
    print(f"Class Weighting Strategy: {strategy}")
    print(f"{'='*60}")
    for cls, count in sorted(label_counts.items()):
        weight = class_weights[cls]
        print(f"Class {cls}: {count:6d} samples | weight: {weight:.4f}")
    print(f"{'='*60}\n")
    
    return sample_weights


def create_weighted_sampler(labels: list[int], strategy: str = "balanced", 
                           num_samples: int = None) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for balanced sampling.
    
    Args:
        labels: List of class labels
        strategy: Weighting strategy ('balanced' or 'inverse_freq')
        num_samples: Number of samples to draw per epoch (default: len(labels))
    
    Returns:
        WeightedRandomSampler
    """
    sample_weights = get_class_weights(labels, strategy)
    
    if num_samples is None:
        num_samples = len(labels)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True  # Allow sampling same image multiple times
    )
    
    return sampler


class ClassBalancedSampler(Sampler):
    """
    Samples equal number of instances from each class per epoch.
    
    - Over-samples minority classes
    - Under-samples majority classes
    """
    
    def __init__(self, labels: list[int], samples_per_class: int = None):
        """
        Args:
            labels: List of class labels
            samples_per_class: Number of samples per class (default: min class size)
        """
        self.labels = np.array(labels)
        self.class_indices = {}
        
        # Group indices by class
        for cls in np.unique(labels):
            self.class_indices[cls] = np.where(self.labels == cls)[0]
        
        # Set samples per class
        if samples_per_class is None:
            # Use minimum class size (under-sampling)
            samples_per_class = min(len(indices) for indices in self.class_indices.values())
        
        self.samples_per_class = samples_per_class
        self.n_classes = len(self.class_indices)
        self.total_samples = self.n_classes * self.samples_per_class
        
        print(f"\n{'='*60}")
        print(f"ClassBalancedSampler: {self.samples_per_class} samples/class")
        print(f"{'='*60}")
        for cls, indices in sorted(self.class_indices.items()):
            print(f"Class {cls}: {len(indices):6d} → {self.samples_per_class:6d} samples/epoch")
        print(f"Total: {self.total_samples} samples/epoch")
        print(f"{'='*60}\n")
    
    def __iter__(self):
        indices = []
        for cls, cls_indices in self.class_indices.items():
            # Sample with replacement if needed
            if len(cls_indices) < self.samples_per_class:
                # Over-sample
                sampled = np.random.choice(cls_indices, self.samples_per_class, replace=True)
            else:
                # Under-sample
                sampled = np.random.choice(cls_indices, self.samples_per_class, replace=False)
            indices.extend(sampled)
        
        # Shuffle
        np.random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return self.total_samples


class StratifiedBatchSampler(Sampler):
    """
    Creates batches with balanced class distribution.
    Each batch has equal number of samples from each class.
    """
    
    def __init__(self, labels: list[int], batch_size: int, drop_last: bool = True):
        """
        Args:
            labels: List of class labels
            batch_size: Total batch size (should be divisible by n_classes)
            drop_last: Drop last incomplete batch
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by class
        self.class_indices = {}
        for cls in np.unique(labels):
            self.class_indices[cls] = np.where(self.labels == cls)[0].tolist()
        
        self.n_classes = len(self.class_indices)
        self.samples_per_class_per_batch = batch_size // self.n_classes
        
        if batch_size % self.n_classes != 0:
            print(f"Warning: batch_size ({batch_size}) not divisible by n_classes ({self.n_classes})")
            print(f"Using {self.samples_per_class_per_batch} samples/class/batch")
        
        print(f"\n{'='*60}")
        print(f"StratifiedBatchSampler: {self.samples_per_class_per_batch} samples/class/batch")
        print(f"Batch size: {self.samples_per_class_per_batch * self.n_classes}")
        print(f"{'='*60}\n")
    
    def __iter__(self):
        # Shuffle indices for each class
        class_iters = {
            cls: iter(np.random.permutation(indices)) 
            for cls, indices in self.class_indices.items()
        }
        
        batches = []
        while True:
            batch = []
            for cls in sorted(self.class_indices.keys()):
                try:
                    # Get samples_per_class_per_batch samples from this class
                    for _ in range(self.samples_per_class_per_batch):
                        batch.append(next(class_iters[cls]))
                except StopIteration:
                    # Re-shuffle when exhausted
                    class_iters[cls] = iter(np.random.permutation(self.class_indices[cls]))
                    try:
                        for _ in range(self.samples_per_class_per_batch):
                            batch.append(next(class_iters[cls]))
                    except StopIteration:
                        # Class is too small
                        if self.drop_last:
                            return iter(batches)
                        break
            
            if len(batch) == self.samples_per_class_per_batch * self.n_classes:
                # Shuffle within batch
                np.random.shuffle(batch)
                batches.append(batch)
            elif not self.drop_last and len(batch) > 0:
                batches.append(batch)
            else:
                break
            
            # Limit number of batches (prevent infinite loop)
            if len(batches) >= 10000:
                break
        
        return iter(batches)
    
    def __len__(self):
        min_samples = min(len(indices) for indices in self.class_indices.values())
        batches_per_class = min_samples // self.samples_per_class_per_batch
        return batches_per_class


def print_sampling_statistics(labels: list[int], sampler, num_epochs: int = 1):
    """
    Print statistics about what the sampler will actually sample.
    
    Args:
        labels: Original labels
        sampler: The sampler to analyze
        num_epochs: Number of epochs to simulate
    """
    print(f"\n{'='*60}")
    print(f"Sampling Statistics ({num_epochs} epoch(s))")
    print(f"{'='*60}")
    
    original_counts = Counter(labels)
    print(f"Original distribution:")
    for cls, count in sorted(original_counts.items()):
        print(f"  Class {cls}: {count:6d} samples")
    
    # Sample and count
    sampled_labels = []
    for _ in range(num_epochs):
        if isinstance(sampler, StratifiedBatchSampler):
            for batch_indices in sampler:
                sampled_labels.extend([labels[i] for i in batch_indices])
        else:
            for idx in sampler:
                sampled_labels.append(labels[idx])
    
    sampled_counts = Counter(sampled_labels)
    print(f"\nSampled distribution:")
    for cls, count in sorted(sampled_counts.items()):
        original = original_counts[cls]
        ratio = count / original
        print(f"  Class {cls}: {count:6d} samples (×{ratio:.2f})")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    print("Example: Imbalanced dataset with 3 classes")
    labels = [0]*100 + [1]*1000 + [2]*50  # Very imbalanced
    
    print("\n" + "="*60)
    print("Method 1: WeightedRandomSampler")
    print("="*60)
    sampler1 = create_weighted_sampler(labels, strategy="balanced", num_samples=300)
    print_sampling_statistics(labels, sampler1, num_epochs=1)
    
    print("\n" + "="*60)
    print("Method 2: ClassBalancedSampler")
    print("="*60)
    sampler2 = ClassBalancedSampler(labels, samples_per_class=100)
    print_sampling_statistics(labels, sampler2, num_epochs=1)
    
    print("\n" + "="*60)
    print("Method 3: StratifiedBatchSampler")
    print("="*60)
    sampler3 = StratifiedBatchSampler(labels, batch_size=30, drop_last=True)
    print_sampling_statistics(labels, sampler3, num_epochs=1)
