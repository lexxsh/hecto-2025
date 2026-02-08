from typing import Callable

from torch.utils.data import DataLoader

from src.config import Config
from src.utils import logger

from .augmentations import init_augmentations, create_cutmix_collate_fn, create_real_fake_mixup_collate_fn
from .base import BaseDataModule
from .dataset import DeepfakeDataset


class DeepfakeDataModule(BaseDataModule):
    def __init__(self, config: Config, preprocess: None | Callable = None):
        super().__init__(
            config, 
            preprocess,
            use_balanced_sampling=config.use_balanced_sampling,
            sampling_strategy=config.sampling_strategy,
        )

    def setup(self, stage: str):
        # Initialize datasets
        if stage == "fit" or stage == "validate":
            logger.print("\n[blue]Creating training dataset")
            
            # Check if validation files are explicitly provided
            has_val_files = hasattr(self.config, 'val_files') and self.config.val_files and len(self.config.val_files) > 0
            
            # Only use automatic split if val_files is not provided
            use_auto_split = self.config.val_split > 0 and not has_val_files
            
            self.train_dataset = DeepfakeDataset(
                self.config.trn_files,
                self.preprocess,
                augmentations=init_augmentations(self.config.augmentations),
                binary=self.config.binary_labels,
                limit_files=self.config.limit_trn_files,
                load_pairs=self.config.load_pairs,
                split="train" if use_auto_split else "full",  # Use full if validation files are provided
                val_split=self.config.val_split,   # Use config value
            )
            self.train_dataset.print_statistics()

            # Create validation dataset from val_files if provided, otherwise use auto split
            if has_val_files:
                logger.print("\n[blue]Creating validation dataset from val_files")
                self.val_dataset = DeepfakeDataset(
                    self.config.val_files,  # Use explicitly provided validation files
                    self.preprocess,
                    shuffle=True,
                    binary=self.config.binary_labels,
                    limit_files=self.config.limit_val_files,
                    split="full",  # No split needed, already separated
                )
                self.val_dataset.print_statistics()
            elif use_auto_split:
                logger.print("\n[blue]Creating validation dataset from automatic split")
                self.val_dataset = DeepfakeDataset(
                    self.config.trn_files,  # Use same files but val split
                    self.preprocess,
                    shuffle=True,
                    binary=self.config.binary_labels,
                    limit_files=self.config.limit_val_files,
                    split="val",  # Use val split
                    val_split=self.config.val_split,  # Use config value
                )
                self.val_dataset.print_statistics()
            else:
                logger.print("\n[yellow]Validation disabled")
                self.val_dataset = None

        if stage == "test":
            logger.print("\nCreating test dataset")
            self.test_dataset = DeepfakeDataset(
                self.config.tst_files,
                self.preprocess,
                augmentations=init_augmentations(self.config.test_augmentations),
                binary=self.config.binary_labels,
                limit_files=self.config.limit_tst_files,
            )
            self.test_dataset.print_statistics()

    def train_dataloader(self):
        # Create collate_fn with augmentations if enabled
        collate_fn = None
        
        if hasattr(self.config, 'augmentations') and self.config.augmentations is not None:
            # Priority: Real-Fake Mixup > CutMix
            # (They are mutually exclusive in one batch)
            
            if self.config.augmentations.real_fake_mixup_prob > 0:
                # Use Real-Fake Mixup (Phase 5 - DFDC 1st Place)
                # Support both paired (DFDC original) and random batch mixing
                use_pairs = self.config.load_pairs  # Use paired mixup if pairs are loaded
                collate_fn = create_real_fake_mixup_collate_fn(
                    mixup_prob=self.config.augmentations.real_fake_mixup_prob,
                    mixup_alpha=self.config.augmentations.real_fake_mixup_alpha,
                    use_pairs=use_pairs,
                )
                mode_str = "PAIRED (same video)" if use_pairs else "RANDOM (batch)"
                logger.print(f"[green]Real-Fake Mixup enabled ({mode_str}): prob={self.config.augmentations.real_fake_mixup_prob}, alpha={self.config.augmentations.real_fake_mixup_alpha}")
            
            elif self.config.augmentations.cutmix_prob > 0:
                # Use CutMix (Phase 4)
                collate_fn = create_cutmix_collate_fn(
                    cutmix_prob=self.config.augmentations.cutmix_prob,
                    cutmix_alpha=self.config.augmentations.cutmix_alpha,
                )
                logger.print(f"[green]CutMix enabled: prob={self.config.augmentations.cutmix_prob}, alpha={self.config.augmentations.cutmix_alpha}")
        
        # Use balanced sampling from BaseDataModule if enabled
        if self.use_balanced_sampling:
            return super().train_dataloader()  # This handles balanced sampling
        
        # Default: standard random sampling with custom collate_fn
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.mini_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
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
