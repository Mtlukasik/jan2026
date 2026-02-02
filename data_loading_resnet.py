"""
Data Loading for CIFAR-10 with ResNet Training.

Features:
- CIFAR-10 with proper normalization
- Training augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip
- Proper train/val/test split (45k/5k/10k)
- SVHN as OOD dataset (normalized with CIFAR-10 stats for fair comparison)
- Saves split indices to disk for reproducibility verification

IMPORTANT: Test set is NEVER used during training or model selection.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
import numpy as np
import json
import os

from config import DataConfig, OODConfig, CalibrationConfig, DEVICE


class DataLoaderManager:
    """Manages data loading for CIFAR-10 experiments with ResNet.
    
    Split Strategy:
        - Original CIFAR-10 train (50k) → train (45k) + val (5k)
        - Original CIFAR-10 test (10k) → test (10k, NEVER used during training)
        - SVHN test → OOD evaluation
    
    The split indices are saved to {save_dir}/data_split.json for:
        1. Reproducibility verification
        2. Ensuring same split across all steps
    """
    
    def __init__(
        self,
        data_config: DataConfig,
        ood_config: OODConfig,
        calibration_config: CalibrationConfig,
        flatten: bool = False,  # Set to False for ResNet (needs 3D images)
        data_dir: str = "./data",
        save_dir: str = None,  # Where to save/load split indices
        val_split: float = 0.1,  # 10% of training data for validation
        seed: int = 42
    ):
        self.data_config = data_config
        self.ood_config = ood_config
        self.calibration_config = calibration_config
        self.flatten = flatten
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.val_split = val_split
        self.seed = seed
        
        # CIFAR-10 normalization
        self.cifar_mean = data_config.cifar10_mean
        self.cifar_std = data_config.cifar10_std
        
        # Create transforms
        self._create_transforms()
        
        # Load datasets with saved/reproducible split
        self._load_datasets()
        
        # Create data loaders
        self._create_loaders()
        
        # Verify no leakage
        self._verify_no_leakage()
    
    def _create_transforms(self):
        """Create transforms for training and testing."""
        
        # Training transform WITH augmentation
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.cifar_mean, self.cifar_std),
        ]
        
        # Validation/Test transform WITHOUT augmentation
        test_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(self.cifar_mean, self.cifar_std),
        ]
        
        # Add flatten if needed (for FC networks)
        if self.flatten:
            train_transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
            test_transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
        
        self.train_transform = transforms.Compose(train_transform_list)
        self.test_transform = transforms.Compose(test_transform_list)
        
        # OOD transform (same as test, uses CIFAR-10 stats for fair comparison)
        self.ood_transform = self.test_transform
    
    def _get_split_path(self) -> Optional[str]:
        """Get path to split indices file."""
        if self.save_dir:
            return os.path.join(self.save_dir, "data_split.json")
        return None
    
    def _save_split(self, train_indices: list, val_indices: list):
        """Save split indices to disk."""
        split_path = self._get_split_path()
        if split_path:
            os.makedirs(os.path.dirname(split_path), exist_ok=True)
            split_data = {
                "seed": self.seed,
                "val_split": self.val_split,
                "n_train": len(train_indices),
                "n_val": len(val_indices),
                "train_indices": train_indices,
                "val_indices": val_indices
            }
            with open(split_path, 'w') as f:
                json.dump(split_data, f)
            print(f"Saved data split to: {split_path}")
    
    def _load_split(self) -> Optional[Tuple[list, list]]:
        """Load split indices from disk if exists."""
        split_path = self._get_split_path()
        if split_path and os.path.exists(split_path):
            with open(split_path, 'r') as f:
                split_data = json.load(f)
            
            # Verify same parameters
            if split_data["seed"] != self.seed:
                print(f"WARNING: Saved split has different seed ({split_data['seed']} vs {self.seed})")
            if split_data["val_split"] != self.val_split:
                print(f"WARNING: Saved split has different val_split ({split_data['val_split']} vs {self.val_split})")
            
            print(f"Loaded existing data split from: {split_path}")
            return split_data["train_indices"], split_data["val_indices"]
        return None
    
    def _load_datasets(self):
        """Load CIFAR-10 with proper train/val/test split."""
        
        # =====================================================================
        # Load full training set (will be split into train + val)
        # =====================================================================
        full_train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # Also load with test transform for validation (no augmentation)
        full_train_dataset_no_aug = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.test_transform  # No augmentation for validation
        )
        
        # =====================================================================
        # Try to load existing split, or create new one
        # =====================================================================
        existing_split = self._load_split()
        
        if existing_split:
            train_indices, val_indices = existing_split
        else:
            # Create new split
            n_total = len(full_train_dataset)
            n_val = int(n_total * self.val_split)
            n_train = n_total - n_val
            
            # Use fixed seed for reproducible split
            generator = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(n_total, generator=generator).tolist()
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
            
            # Save split for future runs
            self._save_split(train_indices, val_indices)
        
        # Create subsets
        self.train_dataset = Subset(full_train_dataset, train_indices)
        self.val_dataset = Subset(full_train_dataset_no_aug, val_indices)  # No augmentation!
        
        # Store indices for verification
        self.train_indices = set(train_indices)
        self.val_indices = set(val_indices)
        
        # =====================================================================
        # Test set (NEVER used during training or model selection)
        # =====================================================================
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        # =====================================================================
        # SVHN as OOD (use test split, normalized with CIFAR-10 stats)
        # =====================================================================
        self.ood_dataset = torchvision.datasets.SVHN(
            root=self.data_dir,
            split='test',
            download=True,
            transform=self.ood_transform
        )
        
        print(f"Data splits:")
        print(f"  Train: {len(self.train_dataset)} (with augmentation)")
        print(f"  Val:   {len(self.val_dataset)} (no augmentation, for model selection)")
        print(f"  Test:  {len(self.test_dataset)} (held out, NEVER used during training)")
        print(f"  OOD:   {len(self.ood_dataset)} (SVHN)")
    
    def _verify_no_leakage(self):
        """Verify train and val sets are disjoint."""
        overlap = self.train_indices & self.val_indices
        if overlap:
            raise RuntimeError(f"DATA LEAKAGE DETECTED! {len(overlap)} indices in both train and val!")
        print("✓ Verified: No data leakage between train/val splits")
    
    def _create_loaders(self):
        """Create DataLoader instances."""
        
        # Only use pin_memory with CUDA
        use_pin_memory = torch.cuda.is_available()
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
            pin_memory=use_pin_memory,
            drop_last=True  # For consistent batch sizes
        )
        
        # Validation loader (for model selection during training)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=use_pin_memory
        )
        
        # Test loader (ONLY for final evaluation)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=use_pin_memory
        )
        
        self.ood_loader = DataLoader(
            self.ood_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=use_pin_memory
        )
    
    def get_rotated_test_loader(self, angle: float) -> DataLoader:
        """Get test loader with rotated images for calibration testing."""
        
        rotated_transform_list = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: transforms.functional.rotate(x, angle)),
            transforms.Normalize(self.cifar_mean, self.cifar_std),
        ]
        
        if self.flatten:
            rotated_transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
        
        rotated_transform = transforms.Compose(rotated_transform_list)
        
        rotated_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=rotated_transform
        )
        
        return DataLoader(
            rotated_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=torch.cuda.is_available()
        )


# =============================================================================
# Unit Tests
# =============================================================================

def test_no_data_leakage():
    """Verify train/val/test are disjoint."""
    config = DataConfig(batch_size=4, num_workers=0)
    ood_config = OODConfig()
    cal_config = CalibrationConfig()
    
    dm = DataLoaderManager(config, ood_config, cal_config, flatten=False)
    
    # Check no overlap
    assert len(dm.train_indices & dm.val_indices) == 0, "Train and val overlap!"
    
    # Check sizes
    assert len(dm.train_indices) + len(dm.val_indices) == 50000, "Total should be 50k"
    
    print("✓ No data leakage test passed")


def test_split_persistence(tmp_dir="/tmp/test_split"):
    """Test that split is saved and loaded correctly."""
    import shutil
    
    config = DataConfig(batch_size=4, num_workers=0)
    ood_config = OODConfig()
    cal_config = CalibrationConfig()
    
    # Clean up
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    
    # Create first time - should save
    dm1 = DataLoaderManager(config, ood_config, cal_config, flatten=False, save_dir=tmp_dir)
    train1 = dm1.train_indices.copy()
    val1 = dm1.val_indices.copy()
    
    # Create second time - should load
    dm2 = DataLoaderManager(config, ood_config, cal_config, flatten=False, save_dir=tmp_dir)
    train2 = dm2.train_indices
    val2 = dm2.val_indices
    
    assert train1 == train2, "Train indices don't match after reload!"
    assert val1 == val2, "Val indices don't match after reload!"
    
    # Clean up
    shutil.rmtree(tmp_dir)
    
    print("✓ Split persistence test passed")


if __name__ == "__main__":
    test_no_data_leakage()
    test_split_persistence()
    print("\n✓ All data loading tests passed!")
