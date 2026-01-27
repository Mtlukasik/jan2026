"""
Data Loading for CIFAR-10 with ResNet Training.

Features:
- CIFAR-10 with proper normalization
- Training augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip
- SVHN as OOD dataset (normalized with CIFAR-10 stats for fair comparison)
- Support for both image (3D) and flattened (1D) inputs
"""

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
import numpy as np

from config import DataConfig, OODConfig, CalibrationConfig, DEVICE


class DataLoaderManager:
    """Manages data loading for CIFAR-10 experiments with ResNet."""
    
    def __init__(
        self,
        data_config: DataConfig,
        ood_config: OODConfig,
        calibration_config: CalibrationConfig,
        flatten: bool = False,  # Set to False for ResNet (needs 3D images)
        data_dir: str = "./data"
    ):
        self.data_config = data_config
        self.ood_config = ood_config
        self.calibration_config = calibration_config
        self.flatten = flatten
        self.data_dir = data_dir
        
        # CIFAR-10 normalization
        self.cifar_mean = data_config.cifar10_mean
        self.cifar_std = data_config.cifar10_std
        
        # Create transforms
        self._create_transforms()
        
        # Load datasets
        self._load_datasets()
        
        # Create data loaders
        self._create_loaders()
    
    def _create_transforms(self):
        """Create transforms for training and testing."""
        
        # Training transform WITH augmentation
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.cifar_mean, self.cifar_std),
        ]
        
        # Test transform WITHOUT augmentation
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
    
    def _load_datasets(self):
        """Load CIFAR-10 and SVHN datasets."""
        
        # CIFAR-10 training set
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # CIFAR-10 test set (no augmentation)
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        # SVHN as OOD (use test split, normalized with CIFAR-10 stats)
        self.ood_dataset = torchvision.datasets.SVHN(
            root=self.data_dir,
            split='test',
            download=True,
            transform=self.ood_transform
        )
        
        print(f"Loaded CIFAR-10: {len(self.train_dataset)} train, {len(self.test_dataset)} test")
        print(f"Loaded SVHN (OOD): {len(self.ood_dataset)} samples")
    
    def _create_loaders(self):
        """Create DataLoader instances."""
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            drop_last=True  # For consistent batch sizes
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=True
        )
        
        self.ood_loader = DataLoader(
            self.ood_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=True
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
            pin_memory=True
        )


# =============================================================================
# Unit Tests
# =============================================================================

def test_data_shapes_resnet():
    """Test data shapes for ResNet (3D images)."""
    config = DataConfig(batch_size=4, num_workers=0)
    ood_config = OODConfig()
    cal_config = CalibrationConfig()
    
    dm = DataLoaderManager(config, ood_config, cal_config, flatten=False)
    
    # Get a batch
    images, labels = next(iter(dm.train_loader))
    
    assert images.shape == (4, 3, 32, 32), f"Expected (4, 3, 32, 32), got {images.shape}"
    assert labels.shape == (4,), f"Expected (4,), got {labels.shape}"
    
    print("✓ ResNet data shape test passed")


def test_data_shapes_flat():
    """Test data shapes for FC networks (flattened)."""
    config = DataConfig(batch_size=4, num_workers=0)
    ood_config = OODConfig()
    cal_config = CalibrationConfig()
    
    dm = DataLoaderManager(config, ood_config, cal_config, flatten=True)
    
    images, labels = next(iter(dm.train_loader))
    
    assert images.shape == (4, 3072), f"Expected (4, 3072), got {images.shape}"
    
    print("✓ Flattened data shape test passed")


def test_ood_loader():
    """Test OOD (SVHN) loader."""
    config = DataConfig(batch_size=4, num_workers=0)
    ood_config = OODConfig()
    cal_config = CalibrationConfig()
    
    dm = DataLoaderManager(config, ood_config, cal_config, flatten=False)
    
    images, labels = next(iter(dm.ood_loader))
    
    assert images.shape == (4, 3, 32, 32), f"Expected (4, 3, 32, 32), got {images.shape}"
    
    print("✓ OOD loader test passed")


def test_normalization():
    """Test that images are properly normalized."""
    config = DataConfig(batch_size=100, num_workers=0)
    ood_config = OODConfig()
    cal_config = CalibrationConfig()
    
    dm = DataLoaderManager(config, ood_config, cal_config, flatten=False)
    
    images, _ = next(iter(dm.test_loader))
    
    # After normalization, mean should be ~0 and std ~1 per channel
    for c in range(3):
        channel_mean = images[:, c, :, :].mean().item()
        channel_std = images[:, c, :, :].std().item()
        
        # Allow some tolerance
        assert abs(channel_mean) < 0.5, f"Channel {c} mean too far from 0: {channel_mean}"
        assert 0.5 < channel_std < 2.0, f"Channel {c} std unexpected: {channel_std}"
    
    print("✓ Normalization test passed")


def test_augmentation():
    """Test that training uses augmentation."""
    config = DataConfig(batch_size=4, num_workers=0)
    ood_config = OODConfig()
    cal_config = CalibrationConfig()
    
    # Check that transforms include augmentation
    dm = DataLoaderManager(config, ood_config, cal_config, flatten=False)
    
    train_transforms = str(dm.train_transform)
    assert "RandomCrop" in train_transforms, "RandomCrop not in training transforms"
    assert "RandomHorizontalFlip" in train_transforms, "RandomHorizontalFlip not in training transforms"
    
    test_transforms = str(dm.test_transform)
    assert "RandomCrop" not in test_transforms, "RandomCrop should not be in test transforms"
    
    print("✓ Augmentation test passed")


if __name__ == "__main__":
    test_data_shapes_resnet()
    test_data_shapes_flat()
    test_ood_loader()
    test_normalization()
    test_augmentation()
    print("\n✓ All data loading tests passed!")
