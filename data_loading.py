"""
Data loading utilities for BNN comparison experiment.

This module handles:
1. CIFAR-10 loading and preprocessing (in-distribution)
2. SVHN loading for OOD detection
3. Rotation transforms for calibration under distribution shift

Design Decisions:
-----------------
- OOD Dataset: We use SVHN (Street View House Numbers) as the OOD dataset for CIFAR-10.
  This is a common choice in the literature because:
  * SVHN has the same image dimensions (32x32x3)
  * It represents a semantically different domain (digits vs objects)
  * It's not trivially separable from CIFAR-10 by simple statistics
  
- Calibration: We use rotated CIFAR-10 images to evaluate calibration under
  distribution shift, following Ovadia et al. (2019) and the paper's approach.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Tuple, Optional, List
import numpy as np

from config import DataConfig, OODConfig, CalibrationConfig, DEVICE


class RotatedDataset(Dataset):
    """Dataset wrapper that applies rotation to images.
    
    Used for evaluating calibration under distribution shift.
    The paper evaluates on rotated MNIST/FashionMNIST digits;
    we apply the same concept to CIFAR-10.
    """
    
    def __init__(self, base_dataset: Dataset, rotation_angle: float):
        """
        Args:
            base_dataset: The original dataset
            rotation_angle: Rotation angle in degrees
        """
        self.base_dataset = base_dataset
        self.rotation_angle = rotation_angle
        self.rotate = transforms.RandomRotation(
            degrees=(rotation_angle, rotation_angle),
            fill=0
        )
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        # Apply deterministic rotation
        if self.rotation_angle != 0:
            # Need to convert to PIL, rotate, convert back
            # For tensors, use functional rotation
            image = transforms.functional.rotate(image, self.rotation_angle)
        return image, label


class FlattenTransform:
    """Transform to flatten images for fully connected networks."""
    
    def __call__(self, x):
        return x.view(-1)


def get_cifar10_transforms(data_config: DataConfig, flatten: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get CIFAR-10 train and test transforms.
    
    Args:
        data_config: Data configuration
        flatten: Whether to flatten images for FC networks
        
    Returns:
        train_transform, test_transform
    """
    normalize = transforms.Normalize(data_config.cifar10_mean, data_config.cifar10_std)
    
    # Training transforms with data augmentation
    train_transform_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ]
    
    # Test transforms (no augmentation)
    test_transform_list = [
        transforms.ToTensor(),
        normalize,
    ]
    
    if flatten:
        train_transform_list.append(FlattenTransform())
        test_transform_list.append(FlattenTransform())
    
    return transforms.Compose(train_transform_list), transforms.Compose(test_transform_list)


def get_svhn_transforms(data_config: DataConfig, ood_config: OODConfig, 
                        use_cifar_stats: bool = True, flatten: bool = True) -> transforms.Compose:
    """Get SVHN transforms for OOD detection.
    
    Args:
        data_config: Data configuration
        ood_config: OOD configuration
        use_cifar_stats: If True, normalize with CIFAR-10 stats for fair comparison
        flatten: Whether to flatten images
        
    Returns:
        transform for SVHN
        
    Note: We normalize SVHN with CIFAR-10 statistics to ensure the model
    sees OOD data in the same normalized space as training data.
    """
    if use_cifar_stats:
        normalize = transforms.Normalize(data_config.cifar10_mean, data_config.cifar10_std)
    else:
        normalize = transforms.Normalize(ood_config.svhn_mean, ood_config.svhn_std)
    
    transform_list = [
        transforms.ToTensor(),
        normalize,
    ]
    
    if flatten:
        transform_list.append(FlattenTransform())
    
    return transforms.Compose(transform_list)


def load_cifar10(data_config: DataConfig, flatten: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 dataset.
    
    Args:
        data_config: Data configuration
        flatten: Whether to flatten images
        
    Returns:
        train_loader, test_loader
    """
    train_transform, test_transform = get_cifar10_transforms(data_config, flatten)
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=data_config.batch_size,
        shuffle=True, 
        num_workers=data_config.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def load_svhn_ood(data_config: DataConfig, ood_config: OODConfig, 
                  flatten: bool = True) -> DataLoader:
    """Load SVHN dataset for OOD detection.
    
    We use the test split of SVHN as OOD data.
    
    Args:
        data_config: Data configuration
        ood_config: OOD configuration
        flatten: Whether to flatten images
        
    Returns:
        ood_loader
    """
    transform = get_svhn_transforms(data_config, ood_config, 
                                    use_cifar_stats=True, flatten=flatten)
    
    # Use test split for OOD evaluation
    ood_dataset = torchvision.datasets.SVHN(
        root='./data', split='test', download=True, transform=transform
    )
    
    ood_loader = DataLoader(
        ood_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True
    )
    
    return ood_loader


def load_rotated_cifar10(data_config: DataConfig, rotation_angle: float,
                         flatten: bool = True) -> DataLoader:
    """Load rotated CIFAR-10 for calibration under distribution shift.
    
    Args:
        data_config: Data configuration
        rotation_angle: Rotation angle in degrees
        flatten: Whether to flatten images
        
    Returns:
        rotated_test_loader
    """
    _, test_transform = get_cifar10_transforms(data_config, flatten=False)
    
    # Load base test dataset without flattening initially
    base_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Create rotated dataset
    rotated_dataset = RotatedDataset(base_dataset, rotation_angle)
    
    # If flattening is needed, we need a custom collate or wrapper
    if flatten:
        class FlattenWrapper(Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                image, label = self.dataset[idx]
                return image.view(-1), label
        
        rotated_dataset = FlattenWrapper(rotated_dataset)
    
    rotated_loader = DataLoader(
        rotated_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True
    )
    
    return rotated_loader


class DataLoaderManager:
    """Centralized manager for all data loaders in the experiment.
    
    This class provides a unified interface for accessing:
    - In-distribution data (CIFAR-10 train/test)
    - OOD data (SVHN)
    - Rotated data for calibration
    """
    
    def __init__(self, data_config: DataConfig, ood_config: OODConfig,
                 calibration_config: CalibrationConfig, flatten: bool = True):
        self.data_config = data_config
        self.ood_config = ood_config
        self.calibration_config = calibration_config
        self.flatten = flatten
        
        # Lazy loading - will be populated on first access
        self._train_loader = None
        self._test_loader = None
        self._ood_loader = None
        self._rotated_loaders = {}
    
    @property
    def train_loader(self) -> DataLoader:
        if self._train_loader is None:
            self._train_loader, self._test_loader = load_cifar10(
                self.data_config, self.flatten
            )
        return self._train_loader
    
    @property
    def test_loader(self) -> DataLoader:
        if self._test_loader is None:
            self._train_loader, self._test_loader = load_cifar10(
                self.data_config, self.flatten
            )
        return self._test_loader
    
    @property
    def ood_loader(self) -> DataLoader:
        if self._ood_loader is None:
            self._ood_loader = load_svhn_ood(
                self.data_config, self.ood_config, self.flatten
            )
        return self._ood_loader
    
    def get_rotated_loader(self, angle: float) -> DataLoader:
        """Get rotated test loader for a specific angle."""
        if angle not in self._rotated_loaders:
            self._rotated_loaders[angle] = load_rotated_cifar10(
                self.data_config, angle, self.flatten
            )
        return self._rotated_loaders[angle]
    
    def get_all_rotated_loaders(self) -> dict:
        """Get all rotated loaders for calibration evaluation."""
        for angle in self.calibration_config.rotation_angles:
            _ = self.get_rotated_loader(angle)
        return self._rotated_loaders


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_cifar10_loading():
    """Test CIFAR-10 data loading."""
    data_config = DataConfig(batch_size=32)
    train_loader, test_loader = load_cifar10(data_config, flatten=True)
    
    # Check batch dimensions
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (32, 3072), f"Expected (32, 3072), got {batch_x.shape}"
    assert batch_y.shape == (32,), f"Expected (32,), got {batch_y.shape}"
    
    # Check label range
    assert batch_y.min() >= 0 and batch_y.max() <= 9, "Labels should be 0-9"
    
    print("✓ CIFAR-10 loading test passed")


def test_svhn_loading():
    """Test SVHN OOD data loading."""
    data_config = DataConfig(batch_size=32)
    ood_config = OODConfig()
    ood_loader = load_svhn_ood(data_config, ood_config, flatten=True)
    
    batch_x, batch_y = next(iter(ood_loader))
    assert batch_x.shape == (32, 3072), f"Expected (32, 3072), got {batch_x.shape}"
    
    print("✓ SVHN loading test passed")


def test_rotated_loading():
    """Test rotated CIFAR-10 loading for calibration."""
    data_config = DataConfig(batch_size=32)
    
    for angle in [0, 45, 90]:
        rotated_loader = load_rotated_cifar10(data_config, angle, flatten=True)
        batch_x, batch_y = next(iter(rotated_loader))
        assert batch_x.shape == (32, 3072), f"Angle {angle}: Expected (32, 3072), got {batch_x.shape}"
    
    print("✓ Rotated CIFAR-10 loading test passed")


def test_data_loader_manager():
    """Test DataLoaderManager unified interface."""
    data_config = DataConfig(batch_size=32)
    ood_config = OODConfig()
    calibration_config = CalibrationConfig(rotation_angles=[0, 45, 90])
    
    manager = DataLoaderManager(data_config, ood_config, calibration_config)
    
    # Test lazy loading
    train_batch, _ = next(iter(manager.train_loader))
    test_batch, _ = next(iter(manager.test_loader))
    ood_batch, _ = next(iter(manager.ood_loader))
    
    assert train_batch.shape[1] == 3072
    assert test_batch.shape[1] == 3072
    assert ood_batch.shape[1] == 3072
    
    # Test rotated loaders
    rotated_loaders = manager.get_all_rotated_loaders()
    assert len(rotated_loaders) == 3
    
    print("✓ DataLoaderManager test passed")


if __name__ == "__main__":
    test_cifar10_loading()
    test_svhn_loading()
    test_rotated_loading()
    test_data_loader_manager()
    print("\nAll data loading tests passed!")
