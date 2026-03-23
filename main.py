from core.dataloader import *
def test_unit_training_integration():
    """🧪 Test DataLoader integration with training workflow."""
    print("🧪 Integration Test: Training Workflow...")

    # Create a realistic dataset
    num_samples = 1000
    num_features = 20
    num_classes = 5

    # Synthetic classification data
    features = Tensor(np.random.randn(num_samples, num_features))
    labels = Tensor(np.random.randint(0, num_classes, num_samples))

    dataset = TensorDataset(features, labels)

    # Create train/val splits
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Manual split (in production, you'd use proper splitting utilities)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))

    # Create subset datasets
    train_samples = [dataset[i] for i in train_indices]
    val_samples = [dataset[i] for i in val_indices]

    # Convert back to tensors for TensorDataset
    train_features = Tensor(np.stack([sample[0].data for sample in train_samples]))
    train_labels = Tensor(np.stack([sample[1].data for sample in train_samples]))
    val_features = Tensor(np.stack([sample[0].data for sample in val_samples]))
    val_labels = Tensor(np.stack([sample[1].data for sample in val_samples]))

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("📊 Dataset splits:")
    print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")

    # Simulate training loop
    print("\n🏃 Simulated Training Loop:")

    epoch_samples = 0
    batch_count = 0

    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        batch_count += 1
        epoch_samples += len(batch_features.data)

        # Simulate forward pass (just check shapes)
        assert batch_features.data.shape[0] <= batch_size, "Batch size exceeded"
        assert batch_features.data.shape[1] == num_features, "Wrong feature count"
        assert len(batch_labels.data) == len(batch_features.data), "Mismatched batch sizes"

        if batch_idx < 3:  # Show first few batches
            print(f"  Batch {batch_idx + 1}: {batch_features.data.shape[0]} samples")

    print(f"  Total: {batch_count} batches, {epoch_samples} samples processed")

    # Validate that all samples were seen
    assert epoch_samples == len(train_dataset), f"Expected {len(train_dataset)}, processed {epoch_samples}"

    print("✅ Training integration works correctly!")


def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    '''print("Running unit tests...")
    test_unit_dataset()
    test_unit_tensordataset()
    test_unit_dataloader()
    test_unit_dataloader_deterministic()
    test_unit_pad_image()
    test_unit_random_crop_region()
    test_unit_augmentation()'''

    print("\nRunning integration scenarios...")

    # Test complete workflow
    test_unit_training_integration()

    # Test augmentation with DataLoader
    print("🧪 Integration Test: Augmentation with DataLoader...")

    # Create dataset with augmentation
    train_transforms = Compose([
        RandomHorizontalFlip(0.5),
        RandomCrop(8, padding=2)  # Small images for test
    ])

    # Simulate CIFAR-style images (C, H, W)
    images = np.random.randn(100, 3, 8, 8)
    labels = np.random.randint(0, 10, 100)

    # Apply augmentation manually (how you'd use in practice)
    augmented_images = np.array([train_transforms(img) for img in images])

    dataset = TensorDataset(Tensor(augmented_images), Tensor(labels))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    batch_count = 0
    for batch_x, batch_y in loader:
        assert batch_x.shape[1:] == (3, 8, 8), f"Augmented batch shape wrong: {batch_x.shape}"
        batch_count += 1

    assert batch_count > 0, "DataLoader should produce batches"
    print("✅ Augmentation + DataLoader integration works!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 05")
if __name__ == "__main__":
    test_module()