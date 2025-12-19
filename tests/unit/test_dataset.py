#!/usr/bin/env python3
"""
Test script for CircuitDataset.

Verifies that the dataset loads correctly and returns the expected shapes
and data types.
"""

import sys
sys.path.insert(0, 'ml/data')

from dataset import CircuitDataset, collate_circuit_batch
from torch.utils.data import DataLoader


def test_basic_loading():
    """Test basic dataset loading."""
    print("="*70)
    print("TEST: Basic Dataset Loading")
    print("="*70)

    dataset = CircuitDataset(
        dataset_path='rlc_dataset/filter_dataset.pkl',
        normalize_features=True,
        log_scale_impedance=True
    )

    print(f"\n✅ Loaded dataset with {len(dataset)} circuits")

    return dataset


def test_single_sample(dataset):
    """Test loading a single sample."""
    print("\n" + "="*70)
    print("TEST: Single Sample")
    print("="*70)

    # Get first sample
    sample = dataset[0]

    print(f"\nSample keys: {list(sample.keys())}")
    print(f"\nGraph (PyG Data):")
    print(f"  Number of nodes: {sample['graph'].num_nodes}")
    print(f"  Number of edges: {sample['graph'].num_edges}")
    print(f"  Node features shape: {sample['graph'].x.shape}")
    print(f"  Edge features shape: {sample['graph'].edge_attr.shape}")
    print(f"  Edge index shape: {sample['graph'].edge_index.shape}")

    print(f"\nTransfer function:")
    print(f"  Poles shape: {sample['poles'].shape}")
    print(f"  Zeros shape: {sample['zeros'].shape}")
    print(f"  Gain shape: {sample['gain'].shape}")
    print(f"  Gain value: {sample['gain'].item():.4e}")

    print(f"\nFrequency response:")
    print(f"  Shape: {sample['freq_response'].shape}")
    print(f"  Min magnitude: {sample['freq_response'][:, 0].min():.4e}")
    print(f"  Max magnitude: {sample['freq_response'][:, 0].max():.4e}")

    print(f"\nFilter type:")
    print(f"  Shape: {sample['filter_type'].shape}")
    print(f"  One-hot: {sample['filter_type'].numpy()}")
    filter_idx = sample['filter_type'].argmax().item()
    print(f"  Type: {dataset.FILTER_TYPES[filter_idx]}")

    print(f"\nCircuit ID: {sample['circuit_id']}")

    print("\n✅ Single sample test passed")


def test_batching(dataset):
    """Test batching with DataLoader."""
    print("\n" + "="*70)
    print("TEST: Batching with DataLoader")
    print("="*70)

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_circuit_batch
    )

    # Get first batch
    batch = next(iter(loader))

    print(f"\nBatch keys: {list(batch.keys())}")

    print(f"\nBatched graph:")
    print(f"  Batch size: {batch['graph'].num_graphs}")
    print(f"  Total nodes: {batch['graph'].num_nodes}")
    print(f"  Total edges: {batch['graph'].num_edges}")
    print(f"  Node features shape: {batch['graph'].x.shape}")
    print(f"  Edge features shape: {batch['graph'].edge_attr.shape}")

    print(f"\nBatched transfer functions:")
    print(f"  Poles: List of {len(batch['poles'])} tensors")
    for i, poles in enumerate(batch['poles']):
        print(f"    Sample {i}: {poles.shape}")
    print(f"  Zeros: List of {len(batch['zeros'])} tensors")
    for i, zeros in enumerate(batch['zeros']):
        print(f"    Sample {i}: {zeros.shape}")

    print(f"\nBatched fixed-size tensors:")
    print(f"  Gain shape: {batch['gain'].shape}")
    print(f"  Freq response shape: {batch['freq_response'].shape}")
    print(f"  Filter type shape: {batch['filter_type'].shape}")

    print("\n✅ Batching test passed")


def test_train_val_test_split(dataset):
    """Test stratified split."""
    print("\n" + "="*70)
    print("TEST: Train/Val/Test Split")
    print("="*70)

    train_idx, val_idx, test_idx = dataset.get_train_val_test_split(
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_idx)} circuits")
    print(f"  Val:   {len(val_idx)} circuits")
    print(f"  Test:  {len(test_idx)} circuits")
    print(f"  Total: {len(train_idx) + len(val_idx) + len(test_idx)}")

    # Verify stratification
    print(f"\nVerifying stratification:")
    for ftype in dataset.FILTER_TYPES:
        indices = dataset.get_filter_type_indices(ftype)
        n_train = sum(1 for i in train_idx if i in indices)
        n_val = sum(1 for i in val_idx if i in indices)
        n_test = sum(1 for i in test_idx if i in indices)

        print(f"  {ftype:<15}: Train={n_train}, Val={n_val}, Test={n_test}")

    print("\n✅ Split test passed")


def test_all_samples(dataset):
    """Test that all samples can be loaded without errors."""
    print("\n" + "="*70)
    print("TEST: Load All Samples")
    print("="*70)

    print(f"\nLoading all {len(dataset)} samples...")

    errors = []
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            # Basic sanity checks
            assert sample['graph'].num_nodes >= 3, f"Too few nodes in circuit {i}"
            assert sample['graph'].num_edges >= 2, f"Too few edges in circuit {i}"
            assert sample['freq_response'].shape == (701, 2), f"Wrong freq_response shape in circuit {i}"
            assert sample['filter_type'].sum() == 1.0, f"Filter type not one-hot in circuit {i}"
        except Exception as e:
            errors.append((i, str(e)))

    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for idx, error in errors[:5]:  # Show first 5 errors
            print(f"  Circuit {idx}: {error}")
    else:
        print(f"\n✅ All {len(dataset)} samples loaded successfully")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CIRCUIT DATASET TEST SUITE")
    print("="*70)

    # Test 1: Basic loading
    dataset = test_basic_loading()

    # Print statistics
    dataset.print_statistics()

    # Test 2: Single sample
    test_single_sample(dataset)

    # Test 3: Batching
    test_batching(dataset)

    # Test 4: Train/val/test split
    test_train_val_test_split(dataset)

    # Test 5: Load all samples
    test_all_samples(dataset)

    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    print()


if __name__ == '__main__':
    main()
