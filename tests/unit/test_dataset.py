"""
Tests for CircuitDataset.

Verifies that the v1 dataset loads correctly and returns the expected
shapes and data types.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from ml.data.dataset import CircuitDataset, collate_circuit_batch


DATASET_PATH = 'rlc_dataset/filter_dataset.pkl'


@pytest.fixture(scope='module')
def dataset():
    return CircuitDataset(dataset_path=DATASET_PATH, normalize_features=True)


class TestBasicLoading:
    def test_dataset_size(self, dataset):
        assert len(dataset) == 1920

    def test_filter_types(self, dataset):
        assert len(dataset.FILTER_TYPES) == 8


class TestSingleSample:
    def test_sample_keys(self, dataset):
        sample = dataset[0]
        for key in ('graph', 'poles', 'zeros', 'gain', 'freq_response', 'filter_type'):
            assert key in sample

    def test_graph_shape(self, dataset):
        sample = dataset[0]
        assert sample['graph'].num_nodes >= 3
        assert sample['graph'].num_edges >= 2
        assert sample['graph'].x.shape[1] == 4      # node features
        assert sample['graph'].edge_attr.shape[1] == 3  # edge features

    def test_freq_response_shape(self, dataset):
        sample = dataset[0]
        assert sample['freq_response'].shape[1] == 2

    def test_filter_type_one_hot(self, dataset):
        sample = dataset[0]
        assert sample['filter_type'].sum().item() == pytest.approx(1.0)


class TestBatching:
    def test_collate(self, dataset):
        loader = DataLoader(
            dataset, batch_size=4, shuffle=False,
            collate_fn=collate_circuit_batch)
        batch = next(iter(loader))

        assert batch['graph'].num_graphs == 4
        assert len(batch['poles']) == 4
        assert len(batch['zeros']) == 4


class TestAllSamples:
    def test_all_loadable(self, dataset):
        """Every sample loads without error and has valid shapes."""
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample['graph'].num_nodes >= 3
            assert sample['graph'].num_edges >= 2
            assert sample['freq_response'].shape[1] == 2
            assert sample['filter_type'].sum().item() == pytest.approx(1.0)
