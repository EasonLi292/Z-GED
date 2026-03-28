"""Tests for circuit vocabulary."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pytest
from ml.models.vocabulary import CircuitVocabulary, COMPONENT_PREFIXES


@pytest.fixture
def vocab():
    return CircuitVocabulary(max_internal=10, max_components=10)


class TestVocabularyStructure:

    def test_pad_is_index_0(self, vocab):
        assert vocab.pad_id == 0
        assert vocab.itos[0] == 'PAD'

    def test_eos_is_index_1(self, vocab):
        assert vocab.eos_id == 1
        assert vocab.itos[1] == 'EOS'

    def test_fixed_nets_at_expected_indices(self, vocab):
        assert vocab.stoi['VSS'] == 2
        assert vocab.stoi['VIN'] == 3
        assert vocab.stoi['VOUT'] == 4
        assert vocab.stoi['VDD'] == 5

    def test_internal_nets_present(self, vocab):
        for i in range(1, 11):
            assert f'INTERNAL_{i}' in vocab.stoi

    def test_components_present(self, vocab):
        for prefix in COMPONENT_PREFIXES:
            for i in range(1, 11):
                assert f'{prefix}{i}' in vocab.stoi

    def test_vocab_size(self, vocab):
        # 1 (PAD) + 1 (EOS) + 4 (fixed nets) + 10 (internal) + 7*10 (components)
        assert vocab.vocab_size == 1 + 1 + 4 + 10 + 70

    def test_deterministic(self):
        v1 = CircuitVocabulary(max_internal=5, max_components=10)
        v2 = CircuitVocabulary(max_internal=5, max_components=10)
        assert v1.tokens == v2.tokens
        assert v1.stoi == v2.stoi


class TestEncodeRoundtrip:

    def test_roundtrip_simple(self, vocab):
        tokens = ['VSS', 'C1', 'VOUT', 'R1', 'VIN', 'R1', 'VOUT', 'C1', 'VSS']
        ids = vocab.encode(tokens)
        decoded = vocab.decode(ids)
        assert decoded == tokens

    def test_roundtrip_with_internal(self, vocab):
        tokens = ['VSS', 'R1', 'INTERNAL_1', 'L1', 'VIN']
        ids = vocab.encode(tokens)
        decoded = vocab.decode(ids)
        assert decoded == tokens

    def test_roundtrip_compound(self, vocab):
        tokens = ['VSS', 'RCL1', 'VOUT', 'RC1', 'VIN', 'RC1', 'VOUT', 'RCL1', 'VSS']
        ids = vocab.encode(tokens)
        decoded = vocab.decode(ids)
        assert decoded == tokens

    def test_unknown_token_raises(self, vocab):
        with pytest.raises(KeyError):
            vocab.encode(['UNKNOWN_TOKEN'])


class TestTokenClassification:

    def test_pad_type(self, vocab):
        assert vocab.token_type('PAD') == 'pad'

    def test_eos_type(self, vocab):
        assert vocab.token_type('EOS') == 'eos'

    def test_net_types(self, vocab):
        for net in ['VSS', 'VIN', 'VOUT', 'VDD', 'INTERNAL_1', 'INTERNAL_10']:
            assert vocab.token_type(net) == 'net'

    def test_component_types(self, vocab):
        for comp in ['R1', 'C5', 'L10', 'RC1', 'RL3', 'CL2', 'RCL1']:
            assert vocab.token_type(comp) == 'component'

    def test_component_type_extraction(self, vocab):
        assert vocab.component_type('R1') == 'R'
        assert vocab.component_type('C10') == 'C'
        assert vocab.component_type('L5') == 'L'
        assert vocab.component_type('RC1') == 'RC'
        assert vocab.component_type('RL3') == 'RL'
        assert vocab.component_type('CL2') == 'CL'
        assert vocab.component_type('RCL1') == 'RCL'
        assert vocab.component_type('VSS') is None
        assert vocab.component_type('PAD') is None

    def test_is_net_is_component(self, vocab):
        assert vocab.is_net(vocab.stoi['VSS'])
        assert vocab.is_net(vocab.stoi['INTERNAL_1'])
        assert not vocab.is_net(vocab.stoi['R1'])
        assert vocab.is_component(vocab.stoi['R1'])
        assert vocab.is_component(vocab.stoi['RCL1'])
        assert not vocab.is_component(vocab.stoi['VSS'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
