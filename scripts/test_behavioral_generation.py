#!/usr/bin/env python3
"""
Test circuit generation with behavioral specifications.

This script tests whether the model can generate circuits that meet
specific behavioral requirements like:
- Cutoff frequency
- Q factor
- Gain
- Filter type

Approach:
1. Find reference circuits in dataset matching specifications
2. Encode them to latent space
3. Sample nearby in latent space (interpolation/perturbation)
4. Generate new circuits
5. Analyze if new circuits have similar behavior

Usage:
    # Test low-pass filter generation around 1kHz cutoff
    python scripts/test_behavioral_generation.py \
        --checkpoint checkpoints/variable_length/20251222_102121/best.pt \
        --filter-type low_pass \
        --target-cutoff 1000 \
        --tolerance 0.5 \
        --num-samples 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import yaml
import argparse
from collections import defaultdict

from ml.models import HierarchicalEncoder
from ml.models.variable_decoder import VariableLengthDecoder
from ml.data import CircuitDataset


def compute_transfer_function_from_poles_zeros(poles, zeros, gain=1.0):
    """Compute transfer function from poles and zeros."""
    poles_complex = poles[:, 0] + 1j * poles[:, 1] if len(poles) > 0 else np.array([])
    zeros_complex = zeros[:, 0] + 1j * zeros[:, 1] if len(zeros) > 0 else np.array([])

    def H(s):
        numerator = complex(gain)
        for z in zeros_complex:
            numerator *= (s - z)

        denominator = complex(1.0)
        for p in poles_complex:
            denominator *= (s - p)

        if abs(denominator) < 1e-10:
            return complex(0.0)

        return numerator / denominator

    return H


def analyze_filter_from_poles_zeros(poles, zeros, gain=1.0, freq_range=(10, 1e8)):
    """
    Analyze filter characteristics from poles and zeros.

    Returns:
        dict with cutoff_freq, filter_type, q_factor, dc_gain, hf_gain
    """
    if len(poles) == 0:
        return None

    H = compute_transfer_function_from_poles_zeros(poles, zeros, gain)

    # Compute frequency response
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 1000)
    omega = 2 * np.pi * freqs
    s_vals = 1j * omega

    response = np.array([abs(H(s)) for s in s_vals])
    response_db = 20 * np.log10(response + 1e-10)

    # DC and HF gains
    dc_gain_db = response_db[0]
    hf_gain_db = response_db[-1]

    # Detect filter type and cutoff
    cutoff_freq = None
    filter_type_inferred = 'unknown'

    if dc_gain_db > hf_gain_db + 10:
        # Low-pass
        filter_type_inferred = 'low_pass'
        threshold = dc_gain_db - 3
        idx = np.where(response_db < threshold)[0]
        if len(idx) > 0:
            cutoff_freq = freqs[idx[0]]
    elif hf_gain_db > dc_gain_db + 10:
        # High-pass
        filter_type_inferred = 'high_pass'
        threshold = hf_gain_db - 3
        idx = np.where(response_db > threshold)[0]
        if len(idx) > 0:
            cutoff_freq = freqs[idx[0]]
    else:
        # Band-pass or band-stop
        peak_idx = np.argmax(response_db)
        if response_db[peak_idx] > dc_gain_db + 3:
            filter_type_inferred = 'band_pass'
        else:
            filter_type_inferred = 'band_stop'

        peak_gain = response_db[peak_idx]
        threshold = peak_gain - 3
        crossings = np.where(np.diff(np.sign(response_db - threshold)))[0]
        if len(crossings) >= 2:
            f_low = freqs[crossings[0]]
            f_high = freqs[crossings[-1]]
            cutoff_freq = np.sqrt(f_low * f_high)

    # Q factor from poles
    q_factor = None
    if len(poles) > 0:
        for pole in poles:
            real, imag = pole
            omega_n = np.sqrt(real**2 + imag**2)
            if abs(omega_n) > 1e-6:
                zeta = -real / omega_n
                q = 1 / (2 * zeta) if abs(zeta) > 1e-6 else float('inf')
                if q_factor is None or (q > 0.5 and q < 100):
                    q_factor = q

    return {
        'cutoff_freq': cutoff_freq,
        'filter_type': filter_type_inferred,
        'q_factor': q_factor,
        'dc_gain_db': dc_gain_db,
        'hf_gain_db': hf_gain_db
    }


def find_circuits_matching_specs(dataset, filter_type, target_cutoff=None, cutoff_tolerance=0.5):
    """
    Find circuits in dataset matching behavioral specifications.

    Args:
        dataset: CircuitDataset
        filter_type: Target filter type
        target_cutoff: Target cutoff frequency (Hz), None = any
        cutoff_tolerance: Tolerance as fraction (0.5 = ±50%)

    Returns:
        List of (circuit_idx, characteristics) tuples
    """
    filter_type_names = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']

    matching_circuits = []

    print(f"\n{'='*70}")
    print(f"SEARCHING DATASET FOR MATCHING CIRCUITS")
    print(f"{'='*70}")
    print(f"Filter type: {filter_type}")
    if target_cutoff:
        cutoff_min = target_cutoff * (1 - cutoff_tolerance)
        cutoff_max = target_cutoff * (1 + cutoff_tolerance)
        print(f"Target cutoff: {target_cutoff:.2e} Hz (±{cutoff_tolerance*100:.0f}%)")
        print(f"Acceptable range: [{cutoff_min:.2e}, {cutoff_max:.2e}] Hz")
    else:
        print(f"Target cutoff: Any")
    print()

    for i in range(len(dataset)):
        circuit = dataset[i]
        circuit_filter_idx = circuit['filter_type'].argmax().item()
        circuit_filter_type = filter_type_names[circuit_filter_idx]

        if circuit_filter_type != filter_type:
            continue

        # Analyze transfer function
        poles = circuit['poles'].numpy()
        zeros = circuit['zeros'].numpy()

        chars = analyze_filter_from_poles_zeros(poles, zeros, gain=1.0)

        if chars is None:
            continue

        # Check cutoff match
        if target_cutoff is not None:
            if chars['cutoff_freq'] is None:
                continue

            cutoff_min = target_cutoff * (1 - cutoff_tolerance)
            cutoff_max = target_cutoff * (1 + cutoff_tolerance)

            if not (cutoff_min <= chars['cutoff_freq'] <= cutoff_max):
                continue

        matching_circuits.append((i, chars))

    print(f"Found {len(matching_circuits)} matching circuits in dataset")

    if matching_circuits:
        print(f"\nMatching circuits:")
        for idx, chars in matching_circuits[:10]:  # Show first 10
            fc_str = f"{chars['cutoff_freq']:.2e} Hz" if chars['cutoff_freq'] else "N/A"
            q_str = f"Q={chars['q_factor']:.2f}" if chars['q_factor'] else ""
            print(f"  Circuit {idx}: fc={fc_str} {q_str}")

    return matching_circuits


def generate_from_reference(
    encoder, decoder, dataset, reference_idx, num_samples, perturbation_scale, device
):
    """
    Generate circuits by perturbing a reference circuit's latent code.

    Args:
        encoder: HierarchicalEncoder
        decoder: VariableLengthDecoder
        dataset: CircuitDataset
        reference_idx: Index of reference circuit
        num_samples: Number of variants to generate
        perturbation_scale: Scale of random perturbation (0.1 = small, 1.0 = large)
        device: Device

    Returns:
        Generated circuits and their latent codes
    """
    circuit = dataset[reference_idx]

    # Encode reference circuit
    graph = circuit['graph'].to(device)
    poles_list = [circuit['poles'].to(device)]
    zeros_list = [circuit['zeros'].to(device)]
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        z, mu, logvar = encoder(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch,
            poles_list,
            zeros_list
        )

    # Generate variants by perturbing latent code
    reference_z = mu  # Use mean of posterior

    # Create perturbations
    perturbations = torch.randn(num_samples, mu.size(1), device=device) * perturbation_scale
    perturbed_z = reference_z + perturbations

    # Decode
    with torch.no_grad():
        outputs = decoder(perturbed_z, hard=True, gt_filter_type=None)

    return outputs, perturbed_z, reference_z


def analyze_generated_circuits(outputs, filter_type_names):
    """Analyze generated circuits and extract their characteristics."""
    pred_topo = outputs['topo_logits'].argmax(dim=-1)
    pred_pole_counts = outputs['pole_count_logits'].argmax(dim=-1)
    pred_zero_counts = outputs['zero_count_logits'].argmax(dim=-1)

    poles_all = outputs['poles_all']
    zeros_all = outputs['zeros_all']

    results = []

    for i in range(len(pred_topo)):
        topo_idx = pred_topo[i].item()
        predicted_type = filter_type_names[topo_idx]

        n_poles = pred_pole_counts[i].item()
        n_zeros = pred_zero_counts[i].item()

        poles = poles_all[i, :n_poles].cpu().numpy() if n_poles > 0 else np.array([]).reshape(0, 2)
        zeros = zeros_all[i, :n_zeros].cpu().numpy() if n_zeros > 0 else np.array([]).reshape(0, 2)

        # Analyze transfer function
        chars = analyze_filter_from_poles_zeros(poles, zeros, gain=1.0)

        results.append({
            'predicted_type': predicted_type,
            'num_poles': n_poles,
            'num_zeros': n_zeros,
            'poles': poles,
            'zeros': zeros,
            'characteristics': chars
        })

    return results


def test_behavioral_generation(
    checkpoint_path, dataset_path, filter_type, target_cutoff,
    cutoff_tolerance, num_samples, perturbation_scale, device
):
    """Test behavioral specification-driven generation."""

    checkpoint_path = Path(checkpoint_path)

    # Load config
    config_path = checkpoint_path.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"\n{'='*70}")
    print("BEHAVIORAL SPECIFICATION TEST")
    print(f"{'='*70}")
    print(f"Target specification:")
    print(f"  Filter type: {filter_type}")
    if target_cutoff:
        print(f"  Cutoff frequency: {target_cutoff:.2e} Hz (±{cutoff_tolerance*100:.0f}%)")
    print(f"  Num variants: {num_samples}")
    print(f"  Perturbation scale: {perturbation_scale}")
    print(f"{'='*70}")

    # Create models
    encoder = HierarchicalEncoder(
        node_feature_dim=config['model']['node_feature_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        gnn_hidden_dim=config['model']['gnn_hidden_dim'],
        gnn_num_layers=config['model']['gnn_num_layers'],
        latent_dim=config['model']['latent_dim'],
        dropout=config['model']['dropout'],
        topo_latent_dim=config['model'].get('topo_latent_dim'),
        values_latent_dim=config['model'].get('values_latent_dim'),
        pz_latent_dim=config['model'].get('pz_latent_dim')
    )

    decoder = VariableLengthDecoder(
        latent_dim=config['model']['latent_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        hidden_dim=config['model']['decoder_hidden_dim'],
        max_poles=config['model'].get('max_poles', 4),
        max_zeros=config['model'].get('max_zeros', 4),
        dropout=config['model']['dropout'],
        topo_latent_dim=config['model'].get('topo_latent_dim'),
        values_latent_dim=config['model'].get('values_latent_dim'),
        pz_latent_dim=config['model'].get('pz_latent_dim')
    )

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    # Load dataset
    dataset = CircuitDataset(
        dataset_path=dataset_path,
        normalize_features=config['data']['normalize'],
        log_scale_impedance=config['data']['log_scale']
    )

    filter_type_names = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']

    # Find reference circuits
    matching_circuits = find_circuits_matching_specs(
        dataset, filter_type, target_cutoff, cutoff_tolerance
    )

    if not matching_circuits:
        print(f"\n❌ No circuits found matching specifications!")
        return

    # Use the first matching circuit as reference
    reference_idx, reference_chars = matching_circuits[0]

    print(f"\n{'='*70}")
    print(f"USING REFERENCE CIRCUIT {reference_idx}")
    print(f"{'='*70}")
    print(f"Filter type: {filter_type}")
    if reference_chars['cutoff_freq']:
        print(f"Cutoff frequency: {reference_chars['cutoff_freq']:.2e} Hz")
    if reference_chars['q_factor']:
        print(f"Q factor: {reference_chars['q_factor']:.2f}")
    print()

    # Generate variants
    print(f"Generating {num_samples} variants...")
    outputs, perturbed_z, reference_z = generate_from_reference(
        encoder, decoder, dataset, reference_idx, num_samples, perturbation_scale, device
    )

    # Analyze results
    results = analyze_generated_circuits(outputs, filter_type_names)

    # Print results
    print(f"\n{'='*70}")
    print(f"GENERATED CIRCUIT ANALYSIS")
    print(f"{'='*70}\n")

    topology_matches = 0
    cutoff_matches = 0
    valid_cutoffs = []

    for i, result in enumerate(results):
        chars = result['characteristics']

        print(f"Variant {i+1}:")
        print(f"  Predicted type: {result['predicted_type']}")
        print(f"  Structure: {result['num_poles']} poles, {result['num_zeros']} zeros")

        if chars:
            if chars['cutoff_freq']:
                print(f"  Cutoff: {chars['cutoff_freq']:.2e} Hz")
                valid_cutoffs.append(chars['cutoff_freq'])

                # Check if within tolerance
                if target_cutoff:
                    cutoff_min = target_cutoff * (1 - cutoff_tolerance)
                    cutoff_max = target_cutoff * (1 + cutoff_tolerance)
                    if cutoff_min <= chars['cutoff_freq'] <= cutoff_max:
                        print(f"    ✅ Within target range!")
                        cutoff_matches += 1
                    else:
                        ratio = chars['cutoff_freq'] / target_cutoff
                        print(f"    ❌ Outside target ({ratio:.2f}x)")

            if chars['q_factor']:
                print(f"  Q factor: {chars['q_factor']:.2f}")

        if result['predicted_type'] == filter_type:
            topology_matches += 1

        print()

    # Summary statistics
    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Topology match: {topology_matches}/{num_samples} ({topology_matches/num_samples:.1%})")

    if target_cutoff and valid_cutoffs:
        print(f"Cutoff match: {cutoff_matches}/{len(valid_cutoffs)} ({cutoff_matches/len(valid_cutoffs):.1%})")

        cutoffs_array = np.array(valid_cutoffs)
        print(f"\nCutoff frequency statistics:")
        print(f"  Target: {target_cutoff:.2e} Hz")
        print(f"  Reference: {reference_chars['cutoff_freq']:.2e} Hz")
        print(f"  Generated mean: {cutoffs_array.mean():.2e} Hz")
        print(f"  Generated std: {cutoffs_array.std():.2e} Hz")
        print(f"  Generated range: [{cutoffs_array.min():.2e}, {cutoffs_array.max():.2e}] Hz")

        # Deviation from target
        deviations = np.abs(cutoffs_array - target_cutoff) / target_cutoff
        print(f"  Mean deviation: {deviations.mean():.1%}")
        print(f"  Median deviation: {np.median(deviations):.1%}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Test behavioral specification generation')

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='rlc_dataset/filter_dataset.pkl')
    parser.add_argument('--filter-type', type=str, required=True,
                       choices=['low_pass', 'high_pass', 'band_pass', 'band_stop'])
    parser.add_argument('--target-cutoff', type=float, default=None,
                       help='Target cutoff frequency in Hz (e.g., 1000 for 1kHz)')
    parser.add_argument('--tolerance', type=float, default=0.5,
                       help='Cutoff tolerance as fraction (0.5 = ±50%%)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of variants to generate')
    parser.add_argument('--perturbation', type=float, default=0.2,
                       help='Perturbation scale (0.1=small, 1.0=large)')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    test_behavioral_generation(
        args.checkpoint,
        args.dataset,
        args.filter_type,
        args.target_cutoff,
        args.tolerance,
        args.num_samples,
        args.perturbation,
        args.device
    )


if __name__ == '__main__':
    main()
