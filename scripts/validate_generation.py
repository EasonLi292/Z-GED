#!/usr/bin/env python3
"""
Validate conditional circuit generation against specifications.

Tests how well the GraphVAE can generate circuits that match:
1. Filter type (low-pass, high-pass, etc.)
2. Cutoff frequency
3. Q factor / bandwidth
4. Component value ranges

Usage:
    python scripts/validate_generation.py --checkpoint checkpoints/best.pt --num-samples 20
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import yaml
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
from collections import defaultdict

from ml.models import HierarchicalEncoder, HybridDecoder
from ml.generation import CircuitSampler
from ml.data import CircuitDataset
from tools.circuit_generator import extract_poles_zeros_gain_analytical


def compute_transfer_function_from_poles_zeros(poles, zeros, gain=1.0):
    """
    Compute transfer function from poles and zeros.

    Args:
        poles: Complex poles [N, 2] as [real, imag]
        zeros: Complex zeros [M, 2] as [real, imag]
        gain: DC or reference gain

    Returns:
        Function H(s) that computes transfer function at frequency s
    """
    # Convert to complex
    poles_complex = poles[:, 0] + 1j * poles[:, 1] if len(poles) > 0 else np.array([])
    zeros_complex = zeros[:, 0] + 1j * zeros[:, 1] if len(zeros) > 0 else np.array([])

    def H(s):
        """Evaluate transfer function at complex frequency s"""
        # H(s) = K * ∏(s - z_i) / ∏(s - p_i)
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


def analyze_filter_characteristics(poles, zeros, gain=1.0, freq_range=(10, 1e8)):
    """
    Analyze filter characteristics from poles and zeros.

    Returns:
        dict with:
            - cutoff_freq: -3dB cutoff frequency (Hz)
            - filter_type: Inferred filter type
            - pole_freqs: Natural frequencies of poles (Hz)
            - q_factor: Quality factor (for resonant filters)
            - dc_gain: Gain at DC (0 Hz)
            - hf_gain: Gain at high frequency
    """
    H = compute_transfer_function_from_poles_zeros(poles, zeros, gain)

    # Compute frequency response
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 1000)
    omega = 2 * np.pi * freqs
    s_vals = 1j * omega

    response = np.array([abs(H(s)) for s in s_vals])
    response_db = 20 * np.log10(response + 1e-10)

    # Find DC and HF gains
    dc_gain_db = response_db[0]
    hf_gain_db = response_db[-1]

    # Find -3dB cutoff
    # Low-pass: find where gain drops 3dB from DC
    # High-pass: find where gain drops 3dB from HF
    cutoff_freq = None
    filter_type_inferred = 'unknown'

    # Detect filter type based on DC vs HF gain
    if dc_gain_db > hf_gain_db + 10:
        # Low-pass (DC gain > HF gain)
        filter_type_inferred = 'low_pass'
        threshold = dc_gain_db - 3
        idx = np.where(response_db < threshold)[0]
        if len(idx) > 0:
            cutoff_freq = freqs[idx[0]]
    elif hf_gain_db > dc_gain_db + 10:
        # High-pass (HF gain > DC gain)
        filter_type_inferred = 'high_pass'
        threshold = hf_gain_db - 3
        # Find from low freq where response crosses threshold
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

        # Find -3dB points
        peak_gain = response_db[peak_idx]
        threshold = peak_gain - 3
        crossings = np.where(np.diff(np.sign(response_db - threshold)))[0]
        if len(crossings) >= 2:
            f_low = freqs[crossings[0]]
            f_high = freqs[crossings[-1]]
            cutoff_freq = np.sqrt(f_low * f_high)  # Geometric mean

    # Compute Q factor from poles
    q_factor = None
    pole_freqs = []
    if len(poles) > 0:
        for pole in poles:
            real, imag = pole
            # Natural frequency: ω_n = sqrt(real^2 + imag^2)
            omega_n = np.sqrt(real**2 + imag**2)
            f_n = omega_n / (2 * np.pi)
            pole_freqs.append(f_n)

            # Q = ω_n / (2 * ζ * ω_n) = 1 / (2 * ζ)
            # ζ = -real / ω_n
            if abs(omega_n) > 1e-6:
                zeta = -real / omega_n
                q = 1 / (2 * zeta) if abs(zeta) > 1e-6 else float('inf')
                if q_factor is None or (q > 0.5 and q < 100):
                    q_factor = q

    return {
        'cutoff_freq': cutoff_freq,
        'filter_type_inferred': filter_type_inferred,
        'pole_freqs': pole_freqs,
        'q_factor': q_factor,
        'dc_gain_db': dc_gain_db,
        'hf_gain_db': hf_gain_db,
        'freq_response': (freqs, response_db)
    }


def validate_generated_circuits(
    sampler: CircuitSampler,
    filter_type: str,
    num_samples: int,
    dataset: CircuitDataset = None
):
    """
    Generate circuits and validate against specifications.

    Args:
        sampler: CircuitSampler instance
        filter_type: Target filter type
        num_samples: Number of circuits to generate
        dataset: Dataset for getting reference circuits

    Returns:
        Validation results dict
    """
    print(f"\n{'='*70}")
    print(f"VALIDATING {filter_type.upper()} FILTER GENERATION")
    print(f"{'='*70}\n")

    # Get reference circuits from dataset
    reference_circuits = []
    filter_type_names = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel']
    if dataset is not None:
        for i in range(len(dataset)):
            circuit = dataset[i]
            # Filter type is one-hot tensor, convert to string
            filter_type_tensor = circuit['filter_type']
            circuit_filter_type = filter_type_names[filter_type_tensor.argmax().item()]
            if circuit_filter_type == filter_type:
                reference_circuits.append(circuit)
        print(f"Found {len(reference_circuits)} reference {filter_type} circuits in dataset")

    # Analyze reference circuits
    ref_characteristics = []
    if reference_circuits:
        print("\nAnalyzing reference circuits...")
        for i, circuit in enumerate(reference_circuits[:5]):  # Limit to 5
            poles = circuit['poles'].cpu().numpy()
            zeros = circuit['zeros'].cpu().numpy()
            gain = circuit.get('gain', 1.0)

            chars = analyze_filter_characteristics(poles, zeros, gain)
            ref_characteristics.append(chars)

            fc_str = f"{chars['cutoff_freq']:.2e} Hz" if chars['cutoff_freq'] else 'N/A'
            q_str = f"{chars['q_factor']:.2f}" if chars['q_factor'] else 'N/A'
            print(f"  Ref {i+1}: fc={fc_str}, Q={q_str}, type={chars['filter_type_inferred']}")

    # Generate new circuits
    print(f"\nGenerating {num_samples} {filter_type} circuits...")
    outputs = sampler.sample_conditional(
        filter_type,
        num_samples,
        temperature=1.0
    )

    # Analyze generated circuits
    generated_characteristics = []
    topology_matches = 0

    graphs = outputs['graphs']
    topo_probs = outputs['topo_probs']
    poles_pred = outputs['poles']
    zeros_pred = outputs['zeros']

    print("\nAnalyzing generated circuits...")
    for i in range(num_samples):
        # Check topology match
        predicted_type = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel'][
            topo_probs[i].argmax().item()
        ]
        confidence = topo_probs[i].max().item()

        if predicted_type == filter_type:
            topology_matches += 1

        # Analyze transfer function
        poles = poles_pred[i].cpu().numpy()
        zeros = zeros_pred[i].cpu().numpy()

        chars = analyze_filter_characteristics(poles, zeros, gain=1.0)
        chars['predicted_type'] = predicted_type
        chars['confidence'] = confidence
        generated_characteristics.append(chars)

        if i < 5:  # Print first 5
            fc_str = f"{chars['cutoff_freq']:.2e} Hz" if chars['cutoff_freq'] else 'N/A'
            q_str = f"{chars['q_factor']:.2f}" if chars['q_factor'] else 'N/A'
            print(f"  Gen {i+1}: fc={fc_str}, Q={q_str}, " +
                  f"type={chars['filter_type_inferred']}, confidence={confidence:.2%}")

    # Compute statistics
    results = {
        'filter_type': filter_type,
        'num_samples': num_samples,
        'topology_accuracy': topology_matches / num_samples,
        'reference_characteristics': ref_characteristics,
        'generated_characteristics': generated_characteristics
    }

    # Compare distributions
    if ref_characteristics:
        ref_cutoffs = [c['cutoff_freq'] for c in ref_characteristics if c['cutoff_freq']]
        gen_cutoffs = [c['cutoff_freq'] for c in generated_characteristics if c['cutoff_freq']]

        if ref_cutoffs and gen_cutoffs:
            ref_median = np.median(ref_cutoffs)
            gen_median = np.median(gen_cutoffs)

            print(f"\nCutoff Frequency Comparison:")
            print(f"  Reference median: {ref_median:.2e} Hz")
            print(f"  Generated median: {gen_median:.2e} Hz")
            print(f"  Ratio: {gen_median/ref_median:.2f}x")

            results['ref_cutoff_median'] = ref_median
            results['gen_cutoff_median'] = gen_median
            results['cutoff_ratio'] = gen_median / ref_median

        ref_qs = [c['q_factor'] for c in ref_characteristics if c['q_factor'] and c['q_factor'] < 100]
        gen_qs = [c['q_factor'] for c in generated_characteristics if c['q_factor'] and c['q_factor'] < 100]

        if ref_qs and gen_qs:
            ref_q_median = np.median(ref_qs)
            gen_q_median = np.median(gen_qs)

            print(f"\nQ Factor Comparison:")
            print(f"  Reference median: {ref_q_median:.2f}")
            print(f"  Generated median: {gen_q_median:.2f}")
            print(f"  Ratio: {gen_q_median/ref_q_median:.2f}x")

            results['ref_q_median'] = ref_q_median
            results['gen_q_median'] = gen_q_median
            results['q_ratio'] = gen_q_median / ref_q_median

    # Filter type inference accuracy
    inferred_matches = sum(1 for c in generated_characteristics if c['filter_type_inferred'] == filter_type)
    print(f"\nFilter Type Inference from Transfer Function:")
    print(f"  Predicted topology: {topology_matches}/{num_samples} ({topology_matches/num_samples:.1%})")
    print(f"  Inferred from poles/zeros: {inferred_matches}/{num_samples} ({inferred_matches/num_samples:.1%})")

    results['inference_accuracy'] = inferred_matches / num_samples

    return results


def plot_comparison(results: dict, output_dir: Path):
    """Plot comparison between reference and generated circuits."""
    filter_type = results['filter_type']
    ref_chars = results['reference_characteristics']
    gen_chars = results['generated_characteristics']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{filter_type.replace("_", " ").title()} Filter Generation Validation', fontsize=14, fontweight='bold')

    # Plot 1: Frequency responses (sample)
    ax = axes[0, 0]
    if ref_chars:
        for i, char in enumerate(ref_chars[:3]):
            if 'freq_response' in char:
                freqs, resp = char['freq_response']
                ax.semilogx(freqs, resp, '--', alpha=0.7, label=f'Ref {i+1}')

    for i, char in enumerate(gen_chars[:3]):
        if 'freq_response' in char:
            freqs, resp = char['freq_response']
            ax.semilogx(freqs, resp, '-', linewidth=2, label=f'Gen {i+1}')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Sample Frequency Responses')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 2: Cutoff frequency distribution
    ax = axes[0, 1]
    ref_cutoffs = [c['cutoff_freq'] for c in ref_chars if c['cutoff_freq']]
    gen_cutoffs = [c['cutoff_freq'] for c in gen_chars if c['cutoff_freq']]

    if ref_cutoffs and gen_cutoffs:
        bins = np.logspace(np.log10(min(ref_cutoffs + gen_cutoffs)),
                          np.log10(max(ref_cutoffs + gen_cutoffs)), 20)
        ax.hist(ref_cutoffs, bins=bins, alpha=0.6, label='Reference', edgecolor='black')
        ax.hist(gen_cutoffs, bins=bins, alpha=0.6, label='Generated', edgecolor='black')
        ax.set_xscale('log')
        ax.set_xlabel('Cutoff Frequency (Hz)')
        ax.set_ylabel('Count')
        ax.set_title('Cutoff Frequency Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 3: Q factor distribution
    ax = axes[1, 0]
    ref_qs = [c['q_factor'] for c in ref_chars if c['q_factor'] and c['q_factor'] < 100]
    gen_qs = [c['q_factor'] for c in gen_chars if c['q_factor'] and c['q_factor'] < 100]

    if ref_qs and gen_qs:
        bins = np.linspace(0, min(10, max(ref_qs + gen_qs)), 20)
        ax.hist(ref_qs, bins=bins, alpha=0.6, label='Reference', edgecolor='black')
        ax.hist(gen_qs, bins=bins, alpha=0.6, label='Generated', edgecolor='black')
        ax.set_xlabel('Q Factor')
        ax.set_ylabel('Count')
        ax.set_title('Q Factor Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: Accuracy metrics
    ax = axes[1, 1]
    metrics = {
        'Topology\nAccuracy': results['topology_accuracy'],
        'Transfer Func\nInference': results.get('inference_accuracy', 0),
    }

    bars = ax.bar(range(len(metrics)), list(metrics.values()), color=['#2ecc71', '#3498db'])
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(list(metrics.keys()))
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Accuracy')
    ax.set_title('Generation Quality Metrics')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / f'validation_{filter_type}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved validation plot: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Validate conditional circuit generation')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='rlc_dataset/filter_dataset.pkl',
                       help='Path to dataset')
    parser.add_argument('--filter-types', type=str, nargs='+',
                       default=['low_pass', 'high_pass', 'band_pass', 'band_stop'],
                       help='Filter types to test')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of circuits to generate per filter type')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/mps/cpu)')

    args = parser.parse_args()

    # Determine device
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print("\n" + "="*70)
    print("CONDITIONAL GENERATION VALIDATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Filter types: {', '.join(args.filter_types)}")
    print(f"Samples per type: {args.num_samples}")
    print("="*70)

    # Load model
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config_path = checkpoint_path.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract branch dimensions
    topo_dim = config['model'].get('topo_latent_dim', config['model']['latent_dim'] // 3)
    values_dim = config['model'].get('values_latent_dim', config['model']['latent_dim'] // 3)
    pz_dim = config['model'].get('pz_latent_dim', config['model']['latent_dim'] // 3)
    branch_dims = (topo_dim, values_dim, pz_dim)

    # Create models
    encoder = HierarchicalEncoder(
        node_feature_dim=config['model']['node_feature_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        gnn_hidden_dim=config['model']['gnn_hidden_dim'],
        gnn_num_layers=config['model']['gnn_num_layers'],
        latent_dim=config['model']['latent_dim'],
        dropout=config['model']['dropout'],
        topo_latent_dim=topo_dim,
        values_latent_dim=values_dim,
        pz_latent_dim=pz_dim
    )

    decoder = HybridDecoder(
        latent_dim=config['model']['latent_dim'],
        edge_feature_dim=config['model']['edge_feature_dim'],
        hidden_dim=config['model']['decoder_hidden_dim'],
        dropout=config['model']['dropout'],
        topo_latent_dim=topo_dim,
        values_latent_dim=values_dim,
        pz_latent_dim=pz_dim
    )

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    print(f"\n✅ Loaded model: {config['model']['latent_dim']}D ({branch_dims[0]}D + {branch_dims[1]}D + {branch_dims[2]}D)")

    # Load dataset
    dataset = None
    if Path(args.dataset).exists():
        dataset = CircuitDataset(
            dataset_path=args.dataset,
            normalize_features=config['data']['normalize'],
            log_scale_impedance=config['data']['log_scale']
        )
        print(f"✅ Loaded dataset: {len(dataset)} circuits")

    # Create sampler
    sampler = CircuitSampler(
        encoder, decoder,
        device=device,
        latent_dim=config['model']['latent_dim'],
        branch_dims=branch_dims
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate each filter type
    all_results = {}
    for filter_type in args.filter_types:
        results = validate_generated_circuits(
            sampler,
            filter_type,
            args.num_samples,
            dataset
        )
        all_results[filter_type] = results

        # Plot comparison
        plot_comparison(results, output_dir)

    # Save summary
    summary = {
        'checkpoint': str(args.checkpoint),
        'model_config': {
            'latent_dim': config['model']['latent_dim'],
            'branch_dims': list(branch_dims)
        },
        'validation_date': datetime.now().isoformat(),
        'num_samples_per_type': args.num_samples,
        'results': {
            filter_type: {
                'topology_accuracy': results['topology_accuracy'],
                'inference_accuracy': results.get('inference_accuracy', None),
                'cutoff_ratio': results.get('cutoff_ratio', None),
                'q_ratio': results.get('q_ratio', None)
            }
            for filter_type, results in all_results.items()
        }
    }

    summary_file = output_dir / 'validation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Saved validation summary: {summary_file}")

    # Print final summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    for filter_type, results in all_results.items():
        print(f"\n{filter_type.upper()}:")
        print(f"  Topology accuracy: {results['topology_accuracy']:.1%}")
        if 'inference_accuracy' in results:
            print(f"  Transfer function inference: {results.get('inference_accuracy', 0):.1%}")
        if 'cutoff_ratio' in results:
            print(f"  Cutoff frequency match: {results.get('cutoff_ratio', 0):.2f}x reference")
        if 'q_ratio' in results:
            print(f"  Q factor match: {results.get('q_ratio', 0):.2f}x reference")

    print("\n✅ Validation complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
