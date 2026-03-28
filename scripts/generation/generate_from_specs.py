"""
Generate circuits from pole/zero specifications.

This script constructs z[4:8] from user-provided pole/zero values using
signed-log normalization, samples z[0:4] from the prior, and generates
circuits using the decoder only (no encoder/dataset needed).

Usage:
    python scripts/generation/generate_from_specs.py \
        --pole-real -1000 --pole-imag 5000 \
        --zero-real 0 --zero-imag 0 \
        --num-samples 5
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from ml.models.constants import PZ_LOG_SCALE
from ml.utils.circuit_ops import walk_to_string, is_valid_walk, generate_walk
from ml.utils.runtime import load_decoder


def signed_log_normalize(x: float, scale: float = PZ_LOG_SCALE) -> float:
    """signed_log(x) = sign(x) * log10(|x| + 1) / scale."""
    if abs(x) < 1e-30:
        return 0.0
    sign = 1.0 if x >= 0 else -1.0
    return sign * np.log10(abs(x) + 1.0) / scale


def poles_zeros_to_latent_pz(pole_real, pole_imag, zero_real, zero_imag):
    """
    Convert raw pole/zero values to 4D normalized latent vector z[4:8].

    Args:
        pole_real: Real part of dominant pole
        pole_imag: |Imaginary part| of dominant pole
        zero_real: Real part of dominant zero
        zero_imag: |Imaginary part| of dominant zero

    Returns:
        Tensor [4] with [sigma_p, omega_p, sigma_z, omega_z]
    """
    sigma_p = signed_log_normalize(pole_real)
    omega_p = signed_log_normalize(abs(pole_imag))
    sigma_z = signed_log_normalize(zero_real)
    omega_z = signed_log_normalize(abs(zero_imag))
    return torch.tensor([sigma_p, omega_p, sigma_z, omega_z], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description='Generate circuits from pole/zero specs')
    parser.add_argument('--pole-real', type=float, default=-1000.0,
                        help='Real part of dominant pole (default: -1000)')
    parser.add_argument('--pole-imag', type=float, default=0.0,
                        help='|Imaginary part| of dominant pole (default: 0)')
    parser.add_argument('--zero-real', type=float, default=0.0,
                        help='Real part of dominant zero (default: 0 = no zero)')
    parser.add_argument('--zero-imag', type=float, default=0.0,
                        help='|Imaginary part| of dominant zero (default: 0 = no zero)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of circuits to generate (default: 5)')
    parser.add_argument('--checkpoint', default='checkpoints/production/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device', default='cpu', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print("=" * 70)
    print("Pole/Zero-Driven Circuit Generation")
    print("=" * 70)

    # Convert poles/zeros to latent
    pz_latent = poles_zeros_to_latent_pz(
        args.pole_real, args.pole_imag,
        args.zero_real, args.zero_imag
    )

    print(f"\nInput pole/zero:")
    print(f"  Pole: {args.pole_real:+.2f} + {args.pole_imag:.2f}j")
    print(f"  Zero: {args.zero_real:+.2f} + {args.zero_imag:.2f}j")
    print(f"  Latent z[4:8]: [{pz_latent[0]:+.4f}, {pz_latent[1]:+.4f}, "
          f"{pz_latent[2]:+.4f}, {pz_latent[3]:+.4f}]")

    device = torch.device(args.device)

    # Load decoder only
    decoder, vocab, _ = load_decoder(
        checkpoint_path=args.checkpoint,
        device=str(device),
    )

    # Generate circuits
    torch.manual_seed(args.seed)
    print(f"\nGenerating {args.num_samples} circuits...")
    print("-" * 70)

    valid_count = 0
    for i in range(args.num_samples):
        # Sample z[0:4] from prior, fix z[4:8] from poles/zeros
        z_topo = torch.randn(4)
        z = torch.cat([z_topo, pz_latent]).to(device)

        tokens = generate_walk(decoder, z, vocab)
        cstr = walk_to_string(tokens, vocab)
        valid = is_valid_walk(tokens, vocab)
        if valid:
            valid_count += 1

        print(f"  Sample {i+1}: `{cstr}` [{'Valid' if valid else 'INVALID'}]")

    print(f"\n  Valid: {valid_count}/{args.num_samples}")
    print("=" * 70)


if __name__ == '__main__':
    main()
