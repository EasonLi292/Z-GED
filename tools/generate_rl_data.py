"""Generate RL low-pass and high-pass filter dataset (240 each)."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import uuid
import numpy as np
import networkx as nx

from tools.circuit_generator import (
    FilterGenerator,
    extract_poles_zeros_gain_analytical,
    create_compact_graph_representation,
)

DATASET_DIR = "rlc_dataset"
NUM_SAMPLES = 240


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    dataset = []

    for filter_type in ['rl_lowpass', 'rl_highpass']:
        print(f"Generating {NUM_SAMPLES} {filter_type} circuits...")

        for i in range(NUM_SAMPLES):
            gen = FilterGenerator()

            if filter_type == 'rl_lowpass':
                char_freq = gen.generate_rl_lowpass_filter()
            else:
                char_freq = gen.generate_rl_highpass_filter()

            poles, zeros, gain = extract_poles_zeros_gain_analytical(
                filter_type, gen.components
            )
            ml_graph = create_compact_graph_representation(gen.graph, filter_type)

            # Synthetic frequency response from poles/zeros
            freqs = np.logspace(1, 8, 100)
            s = 1j * 2 * np.pi * freqs
            H = np.ones_like(s, dtype=complex) * gain
            for z in zeros:
                H *= (s - z)
            for p in poles:
                H /= (s - p)

            data_point = {
                'id': uuid.uuid4().hex,
                'filter_type': filter_type,
                'characteristic_frequency': char_freq,
                'components': gen.components,
                'graph_adj': nx.adjacency_data(ml_graph),
                'frequency_response': {
                    'freqs': freqs,
                    'H_magnitude': np.abs(H),
                    'H_phase': np.angle(H),
                    'H_complex': H,
                },
                'label': {
                    'poles': poles if poles is not None else [],
                    'zeros': zeros if zeros is not None else [],
                    'gain': gain if gain is not None else 0.0,
                },
            }
            dataset.append(data_point)

            if (i + 1) % 60 == 0:
                print(f"  {filter_type} {i+1}/{NUM_SAMPLES} | fc={char_freq:.2f} Hz")

    output_file = os.path.join(DATASET_DIR, "rl_dataset.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\nSaved {len(dataset)} circuits to {output_file}")

    # Verify
    for ft in ['rl_lowpass', 'rl_highpass']:
        subset = [d for d in dataset if d['filter_type'] == ft]
        freqs = [d['characteristic_frequency'] for d in subset]
        print(f"  {ft}: n={len(subset)}, freq range [{min(freqs):.1f}, {max(freqs):.1f}] Hz")


if __name__ == '__main__':
    main()
