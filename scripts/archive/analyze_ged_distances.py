#!/usr/bin/env python3
"""Analyze GED distances within and between filter types."""

import pickle
import numpy as np
from tools.graph_edit_distance import CircuitGED, load_graph_from_dataset


def load_dataset(path='rlc_dataset/filter_dataset.pkl'):
    """Load circuit dataset from pickle file."""
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def analyze_ged_distances():
    """Compute within-type and between-type GED averages."""
    dataset = load_dataset()
    ged_calc = CircuitGED()

    # Group by filter type
    by_type = {}
    for sample in dataset:
        ftype = sample['filter_type']
        if ftype not in by_type:
            by_type[ftype] = []
        by_type[ftype].append(sample)

    # Sample circuits per type
    filter_types = sorted(by_type.keys())
    samples_per_type = 5
    sampled = {ft: by_type[ft][:samples_per_type] for ft in filter_types}

    print("="*80)
    print("GED DISTANCE ANALYSIS: WITHIN-TYPE vs BETWEEN-TYPE")
    print("="*80)
    print(f"\nFilter types: {filter_types}")
    print(f"Samples per type: {samples_per_type}")
    print()

    # Convert to graphs
    graphs_by_type = {}
    for ft in filter_types:
        graphs_by_type[ft] = [load_graph_from_dataset(s['graph_adj'])
                              for s in sampled[ft]]

    # Compute within-type GEDs
    print("\n" + "="*80)
    print("WITHIN-TYPE DISTANCES (same filter type)")
    print("="*80)
    within_results = {}

    for ft in filter_types:
        graphs = graphs_by_type[ft]
        within_geds = []

        for i in range(len(graphs)):
            for j in range(i+1, len(graphs)):
                ged = ged_calc.compute_ged(graphs[i], graphs[j], timeout=10)
                within_geds.append(ged)

        avg_within = np.mean(within_geds) if within_geds else 0
        std_within = np.std(within_geds) if within_geds else 0
        within_results[ft] = (avg_within, std_within)

        print(f"{ft:<20} avg={avg_within:>8.4f}  std={std_within:>8.4f}")

    # Compute between-type GEDs
    print("\n" + "="*80)
    print("BETWEEN-TYPE DISTANCES (different filter types)")
    print("="*80)
    between_results = {}

    for i, ft1 in enumerate(filter_types):
        for j, ft2 in enumerate(filter_types):
            if i >= j:
                continue

            graphs1 = graphs_by_type[ft1]
            graphs2 = graphs_by_type[ft2]

            between_geds = []
            for g1 in graphs1:
                for g2 in graphs2:
                    ged = ged_calc.compute_ged(g1, g2, timeout=10)
                    between_geds.append(ged)

            avg_between = np.mean(between_geds)
            std_between = np.std(between_geds)
            between_results[(ft1, ft2)] = (avg_between, std_between)

            print(f"{ft1:<15} <-> {ft2:<15} avg={avg_between:>8.4f}  std={std_between:>8.4f}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    within_avgs = [v[0] for v in within_results.values()]
    between_avgs = [v[0] for v in between_results.values()]

    print(f"\nWithin-type distances:")
    print(f"  Mean:   {np.mean(within_avgs):>8.4f}")
    print(f"  Median: {np.median(within_avgs):>8.4f}")
    print(f"  Min:    {np.min(within_avgs):>8.4f}")
    print(f"  Max:    {np.max(within_avgs):>8.4f}")

    print(f"\nBetween-type distances:")
    print(f"  Mean:   {np.mean(between_avgs):>8.4f}")
    print(f"  Median: {np.median(between_avgs):>8.4f}")
    print(f"  Min:    {np.min(between_avgs):>8.4f}")
    print(f"  Max:    {np.max(between_avgs):>8.4f}")

    print(f"\nSeparation ratio (between/within):")
    print(f"  {np.mean(between_avgs) / np.mean(within_avgs):.2f}x")

    # Highlight specific comparisons
    print("\n" + "="*80)
    print("KEY COMPARISONS FOR ML")
    print("="*80)

    # Low-pass vs High-pass (check both orderings)
    lp_hp_key = None
    if ('low_pass', 'high_pass') in between_results:
        lp_hp_key = ('low_pass', 'high_pass')
    elif ('high_pass', 'low_pass') in between_results:
        lp_hp_key = ('high_pass', 'low_pass')

    if lp_hp_key:
        lp_hp_dist, lp_hp_std = between_results[lp_hp_key]
        lp_within, _ = within_results['low_pass']
        hp_within, _ = within_results['high_pass']

        print(f"\nLow-pass <-> High-pass:")
        print(f"  Between-type distance: {lp_hp_dist:.4f} ± {lp_hp_std:.4f}")
        print(f"  Low-pass within:       {lp_within:.4f}")
        print(f"  High-pass within:      {hp_within:.4f}")
        print(f"  Separation ratio:      {lp_hp_dist / max(lp_within, hp_within):.2f}x")

        if lp_hp_dist > max(lp_within, hp_within) * 3:
            print(f"  ✅ GOOD separation (>3x) - ML should distinguish easily")
        elif lp_hp_dist > max(lp_within, hp_within):
            print(f"  ⚠️  MODERATE separation ({lp_hp_dist / max(lp_within, hp_within):.2f}x) - ML may need careful training")
        else:
            print(f"  ❌ POOR SEPARATION - CRITICAL ISSUE!")
            print(f"     Between-type distance is SMALLER than within-type!")
            print(f"     ML will confuse low-pass and high-pass filters!")

    # Band-pass vs Band-stop
    if ('band_pass', 'band_stop') in between_results:
        bp_bs_dist, bp_bs_std = between_results[('band_pass', 'band_stop')]
        bp_within, _ = within_results['band_pass']
        bs_within, _ = within_results['band_stop']

        print(f"\nBand-pass <-> Band-stop:")
        print(f"  Between-type distance: {bp_bs_dist:.4f} ± {bp_bs_std:.4f}")
        print(f"  Band-pass within:      {bp_within:.4f}")
        print(f"  Band-stop within:      {bs_within:.4f}")
        print(f"  Separation ratio:      {bp_bs_dist / max(bp_within, bs_within):.2f}x")

        if bp_bs_dist < max(bp_within, bs_within):
            print(f"  ❌ POOR SEPARATION - Classes overlap!")

    print("\n" + "="*80)


if __name__ == '__main__':
    analyze_ged_distances()
