"""Constants for circuit models."""

# Pole/zero normalization scale: signed_log(x) = sign(x) * log10(|x| + 1) / PZ_LOG_SCALE
PZ_LOG_SCALE = 7.0

# Filter type ordering (must match dataset)
FILTER_TYPES = ['low_pass', 'high_pass', 'band_pass', 'band_stop', 'rlc_series', 'rlc_parallel', 'lc_lowpass', 'cl_highpass']

# Circuit templates for each filter type
CIRCUIT_TEMPLATES = {
    'low_pass': {
        'num_nodes': 3,
        # Canonical order: sort by (source, target)
        'edges': [(0, 2), (1, 2), (2, 0), (2, 1)],
        'num_components': 2,  # R, C
        'node_types': [0, 1, 2]  # [GND, VIN, VOUT]
    },
    'high_pass': {
        'num_nodes': 3,
        'edges': [(0, 2), (1, 2), (2, 0), (2, 1)],
        'num_components': 2,  # C, R
        'node_types': [0, 1, 2]
    },
    'band_pass': {
        'num_nodes': 4,
        # Canonical order: (0,2), (0,3), (1,2), (2,0), (2,1), (2,3), (3,0)
        'edges': [(0, 2), (0, 3), (1, 2), (2, 0), (2, 1), (2, 3), (3, 0)],
        'num_components': 3,  # R, L, C
        'node_types': [0, 1, 2, 3]  # [GND, VIN, VOUT, INTERNAL]
    },
    'band_stop': {
        'num_nodes': 5,
        # Canonical order: sorted by (source, target)
        'edges': [(0, 2), (0, 4), (1, 2), (2, 0), (2, 1), (2, 3), (3, 2), (3, 4), (4, 0)],
        'num_components': 6,
        'node_types': [0, 1, 2, 3, 3]  # [GND, VIN, VOUT, INTERNAL, INTERNAL]
    },
    'rlc_series': {
        'num_nodes': 5,
        # Canonical order: sorted by (source, target)
        'edges': [(0, 2), (0, 4), (1, 2), (2, 0), (2, 1), (2, 3), (3, 2), (3, 4), (4, 0)],
        'num_components': 4,  # R, L, C, R_series
        'node_types': [0, 1, 2, 3, 3]
    },
    'rlc_parallel': {
        'num_nodes': 4,
        # Canonical order: (0,2), (0,3), (1,2), (2,0), (2,1), (2,3), (3,0)
        'edges': [(0, 2), (0, 3), (1, 2), (2, 0), (2, 1), (2, 3), (3, 0)],
        'num_components': 4,  # R, L, C, R_parallel
        'node_types': [0, 1, 2, 3]
    },
    'lc_lowpass': {
        'num_nodes': 3,
        'edges': [(0, 2), (1, 2), (2, 0), (2, 1)],
        'num_components': 2,  # L, C
        'node_types': [0, 1, 2]
    },
    'cl_highpass': {
        'num_nodes': 3,
        'edges': [(0, 2), (1, 2), (2, 0), (2, 1)],
        'num_components': 2,  # C, L
        'node_types': [0, 1, 2]
    }
}
