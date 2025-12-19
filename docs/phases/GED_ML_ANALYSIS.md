# GED Analysis for ML-Based Circuit Generation

## Problem Summary

**Critical Issue**: Low-pass and high-pass filters have nearly identical Graph Edit Distance (GED), which will severely impact ML classification performance.

### Key Findings

```
Low-pass <-> High-pass:
  Between-type distance: 0.0357
  Low-pass within-type:  0.0473
  High-pass within-type: 0.0096
  Separation ratio:      0.75x  ❌ CRITICAL
```

**The between-class distance is SMALLER than the within-class distance!** This means two low-pass filters can be more different from each other than a low-pass filter is from a high-pass filter.

## Root Cause Analysis

### Identical Graph Structures

Both low-pass and high-pass RC filters have:
- **Same topology**: 3 nodes (GND, VIN, VOUT), 2 edges
- **Same components**: 1 resistor (R), 1 capacitor (C)
- **Only difference**: Which edge has which component

**Low-pass filter:**
```
VIN ---R---> VOUT
        |
        C
        |
       GND
```
- Edge (VIN, VOUT): Resistor (G = conductance)
- Edge (VOUT, GND): Capacitor (C)

**High-pass filter:**
```
VIN ---C---> VOUT
        |
        R
        |
       GND
```
- Edge (VIN, VOUT): Capacitor (C)
- Edge (VOUT, GND): Resistor (G = conductance)

### Why Current GED Fails

The current GED implementation:
1. ✅ Correctly identifies identical graph topology (0 cost)
2. ✅ Computes impedance distance for edge substitution
3. ❌ **Does NOT** capture that components are in different functional positions

The impedance distance `d = sqrt(w_C*(C1-C2)² + w_G*(G1-G2)² + w_L*(L1-L2)²)` only measures component VALUE differences, not component PLACEMENT differences.

### Detailed Example

```
Low-pass #1:
  Edge (GND, VOUT):  C = 2.00e-08
  Edge (VIN, VOUT):  G = 8.56e-03

High-pass #1:
  Edge (GND, VOUT):  G = 5.00e-05
  Edge (VIN, VOUT):  C = 4.73e-08

GED = 0.0118 (very small!)
```

Even though these circuits have completely different frequency responses, the GED sees them as very similar because the graph structure is identical.

## Impact on ML Performance

### Will It Affect ML?

**YES - Severely**

An ML model trained on GED features alone will:
- ❌ **Struggle to distinguish** low-pass from high-pass filters
- ❌ **High classification error** between these two classes
- ❌ **Poor generative performance** - may generate wrong filter type
- ⚠️  **May still work** for other filter types with distinct topologies:
  - Band-pass vs Band-stop: 2.0285 distance (1.12x separation)
  - Most other comparisons: 2.5+ distance (good separation)

### Your Goal: Specification → Circuit Generation

For your use case ("tell a circuit specification, then appropriate circuit is generated"), the low-pass/high-pass confusion is **CRITICAL** because:

1. **User specifies**: "I want a low-pass filter at 1kHz"
2. **ML model** might generate a high-pass filter (GED says they're nearly identical)
3. **Result**: Wrong circuit type, even though frequency might be correct

## Solutions

### Option 1: Enhanced GED with Positional Information ⭐ RECOMMENDED

Modify the GED edge matching to encode edge position/connectivity:

```python
def _edge_positional_cost(self, edge1, edge2, graph1, graph2):
    """
    Add cost based on edge position in circuit.

    Distinguish between:
    - Input edges (connected to VIN)
    - Output edges (connected to VOUT)
    - Ground edges (connected to GND)
    """
    u1, v1 = edge1
    u2, v2 = edge2

    # Get node types
    type1_u = get_node_type(graph1, u1)  # GND, VIN, VOUT, or INTERNAL
    type1_v = get_node_type(graph1, v1)
    type2_u = get_node_type(graph2, u2)
    type2_v = get_node_type(graph2, v2)

    # Edges must connect same node types
    edge_signature1 = frozenset([type1_u, type1_v])
    edge_signature2 = frozenset([type2_u, type2_v])

    if edge_signature1 != edge_signature2:
        return float('inf')  # Cannot match edges in different positions

    # Regular impedance distance for edges in same position
    return self.impedance_distance(...)
```

**Impact**: This would increase low-pass ↔ high-pass GED dramatically (to ~1.0+), providing clear separation.

**Pros**:
- Solves the root cause
- Pure GED-based ML would work
- Maintains graph-theoretic foundations

**Cons**:
- Requires modifying GED implementation
- More restrictive matching (may increase computational cost)

### Option 2: Hybrid Features (GED + Transfer Function) ⭐ RECOMMENDED

Combine GED with functional circuit characteristics:

```python
features = {
    'ged_distance': ged_to_reference,
    'num_poles': len(poles),
    'num_zeros': len(zeros),
    'pole_frequencies': [abs(p) for p in poles],
    'zero_frequencies': [abs(z) for z in zeros],
    'dc_gain': abs(H(0)),
    'hf_gain': abs(H(inf)),
    'filter_order': max(len(poles), len(zeros))
}
```

**Why this works**:
- Low-pass: pole at low freq, zero at infinity → DC gain > HF gain
- High-pass: zero at DC, pole at high freq → HF gain > DC gain
- These functional features are VERY different even though GED is small

**Pros**:
- Best of both worlds: topology (GED) + function (transfer function)
- Most robust for generation task
- You already have transfer function data!

**Cons**:
- More complex feature engineering
- Need to ensure both feature types are properly weighted

### Option 3: Adjust GED Weights

Try increasing impedance distance weights to amplify small differences:

```python
ged_calc = CircuitGED(
    w_C=1e15,      # Was 1e12, increase by 1000x
    w_G=1e5,       # Was 1e2, increase by 1000x
    w_L_inv=1.0    # Was 1e-3, increase by 1000x
)
```

**Pros**:
- Simplest to implement (just parameter tuning)
- No code changes needed

**Cons**:
- May not solve the fundamental problem (components are in different positions)
- Could negatively affect other filter type comparisons
- Empirical tuning without theoretical justification

### Option 4: Use Transfer Function Features Only

Skip GED entirely, use only frequency-domain features:

```python
features = {
    'poles': [...],
    'zeros': [...],
    'magnitude_response': [|H(f1)|, |H(f2)|, ...],
    'phase_response': [∠H(f1), ∠H(f2), ...],
    'filter_type_classifier': neural_network(magnitude_response)
}
```

**Pros**:
- Directly captures functional behavior
- Clear discrimination between filter types

**Cons**:
- Loses graph structure information (topology)
- May struggle with circuits that have similar frequency response but different implementations
- Your circuit generation still needs to output a graph structure

## Recommended Approach

### For Your Use Case: **Option 2 (Hybrid) + Option 1 (Enhanced GED)**

1. **Short-term (Quick fix)**:
   - Use Option 2: Add transfer function features to your ML pipeline
   - Features to add:
     - DC gain vs HF gain ratio
     - Pole/zero locations
     - Magnitude response at key frequencies
   - This will immediately solve the classification problem

2. **Long-term (Better GED)**:
   - Implement Option 1: Enhance GED with positional edge matching
   - This makes GED itself more meaningful for circuit similarity
   - Better for retrieval, clustering, and nearest-neighbor search

### Implementation Priority

```python
# Phase 1: Hybrid features (2-3 days)
def extract_features(circuit):
    G = load_graph_from_dataset(circuit['graph_adj'])

    # GED to prototype circuits
    ged_features = {
        f'ged_to_{ftype}': min([
            ged_calc.compute_ged(G, prototype)
            for prototype in prototypes[ftype]
        ])
        for ftype in ['low_pass', 'high_pass', 'band_pass', 'band_stop']
    }

    # Transfer function features
    tf_features = {
        'dc_gain': circuit['dc_gain'],
        'hf_gain': circuit['hf_gain'],
        'gain_ratio': circuit['dc_gain'] / (circuit['hf_gain'] + 1e-10),
        'cutoff_freq': circuit['characteristic_frequency'],
        'num_poles': len(circuit['poles']),
        'num_zeros': len(circuit['zeros'])
    }

    return {**ged_features, **tf_features}

# Phase 2: Enhanced GED (1 week)
# Implement positional edge matching in graph_edit_distance.py
```

## Expected Results

### With Current GED Only
- Low-pass/high-pass classification accuracy: ~60-70% (near random)
- Overall accuracy: ~75-80% (other classes are OK)

### With Hybrid Features
- Low-pass/high-pass classification accuracy: ~95-99%
- Overall accuracy: ~95-99%
- **Meets your generation requirements** ✅

### With Enhanced GED
- Low-pass/high-pass GED separation: 0.75x → **5-10x**
- All filter types clearly separated
- **Enables pure GED-based ML** ✅

## Conclusion

**Answer to your question**: Yes, the low GED between low-pass and high-pass will significantly affect ML function. Your ML model will struggle to distinguish between these two filter types using GED alone.

**Solution**: Use a hybrid approach combining GED (for topology) with transfer function features (for functionality). This will enable reliable specification-to-circuit generation.

The high GED values for other filter comparisons (band-pass, band-stop, etc.) are actually GOOD - they indicate clear separation between those classes. The low-pass/high-pass issue is specific and fixable.
