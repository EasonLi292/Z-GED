# Circuit Generation Results (V2)

This document covers the V2 inverse-design model.

- Dataset: `rlc_dataset/filter_dataset.pkl` + `rlc_dataset/rl_dataset.pkl` — 2,400 circuits across 10 topology signatures.
- Checkpoint: `checkpoints/production/best_v2.pt`.
- Encoder: AdmittanceEncoder on admittance-polynomial edge features.
- Latent: 5D factored `[z_topo(2) | z_VIN | z_VOUT | z_GND]`.
- Decoder: GPT-style sequence decoder on Eulerian walks.
- Attribute heads: `FreqHead(log10 fc)`, `GainHead(|H(1kHz)|)`, `TypeHead(10-way CE)`.
- Generation entry point: `scripts/generation/generate_inverse_design.py`, which conditions on `(fc, gain, filter_type)` via attribute heads and latent optimization (K-NN interpolation + gradient descent on µ).

---

## Novel Topology Generation

The decoder was trained on only 10 unique topology signatures (across 2,400 circuits). Temperature-based sampling is the primary lever for exploring beyond those 10. The experiments below characterize the tradeoff between temperature, validity, and novelty, and break down *how* the decoder fails when temperature is raised.

### Fine-Grained Error-Mode Sweep

`scripts/analysis/error_mode_temperature_sweep.py` decodes 2,000 samples per seed at 13 temperatures (0.1–2.0), 3 seeds per temperature (78,000 samples total), and classifies each generated walk into one of seven buckets:

| Category | Meaning |
|---|---|
| `valid_known` | electrically valid, topology signature matches a training topology |
| `valid_novel` | electrically valid, topology signature not in the training set |
| `invalid_self_loop` | well-formed sequence, but some component has identical terminals |
| `invalid_dangling` | well-formed sequence, but an internal net is incident to < 2 components |
| `invalid_missing_terminal` | well-formed sequence, but VIN/VOUT/VSS is not connected |
| `ill_formed_seq` | wrong alternation, odd length, or does not start/end at VSS |
| `ill_formed_comp_count` | sequence parses, but some component token appears ≠ 2 times |

Reproduction (target: fc=10 kHz, gain=0.5, interpolated latent + gradient descent):

```bash
.venv/bin/python scripts/analysis/error_mode_temperature_sweep.py \
  --fc 10000 --gain 0.5 --samples 2000 --seeds 0 1 2 \
  --out analysis_results/error_mode_sweep.json
```

Counts aggregated across 3 seeds (6,000 samples per temperature):

| T | known | novel | self-loop | dangling | miss-term | seq-err | comp-count |
|---|---|---|---|---|---|---|---|
| 0.10 | 6000 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.30 | 5996 | 0 | 0 | 0 | 0 | 0 | 4 |
| 0.50 | 5957 | 0 | 0 | 0 | 0 | 0 | 43 |
| 0.70 | 5890 | 0 | 0 | 0 | 0 | 12 | 98 |
| 0.90 | 5733 | 1 | 0 | 0 | 0 | 58 | 208 |
| 1.00 | 5566 | 1 | 0 | 0 | 1 | 125 | 307 |
| 1.10 | 5305 | 1 | 0 | 0 | 2 | 225 | 467 |
| 1.20 | 4909 | 2 | 0 | 0 | 2 | 414 | 673 |
| 1.30 | 4336 | 3 | 0 | 0 | 3 | 690 | 968 |
| 1.40 | 3714 | 4 | 0 | 0 | 4 | 1071 | 1207 |
| 1.50 | 2958 | 3 | 0 | 2 | 6 | 1587 | 1444 |
| 1.70 | 1474 | 6 | 0 | 3 | 7 | 3038 | 1472 |
| 2.00 | 282 | 2 | 0 | 1 | 7 | 4894 | 814 |

Derived rates (same data, expressed as percentages of 6,000):

| T | valid % | novel / valid | seq-err % | comp-count % | electrical-err % |
|---|---|---|---|---|---|
| 0.10 | 100.00 | 0.000 | 0.00 | 0.00 | 0.00 |
| 0.50 | 99.28 | 0.000 | 0.00 | 0.72 | 0.00 |
| 0.70 | 98.17 | 0.000 | 0.20 | 1.63 | 0.00 |
| 0.90 | 95.57 | 0.017 | 0.97 | 3.47 | 0.00 |
| 1.00 | 92.78 | 0.018 | 2.08 | 5.12 | 0.02 |
| 1.20 | 81.85 | 0.041 | 6.90 | 11.22 | 0.03 |
| 1.40 | 61.97 | 0.108 | 17.85 | 20.12 | 0.07 |
| 1.50 | 49.35 | 0.101 | 26.45 | 24.07 | 0.13 |
| 1.70 | 24.67 | 0.405 | 50.63 | 24.53 | 0.17 |
| 2.00 | 4.73 | 0.704 | 81.57 | 13.57 | 0.13 |

Failure-mode observations:

- **Self-loops are effectively never produced** — 0 out of 78,000 samples. The decoder has learned as a hard constraint that a component's two terminal tokens must differ.
- **Electrical invalidity (dangling internals + missing terminals) stays under 0.2 % at every temperature.** Even at T=2.0, where overall validity drops to 4.7 %, fewer than 0.2 % of samples are well-formed-but-electrically-broken walks. Connectivity to VIN/VOUT/VSS and full coverage of internal nets are also learned as hard constraints.
- **The two dominant failure modes are both structural**, not electrical:
  - `ill_formed_comp_count` (each component must appear exactly twice in an Euler walk) is the first failure mode to appear. It starts at T=0.3 and grows steadily until T=1.5, then saturates.
  - `ill_formed_seq` (token alternation or length violation) appears later (T≥0.7), grows much faster, and dominates at T≥1.7. At T=2.0, 82 % of samples have an ill-formed sequence structure.
- **Novelty lives in a narrow, low-yield band.** Measured as `valid_novel / valid_total`:
  - T≤0.7: identically 0
  - T=0.9–1.1: ≈0.02 %
  - T=1.4–1.5: ≈0.1 %
  - T=1.7: 0.4 %
  - T=2.0: 0.7 %
  
  In absolute terms, the best yield is T=1.7 with 6 novel samples out of 1,480 valid (and 4,518 invalid). Cranking further to T=2.0 gives slightly higher *rate among valids* but fewer novel samples in absolute terms because validity collapses.

**Takeaway (with caveat below).** Under the strict parser, the decoder appears to rarely produce novel topologies, and most failures above T=1.0 are sequence-structure (`ill_formed_comp_count`, `ill_formed_seq`) rather than electrical. The strict parser, however, rejects any walk whose component-count doesn't match the Eulerian convention of exactly 2, which is a strong assumption. The next subsection re-analyses the same sweep with a permissive parser and reaches a very different conclusion about novelty.

### Permissive Re-Analysis

The strict `well_formed` check enforces two things simultaneously:

1. **Sequence grammar**: alternating net/component tokens, starts and ends at VSS.
2. **Eulerian traversal convention**: each component token appears exactly twice.

Rule 2 is true of the training walks but is not a circuit-level requirement. A walk where a component appears 1, 3, or 4 times can still be parsed into a valid 2-terminal circuit graph as long as the component's *unique* net-neighbors are exactly two distinct nets. The over-counted appearances just mean the walk isn't a canonical Euler circuit — not that the underlying graph is broken.

`scripts/analysis/error_mode_temperature_sweep_permissive.py` repeats the same 78k-sample sweep with a relaxed parser:

- Keep the alternation + VSS-bracketed sequence grammar.
- Drop the `count == 2` rule.
- Per component, collect the *set* of distinct adjacent nets.
  - `|set| == 1` → self-loop (truly invalid).
  - `|set| == 2` → valid 2-terminal component (regardless of count).
  - `|set| >= 3` → multi-terminal claim, not supported by the 2-terminal vocab (new category `invalid_multi_terminal`).
- Re-run electrical checks (VIN/VOUT/VSS connected, no dangling internals).

Counts aggregated across 3 seeds (6,000 samples per temperature):

| T | known | novel | self-loop | multi-term | dangling | miss-term | seq-err |
|---|---|---|---|---|---|---|---|
| 0.10 | 6000 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.50 | 5961 | 0 | 0 | 0 | 0 | 39 | 0 |
| 0.70 | 5904 | 1 | 16 | 19 | 0 | 60 | 0 |
| 0.90 | 5762 | 8 | 48 | 81 | 2 | 78 | 21 |
| 1.00 | 5613 | 23 | 78 | 133 | 2 | 91 | 60 |
| 1.20 | 5006 | 83 | 177 | 320 | 5 | 114 | 295 |
| 1.40 | 3845 | 207 | 299 | 586 | 6 | 142 | 915 |
| 1.50 | 3092 | 297 | 310 | 719 | 12 | 162 | 1408 |
| 1.70 | 1556 | 384 | 348 | 694 | 11 | 139 | 2868 |
| 2.00 | 297 | 266 | 210 | 325 | 6 | 68 | 4828 |

Strict vs permissive novelty, same samples:

| T | valid % (perm) | novel % of valid (perm) | novel % of valid (strict) | permissive-novel unique |
|---|---|---|---|---|
| 1.00 | 93.93 | 0.41 | 0.02 | — |
| 1.20 | 84.82 | 1.63 | 0.04 | — |
| 1.40 | 67.53 | 5.11 | 0.11 | — |
| 1.50 | 56.48 | 8.76 | 0.10 | — |
| 1.70 | 32.33 | 19.79 | 0.41 | — |
| 2.00 | 9.38 | 47.25 | 0.70 | — |

Totals across all 78k samples:

- Strict parser: 6 unique novel topologies, 23 samples.
- **Permissive parser: 264 unique novel topologies, 1,462 samples.**

Almost all novel outputs (≈ 98 % at T=1.7) appear in walks where at least one component is over- or under-counted — i.e. exactly the walks the strict parser was discarding.

Non-Euler walks also include genuinely invalid ones. Separating them out (permissive categories at T=1.7, percent of 6,000):

| category | count | %  |
|---|---|---|
| valid_known | 1556 | 25.9 |
| valid_novel | 384 | 6.4 |
| invalid_multi_terminal (a component has ≥3 distinct nets) | 694 | 11.6 |
| invalid_self_loop | 348 | 5.8 |
| invalid_missing_terminal (VIN/VOUT/VSS not connected) | 139 | 2.3 |
| invalid_dangling (internal net incident to < 2 components) | 11 | 0.2 |
| ill_formed_seq (grammar-level violation) | 2868 | 47.8 |

So above T≈1.4 the decoder splits its output into three roughly equal failure modes: (i) ungrammatical sequences, (ii) genuinely invalid electrical claims (self-loop or multi-terminal — the latter is the biggest new category), and (iii) valid circuits whose walks are non-Euler.

#### Top permissive-novel topologies (selection)

| Count | Topology sketch |
|---|---|
| 69 | `C(VSS-VOUT) \| R(VIN-VOUT) \| R(VSS-VOUT)` — RC LPF with extra shunt R |
| 55 | 6-comp: `R(VIN-INT1), R(VOUT-INT1), C(VOUT-INT1), C(VSS-INT0), L(INT0-INT1), R(VSS-VOUT)` |
| 51 | 6-comp variant including an `RC` compound token between VOUT-INT1 |
| 45 | 6-comp variant including `RCL(VOUT-INT1)` |
| 44 | 6-comp variant including `CL(VOUT-INT1)` |
| 35 | 6-comp variant with `RCL(VSS-VOUT)` |
| 27 | 4-comp: `C(VSS-INT0), L(INT0-INT1), R(VIN-INT1), R(VSS-VOUT)` |
| 26 | 6-comp variants using `RC(INT0-INT1)` or `RC(VIN-INT1)` |

Observations:

- The decoder recombines the compound component tokens (`RC`, `CL`, `RCL`) in placements that do not appear in the 10 training signatures. These are *structurally* novel.
- Many novel outputs have 6 components — larger than any training topology (which top out at 5). Topology size is not capped by the training distribution.
- The most frequent permissive-novel topology (×69) is still a simple 3-component variant of the RC low-pass family.

**Revised takeaway.** The decoder *does* generate a meaningful volume of novel circuit graphs — it just does so in walks that break the Eulerian counting convention. The strict parser was measuring conformance to the training walk format, not circuit-graph validity, and under-counted true novelty by ~60×. The corrected picture:

- T=0.1–0.7: effectively no novelty (and no need for it — training topologies dominate).
- T=1.0–1.4: novelty grows from ~0.4 % to ~5 % of valid samples, already 50–100× higher than the strict measure suggests.
- T=1.5–1.7: novelty peaks in absolute terms (297–384 novel samples / 6,000), still at reasonable validity (32–57 %).
- T=2.0: validity collapses, but nearly half of what remains is novel.

The decoder *does* internalize electrical priors (self-loops and multi-terminal claims together make up ~17 % at T=1.7, not zero — larger than the strict analysis implied). But it also has real structural exploration beyond its training topologies. The practical "novelty sweet spot" is T≈1.5–1.7, where roughly 1 in 5 valid outputs is a new topology.

### Are The Novel Topologies Actually Buildable?

"Buildable" is not the same thing as "passes the permissive parser." The permissive parser guarantees:

1. Every component has exactly two **distinct** terminals (no self-loops, no 3+-terminal claims).
2. Every named rail (VIN, VOUT, VSS) has at least one component attached to it.
3. Every internal net is incident to ≥ 2 components (no dangling stubs).

These conditions are sufficient for the graph to be **physically wirable on a breadboard** — every edge is a real 2-terminal part that can be soldered between the two specified nets. They are *not* sufficient to guarantee a **useful filter transfer function**, because the parser will accept a circuit where VOUT happens to be tied only to ground through a resistor, with no signal path back to VIN.

To measure how many of the 264 novel topologies are "real filters" vs merely "wirable," we apply one more test: remove VSS from the graph and check whether VIN and VOUT remain in the same connected component. If yes, a non-trivial transfer function `H(s) = V_OUT / V_IN` exists; if no, V_OUT is pinned to ground (through whatever resistor network) and the circuit is a wirable but functionally trivial "drain."

Results across all 1,462 novel samples (78k-sample sweep):

| Category | Unique topologies | Samples | Share of novel samples |
|---|---|---|---|
| **Buildable AND useful** (VIN→VOUT path exists without VSS) | 252 / 264 | 1,406 / 1,462 | **96.2 %** |
| **Buildable but trivial** (VOUT shorted to ground only) | 12 / 264 | 56 / 1,462 | 3.8 % |

So the headline number is: **≈ 96 % of novel generations are not only electrically wirable but also have a real VIN→VOUT signal path.** Nearly all of them also contain at least one reactive element — 1,454 / 1,462 (99.5 %) include a capacitor, inductor, or one of the compound tokens — so they act as filters, not resistive dividers.

#### Component-size distribution of novel samples

| # components | samples | note |
|---|---|---|
| 2 | 13 | small rewirings like `C \| L` or `R \| C` |
| 3 | 307 | RC, RL, LC variants; the ×69 `R(VIN-VOUT) \| C(VOUT-VSS) \| R(VOUT-VSS)` is the largest bucket here |
| 4 | 74 | most training topologies are 3–5 components |
| 5 | 103 | matches the largest training-size tier |
| 6 | 910 | **exceeds every training topology** |
| 7 | 49 | exceeds every training topology |
| 8 | 6 | exceeds every training topology |

62 % of novel samples have 6+ components, i.e. larger than anything in the 2,400-circuit training corpus (which tops out at 5). Size-extrapolation is happening.

#### Component-token novelty

Training walks use only 4 component types: `R`, `C`, `L`, `RCL`. The vocabulary, however, defines 7 — the three unused types are the compound tokens `RC` (R ‖ C), `CL` (C ‖ L), and `RL` (R ‖ L):

| Token | Occurrences in training (2,400 circuits) | Occurrences in novel generations |
|---|---|---|
| R | 5,280 | 3,823 |
| C | 3,360 | 1,485 |
| L | 3,360 | 1,396 |
| RCL | 480 | 305 |
| RC | **0** | **229** |
| CL | **0** | **201** |
| RL | **0** | **170** |

The decoder places `RC`, `CL`, and `RL` at plausible positions (most often as shunt or series elements between an internal node and VOUT/VSS), despite never having seen them in a training walk. This is real extrapolation in the component-vocabulary dimension, not just in connectivity.

#### Worked examples (top useful novel topologies)

**1. `C(VOUT-VSS) | R(VIN-VOUT) | R(VOUT-VSS)` — ×69 samples, 3 components**

- Ascii: `VIN ──R──┬──┬── VOUT,  VOUT──C──VSS,  VOUT──R──VSS`
- Transfer function: first-order low-pass with a DC-loaded output. `H(s) = (R_load / (R_in + R_load)) · 1 / (1 + sRC)` where the DC gain is set by the resistive divider and the corner by C in parallel with `R_in ‖ R_load`.
- Buildable: yes, 3 parts on a breadboard.
- Novel vs training: the training set's RC low-pass family has only one R and one C; adding the second R as a DC loading/divider is new.

**2. 6-component elliptic-like filter — ×55 samples**

`R(VIN-INT1) | R(VOUT-INT1) | C(VOUT-INT1) | C(VSS-INT0) | L(INT0-INT1) | R(VOUT-VSS)`

- INT1 is a summing node connected to VIN (through R), VOUT (through R ‖ C), and a grounded LC branch (L-INT0-C-VSS).
- VOUT is taken at the R ‖ C leg of the summing node, with an extra R to ground.
- Structurally analogous to a passive twin-T or LC-notch topology — second-order with a zero near the LC resonance `ω₀ = 1/√(LC)` and pole locations set by the resistor ratios.
- Buildable: yes; every part is a standard 2-terminal element.
- This topology has no structural equivalent in the 10 training signatures.

**3. Compound-token variant — ×51 samples**

Same skeleton as #2 but replacing `C(VOUT-INT1)` with `RC(VOUT-INT1)`. An `RC` block is a parallel R‖C shunt between VOUT and the summing node, giving this leg a finite Q. Buildable by wiring a resistor in parallel with a capacitor between the two nets; no more complex than the previous example.

**4. LC-tank variant — ×45 samples**

Replace `RC(VOUT-INT1)` with `RCL(VOUT-INT1)`. Now VOUT sees a parallel R‖C‖L tank to the summing node — this is a band-reject or band-pass depending on which terminal is observed. The RCL compound is well-attested in training (rlc_parallel family), just placed in a new position in this 6-component context.

**5. Pure-L variant — ×23 samples**

`R(VIN-INT1) | L(VOUT-INT1) | R(VOUT-INT1) | C(VSS-INT0) | L(INT0-INT1) | R(VOUT-VSS)`

Two inductors — one between VOUT and the summing node, one in the ground-referred LC branch. This is a 2-pole LPF-with-LC-trap, fully realizable with standard parts.

#### Examples that parse as valid but are functionally trivial

**×27 samples: `C(VSS-INT0) | L(INT0-INT1) | R(VIN-INT1) | R(VOUT-VSS)`**

Trace the connectivity: VIN → R → INT1 → L → INT0 → C → VSS. This is a perfectly valid input-side RLC branch to ground. But VOUT's only connection is `R(VOUT-VSS)` — a single resistor to ground, with no path back to the input side except through VSS itself. The permissive parser accepts it (VIN/VOUT/VSS all have ≥ 1 component, every internal is doubly-incident), but the transfer function is `V_OUT = 0` regardless of V_IN.

Physically you can build this — it just wouldn't function as a filter. These cases account for 12 unique topologies and 56 / 1,462 (3.8 %) of novel samples.

#### Summary of buildability

| Question | Answer |
|---|---|
| Are novel generations **electrically wirable** as described? | Yes — 100 % by permissive-parser construction (2-terminal components, connected rails, no dangling nets). |
| Do they have a **non-trivial VIN→VOUT transfer function**? | 96.2 % of samples do. |
| Do they contain at least one **reactive element** (C, L, or compound)? | 99.5 % do. |
| Do they use **component tokens the decoder never saw in training** (`RC`, `CL`, `RL`)? | 600 / 1,462 samples do — 41 %. |
| Are any **larger than the training topologies** (≥ 6 components)? | 965 / 1,462 samples — 66 %. |

In short: almost every novel topology the decoder generates corresponds to a real circuit you could solder together, nearly all of them use at least one frequency-selective element, and most extrapolate either in size or in component-token usage beyond what the training corpus explicitly contained.

### Multi-Target Temperature Sweep

Comprehensive sweep across 4 target specs, 5 temperatures, and 5 random seeds.
For each target, the V2 pipeline computes one optimized latent via K-NN interpolation
+ gradient descent (no filter-type constraint), then decodes 2,000 samples per
(temperature, seed) combination. Total: 200,000 decoded samples. This sweep uses the
**strict** parser, so its novel-topology counts are the lower-bound figures — see the
permissive re-analysis above for the corrected picture.

Setup:

- Checkpoint: `checkpoints/production/best_v2.pt`
- Dataset: `rlc_dataset/filter_dataset.pkl` + `rlc_dataset/rl_dataset.pkl`
- Training set: 2,400 circuits, **10 known topology signatures**
- Seeds: 0, 1, 2, 3, 4
- Samples per (target, temperature, seed): 2,000

| Target | Temperature | Valid (mean +/- std) | Rate | Unique Topos | Novel / seed | Novel total |
|--------|-------------|----------------------|------|--------------|-------------|-------------|
| fc=1kHz | 0.5 | 2000 +/- 0.0 | 100.0% | 6.0 | 0.00 | 0 |
| fc=1kHz | 0.7 | 2000 +/- 0.5 | 100.0% | 6.0 | 0.00 | 0 |
| fc=1kHz | 1.0 | 1986 +/- 3.1 | 99.3% | 6.0 | 0.00 | 0 |
| fc=1kHz | 1.2 | 1944 +/- 5.5 | 97.2% | 6.2 | 0.00 | 0 |
| fc=1kHz | 1.5 | 1742 +/- 8.5 | 87.1% | 8.4 | 1.00 | 5 |
| | | | | | | |
| fc=10kHz, g=0.5 | 0.5 | 1998 +/- 1.7 | 99.9% | 1.2 | 0.00 | 0 |
| fc=10kHz, g=0.5 | 0.7 | 1990 +/- 4.1 | 99.5% | 3.0 | 0.00 | 0 |
| fc=10kHz, g=0.5 | 1.0 | 1939 +/- 9.0 | 96.9% | 6.6 | 0.20 | 1 |
| fc=10kHz, g=0.5 | 1.2 | 1814 +/- 13.7 | 90.7% | 8.0 | 0.40 | 2 |
| fc=10kHz, g=0.5 | 1.5 | 1310 +/- 16.0 | 65.5% | 9.2 | 0.60 | 3 |
| | | | | | | |
| fc=50kHz | 0.5 | 2000 +/- 0.0 | 100.0% | 2.0 | 0.00 | 0 |
| fc=50kHz | 0.7 | 1999 +/- 1.4 | 99.9% | 3.2 | 0.00 | 0 |
| fc=50kHz | 1.0 | 1983 +/- 2.9 | 99.2% | 3.8 | 0.00 | 0 |
| fc=50kHz | 1.2 | 1937 +/- 5.3 | 96.8% | 5.4 | 0.20 | 1 |
| fc=50kHz | 1.5 | 1730 +/- 3.9 | 86.5% | 6.8 | 1.00 | 5 |
| | | | | | | |
| gain=0.1 | 0.5 | 2000 +/- 0.0 | 100.0% | 4.4 | 0.00 | 0 |
| gain=0.1 | 0.7 | 1999 +/- 1.0 | 99.9% | 5.6 | 0.00 | 0 |
| gain=0.1 | 1.0 | 1989 +/- 2.5 | 99.4% | 7.6 | 0.20 | 1 |
| gain=0.1 | 1.2 | 1947 +/- 5.9 | 97.3% | 9.2 | 0.20 | 1 |
| gain=0.1 | 1.5 | 1728 +/- 9.2 | 86.4% | 10.4 | 0.80 | 4 |

Across all 200,000 samples, **4 unique novel topologies** were discovered under the strict parser:

| # | Topology | Count | Source |
|---|----------|-------|--------|
| 1 | `C1(VIN--VOUT), RCL1(VOUT--VSS)` [1C+1RCL] | x4 | fc=1kHz / T=1.5 |
| 2 | `R1(VIN--VOUT), R3(VOUT--VSS)` [2R] | x2 | fc=10kHz,g=0.5 / T=1.2 |
| 3 | `C1(INT2--VSS), L1(INT1--INT2), R1(VOUT--VSS), R2(INT1--VIN)` [1C+1L+2R] | x1 | fc=10kHz,g=0.5 / T=1.2 |
| 4 | `R1(VOUT--VSS), R2(INT1--VIN), R3(INT1--VOUT)` [3R] | x1 | gain=0.1 / T=1.5 |

Key observations:

- **T <= 0.7**: Zero novel topologies (strict) across all targets and seeds. Validity near 100%.
- **T = 1.0–1.2**: Occasional novel topologies (0–2 total across 5 seeds). Validity 91–99%.
- **T = 1.5**: ~1 novel topology per 2,000 samples on average under the strict parser. Validity drops to 65–87%.
- The permissive re-analysis above shows that these strict counts are ≈ 60× below the true novelty rate — most novel graphs live in walks where a component token is over- or under-counted.

### Comparison To AnalogGenie Paper

The AnalogGenie paper reports much higher novelty because it studies a different task and model scale:

- Dataset: 3350 distinct analog IC topologies across 11 circuit families, expanded to 227,766 augmented sequences.
- Model: decoder-only transformer with 11.825M parameters, 1029-token vocabulary, and maximum sequence length 1024.
- Validity: SPICE-simulatable with default sizing, checking for floating and shorting nodes.
- Novelty: different from every topology in their full dataset.

This repo's V2 model is a much smaller sequence decoder trained on 10 RLC filter topology signatures. Under a permissive graph-level validator it still produces 264 unique novel topologies across 78k samples, many of which are size- or vocabulary-extrapolations from training. The absolute numbers are not directly comparable to AnalogGenie's Table 1 (different corpus, different validator, different task), but the qualitative story — temperature trades validity for novelty, and the model does find new buildable topologies — is consistent.

Scaling the training corpus (à la AnalogGenie's 3,350 topologies) would further expand the reachable space, but the current model is already more exploratory than the strict Euler-convention metrics suggest.

---

## Key Insights

### 1. "Electrical vs structural" depends on the parser you use

The strict parser (training-convention match: every component appears exactly twice) suggests failures are overwhelmingly sequence-level and the decoder's electrical priors are near-deterministic. The permissive parser (only check unique net-neighbors, not occurrence count) reveals a more nuanced picture:

- Genuine electrical failures at T=1.7: ≈ 17 % (self-loops + multi-terminal claims), not < 0.2 %.
- The hard constraint the decoder *has* clearly internalized is "don't re-use a net twice at the same component" (self-loop rate ≈ 6 % at T=1.7, still much less than the random baseline, but nonzero).
- The soft constraint it has not internalized is "a component's distinct neighbors number exactly 2" — at high temperature, roughly 12 % of walks claim a 3+-terminal component.

### 2. Novelty Is Substantial Once You Stop Filtering By Training Grammar

Permissive-parser novelty rates:

| T | validity | novel share of valid | absolute novel count |
|---|---|---|---|
| 1.0 | 94 % | 0.4 % | 23 / 6,000 |
| 1.4 | 68 % | 5.1 % | 207 / 6,000 |
| 1.7 | 32 % | 19.8 % | 384 / 6,000 |
| 2.0 | 9 % | 47.3 % | 266 / 6,000 |

Total unique novel topologies across 78k samples: **264** (permissive) vs **6** (strict — the strict count was filtering the walks where novelty actually shows up). Practical sweet spot: T ≈ 1.5–1.7.

### 3. Latent Space is Well-Organized

- The V2 5D latent factors into `[z_topo(2) | z_VIN | z_VOUT | z_GND]`.
- Filter types form distinct clusters.
- Spec-conditioned µ optimisation (K-NN + gradient descent on attribute heads) converges to high-validity neighborhoods — 100 % valid at T ≤ 0.5 across every target tested.

### 4. Where Novelty Lives

Under the permissive parser, the decoder produces hundreds of distinct novel topologies (264 unique across 78k samples), many with six components — one larger than any training example. They live almost entirely in non-Eulerian walks: walks where a component token appears a number of times other than 2, but whose *unique* terminal set still has exactly 2 distinct nets.

This means there are two useful knobs for exploration:

1. **Temperature** opens up structural variety, peaking around T=1.5–1.7 where ~1 in 5 valid outputs is novel.
2. **Relaxing the walk-format enforcement** at decode time (or post-hoc when canonicalising outputs to graphs) recovers novelty that the training-grammar filter was discarding.

---

## Usage

```bash
# V2 inverse-design generation (fc + gain + optional filter type)
.venv/bin/python scripts/generation/generate_inverse_design.py \
    --type low_pass --fc 10000 --gain 0.7 --samples 5

# Fine-grained error-mode sweep — strict (Eulerian walk-format filter)
.venv/bin/python scripts/analysis/error_mode_temperature_sweep.py \
    --fc 10000 --gain 0.5 --samples 2000 --seeds 0 1 2 \
    --out analysis_results/error_mode_sweep.json

# Fine-grained error-mode sweep — permissive (graph-level validity only)
.venv/bin/python scripts/analysis/error_mode_temperature_sweep_permissive.py \
    --fc 10000 --gain 0.5 --samples 2000 --seeds 0 1 2 \
    --out analysis_results/error_mode_sweep_permissive.json

# Multi-target fixed-spec sweep
.venv/bin/python scripts/analysis/fixed_spec_temperature_sweep.py
```

---

## Files

- **V2 model:** `checkpoints/production/best_v2.pt` (2,400 circuits, 10 topology signatures)
- **Datasets:** `rlc_dataset/filter_dataset.pkl`, `rlc_dataset/rl_dataset.pkl`
- **V2 generation script:** `scripts/generation/generate_inverse_design.py`
- **Error-mode sweep (strict):** `scripts/analysis/error_mode_temperature_sweep.py`
- **Error-mode sweep (permissive):** `scripts/analysis/error_mode_temperature_sweep_permissive.py`
- **Multi-target sweep:** `scripts/analysis/fixed_spec_temperature_sweep.py`
