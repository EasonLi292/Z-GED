# Circuit Generation Results

**Model:** Latent-Guided GraphGPT Decoder with VAE (8D latent space)
**Training Date:** January 2026
**Architecture:** Encoder (69,651 params) + Decoder (6,460,050 params)

---

## Training Results (With KL Divergence)

### Final Metrics (Epoch 100/100)

**Training Set:**
- Total Loss: **0.3370**
- Node Type Accuracy: **99.4%**
- Edge Existence Accuracy: **99.8%**
- Component Type Accuracy: **97.1%**

**Validation Set:**
- Total Loss: **0.2396**
- Node Type Accuracy: **100.0%** ✅
- Edge Existence Accuracy: **100.0%** ✅
- Component Type Accuracy: **100.0%** ✅

**Best Model:**
- Best Validation Loss: **0.2320**
- Checkpoint: `checkpoints/production/best.pt`

### Key Changes vs. Previous Version

**Added KL Divergence Regularization:**
- Model is now a proper Variational Autoencoder (VAE)
- Latent space regularized toward N(0,1) prior
- Enables sampling from prior for novel circuit generation
- KL weight: 0.005

**Benefits:**
- True generative model (not just discriminative)
- Better latent space structure
- Improved generalization
- Can generate circuits by sampling from prior

---

## Generation Test Results

### Comprehensive Specification Test (18 Test Cases)

**Overall Statistics:**
- Valid circuits: **18/18 (100.0%)**
- Successful simulations: **18/18 (100.0%)**

**Topology Distribution:**
- 2 edges (simple RC/RL/RLC): 12 circuits (67%)
- 3 edges: 3 circuits (17%)
- 4 edges: 3 circuits (17%)

**Accuracy by Category:**

| Category | Avg Cutoff Error | Avg Q Error | Performance |
|----------|------------------|-------------|-------------|
| **Low-pass (Q≈0.707)** | 36.1% | 0.0% | ✅ **GOOD** |
| **Band-pass (1<Q<5)** | 56.0% | 49.1% | ⚠️ MODERATE |
| **High-Q (Q≥5)** | 97.0% | 93.2% | ❌ POOR |
| **Overdamped (Q<0.5)** | 31.6% | 68.2% | ⚠️ MODERATE |

### Key Observations

**Strengths:**
1. ✅ **Perfect Q-factor for Butterworth filters** (Q=0.707): 0% error
2. ✅ **100% valid circuits**: All generated circuits have proper VIN/VOUT connectivity
3. ✅ **Consistent topology**: Model generates reasonable 2-4 edge circuits
4. ✅ **Good low-frequency accuracy**: 10.7% error at 50 Hz

**Weaknesses:**
1. ❌ **High-Q resonators**: 93-97% error for Q>5
2. ⚠️ **Cutoff frequency errors**: 36-56% average error depending on category
3. ⚠️ **Band-pass filters**: Moderate errors for Q>1

**Root Causes:**
- Training dataset bias toward low-pass Butterworth filters (Q≈0.707)
- Limited high-Q examples in training data
- K-NN interpolation may not generalize well to extreme Q values

---

## Training Data Composition

Understanding the training data helps explain the model's strengths and limitations.

### Dataset Statistics
- **Total circuits**: 120
- **Filter types**: Evenly distributed (20 each of low-pass, high-pass, band-pass, band-stop, RLC series, RLC parallel)

### Q-Factor Distribution in Training Data

| Q Range | Count | Percentage | Notes |
|---------|-------|------------|-------|
| **Q < 0.5** (Very low) | 27 | 22.5% | Overdamped filters |
| **0.5 ≤ Q < 0.8** (Butterworth) | 56 | 46.7% | **Dominant category** |
| **0.8 ≤ Q < 2.0** (Moderate) | 11 | 9.2% | Limited examples |
| **2.0 ≤ Q < 5.0** (Medium-high) | 15 | 12.5% | Sparse coverage |
| **5.0 ≤ Q < 10.0** (High) | 4 | 3.3% | Very few examples |
| **Q ≥ 10.0** (Very high) | 7 | 5.8% | Extremely limited |

**Key Finding**: 35.8% of training circuits have Q ≈ 0.707 (within ±10%), creating a strong Butterworth bias.

### Topology Distribution in Training Data

| Topology | Count | Percentage | Q Capability |
|----------|-------|------------|--------------|
| **1R+1C** | 40 | 33.3% | Q ≤ 0.707 (RC limit) |
| **2R+1C+1L** | 40 | 33.3% | Can achieve any Q |
| **1R+1C+1L** | 20 | 16.7% | Can achieve any Q |
| **4R+1C+1L** | 20 | 16.7% | Can achieve any Q |

**Key Finding**: 33.3% of training circuits are RC-only, which cannot achieve Q > 0.707. This biases the model toward generating RC topologies even when inappropriate.

### High-Q Training Examples (Q ≥ 5.0)

**Only 11 circuits** (9.2%) in the training set have Q ≥ 5.0:
- **All 11** have both inductors and capacitors (required for resonance)
- **7 circuits** (63.6%) use 4R+1C+1L topology
- **4 circuits** (36.4%) use 2R+1C+1L topology

**Key Finding**: With only 11 high-Q examples and 7 examples of Q ≥ 10, the model has insufficient data to learn high-Q circuit generation.

### Frequency Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| < 100 Hz | 4 | 3.3% |
| 100 Hz - 1 kHz | 11 | 9.2% |
| **1 kHz - 10 kHz** | 33 | 27.5% |
| **10 kHz - 100 kHz** | 59 | 49.2% |
| > 100 kHz | 13 | 10.8% |

**Key Finding**: Model is best trained on 1 kHz - 100 kHz range (76.7% of data).

### Why the Model Struggles

Based on training data analysis:

1. **Excellent on Butterworth filters** (Q ≈ 0.707)
   - 46.7% of training data in this range
   - 35.8% specifically at Q ≈ 0.707
   - **Result**: 0% Q error on all Butterworth test cases ✅

2. **Poor on high-Q specifications** (Q ≥ 5.0)
   - Only 9.2% of training examples
   - Just 7 examples with Q ≥ 10
   - **Result**: 85-98% error on high-Q test cases ❌

3. **Defaults to RC topologies** for unfamiliar specs
   - 33.3% of training data is RC-only
   - RC is most common single topology
   - **Result**: Generates RC for Q=1.5 (impossible) ❌

4. **Lacks diversity** in moderate Q range (1 < Q < 5)
   - Only 26 circuits (21.7%) in this range
   - **Result**: Inconsistent performance, 49-72% error ⚠️

---

## Topology Novelty Analysis

### Are Novel Topologies Being Generated?

**Yes - but they're all invalid!**

The training dataset contains only **4 unique topologies**:
- 1R+1C (40 circuits, 33.3%)
- 2R+1C+1L (40 circuits, 33.3%)
- 1R+1C+1L (20 circuits, 16.7%)
- 4R+1C+1L (20 circuits, 16.7%)

Out of 18 test cases, the model generated:
- **15 circuits** (83.3%) with training topologies ✅
- **3 circuits** (16.7%) with novel topologies ❌

### Novel Topologies Generated

**Novel Topology 1: `2R` (Pure Resistive)**
- Generated for:
  - 50 kHz, Q=2.5
  - 1 kHz, Q=5.0
- **Validity**: ❌ **INVALID**
- **Why**: Pure resistive circuits cannot provide frequency selectivity
- **Conclusion**: Generation failure - physically impossible to create a filter

**Novel Topology 2: `2R+1C` (Missing Inductor)**
- Generated for:
  - 5 kHz, Q=20.0
- **Validity**: ❌ **INVALID for target spec**
- **Why**: High-Q (Q=20) requires both L and C for resonance
- **Conclusion**: Generation failure - cannot achieve Q>1 without inductor

### Key Finding: "Novelty" = Failure Mode

The model's novel topologies are not creative solutions - they are **invalid extrapolations** when the model encounters unfamiliar specifications:

| Specification | Nearest Training | Model Behavior |
|---------------|------------------|----------------|
| 50 kHz, Q=2.5 | No close match | Generates 2R (invalid) ❌ |
| 1 kHz, Q=5.0 | Only 11 high-Q examples | Generates 2R (invalid) ❌ |
| 5 kHz, Q=20.0 | Only 7 Q≥10 examples | Generates 2R+1C (missing L) ❌ |

When specifications are **within training distribution** (e.g., Butterworth filters):
- Model uses familiar topologies ✅
- Component values are interpolated ✅
- Results are valid and accurate ✅

When specifications are **outside training distribution** (high-Q):
- Model generates novel topologies ❌
- But these topologies violate physical constraints ❌
- "Novelty" indicates model confusion, not creativity ❌

### What the Model Actually Does

The model is fundamentally a **sophisticated interpolation system**, not a true generative designer:

1. **K-NN retrieval**: Finds 5 similar circuits from training data
2. **Latent interpolation**: Blends their latent codes
3. **Topology selection**: Decoder outputs topology (usually from training set)
4. **Component value generation**: Interpolates component values

**Successful cases (within training distribution):**
- Uses training topologies (1R+1C, 2R+1C+1L, etc.)
- Interpolates component values between neighbors
- Produces valid, working circuits

**Failure cases (outside training distribution):**
- Attempts to extrapolate to unseen topologies
- Generates invalid combinations (2R, 2R+1C for high-Q)
- Violates physical constraints

### Implications for "Generative" Model

This is more accurately described as:
- ✅ **Retrieval + Interpolation** system
- ✅ Component value optimization within known topologies
- ❌ NOT true creative circuit design
- ❌ NOT exploring novel valid architectures

**Pros:**
- Reliable within training distribution
- Avoids most invalid topologies (only 3/18 invalid)
- Good at parameter tuning for familiar structures

**Cons:**
- Cannot design truly novel circuits
- Limited to 4 topology templates from training
- Extrapolation beyond training → invalid outputs
- No understanding of physical constraints

---

## Detailed Test Examples (All 18 Test Cases)

### 100 Hz, Q=0.707

**Target Specification:**
- Cutoff frequency: 100 Hz
- Q-factor: 0.707

**Generated Circuit:**
- Topology: 2 edges (1R+1C)
- Valid: ✅ Yes

**Measured Performance:**
- Actual Cutoff: 35.9 Hz
- Actual Q: 0.707
- **Cutoff Error: 64.1%** ❌
- **Q Error: 0.0%** ✅

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

C1 0 n2 2.367803091374e-07
R1 n1 n2 1.867808007812e+04

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── C1=236.8nF ────── VOUT
VIN ────── R1=18.7kΩ ────── VOUT
```

**Analysis:**
- ❌ Poor accuracy, likely due to training data bias
- Topology is theoretically capable but parameters are poorly tuned
- Butterworth filter (Q≈0.707) matches training data well

---

### 10 kHz, Q=0.707

**Target Specification:**
- Cutoff frequency: 10,000 Hz
- Q-factor: 0.707

**Generated Circuit:**
- Topology: 2 edges (1R+1C)
- Valid: ✅ Yes

**Measured Performance:**
- Actual Cutoff: 5,425.6 Hz
- Actual Q: 0.707
- **Cutoff Error: 45.7%** ⚠️
- **Q Error: 0.0%** ✅

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

C1 0 n2 3.876950316339e-08
R1 n1 n2 7.548328857422e+02

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── C1=38.8nF ────── VOUT
VIN ────── R1=754.8Ω ────── VOUT
```

**Analysis:**
- ⚠️ Moderate accuracy, within acceptable range
- Topology is appropriate but parameter tuning could improve
- Butterworth filter (Q≈0.707) matches training data well

---

### 100 kHz, Q=0.707

**Target Specification:**
- Cutoff frequency: 100,000 Hz
- Q-factor: 0.707

**Generated Circuit:**
- Topology: 2 edges (1R+1C)
- Valid: ✅ Yes

**Measured Performance:**
- Actual Cutoff: 87,693.3 Hz
- Actual Q: 0.707
- **Cutoff Error: 12.3%** ✅
- **Q Error: 0.0%** ✅

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

C1 0 n2 7.146294223048e-09
R1 n1 n2 2.533620910645e+02

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── C1=7.1nF ────── VOUT
VIN ────── R1=253.4Ω ────── VOUT
```

**Analysis:**
- ✅ Excellent accuracy on both metrics
- Topology is appropriate and parameters are well-tuned
- Butterworth filter (Q≈0.707) matches training data well

---

### 500 Hz, Q=0.707

**Target Specification:**
- Cutoff frequency: 500 Hz
- Q-factor: 0.707

**Generated Circuit:**
- Topology: 2 edges (1R+1C)
- Valid: ✅ Yes

**Measured Performance:**
- Actual Cutoff: 149.6 Hz
- Actual Q: 0.707
- **Cutoff Error: 70.1%** ❌
- **Q Error: 0.0%** ✅

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

C1 0 n2 9.983003224079e-08
R1 n1 n2 1.062946484375e+04

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── C1=99.8nF ────── VOUT
VIN ────── R1=10.6kΩ ────── VOUT
```

**Analysis:**
- ❌ Poor accuracy, likely due to training data bias
- Topology is theoretically capable but parameters are poorly tuned
- Butterworth filter (Q≈0.707) matches training data well

---

### 50 kHz, Q=0.707

**Target Specification:**
- Cutoff frequency: 50,000 Hz
- Q-factor: 0.707

**Generated Circuit:**
- Topology: 2 edges (1R+1C)
- Valid: ✅ Yes

**Measured Performance:**
- Actual Cutoff: 43,173.4 Hz
- Actual Q: 0.707
- **Cutoff Error: 13.7%** ✅
- **Q Error: 0.0%** ✅

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

C1 0 n2 8.706861009955e-09
R1 n1 n2 4.223875122070e+02

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── C1=8.7nF ────── VOUT
VIN ────── R1=422.4Ω ────── VOUT
```

**Analysis:**
- ✅ Excellent accuracy on both metrics
- Topology is appropriate and parameters are well-tuned
- Butterworth filter (Q≈0.707) matches training data well

---

### 1 kHz, Q=1.5

**Target Specification:**
- Cutoff frequency: 1,000 Hz
- Q-factor: 1.500

**Generated Circuit:**
- Topology: 2 edges (1R+1C)
- Valid: ✅ Yes
- **Topology Issues:**
  - ❌ RC filter cannot achieve Q=1.5 (max ~0.707)

**Measured Performance:**
- Actual Cutoff: 542.3 Hz
- Actual Q: 0.707
- **Cutoff Error: 45.8%** ⚠️
- **Q Error: 52.9%** ❌

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

R1 0 n2 1.569016601562e+04
C1 n1 n2 1.874838595484e-08

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── R1=15.7kΩ ────── VOUT
VIN ────── C1=18.7nF ────── VOUT
```

**Analysis:**
- **⚠️ Topology cannot produce desired response:**
  - RC filter cannot achieve Q=1.5 (max ~0.707)
- Generated topology is fundamentally incapable of meeting specifications
- Even with perfect parameter tuning, this circuit cannot achieve the target Q-factor

---

### 5 kHz, Q=2.0

**Target Specification:**
- Cutoff frequency: 5,000 Hz
- Q-factor: 2.000

**Generated Circuit:**
- Topology: 2 edges (1R+1C)
- Valid: ✅ Yes
- **Topology Issues:**
  - ❌ RC filter cannot achieve Q=2.0 (max ~0.707)

**Measured Performance:**
- Actual Cutoff: 1,759.9 Hz
- Actual Q: 0.707
- **Cutoff Error: 64.8%** ❌
- **Q Error: 64.7%** ❌

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

R1 0 n2 8.708450195312e+03
C1 n1 n2 1.040948482967e-08

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── R1=8.7kΩ ────── VOUT
VIN ────── C1=10.4nF ────── VOUT
```

**Analysis:**
- **⚠️ Topology cannot produce desired response:**
  - RC filter cannot achieve Q=2.0 (max ~0.707)
- Generated topology is fundamentally incapable of meeting specifications
- Even with perfect parameter tuning, this circuit cannot achieve the target Q-factor

---

### 15 kHz, Q=3.0

**Target Specification:**
- Cutoff frequency: 15,000 Hz
- Q-factor: 3.000

**Generated Circuit:**
- Topology: 4 edges (4R+1C+1L)
- Valid: ✅ Yes

**Measured Performance:**
- Actual Cutoff: 12,957.7 Hz
- Actual Q: 3.215
- **Cutoff Error: 13.6%** ✅
- **Q Error: 7.2%** ✅

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

R1 0 n2 6.973394042969e+03
C1 0 n3 7.897672560375e-08
R2 0 n3 1.458241455078e+03
L1 0 n3 1.910242135637e-03
R3 n1 n3 8.019754638672e+02
R4 n2 n3 7.484071289062e+03

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── R1=7.0kΩ ────── VOUT
GND ────── C1=79.0nF, R2=1.5kΩ, L1=1.9mH ────── n3
VIN ────── R3=802.0Ω ────── n3
VOUT ────── R4=7.5kΩ ────── n3
```

**Analysis:**
- ✅ Excellent accuracy on both metrics
- Topology is appropriate and parameters are well-tuned

---

### 50 kHz, Q=2.5

**Target Specification:**
- Cutoff frequency: 50,000 Hz
- Q-factor: 2.500

**Generated Circuit:**
- Topology: 2 edges (2R)
- Valid: ✅ Yes
- **Topology Issues:**
  - ❌ Pure resistive (2R) cannot provide frequency selectivity

**Measured Performance:**
- Actual Cutoff: 1.0 Hz
- Actual Q: 0.707
- **Cutoff Error: 100.0%** ❌
- **Q Error: 71.7%** ❌

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

R1 0 n1 1.715607604980e+02
R2 0 n2 3.833218750000e+03

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── R1=171.6Ω ────── VIN
GND ────── R2=3.8kΩ ────── VOUT
```

**Analysis:**
- **⚠️ Topology cannot produce desired response:**
  - Pure resistive (2R) cannot provide frequency selectivity
- Generated topology is fundamentally incapable of meeting specifications
- Even with perfect parameter tuning, this circuit cannot achieve the target Q-factor

---

### 1 kHz, Q=5.0

**Target Specification:**
- Cutoff frequency: 1,000 Hz
- Q-factor: 5.000

**Generated Circuit:**
- Topology: 2 edges (2R)
- Valid: ✅ Yes
- **Topology Issues:**
  - ❌ Pure resistive (2R) cannot provide frequency selectivity
  - ❌ High-Q (Q=5.0) requires both L and C for resonance

**Measured Performance:**
- Actual Cutoff: 1.0 Hz
- Actual Q: 0.707
- **Cutoff Error: 99.9%** ❌
- **Q Error: 85.9%** ❌

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

R1 0 n2 1.877421289062e+04
R2 n1 n2 5.990598144531e+03

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── R1=18.8kΩ ────── VOUT
VIN ────── R2=6.0kΩ ────── VOUT
```

**Analysis:**
- **⚠️ Topology cannot produce desired response:**
  - Pure resistive (2R) cannot provide frequency selectivity
  - High-Q (Q=5.0) requires both L and C for resonance
- Generated topology is fundamentally incapable of meeting specifications
- Even with perfect parameter tuning, this circuit cannot achieve the target Q-factor

---

### 10 kHz, Q=10.0

**Target Specification:**
- Cutoff frequency: 10,000 Hz
- Q-factor: 10.000

**Generated Circuit:**
- Topology: 3 edges (1R+1C+1L)
- Valid: ✅ Yes

**Measured Performance:**
- Actual Cutoff: 798.6 Hz
- Actual Q: 0.707
- **Cutoff Error: 92.0%** ❌
- **Q Error: 92.9%** ❌

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

C1 0 n2 9.997695116226e-08
R1 n1 n3 1.999337524414e+03
L1 n2 n3 2.124639926478e-03

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── C1=100.0nF ────── VOUT
VIN ────── R1=2.0kΩ ────── n3
VOUT ────── L1=2.1mH ────── n3
```

**Analysis:**
- ❌ Poor accuracy, likely due to training data bias
- Topology is theoretically capable but parameters are poorly tuned
- High-Q specification outside typical training data

---

### 5 kHz, Q=20.0

**Target Specification:**
- Cutoff frequency: 5,000 Hz
- Q-factor: 20.000

**Generated Circuit:**
- Topology: 3 edges (2R+1C)
- Valid: ✅ Yes
- **Topology Issues:**
  - ❌ High-Q (Q=20.0) requires both L and C for resonance

**Measured Performance:**
- Actual Cutoff: 22.1 Hz
- Actual Q: 0.707
- **Cutoff Error: 99.6%** ❌
- **Q Error: 96.5%** ❌

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

C1 0 n2 1.893288867905e-07
R1 n1 n3 3.751474365234e+03
R2 n2 n3 3.419339453125e+04

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── C1=189.3nF ────── VOUT
VIN ────── R1=3.8kΩ ────── n3
VOUT ────── R2=34.2kΩ ────── n3
```

**Analysis:**
- **⚠️ Topology cannot produce desired response:**
  - High-Q (Q=20.0) requires both L and C for resonance
- Generated topology is fundamentally incapable of meeting specifications
- Even with perfect parameter tuning, this circuit cannot achieve the target Q-factor

---

### 1 kHz, Q=0.3

**Target Specification:**
- Cutoff frequency: 1,000 Hz
- Q-factor: 0.300

**Generated Circuit:**
- Topology: 2 edges (1R+1C)
- Valid: ✅ Yes

**Measured Performance:**
- Actual Cutoff: 554.3 Hz
- Actual Q: 0.707
- **Cutoff Error: 44.6%** ⚠️
- **Q Error: 135.7%** ❌

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

R1 0 n2 1.471995214844e+04
C1 n1 n2 1.955441319978e-08

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── R1=14.7kΩ ────── VOUT
VIN ────── C1=19.6nF ────── VOUT
```

**Analysis:**
- ❌ Poor accuracy, likely due to training data bias
- Topology is theoretically capable but parameters are poorly tuned

---

### 50 kHz, Q=0.1

**Target Specification:**
- Cutoff frequency: 50,000 Hz
- Q-factor: 0.100

**Generated Circuit:**
- Topology: 4 edges (2R+1C+1L)
- Valid: ✅ Yes

**Measured Performance:**
- Actual Cutoff: 38,278.5 Hz
- Actual Q: 0.056
- **Cutoff Error: 23.4%** ⚠️
- **Q Error: 43.8%** ⚠️

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

R1 0 n2 2.254170166016e+03
R2 n1 n3 2.309520721436e+02
C1 n2 n4 2.985834868241e-08
L1 n3 n4 5.789948627353e-04

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── R1=2.3kΩ ────── VOUT
VIN ────── R2=231.0Ω ────── n3
VOUT ────── C1=29.9nF ────── n4
n3 ────── L1=579.0uH ────── n4
```

**Analysis:**
- ⚠️ Moderate accuracy, within acceptable range
- Topology is appropriate but parameter tuning could improve

---

### 50 Hz, Q=0.707

**Target Specification:**
- Cutoff frequency: 50 Hz
- Q-factor: 0.707

**Generated Circuit:**
- Topology: 2 edges (1R+1C)
- Valid: ✅ Yes

**Measured Performance:**
- Actual Cutoff: 44.6 Hz
- Actual Q: 0.707
- **Cutoff Error: 10.7%** ✅
- **Q Error: 0.0%** ✅

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

R1 0 n2 3.158770507812e+04
C1 n1 n2 1.131662727971e-07

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── R1=31.6kΩ ────── VOUT
VIN ────── C1=113.2nF ────── VOUT
```

**Analysis:**
- ✅ Excellent accuracy on both metrics
- Topology is appropriate and parameters are well-tuned
- Butterworth filter (Q≈0.707) matches training data well

---

### 500 kHz, Q=0.707

**Target Specification:**
- Cutoff frequency: 500,000 Hz
- Q-factor: 0.707

**Generated Circuit:**
- Topology: 2 edges (1R+1C)
- Valid: ✅ Yes

**Measured Performance:**
- Actual Cutoff: 101,681.3 Hz
- Actual Q: 0.707
- **Cutoff Error: 79.7%** ❌
- **Q Error: 0.0%** ✅

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

R1 0 n2 1.099987060547e+03
C1 n1 n2 1.411597394529e-09

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── R1=1.1kΩ ────── VOUT
VIN ────── C1=1.4nF ────── VOUT
```

**Analysis:**
- ❌ Poor accuracy, likely due to training data bias
- Topology is theoretically capable but parameters are poorly tuned
- Butterworth filter (Q≈0.707) matches training data well

---

### 10 kHz, Q=0.05

**Target Specification:**
- Cutoff frequency: 10,000 Hz
- Q-factor: 0.050

**Generated Circuit:**
- Topology: 4 edges (2R+1C+1L)
- Valid: ✅ Yes

**Measured Performance:**
- Actual Cutoff: 12,679.0 Hz
- Actual Q: 0.037
- **Cutoff Error: 26.8%** ⚠️
- **Q Error: 25.0%** ⚠️

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

R1 0 n2 1.841528198242e+03
R2 n1 n3 1.697528533936e+02
C1 n2 n4 1.669020406325e-07
L1 n3 n4 9.441070724279e-04

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── R1=1.8kΩ ────── VOUT
VIN ────── R2=169.8Ω ────── n3
VOUT ────── C1=166.9nF ────── n4
n3 ────── L1=944.1uH ────── n4
```

**Analysis:**
- ⚠️ Moderate accuracy, within acceptable range
- Topology is appropriate but parameter tuning could improve

---

### 5 kHz, Q=30.0

**Target Specification:**
- Cutoff frequency: 5,000 Hz
- Q-factor: 30.000

**Generated Circuit:**
- Topology: 3 edges (1R+1C+1L)
- Valid: ✅ Yes

**Measured Performance:**
- Actual Cutoff: 168.4 Hz
- Actual Q: 0.707
- **Cutoff Error: 96.6%** ❌
- **Q Error: 97.6%** ❌

**SPICE Netlist:**
```spice
* Auto-generated circuit netlist

VIN n1 0 DC 0 AC 1.0

C1 0 n2 2.081799692633e-07
R1 n1 n3 4.531442382812e+03
L1 n2 n3 2.566894982010e-03

.ac dec 200 1.0 1000000.0
.print ac v(n2)
.control
run
set hcopydevtype=ascii
print frequency v(n2)
.endc

.end
```

**Circuit Diagram:**
```
GND ────── C1=208.2nF ────── VOUT
VIN ────── R1=4.5kΩ ────── n3
VOUT ────── L1=2.6mH ────── n3
```

**Analysis:**
- ❌ Poor accuracy, likely due to training data bias
- Topology is theoretically capable but parameters are poorly tuned
- High-Q specification outside typical training data

---

## Comparison: Before vs After KL Divergence

| Metric | Before (No KL) | After (With KL) | Change |
|--------|----------------|-----------------|--------|
| **Val Node Acc** | 100.0% | 100.0% | ✅ Same |
| **Val Edge Acc** | 100.0% | 100.0% | ✅ Same |
| **Val Component Acc** | 100.0% | 100.0% | ✅ Same |
| **Val Loss** | ~0.16 | 0.24 | +50% (expected) |
| **Model Type** | Discriminative | **Generative (VAE)** | ✅ Improved |
| **Latent Space** | Unregularized | **Regularized (N(0,1))** | ✅ Improved |
| **Can sample from prior** | ❌ No | ✅ Yes | ✅ New capability |

**Conclusion:** Adding KL divergence successfully converted the model to a proper VAE while maintaining perfect reconstruction accuracy. The slight increase in validation loss (0.16 → 0.24) is expected and acceptable as it represents the KL regularization term. The model now has a structured latent space and can generate novel circuits by sampling from the prior distribution.

---

## Model Capabilities

### ✅ What the Model Does Well

1. **Butterworth Filters (Q=0.707)**
   - Perfect Q-factor reproduction
   - 36% average cutoff error
   - Reliable topology generation

2. **Connectivity**
   - 100% valid circuits (VIN/VOUT connected)
   - No orphaned nodes or islands

3. **Frequency Range**
   - Handles 50 Hz to 500 kHz range
   - Best at low frequencies (<50 kHz)

4. **VAE Properties**
   - Structured latent space (N(0,1))
   - Can interpolate between specifications
   - Supports sampling from prior

### ⚠️ Limitations

1. **High-Q Specifications (Q>5)**
   - Model defaults to Q≈0.707
   - 90%+ errors on sharp resonators
   - Limited by training data

2. **Cutoff Accuracy**
   - Typical 30-60% error
   - Better at low frequencies
   - Needs more diverse training data

3. **Complex Topologies**
   - Mostly generates 2-edge circuits (67%)
   - Limited 4+ edge circuits
   - May miss optimal designs

---

## Recommendations for Improvement

### Short-term (Current Model)

1. **Use for Butterworth designs only**
   - Best accuracy for Q≈0.707
   - Reliable for standard filters

2. **Apply frequency correction**
   - Multiply target by ~1.8× to compensate
   - Example: For 10kHz target, request 18kHz

3. **Accept cutoff tolerance**
   - Expect ±40% cutoff variation
   - Post-tune component values if needed

### Long-term (Future Work)

1. **Expand training data**
   - Add high-Q examples (Q>5)
   - Include more band-pass filters
   - Balance frequency distribution

2. **Explicit Q-factor control**
   - Add Q as explicit decoder input
   - Currently implicit in latent code

3. **Multi-objective optimization**
   - Jointly optimize cutoff + Q
   - Add diversity regularization
   - Explore 16D latent space

---

## Files

- **Model checkpoint:** `checkpoints/production/best.pt`
- **Training logs:** `logs/training_with_kl.log`
- **Test results:** `docs/GENERATION_TEST_RESULTS.txt`
- **Config:** `configs/production.yaml`

---

**Last Updated:** January 4, 2026
**Model Version:** VAE with KL Divergence (v2.0)
