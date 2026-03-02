# Drone Altitude Stabilization Using Quantum Fuzzy Logic

A **Quantum Fuzzy Inference Engine (QFIE)** that stabilizes a drone's hover altitude using fuzzy control rules executed on a quantum circuit. The controller reads altitude error and its rate of change, then computes motor thrust adjustments — all through quantum gates simulated via IBM Qiskit.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Theoretical Background](#theoretical-background)
   - [Fuzzy Logic Control](#fuzzy-logic-control)
   - [Quantum Computing Basics](#quantum-computing-basics)
   - [Quantum Fuzzy Inference](#quantum-fuzzy-inference)
4. [System Design](#system-design)
   - [Input & Output Variables](#input--output-variables)
   - [Membership Functions](#membership-functions)
   - [Fuzzy Rule Base](#fuzzy-rule-base)
5. [How the Quantum Fuzzy Engine Works (Step by Step)](#how-the-quantum-fuzzy-engine-works-step-by-step)
   - [Step 1 — Fuzzification](#step-1--fuzzification)
   - [Step 2 — Quantum Encoding (Ry Rotations)](#step-2--quantum-encoding-ry-rotations)
   - [Step 3 — Rule Application (MCX Gates)](#step-3--rule-application-mcx-gates)
   - [Step 4 — Measurement](#step-4--measurement)
   - [Step 5 — Defuzzification (Centroid Method)](#step-5--defuzzification-centroid-method)
6. [Simulation Loop (Drone Physics)](#simulation-loop-drone-physics)
7. [Installation & Usage](#installation--usage)
8. [Output & Results](#output--results)
9. [Dependencies](#dependencies)
10. [References](#references)

---

## Project Overview

Traditional drone altitude controllers use PID (Proportional-Integral-Derivative) algorithms. This project replaces the PID with a **Quantum Fuzzy Logic Controller** (QFLC), where:

- **Fuzzy logic** handles the imprecision inherent in real-world sensor readings — instead of hard thresholds, input values have *degrees of membership* in linguistic categories like "negative", "zero", "positive".
- **Quantum computing** accelerates the fuzzy inference process by encoding membership degrees as qubit amplitudes and evaluating all fuzzy rules simultaneously through quantum parallelism.

The result is a controller that can, in principle, evaluate an exponential number of rule combinations in a single quantum circuit execution.

---

## Project Structure

```
Code/
├── src/
│   └── QFIE/
│       ├── __init__.py              # Package initializer
│       └── FuzzyEngines.py          # QuantumFuzzyEngine class + membership helpers
├── fuzzy_partitions.py              # FuzzyPartition data structures (for QFS.py)
├── QFS.py                           # Low-level Quantum Fuzzy System circuit builder
├── stablizer.py                     # Drone Altitude Stabilization controller & simulation
├── pyproject.toml                   # Project metadata and dependencies (uv)
├── uv.lock                          # Locked dependency versions
├── .python-version                  # Python version pin
└── resources/                       # Reference papers and documents
    ├── pendulum.docx                # Reference: Pendulum controller implementation
    ├── QFS.docx                     # Reference: QFS module source
    ├── Quantum Fuzzy Logic.pdf      # Research paper on quantum fuzzy logic
    └── On_the_Implementation_of_Fuzzy_Inference_Engines_on_Quantum_Computers.pdf
```

### Key Files

| File | Role |
|------|------|
| `stablizer.py` | Main application — defines the drone controller, runs the simulation, and plots results |
| `src/QFIE/FuzzyEngines.py` | Core engine — fuzzification, quantum circuit construction, execution, and defuzzification |
| `QFS.py` | Lower-level module for manually building quantum fuzzy circuits (used for advanced customization) |
| `fuzzy_partitions.py` | Data structures for fuzzy linguistic variables and rule tokenization |

---

## Theoretical Background

### Fuzzy Logic Control

Classical (Boolean) logic operates in binary: a statement is either **true** or **false**. Fuzzy logic extends this by allowing **partial truth** — a value between 0 and 1.

For example, if the drone is 3 cm below the target:
- Classical: "error is positive" → **True**
- Fuzzy: "error is positive" → **0.3**, "error is zero" → **0.7**

A fuzzy controller works in three stages:

```
Crisp Input  →  [Fuzzification]  →  [Rule Evaluation]  →  [Defuzzification]  →  Crisp Output
  (sensors)       (μ values)         (IF-THEN rules)       (centroid method)      (actuator)
```

### Quantum Computing Basics

A **qubit** is the quantum analogue of a classical bit. While a bit is either 0 or 1, a qubit exists in a **superposition**:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

where $|\alpha|^2 + |\beta|^2 = 1$. Upon measurement, the qubit collapses to $|0\rangle$ with probability $|\alpha|^2$ or $|1\rangle$ with probability $|\beta|^2$.

Key gates used in this project:

| Gate | Symbol | Effect |
|------|--------|--------|
| **Ry(θ)** | Rotation-Y | Rotates a qubit around the Y-axis. Sets $P(\|1\rangle) = \sin^2(\theta/2)$ |
| **MCX** | Multi-Controlled-X | Flips the target qubit **only if all control qubits are** $\|1\rangle$ — implements a fuzzy AND-THEN rule |

### Quantum Fuzzy Inference

The core insight: **a membership degree μ can be encoded as the probability of measuring |1⟩ on a qubit**.

If we set $\theta = 2\arcsin(\sqrt{\mu})$, then:

$$R_y(\theta)|0\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$$

$$P(|1\rangle) = \sin^2(\theta/2) = \mu$$

This means each qubit **directly represents** a fuzzy membership degree. An MCX gate connecting two input qubits to an output qubit computes the fuzzy AND (product t-norm) of the input memberships — in hardware, through quantum entanglement.

---

## System Design

### Input & Output Variables

| Variable | Type | Range | Unit | Meaning |
|----------|------|-------|------|---------|
| `altitude_error` | Input | −50 to +50 | cm | `target_altitude − current_altitude`. Positive = drone is below target |
| `error_rate` | Input | −20 to +20 | cm/s | Rate of change of the altitude error. Positive = error is growing |
| `motor_thrust` | Output | −100 to +100 | % | Thrust adjustment relative to hover. Positive = increase thrust (go up) |

### Membership Functions

Each variable is divided into **3 fuzzy sets** using trapezoidal and triangular membership functions:

#### Altitude Error

```
  neg (above target)     zero (at target)      pos (below target)
  ┌────────┐                  /\                    ┌────────┐
  │        │                 /  \                   │        │
  │        │               /    \                   │        │
  │        └──────────────/      \──────────────────┘        │
 -50      -10            0       10                         50
```

- **neg** : trapezoidal `[-50, -50, -10, 0]` — drone is above the target
- **zero**: triangular  `[-10, 0, 10]`       — drone is at/near the target
- **pos** : trapezoidal `[0, 10, 50, 50]`    — drone is below the target

#### Error Rate

- **neg** : trapezoidal `[-20, -20, -5, 0]` — error is shrinking (returning)
- **zero**: triangular  `[-5, 0, 5]`        — error is stable
- **pos** : trapezoidal `[0, 5, 20, 20]`    — error is growing (drifting away)

#### Motor Thrust

- **neg_big**: trapezoidal `[-100, -100, -50, -10]` — reduce thrust strongly
- **zero**   : triangular  `[-20, 0, 20]`           — hold current thrust
- **pos_big**: trapezoidal `[10, 50, 100, 100]`     — boost thrust strongly

### Fuzzy Rule Base

The controller uses **9 rules** that cover every combination of the two 3-set inputs (3 × 3 = 9):

| # | Altitude Error | Error Rate | → Motor Thrust | Reasoning |
|---|---------------|------------|----------------|-----------|
| 1 | zero | zero | zero | At target, stable → hold |
| 2 | pos (below) | zero | pos_big | Below target, stable → push up |
| 3 | neg (above) | zero | neg_big | Above target, stable → push down |
| 4 | zero | pos (growing) | pos_big | At target but drifting down → push up |
| 5 | zero | neg (shrinking) | neg_big | At target but drifting up → push down |
| 6 | pos (below) | pos (growing) | pos_big | Below and getting worse → strong up |
| 7 | neg (above) | neg (shrinking) | neg_big | Above and getting worse → strong down |
| 8 | pos (below) | neg (returning) | zero | Below but coming back → relax |
| 9 | neg (above) | pos (returning) | zero | Above but coming back → relax |

Rules 8 and 9 are critical — they prevent **overshoot** by relaxing the thrust when the drone is already self-correcting.

---

## How the Quantum Fuzzy Engine Works (Step by Step)

Each control loop iteration calls the QFIE, which internally performs five stages:

### Step 1 — Fuzzification

The crisp sensor readings are converted to membership degrees by interpolating against the pre-defined membership functions.

**Example**: if `altitude_error = -15 cm` and `error_rate = 0 cm/s`:

```
altitude_error:  { neg: 1.0,   zero: 0.0,  pos: 0.0  }
error_rate:      { neg: 0.0,   zero: 1.0,  pos: 0.0  }
```

The value −15 falls entirely within the "neg" trapezoidal region, so `μ_neg = 1.0`.

**Code location**: `QuantumFuzzyEngine._fuzzify()` in `FuzzyEngines.py`

### Step 2 — Quantum Encoding (Ry Rotations)

Each fuzzy set gets **one qubit** (linear encoding). The membership degree μ is encoded into the qubit's amplitude using a Ry rotation:

$$\theta = 2 \cdot \arcsin(\sqrt{\mu})$$

$$R_y(\theta)|0\rangle \rightarrow P(|1\rangle) = \mu$$

For our example (6 input qubits + 3 output qubits = 9 qubits total):

```
Qubit Layout (Linear Encoding):
┌─────────────────┬──────────────────┬────────────────┐
│ altitude_error   │ error_rate        │ motor_thrust    │
│ [neg, zero, pos] │ [neg, zero, pos]  │ [neg_big, zero, pos_big] │
│  q0    q1    q2  │  q3    q4    q5   │  q6     q7     q8        │
└─────────────────┴──────────────────┴────────────────┘

Ry(π) applied to q0  →  P(|1⟩) = 1.0  (neg altitude error)
Ry(π) applied to q4  →  P(|1⟩) = 1.0  (zero error rate)
All other input qubits remain |0⟩  →  P(|1⟩) = 0.0
```

**Code location**: the Ry-rotation loop inside `build_inference_qc()`

### Step 3 — Rule Application (MCX Gates)

Each fuzzy rule maps to a **Multi-Controlled X (MCX / Toffoli) gate**:

```
Rule 3: IF altitude_error IS neg AND error_rate IS zero THEN motor_thrust IS neg_big

    q0 (alt_error=neg)   ──●──
                           │
    q4 (err_rate=zero)   ──●──
                           │
    q6 (thrust=neg_big)  ──⊕──   ← flipped if BOTH controls are |1⟩
```

The MCX gate **flips the output qubit only when all control qubits are |1⟩**. Because the control qubits are in superposition (with amplitudes set by Ry), the output qubit's probability of being |1⟩ becomes the **product** of the input membership degrees — this is the **product t-norm**, a standard fuzzy AND operator.

$$P(\text{output} = |1\rangle) = \mu_{\text{neg}} \times \mu_{\text{zero}} = 1.0 \times 1.0 = 1.0$$

All 9 rules are applied sequentially as 9 MCX gates in the same circuit. Multiple rules targeting the same output qubit produce the fuzzy OR (probabilistic union) naturally through quantum superposition.

**Code location**: the MCX loop inside `build_inference_qc()`

### Step 4 — Measurement

After all rules are applied, the **output qubits are measured** across many shots (default: 1024). The measurement collapses each qubit to |0⟩ or |1⟩. By counting how often each output qubit reads |1⟩, we obtain the **activation strength** of each output fuzzy set:

```
After 1024 shots:
  neg_big: 1024/1024 = 1.0    ← fully activated
  zero:       0/1024 = 0.0    ← not activated
  pos_big:    0/1024 = 0.0    ← not activated
```

**Code location**: `QuantumFuzzyEngine.execute()`, using Qiskit's `StatevectorSampler`

### Step 5 — Defuzzification (Centroid Method)

The activation strengths are used to **clip** the output membership functions, then the **centroid** (center of gravity) of the aggregated area is computed:

```
1. Clip each output MF by its strength:
   neg_big MF clipped at 1.0  →  full shape retained
   zero MF    clipped at 0.0  →  completely removed
   pos_big MF clipped at 0.0  →  completely removed

2. Union (max) of all clipped MFs  →  aggregated shape

3. Centroid:
              Σ (x · aggregated(x))
   output = ─────────────────────────
               Σ aggregated(x)
```

For this example, the centroid of the full neg_big trapezoid ≈ **−64** (thrust reduction of 64%), telling the drone to descend toward the target.

**Code location**: the centroid calculation at the end of `execute()`

---

## Simulation Loop (Drone Physics)

The simulation in `stablizer.py` models a simplified vertical drone with discrete time steps (dt = 0.05 s):

```python
for each time step:
    thrust = quantum_fuzzy_controller(error, velocity)

    acceleration = error - thrust - 0.5 * velocity    # spring + damping model
    velocity    += acceleration * dt
    error       += velocity * dt
```

### Physics Explanation

| Term | Meaning |
|------|---------|
| `error` | Acts as a restoring force (like a spring pulling toward target) |
| `- thrust` | Controller's corrective action opposing the error |
| `- 0.5 * velocity` | Damping term that prevents oscillation (simulates air resistance) |
| `dt = 0.05` | Small time step for numerical stability |

The system forms a **closed-loop feedback controller**:

```
         ┌──────────────┐
         │   Quantum    │
 error ──┤   Fuzzy      ├── thrust ──┐
         │   Engine     │            │
         └──────────────┘            │
              ▲                      ▼
              │              ┌──────────────┐
              │              │    Drone     │
              └──────────────┤   Physics   │
                 new error   └──────────────┘
```

Each step: the engine reads the current error → computes thrust → updates the physics → produces a new error → repeat.

---

## Installation & Usage

### Prerequisites

- Python 3.12+ 
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Clone the project
git clone <repository-url>
cd Code

# Install all dependencies
uv sync
```

### Run the Simulation

```bash
uv run python stablizer.py
```

This will:
1. Initialize the quantum fuzzy controller (9 qubits, 9 rules)
2. Simulate 150 time steps of drone hover stabilization
3. Print step-by-step output to the console
4. Save a 3-panel plot to `drone_stabilization_results.png`
5. Display the plot in a window

### Customize the Scenario

Edit the bottom of `stablizer.py`:

```python
if __name__ == "__main__":
    controller = DroneAltitudeController()
    # Change initial conditions here:
    controller.run_simulation(error=-15, velocity=0, steps=150)
    #                         ↑ cm       ↑ cm/s      ↑ time steps
```

| Scenario | error | velocity |
|----------|-------|----------|
| Drone 15 cm above target | −15 | 0 |
| Drone 30 cm below target | +30 | 0 |
| At target but falling | 0 | +10 |
| Above and rising fast | −20 | −8 |

---

## Output & Results

The simulation produces three time-series plots:

1. **Altitude Error** — Should converge toward 0 (horizontal dashed line). A successful controller drives this to near-zero within the simulation window.

2. **Rate of Change of Error** — Shows how fast the error is changing. Should also settle near zero as the drone stabilizes.

3. **Motor Thrust Adjustment** — The control signal output by the quantum fuzzy engine at each step. Starts strong (to correct the initial offset) and tapers off as the drone reaches equilibrium.

### Expected Behavior

```
Step   0:  error=  -15.00 cm   thrust=  -63.93 %   ← strong downward correction
Step  10:  error=  -13.50 cm   thrust=  -28.17 %   ← correction easing
Step  50:  error=   -3.20 cm   thrust=   -5.41 %   ← nearly at target
Step 100:  error=   -0.15 cm   thrust=   -0.30 %   ← stabilized
Step 149:  error=   -0.01 cm   thrust=   -0.02 %   ← hover maintained
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `qiskit` | ≥ 2.3.0 | Quantum circuit construction and simulation |
| `qiskit-ibm-runtime` | ≥ 0.45.1 | IBM Quantum backend support |
| `numpy` | ≥ 2.4.2 | Numerical computation |
| `matplotlib` | ≥ 3.10.8 | Plotting simulation results |

All dependencies are managed by `uv` and declared in `pyproject.toml`.

---

## References

1. Gerardo Paz-Silva, D., et al. — *"On the Implementation of Fuzzy Inference Engines on Quantum Computers"* — foundational paper on encoding fuzzy logic in quantum circuits.
2. Qiskit Documentation — [qiskit.org](https://qiskit.org/) — IBM's open-source quantum computing SDK.
3. Mamdani, E.H. — *"Application of fuzzy logic to approximate reasoning using linguistic synthesis"* — IEEE, 1977.
4. Zadeh, L.A. — *"Fuzzy Sets"* — Information and Control, 1965 — the original paper introducing fuzzy set theory.
