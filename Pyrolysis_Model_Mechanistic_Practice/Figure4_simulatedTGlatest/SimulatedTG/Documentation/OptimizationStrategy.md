# Multi-Mechanism Optimization Strategy

The simulation system utilizes a state-of-the-art **Modular Adaptive Multi-Mechanism Unified GA Framework** to resolve optimal solid-state pyrolysis kinetic pathways. By integrating deep neural network predictions ($E_a(T)$, $w_{\infty}$) with a multi-objective penalized genetic algorithm, the system identifies the most physically consistent reaction mechanisms without human bias.

---

## I. Unified Genetic Algorithm (GA) Optimization

The optimization is framed as a **Mixed-Integer Nonlinear Programming (MINLP)** problem solved using MATLAB's `ga` solver. It co-optimizes discrete mechanism selections, continuous mechanism exponents ($n$), and branch weight fractions ($w_i$).

### 1. Chromosome Representation (GA Variables)
The GA vector `X` is dynamically sized depending on the multi-mechanism mode:
*   **Series Mode**:
    $$\text{nvars} = 1 + 2 \times k$$
    *   $X_1$: Dummy weight factor.
    *   $X_{2 \dots k+1}$: Continuous exponents $n$ for each step.
    *   $X_{k+2 \dots 2k+1}$: Discrete mechanism library indices (`intcon`).
*   **Parallel Mode**:
    $$\text{nvars} = 3 \times k$$
    *   $X_{1 \dots k}$: Branch weights $w_i$.
    *   $X_{k+1 \dots 2k}$: Continuous exponents $n$ for each branch.
    *   $X_{2k+1 \dots 3k}$: Discrete mechanism library indices (`intcon`).
*   **Hybrid Mode** ($P$ branches, $S$ steps per branch):
    $$\text{nvars} = P + 2 \times (P \times S)$$
    *   $X_{1 \dots P}$: Branch weights $w_j$.
    *   $X_{P+1 \dots P+PS}$: Exponents $n$ for each step.
    *   $X_{P+PS+1 \dots nvars}$: Discrete mechanism library indices (`intcon`).

---

## II. Double Closed-Loop Safeguards for Mass Conservation

In Parallel and Hybrid modes, the physical requirement that branch weight fractions sum to exactly 1 ($\sum w_i = 1$) is enforced through two redundant safeguards to bypass MATLAB's MINLP equality constraint limitations:

1.  **Safeguard 1 (Calculation End - `multiModelFitness.m`)**:
    Before evaluating the kinetic integration and error, the chromosome weights slice is normalized locally:
    $$w_i = \frac{X_i}{\sum_{j=1}^{k} X_j}$$
    This guarantees that the simulated TG curve always represents physical mass conservation.
2.  **Safeguard 2 (Output End - `reconstructMultiModelParams.m`)**:
    During final parameter reconstruction and saving, the weights are explicitly normalized to prevent numerical floating-point noise from violating conservation in output datasets.

---

## III. DTG-Driven Adaptive Weighting & Multi-Objective Penalties

To prevent over-fitting and ensure physical correctness, the error metric in `calculateErrorWithKinetics.m` does not use naive RMSE. It evaluates a highly penalized, multi-objective fitness value:

### 1. DTG-Driven Adaptive Weighting
The weight $W(T)$ at each temperature point is calculated dynamically from the experimental derivative thermogravimetry (DTG) curve:
*   **Base Weight**: Scaled from 1 to 6 proportionally to mass loss rate magnitude:
    $$W_{\text{base}}(T) = 1 + 5 \times \frac{|\text{DTG}(T)|}{\text{DTG}_{\text{max}}}$$
*   **Peak Acceleration Boost**: Multiplies weights by $1.3\times$ in active decomposition zones ($\pm 30^{\circ}\text{C}$ around DTG peaks) to force precise reaction-profile alignment.
*   **Moisture Suppression**: Suppresses weights to $0.3 - 0.5$ in low-activity zones ($T < 150^{\circ}\text{C}$) to ignore drying effects.
*   **Lignin Tail Boost**: Imposes a minimum weight of $2.5$ at high temperatures ($T > 450^{\circ}\text{C}$) to force the solver to respect continuous lignin slow-devolatilization.

### 2. The Comprehensive Fitness Formulation
The total fitness error minimized by the GA solver is:
$$\text{Fitness} = \text{Weighted RMSE} + \text{Shape Penalty} + \text{Range Penalty} + \text{Endpoint Penalty} + T_{50}\text{ Penalty} + \text{DTG Penalty}$$

*   **Weighted RMSE**:
    $$\text{Weighted RMSE} = \sqrt{\frac{1}{N} \sum_{j=1}^{N} W(T_j) \cdot \left(\frac{w_{\text{pred}}(T_j) - w_{\text{exp}}(T_j)}{w_{\text{exp}}(T_j) + 10^{-6}}\right)^2} \times 100$$
*   **Shape Penalty**: $+50$ if the predicted curve is non-monotonic.
*   **Range Penalty**: $+30$ or $+20$ if the predicted temperature span deviates from the experimental boundaries ($20-900^{\circ}\text{C}$).
*   **Endpoint Penalty**: $\text{endpoint\_error} \times 5$ if the final weight deviates from the neural network char yield prediction ($w_{\infty}$).
*   **$T_{50}$ Midpoint Penalty**: Aligns the midpoint temperature of decomposition:
    $$\text{Penalty}_{T_{50}} = 0.15 \times |T_{50, \text{pred}} - T_{50, \text{exp}}|$$
*   **DTG Rate Matching Penalty**: Forces mass loss rates (derivatives) to match:
    $$\text{Penalty}_{\text{DTG}} = 100.0 \times \sqrt{\frac{1}{M} \sum_{\text{focus}} (\text{DTG}_{\text{pred}} - \text{DTG}_{\text{exp}})^2}$$

---

## IV. Two-Stage Coarse-to-Fine GA Presets (Optional)

For large-scale, high-performance computing (HPC) environments scaling up to 192 cores, the framework supports a **Two-Stage Coarse-to-Fine GA screening** controlled by `performanceConfig.m`:

1.  **Stage 1: Coarse Screening**
    Runs a fast GA with low population size and low generation limits (`multiCoarse` presets) across all combinations to quickly rank and identify the Top-N mechanism topologies.
2.  **Stage 2: Fine Optimization**
    Applies high-resolution GA parameters (`multiFine` presets: high population, tight tolerances) to the Top-N candidates to obtain mathematically precise parameter convergence.