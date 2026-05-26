# Kinetic Model Categories & Unified Mechanism Library

The simulation system utilizes a unified heterogeneous kinetic mechanism library. Mechanisms are structured using a `"category|modelType"` format to dynamically resolve mathematical equations during optimization.

---

## I. Multi-Mechanism Integration Formulations

To model complex biomass pyrolysis (composed of cellulose, hemicellulose, lignin, and transport processes), the codebase supports three integration topologies:

### 1. Series Mode (Consecutive Bottlenecks / Mixed Control)
*   **Mathematical Equation:**
    $$G_{\text{eff}}(\alpha) = \sum_{i=1}^{k} G_i(\alpha)$$
*   **Physical Significance:** Captures consecutive chemical or physical processes occurring on the same reacting particle (e.g., a chemical cracking reaction step followed by macro-pore vapor mass diffusion). The total reaction time is the sum of the times required for each step, which translates directly to the additive sum of their integral kinetic functions $G(\alpha)$.
*   **Optimization Advantage:** Bypasses linear weight constraints, ensuring an extremely smooth search space that consistently achieves a global minimum error of **~15.1% to 15.4%**.

### 2. Parallel Mode (Independent Multi-Component Degradation)
*   **Mathematical Equation:**
    $$G_{\text{eff}}(\alpha) = \sum_{i=1}^{k} w_i G_i(\alpha) \quad \text{subject to} \quad \sum_{i=1}^{k} w_i = 1$$
*   **Physical Significance:** Models the independent, simultaneous thermal degradation of pseudo-components within a mixture (e.g., cellulose, hemicellulose, lignin). Each component reacts according to its own mechanism weight fraction ($w_i$).
*   **Mathematical Note:** In this codebase's single-$E_a(T)$ setup, the numeric derivative resolves this to a weighted time-addition scheme, physically representing series-like behavior under a single energetic constraint.

### 3. Hybrid Mode (Parallel-Series Multi-Stage)
*   **Mathematical Equation:**
    $$\alpha_{\text{total}} = \sum w_j \alpha_j \quad \text{where branch } j \text{ follows} \quad G_{\text{eff}, j}(\alpha_j) = \sum G_{k}(\alpha_j)$$
*   **Physical Significance:** Macroscopically models parallel component degradation while microscopically allowing specific branches (e.g., lignin decomposition) to experience sequential reaction-diffusion limitations.

---

## II. The 14 Candidate Mechanisms in the Unified Library

The candidate pool in `mechanismLibrary.m` contains the following 14 concrete models:

### 1. Diffusion Models
Diffusion models describe solid-state reactions where mass transfer (diffusion of volatile gases or reactants through the product layer) is the rate-limiting step.
*   **Parabolic 1D** (`diffusion|parabolic_1d`):
    $$G(\alpha) = \alpha^2$$
*   **Valensi 2D** (`diffusion|valensi_2d`):
    $$G(\alpha) = \alpha + (1-\alpha)\ln(1-\alpha)$$
*   **Jander 2D** (`diffusion|jander_2d`):
    $$G(\alpha) = \left[1 - (1-\alpha)^{1/2}\right]^n$$
*   **Jander 3D (Spherical)** (`diffusion|jander_3d`):
    $$G(\alpha) = \left[1 - (1-\alpha)^{1/3}\right]^n$$
*   **Ginstling-Brounshtein 3D** (`diffusion|ginstling_brounshtein_3d`):
    $$G(\alpha) = 1 - \frac{2}{3}\alpha - (1-\alpha)^{2/3}$$
*   **Anti-Jander 3D** (`diffusion|anti_jander_3d`):
    $$G(\alpha) = \left[(1+\alpha)^{1/3} - 1\right]^2$$
*   **Zhuralev-Lesokin-Tempelman 3D** (`diffusion|zhuralev_lesokin_tempelman_3d`):
    $$G(\alpha) = \left[(1-\alpha)^{-1/3} - 1\right]^2$$

### 2. Nucleation & Growth Models
These models describe processes where reactions initiate at active nucleation sites and propagate through solid growth.
*   **Avrami-Erofeev** (`nucleation|avrami_erofeev`):
    $$G(\alpha) = \left[-\ln(1-\alpha)\right]^n$$
*   **Prout-Tomkins** (`nucleation|prout_tomkins`):
    $$G(\alpha) = \ln\left(\frac{\alpha}{1-\alpha}\right)$$

### 3. Power Law Models
*   **Mapel Power** (`powerlaw|mapel_power`):
    $$G(\alpha) = \alpha^n$$

### 4. Geometrical Contraction Models
These models assume that nucleation occurs instantaneously over the entire surface, and the reaction is controlled by the inward advancement of the reaction interface.
*   **Contracting Cylinder (2D)** (`geometrical|contracting_cylinder`):
    $$G(\alpha) = 1 - (1-\alpha)^{1/2}$$
*   **Contracting Sphere (3D)** (`geometrical|contracting_sphere`):
    $$G(\alpha) = 1 - (1-\alpha)^{1/3}$$

### 5. Reaction Order Models
Classical homogeneous-like reaction kinetics applied to heterogeneous solid decompositions.
*   **First Order (F1)** (`reaction_order|first_order`):
    $$G(\alpha) = -\ln(1-\alpha)$$
*   **N-th Order (Fn)** (`reaction_order|nth_order`):
    $$G(\alpha) = \frac{(1-\alpha)^{1-n} - 1}{n-1}$$

---

## III. Parameter Bounds for Optimization

During GA optimization, continuous mechanism parameters (exponents $n$) are strictly bound to ensure physical validity:
*   **Diffusion Exponent ($n$)**: $0.1 \le n \le 10.0$
*   **Nucleation Exponent ($n$)**: $0.1 \le n \le 10.0$
*   **Power Law Exponent ($n$)**: $0.1 \le n \le 10.0$
*   **Reaction Order ($n$)**: $0.1 \le n \le 10.0$
*   *Note: Geometrical and First-Order models do not require exponent parameters.*