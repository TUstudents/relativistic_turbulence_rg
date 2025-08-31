# MSRJD Formalism for Stochastic Hydrodynamics: Path Integrals and Field Theory

## Executive Summary

The Martin-Siggia-Rose-Janssen-De Dominicis (MSRJD) formalism transforms stochastic differential equations into a quantum field theory framework, enabling the application of powerful techniques like renormalization group analysis. This report provides a complete guide to constructing the MSRJD action for the stochastic Israel-Stewart equations, deriving Feynman rules, and implementing perturbative calculations.

## 1. Foundational Concepts

### 1.1 From Stochastic Equations to Path Integrals

Consider a general stochastic differential equation:
$$\partial_t \phi_i + F_i[\phi] = \xi_i$$

where $\xi_i$ is Gaussian white noise with correlator:
$$\langle \xi_i(x,t) \xi_j(x',t') \rangle = 2D_{ij} \delta(x-x')\delta(t-t')$$

The probability of a trajectory $\phi(x,t)$ is given by the path integral:
$$P[\phi] = \int \mathcal{D}\xi \, P[\xi] \prod_{x,t} \delta\left(\partial_t\phi_i + F_i[\phi] - \xi_i\right)$$

### 1.2 The Jacobian Trick

The key insight: represent the delta function using an auxiliary field (response field) $\tilde{\phi}_i$:
$$\delta(G) = \int \mathcal{D}\tilde{\phi} \exp\left(i\int dx dt \, \tilde{\phi}_i G_i\right)$$

This transforms the probability into:
$$P[\phi] = \int \mathcal{D}\tilde{\phi} \mathcal{D}\xi \exp\left(-\frac{1}{4D}\int \xi_i D^{-1}_{ij} \xi_j + i\int \tilde{\phi}_i(\partial_t\phi_i + F_i[\phi] - \xi_i)\right)$$

### 1.3 Integrating Out the Noise

Performing the Gaussian integral over $\xi$ yields the MSRJD action:
$$S[\phi, \tilde{\phi}] = \int dx dt \left[\tilde{\phi}_i(\partial_t\phi_i + F_i[\phi]) - \tilde{\phi}_i D_{ij} \tilde{\phi}_j\right]$$

## 2. MSRJD for Israel-Stewart Equations

### 2.1 Field Content and Structure

For the relativistic Israel-Stewart system, we have:

**Physical fields:**
- $\rho(x^\mu)$ - energy density
- $u^\mu(x^\nu)$ - four-velocity (subject to constraint $u^\mu u_\mu = -c^2$)
- $\pi^{\mu\nu}(x^\alpha)$ - shear stress tensor
- $\Pi(x^\mu)$ - bulk pressure
- $q^\mu(x^\nu)$ - heat flux

**Response fields:**
- $\tilde{\rho}, \tilde{u}_\mu, \tilde{\pi}_{\mu\nu}, \tilde{\Pi}, \tilde{q}_\mu$ - conjugate to each physical field

### 2.2 Handling Constraints

The velocity constraint $u^\mu u_\mu = -c^2$ requires special treatment:

**Method 1: Lagrange Multiplier**
$$S = S_{\text{unconstrained}} + \int d^4x \, \lambda(u^\mu u_\mu + c^2)$$

**Method 2: Parameterization**
Express $u^\mu$ in terms of unconstrained variables:
$$u^\mu = c\gamma(1, \vec{\beta}), \quad \gamma = (1-\beta^2)^{-1/2}$$

**Method 3: Gauge Fixing (Recommended)**
Work in the fluid rest frame where $u^\mu = (c, 0, 0, 0)$ locally, treating deviations perturbatively.

### 2.3 The Complete MSRJD Action

The action for stochastic Israel-Stewart equations:

$$S = \int d^4x \left[S_{\text{conservation}} + S_{\text{relaxation}} + S_{\text{noise}}\right]$$

**Conservation Part:**
$$S_{\text{conservation}} = \tilde{T}_{\mu\nu} \partial_\mu T^{\mu\nu}$$

where $\tilde{T}_{\mu\nu}$ is the response field for the stress-energy tensor.

**Relaxation Part:**
$$S_{\text{relaxation}} = \tilde{\pi}_{\mu\nu}\left(\tau_\pi \dot{\pi}^{\langle\mu\nu\rangle} + \pi^{\mu\nu} - 2\eta\sigma^{\mu\nu}\right) + \text{similar for } \Pi, q^\mu$$

**Noise Part:**
$$S_{\text{noise}} = -\tilde{\pi}_{\mu\nu} D^{\pi}_{\mu\nu\alpha\beta} \tilde{\pi}_{\alpha\beta} - \tilde{\Pi} D^{\Pi} \tilde{\Pi} - \tilde{q}_\mu D^q_{\mu\nu} \tilde{q}_\nu$$

### 2.4 Covariant Noise Correlators

The noise correlators must respect Lorentz symmetry:

**Shear noise:**
$$D^{\pi}_{\mu\nu\alpha\beta} = 2k_B T\eta \left(P_{\mu\nu\alpha\beta} + \text{relaxation corrections}\right)$$

where $P_{\mu\nu\alpha\beta}$ is the transverse-traceless projector:
$$P_{\mu\nu\alpha\beta} = \frac{1}{2}\left(\Delta_{\mu\alpha}\Delta_{\nu\beta} + \Delta_{\mu\beta}\Delta_{\nu\alpha} - \frac{2}{3}\Delta_{\mu\nu}\Delta_{\alpha\beta}\right)$$

**Bulk noise:**
$$D^{\Pi} = 2k_B T\zeta$$

**Heat flux noise:**
$$D^q_{\mu\nu} = 2k_B T^2\kappa \Delta_{\mu\nu}$$

## 3. Perturbative Expansion and Feynman Rules

### 3.1 Quadratic Action and Propagators

Expanding around equilibrium $\phi_0$ to quadratic order:

$$S^{(2)} = \frac{1}{2}\int \frac{d^4k}{(2\pi)^4} \Phi^*_a(k) G^{-1}_{ab}(k) \Phi_b(k)$$

where $\Phi = (\phi, \tilde{\phi})^T$ is the doubled field vector.

**Propagator Structure:**
The propagator matrix has the block form:
$$G = \begin{pmatrix}
0 & G^A \\
G^R & G^K
\end{pmatrix}$$

where:
- $G^R$ = Retarded propagator (physical response)
- $G^A$ = Advanced propagator = $(G^R)^*$
- $G^K$ = Keldysh propagator (noise/fluctuations)

### 3.2 Explicit Propagators for IS Theory

**Velocity-Velocity Propagator:**
$$G^R_{u^i u^j}(\omega, k) = \frac{P^T_{ij}(k)}{-i\omega + \nu_\perp k^2} + \frac{P^L_{ij}(k)}{-i\omega + \Gamma_s k^2 + i c_s k}$$

where:
- $P^T_{ij} = \delta_{ij} - k_i k_j/k^2$ (transverse projector)
- $P^L_{ij} = k_i k_j/k^2$ (longitudinal projector)
- $\nu_\perp = \eta/\rho$ (kinematic viscosity)
- $\Gamma_s = (4\eta/3 + \zeta)/\rho$ (sound damping)

**Shear Stress Propagator:**
$$G^R_{\pi^{ij}\pi^{kl}}(\omega, k) = \frac{2\eta P^{TT}_{ijkl}}{1 - i\omega\tau_\pi + \tau_\pi\nu_\perp k^2}$$

where $P^{TT}_{ijkl}$ is the transverse-traceless projector.

**Mixed Propagators (Cross-correlations):**
$$G^R_{u^i\pi^{jk}}(\omega, k) = \frac{i k^{(j}\delta^{k)i}}{-i\omega + \nu_\perp k^2} \cdot \frac{1}{\rho(1 - i\omega\tau_\pi)}$$

### 3.3 Interaction Vertices

The non-linear terms in the IS equations generate interaction vertices:

**Three-point vertices:**

1. **Advection vertex** ($u$-$u$-$\tilde{u}$):
$$V^{\text{adv}}_{\mu\nu\alpha} = ik_\mu \delta_{\nu\alpha}$$

2. **Shear-velocity coupling** ($\pi$-$u$-$\tilde{\pi}$):
$$V^{\pi u}_{\mu\nu,\alpha\beta,\gamma} = i(k_\alpha \delta_{\beta\gamma}\delta_{\mu\nu} + \text{permutations})$$

3. **Relaxation-expansion coupling** ($\pi$-$u$-$\tilde{\pi}$):
$$V^{\text{relax}}_{\mu\nu,\alpha,\beta\gamma} = -\tau_\pi k_\alpha P_{\mu\nu\beta\gamma}$$

**Four-point vertices:**

From terms like $\pi_{\mu\alpha}\pi^{\alpha\nu}$ in the energy-momentum tensor:
$$V^{(4)}_{\pi\pi uu} = \text{tensor structure encoding } \pi^2 \text{ contributions}$$

### 3.4 Feynman Diagram Rules

1. **Lines:** 
   - Solid lines = physical field propagators
   - Dashed lines = response field propagators
   - Wavy lines = noise correlators

2. **Vertices:**
   - Each vertex contributes a factor from the interaction Lagrangian
   - Conserve momentum at each vertex
   - Sum over all internal indices

3. **Loops:**
   - Integrate over loop momenta: $\int \frac{d^4q}{(2\pi)^4}$
   - Include symmetry factors

4. **External legs:**
   - Physical fields: amputate propagators
   - Response fields: set to unity (classical source)

## 4. One-Loop Calculations

### 4.1 Self-Energy Corrections

The one-loop self-energy for the velocity field:

$$\Sigma^{(1)}_{u^i u^j}(k) = \int \frac{d^4q}{(2\pi)^4} V^{\text{adv}}_{i\alpha\beta}(k,q) G^R_{u^\alpha u^\gamma}(q) G^K_{u^\beta u^\delta}(k-q) V^{\text{adv}}_{j\gamma\delta}(-k,-q)$$

This integral has the structure:
$$\Sigma^{(1)} \sim g \int_{\Lambda/b}^{\Lambda} \frac{d^3q}{(2\pi)^3} \int \frac{d\omega}{2\pi} \frac{1}{(-i\omega + \nu q^2)(-i(\omega-\Omega) + \nu|k-q|^2)}$$

### 4.2 Frequency Integration via Residues

The frequency integral can be evaluated by closing the contour:

$$\int \frac{d\omega}{2\pi} \frac{1}{(-i\omega + a)(-i\omega + b)} = \frac{i}{a-b}$$

This reduces the one-loop integrals to 3D momentum integrals.

### 4.3 Vertex Corrections

The one-loop correction to the advection vertex:

$$\delta V^{\text{adv}} = g^2 \int \frac{d^4q}{(2\pi)^4} \text{[Product of three propagators and two vertices]}$$

These corrections renormalize the effective coupling constant $g$.

## 5. Renormalization and Beta Functions

### 5.1 Wilsonian RG Procedure

1. **Integrate out high-momentum modes:** $\Lambda/b < |q| < \Lambda$
2. **Rescale coordinates:** $x' = x/b$, $t' = t/b^z$
3. **Rescale fields:** $\phi' = b^{\Delta_\phi}\phi$
4. **Read off running couplings**

### 5.2 Scaling Dimensions

**Engineering dimensions** (from dimensional analysis):
- $[\rho] = \text{energy}/\text{volume} = 4$
- $[u^\mu] = 0$ (dimensionless)
- $[\pi^{\mu\nu}] = 2$
- $[\eta] = 1$
- $[\tau_\pi] = -1$

**Anomalous dimensions** (from loop corrections):
- $\gamma_\rho = \text{coefficient} \times g^2 + O(g^3)$
- $\gamma_\eta = a_d g + O(g^2)$
- $\gamma_{\tau} = b_d g + O(g^2)$

### 5.3 Beta Functions

The RG flow equations:

$$\frac{dg}{d\ell} = \beta_g = \epsilon g - c_d g^2 + O(g^3)$$

$$\frac{d\eta}{d\ell} = \beta_\eta = \eta(z-1+\gamma_\eta)$$

$$\frac{d\tau_\pi}{d\ell} = \beta_{\tau} = \tau_\pi(-z + \gamma_\tau)$$

where $\epsilon = 4-d$ and $\ell = \ln(b)$.

## 6. Implementation Strategy

### 6.1 Symbolic Computation Pipeline

```python
# Pseudocode structure
class MSRJDCalculator:
    def __init__(self, theory='Israel-Stewart'):
        self.fields = self.define_fields()
        self.action = self.construct_action()
        
    def construct_action(self):
        S_quad = self.quadratic_action()
        S_int = self.interaction_action()
        return S_quad + S_int
    
    def derive_propagators(self):
        G_inv = self.inverse_propagator_matrix()
        return invert_matrix(G_inv)
    
    def calculate_one_loop(self, diagram_type):
        # Symbolic integration
        integrand = self.construct_integrand(diagram_type)
        freq_integrated = self.integrate_frequency(integrand)
        return self.momentum_shell_integral(freq_integrated)
```

### 6.2 Numerical Integration Techniques

**Momentum Shell Integration:**

```python
def momentum_shell_integral(integrand, Lambda, b):
    """
    Integrate over the shell Lambda/b < |q| < Lambda
    """
    def spherical_integral(q_mag):
        # Angular average
        angular_avg = angular_integration(integrand, q_mag)
        return q_mag**2 * angular_avg
    
    result, error = quad(spherical_integral, Lambda/b, Lambda)
    return result
```

**Handling IR/UV Divergences:**

1. **UV**: Already regulated by the cutoff $\Lambda$
2. **IR**: Introduce small mass $m$ or use dimensional regularization
3. **Collinear**: Special care needed for light-cone singularities

### 6.3 Validation Checks

1. **Ward Identities**: Verify conservation laws are preserved
2. **Symmetry**: Check Lorentz covariance of results
3. **Known Limits**: Reproduce Navier-Stokes RG in appropriate limit
4. **Numerical Convergence**: Test sensitivity to cutoff and discretization

## 7. Advanced Topics

### 7.1 Composite Operators

For structure functions and turbulence observables:

$$S_n(r) = \langle |\delta u(r)|^n \rangle$$

Requires composite operator renormalization:
$$[u^n]_R = Z_n(\Lambda) u^n$$

### 7.2 Operator Product Expansion

For analyzing intermittency:
$$u(x)u(y) \sim \sum_i C_i(x-y) \mathcal{O}_i\left(\frac{x+y}{2}\right)$$

The Wilson coefficients $C_i$ contain the scaling information.

### 7.3 Two-Loop and Beyond

Higher-loop calculations require:
- Automated diagram generation
- Tensor algebra packages
- Parallel computation for complex integrals

## 8. Summary and Key Insights

The MSRJD formalism transforms the stochastic Israel-Stewart equations into a field theory where:

1. **Causality is Manifest**: Retarded propagators automatically encode causal structure
2. **Symmetry is Preserved**: Lorentz covariance simplifies vertex structure
3. **Fluctuations are Systematic**: Noise enters through Keldysh propagators
4. **RG is Natural**: The path integral formulation enables standard RG techniques

The main technical challenges are:
- Handling the velocity constraint properly
- Managing the complexity of tensor indices
- Efficient computation of loop integrals
- Extracting physical observables from field correlators

The relativistic structure provides crucial advantages over Galilean theories, potentially resolving long-standing puzzles in turbulence theory through its superior symmetry properties and causal structure.