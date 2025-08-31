# Relativistic RG Turbulence: Complete Task List for Planning Agent

## Project Mission Statement
**Objective**: Implement a complete Renormalization Group analysis of the relativistic Israel-Stewart equations to derive universal properties of turbulence from first principles.

---

## PHASE 1: FOUNDATION AND INFRASTRUCTURE

### Task 1.1: Project Setup and Core Architecture
**Objective**: Establish project structure and core mathematical infrastructure

**Implementation Requirements**:
- Create modular Python package structure
- Set up development environment with dependencies
- Implement configuration management system

**Deliverables**:
```
Project structure with:
- pyproject.toml  with all dependencies
- pytest.ini for test configuration
- .github/workflows/ci.yml for continuous integration
```

**Output**: Installable Python package `rtrg`

**Test**: 
```bash
source .venv/bin/activate
uv pip install -e . && python -c "import rtrg; print(rtrg.__version__)"
```

---

### Task 1.2: Lorentz Tensor Algebra System
**Objective**: Implement complete tensor manipulation framework for relativistic calculations

**Equations to Implement**:
- Metric tensor: $g_{\mu\nu} = \text{diag}(-1, +1, +1, +1)$
- Tensor contraction: $A^{\mu}B_{\mu} = g_{\mu\nu}A^{\mu}B^{\nu}$
- Christoffel symbols: $\Gamma^{\lambda}_{\mu\nu} = \frac{1}{2}g^{\lambda\rho}(\partial_{\mu}g_{\rho\nu} + \partial_{\nu}g_{\rho\mu} - \partial_{\rho}g_{\mu\nu})$

**Implementation Details**:
```python
class LorentzTensor:
    - __init__(components, indices, covariant_indices, contravariant_indices)
    - contract(other, index_pairs) -> LorentzTensor
    - raise_index(index_position) -> LorentzTensor
    - lower_index(index_position) -> LorentzTensor
    - symmetrize() -> LorentzTensor
    - antisymmetrize() -> LorentzTensor
    - trace() -> scalar or LorentzTensor
```

**Output**: 
- Tensor objects with automatic index management
- Covariant derivative operations

**Visualization**:
```python
def visualize_tensor_network(expression):
    """Draw tensor contraction diagram"""
    # Use networkx to show tensor network
```

**Tests**:
1. Verify $u^{\mu}u_{\mu} = -c^2$ for four-velocity
2. Check metric properties: $g^{\mu\rho}g_{\rho\nu} = \delta^{\mu}_{\nu}$
3. Validate Bianchi identities

---

### Task 1.3: Field Definition Framework
**Objective**: Define all fields in the Israel-Stewart theory with proper quantum numbers

**Fields to Implement**:
| Field | Symbol | Indices | Dimension | Symmetry |
|-------|--------|---------|-----------|----------|
| Energy density | $\rho$ | - | 4 | Scalar |
| Four-velocity | $u^{\mu}$ | μ | 0 | Vector |
| Shear stress | $\pi^{\mu\nu}$ | μ,ν | 2 | Symmetric, Traceless |
| Bulk pressure | $\Pi$ | - | 2 | Scalar |
| Heat flux | $q^{\mu}$ | μ | 3 | Orthogonal to $u^{\mu}$ |

**Implementation**:
```python
@dataclass
class Field:
    name: str
    latex_symbol: str
    lorentz_indices: List[str]
    engineering_dimension: float
    canonical_dimension: float
    constraints: List[Constraint]
    
class ResponseField(Field):
    """MSRJD response field $\tilde{\phi}$"""
    physical_field: Field
```

**Output**: Field registry with all IS fields and response fields

**Visualization**:
```python
def plot_field_hierarchy():
    """Graphical representation of field relationships"""
    # Tree diagram showing physical ↔ response field pairs
```

**Tests**:
1. Verify dimension counting: $[\rho] + [u^{\mu}] = 4$
2. Check constraint satisfaction
3. Validate response field properties

---

### Task 1.4: Israel-Stewart Equations Implementation
**Objective**: Encode the complete Israel-Stewart equation system symbolically

**Equations to Implement**:

**Conservation Laws**:
$$\partial_{\mu}T^{\mu\nu} = 0$$
where
$$T^{\mu\nu} = \rho u^{\mu}u^{\nu} + (p + \Pi)\Delta^{\mu\nu} + \pi^{\mu\nu} + q^{\mu}u^{\nu} + q^{\nu}u^{\mu}$$

**Relaxation Equations**:
$$\tau_{\pi}\dot{\pi}^{\langle\mu\nu\rangle} + \pi^{\mu\nu} = 2\eta\sigma^{\mu\nu} - \tau_{\pi}\pi^{\mu\nu}\theta + \lambda_{\pi\Pi}\Pi\sigma^{\mu\nu} + \cdots$$

$$\tau_{\Pi}\dot{\Pi} + \Pi = -\zeta\theta - \tau_{\Pi}\Pi\theta + \cdots$$

$$\tau_{q}\dot{q}^{\langle\mu\rangle} + q^{\mu} = -\kappa T\nabla^{\mu}\alpha - \tau_{q}q^{\mu}\theta + \cdots$$

**Implementation**:
```python
class IsraelStewartEquations:
    def __init__(self, parameters: ISParameters):
        self.setup_thermodynamics()
        self.setup_kinematics()
        
    def stress_energy_tensor(self) -> SymbolicTensor:
        """Construct T^{μν}"""
        
    def conservation_laws(self) -> List[SymbolicEquation]:
        """∂_μ T^{μν} = 0"""
        
    def relaxation_equations(self) -> Dict[str, SymbolicEquation]:
        """Relaxation for π^{μν}, Π, q^μ"""
```

**Output**: 
- Symbolic IS equation system
- Parameter dependency graph

**Visualization**:
```python
def plot_equation_network():
    """Show coupling between equations"""
    # Directed graph showing field dependencies
```

**Tests**:
1. Verify thermodynamic consistency
2. Check causality: characteristic speeds < c
3. Validate equilibrium limits

---

### Task 1.5: Linearization Module
**Objective**: Linearize IS equations for stability analysis and propagator calculation

**Equations**:
Expand around equilibrium: $\phi = \phi_0 + \delta\phi$

$$\partial_t\delta\rho + \rho_0\nabla\cdot\delta\vec{v} = 0$$
$$\rho_0\partial_t\delta v^i + \nabla^i\delta p + \nabla_j\delta\pi^{ij} = 0$$
$$\tau_{\pi}\partial_t\delta\pi^{ij} + \delta\pi^{ij} = 2\eta\nabla^{(i}\delta v^{j)}$$

**Implementation**:
```python
class LinearizedIS:
    def __init__(self, background_state: Dict):
        self.background = background_state
        
    def linearize_field(self, field: Field) -> LinearField:
        """φ = φ_0 + δφ"""
        
    def dispersion_relation(self, k: float, mode: str) -> complex:
        """ω(k) for different modes"""
        
    def characteristic_polynomial(self, omega, k) -> Polynomial:
        """Det[M(ω,k)] = 0"""
```

**Output**: 
- Dispersion relations ω(k)
- Stability analysis results

**Visualization**:
```python
def plot_dispersion_relations():
    """Plot ω(k) for all modes"""
    # Multi-panel plot: Re(ω) and Im(ω) vs k
    # Show sound modes, diffusive modes, shear modes
```

**Tests**:
1. Sound speed: $\lim_{k\to 0} \omega/k = c_s$
2. Damping: $\text{Im}(\omega) > 0$ (stability)
3. Compare with known hydrodynamic modes

---

## PHASE 2: MSRJD ACTION AND FIELD THEORY

### Task 2.1: MSRJD Action Construction
**Objective**: Build path integral formulation for stochastic IS equations

**Theory**: Transform stochastic equations to field theory via
$$P[\phi] = \int \mathcal{D}\tilde{\phi}\mathcal{D}\phi \exp(-S[\phi, \tilde{\phi}])$$

**Action Components**:
$$S = S_{\text{det}} + S_{\text{noise}}$$
$$S_{\text{det}} = \int d^4x \, \tilde{\phi}_i\left(\partial_t\phi_i + F_i[\phi]\right)$$
$$S_{\text{noise}} = -\int d^4x \, \tilde{\phi}_i D_{ij} \tilde{\phi}_j$$

**Implementation**:
```python
class MSRJDAction:
    def __init__(self, is_system: IsraelStewartSystem):
        self.deterministic_part = self.build_deterministic_action()
        self.noise_part = self.build_noise_action()
        
    def build_deterministic_action(self) -> SymbolicAction:
        """Construct S_det from IS equations"""
        
    def build_noise_action(self) -> SymbolicAction:
        """Construct S_noise with FDT-consistent correlators"""
        
    def expand_in_fields(self, order: int) -> Dict[int, SymbolicExpression]:
        """Taylor expand to given order"""
```

**Output**: 
- Symbolic action S[φ, φ̃]
- Expansion coefficients for vertices

**Visualization**:
```python
def visualize_action_structure():
    """Show action terms graphically"""
    # Hierarchical diagram of action components
```

**Tests**:
1. Verify causality structure
2. Check FDT relations
3. Validate symmetries

---

### Task 2.2: Propagator Calculation
**Objective**: Extract propagators from quadratic action

**Key Propagators**:

**Velocity-Velocity**:
$$G^R_{u^i u^j}(\omega, k) = \frac{P^T_{ij}(k)}{-i\omega + \nu k^2} + \frac{P^L_{ij}(k)}{-i\omega + \Gamma_s k^2 + ic_s k}$$

**Shear Stress**:
$$G^R_{\pi^{ij}\pi^{kl}}(\omega, k) = \frac{2\eta P^{TT}_{ijkl}}{1 - i\omega\tau_{\pi} + \tau_{\pi}\nu k^2}$$

**Implementation**:
```python
class PropagatorCalculator:
    def __init__(self, quadratic_action: QuadraticAction):
        self.G_inv = self.construct_inverse_propagator_matrix()
        
    def calculate_propagator(self, field1: Field, field2: Field, 
                            omega: complex, k: np.ndarray) -> complex:
        """G_12(ω, k)"""
        
    def retarded_propagator(self, omega, k) -> np.ndarray:
        """G^R(ω, k) matrix"""
        
    def keldysh_propagator(self, omega, k, temperature) -> np.ndarray:
        """G^K(ω, k, T) with FDT"""
```

**Output**: 
- Propagator matrices G^R, G^A, G^K
- Spectral functions

**Visualization**:
```python
def plot_propagator_spectrum():
    """Visualize propagator pole structure"""
    # Contour plot of |G(ω, k)| in complex ω plane
    # Show poles and branch cuts
```

**Tests**:
1. Kramers-Kronig relations
2. Sum rules
3. Correct pole locations

---

### Task 2.3: Vertex Extraction
**Objective**: Identify all interaction vertices from the action

**Vertices to Extract**:

**Three-point**:
- Advection: $V_{uuu} \sim u^{\mu}\partial_{\mu}u^{\nu}$
- Stress coupling: $V_{\pi uu} \sim \pi^{\mu\nu}\nabla_{\mu}u_{\nu}$

**Four-point**:
- Nonlinear stress: $V_{\pi\pi uu} \sim \pi^{\mu\alpha}\pi_{\alpha}^{\nu}$

**Implementation**:
```python
class VertexExtractor:
    def __init__(self, action: MSRJDAction):
        self.vertices = {}
        
    def extract_cubic_vertices(self) -> Dict[str, Vertex]:
        """All 3-point interactions"""
        
    def extract_quartic_vertices(self) -> Dict[str, Vertex]:
        """All 4-point interactions"""
        
    def vertex_tensor_structure(self, vertex: Vertex) -> TensorStructure:
        """Lorentz structure of vertex"""
```

**Output**: 
- Vertex catalog with Feynman rules
- Coupling constants

**Visualization**:
```python
def draw_feynman_vertices():
    """Draw all vertex types"""
    # Use matplotlib patches to draw Feynman diagrams
```

**Tests**:
1. Ward identities: $k_{\mu}V^{\mu...} = 0$
2. Symmetry factors correct
3. Coupling constant dimensions

---

## PHASE 3: ONE-LOOP RENORMALIZATION

### Task 3.1: One-Loop Diagram Generator
**Objective**: Generate and classify all one-loop diagrams

**Diagrams to Generate**:
1. Self-energy: $\Sigma(k) = \bigcirc$
2. Vertex correction: $\delta V = \triangle$
3. Tadpoles: $\bigodot$

**Implementation**:
```python
class DiagramGenerator:
    def __init__(self, vertices: Dict, propagators: Dict):
        self.diagrams = []
        
    def generate_one_loop_diagrams(self, external_legs: List) -> List[Diagram]:
        """All 1-loop diagrams with given external legs"""
        
    def compute_symmetry_factor(self, diagram: Diagram) -> float:
        """Combinatorial factor"""
        
    def is_one_particle_irreducible(self, diagram: Diagram) -> bool:
        """Check if 1PI"""
```

**Output**: 
- List of Feynman diagrams
- Symmetry factors

**Visualization**:
```python
def draw_all_one_loop_diagrams():
    """Gallery of all 1-loop contributions"""
    # Grid layout showing all diagrams
```

**Tests**:
1. Count matches field theory expectation
2. No missing or duplicate diagrams
3. Symmetry factors correct

---

### Task 3.2: Loop Integration Engine
**Objective**: Compute one-loop integrals symbolically and numerically

**Integrals to Compute**:
$$I = \int \frac{d^4q}{(2\pi)^4} \frac{f(q)}{(q^2 + m_1^2)((k-q)^2 + m_2^2)}$$

**Frequency Integration**:
$$\int_{-\infty}^{\infty} \frac{d\omega}{2\pi} \frac{1}{(-i\omega + a)(-i\omega + b)} = \frac{i}{a-b}$$

**Implementation**:
```python
class LoopIntegrator:
    def __init__(self, cutoff: float, method: str = 'wilsonian'):
        self.cutoff = cutoff
        
    def integrate_frequency(self, integrand: Expression) -> Expression:
        """Residue theorem for ω integration"""
        
    def integrate_momentum_shell(self, integrand: Callable, 
                                 Lambda: float, b: float) -> float:
        """∫_{Λ/b}^{Λ} d³q q² f(q)"""
        
    def regularize_divergence(self, integral: float, 
                             regulator: str) -> float:
        """Handle UV/IR divergences"""
```

**Output**: 
- One-loop corrections δη, δτ, δg
- Anomalous dimensions γ_i

**Visualization**:
```python
def plot_loop_integrand():
    """Visualize integrand structure"""
    # 3D plot of integrand in (q_x, q_y) plane
    # Show peaks and cutoff regions
```

**Tests**:
1. Reproduce known scalar integrals
2. Numerical vs analytical comparison
3. Cutoff independence of physical results

---

### Task 3.3: Beta Function Extraction
**Objective**: Extract RG beta functions from one-loop corrections

**Beta Functions**:
$$\beta_g = \epsilon g - a_d g^2 + O(g^3)$$
$$\beta_{\eta} = \eta(z - 1 + \gamma_{\eta})$$
$$\beta_{\tau} = \tau(-z + \gamma_{\tau})$$

**Implementation**:
```python
class BetaFunctionExtractor:
    def __init__(self, one_loop_corrections: Dict):
        self.corrections = one_loop_corrections
        
    def extract_anomalous_dimension(self, field: str) -> float:
        """γ = -d ln Z/d ln μ"""
        
    def construct_beta_function(self, coupling: str) -> Callable:
        """β(g) from corrections"""
        
    def verify_ward_identities(self) -> bool:
        """Check gauge invariance"""
```

**Output**: 
- Beta functions β_i(g_j)
- Anomalous dimensions γ_i

**Visualization**:
```python
def plot_beta_functions():
    """Plot β(g) vs g"""
    # Show zeros (fixed points)
    # Indicate flow direction
```

**Tests**:
1. β(0) = 0 (Gaussian fixed point)
2. Sign of β determines stability
3. Consistency with symmetries

---

## PHASE 4: RG FLOW AND FIXED POINTS

### Task 4.1: RG Flow Integration
**Objective**: Integrate RG flow equations to find trajectories

**Flow Equations**:
$$\frac{dg_i}{d\ell} = \beta_i(g_1, ..., g_n)$$

**Implementation**:
```python
class RGFlowIntegrator:
    def __init__(self, beta_functions: Dict[str, Callable]):
        self.beta = beta_functions
        
    def integrate_flow(self, initial: Dict, l_max: float) -> Trajectory:
        """Solve dg/dl = β(g)"""
        
    def find_separatrices(self) -> List[Trajectory]:
        """Critical trajectories between basins"""
        
    def compute_basin_of_attraction(self, fixed_point: Dict) -> Region:
        """Parameter region flowing to FP"""
```

**Output**: 
- Flow trajectories g_i(ℓ)
- Basin boundaries

**Visualization**:
```python
def plot_rg_flow():
    """RG flow in coupling space"""
    # Stream plot in (g, η) plane
    # Mark fixed points
    # Color by flow speed
```

**Tests**:
1. Flow preserves constraints (η > 0)
2. Fixed points are stationary
3. Trajectories don't cross

---

### Task 4.2: Fixed Point Finder
**Objective**: Locate and classify all RG fixed points

**Fixed Point Equation**:
$$\beta_i(g_*) = 0 \quad \forall i$$

**Implementation**:
```python
class FixedPointAnalyzer:
    def __init__(self, beta_functions: Dict):
        self.beta = beta_functions
        
    def find_fixed_points(self, initial_guesses: List) -> List[FixedPoint]:
        """Solve β(g*) = 0"""
        
    def linear_stability_analysis(self, fp: FixedPoint) -> StabilityResult:
        """Eigenvalues of ∂β_i/∂g_j"""
        
    def classify_fixed_point(self, fp: FixedPoint) -> str:
        """UV/IR attractive, saddle, etc."""
```

**Output**: 
- Fixed point locations g*
- Stability eigenvalues λ_i
- Critical exponents

**Visualization**:
```python
def plot_fixed_point_stability():
    """Eigenvalue spectrum at FP"""
    # Complex plane plot of eigenvalues
    # Indicate stable/unstable directions
```

**Tests**:
1. Gaussian FP at g=0
2. Number of FPs matches theory
3. Eigenvalues determine flow near FP

---

### Task 4.3: Universal Exponent Calculation
**Objective**: Extract universal scaling exponents at fixed points

**Critical Exponents**:
- Correlation length: $\nu = -1/\lambda_{\text{relevant}}$
- Dynamic exponent: $z = 2 - \gamma_{\eta}|_{g*}$
- Energy spectrum: $\alpha = 5/3 - \delta$

**Implementation**:
```python
class UniversalExponents:
    def __init__(self, fixed_point: FixedPoint, stability: StabilityResult):
        self.fp = fixed_point
        self.stability = stability
        
    def correlation_length_exponent(self) -> float:
        """ν from relevant eigenvalue"""
        
    def dynamic_exponent(self) -> float:
        """z from anomalous dimensions"""
        
    def structure_function_exponents(self, n: int) -> float:
        """ζ_n for S_n(r) ~ r^{ζ_n}"""
```

**Output**: 
- Universal exponents ν, z, ζ_n
- Amplitude ratios

**Visualization**:
```python
def plot_scaling_exponents():
    """ζ_n vs n (multifractal spectrum)"""
    # Compare with K41: ζ_n = n/3
    # Show deviations (intermittency)
```

**Tests**:
1. ζ_2 ≈ 2/3 (close to Kolmogorov)
2. Convexity: ζ_n is convex
3. Consistency relations between exponents

---

## PHASE 5: NON-RELATIVISTIC LIMIT

### Task 5.1: Systematic c→∞ Limit
**Objective**: Extract non-relativistic turbulence from relativistic fixed point

**Scaling Procedure**:
- $u^{\mu} = \gamma(c, \vec{v}) \approx (c, \vec{v}) + O(v^2/c)$
- $\tau \sim 1/c^2$
- $g \sim \sqrt{c_s/c}$

**Implementation**:
```python
class NonRelativisticLimit:
    def __init__(self, relativistic_fp: FixedPoint):
        self.rel_fp = relativistic_fp
        
    def scale_parameters_with_c(self, c: float) -> Dict:
        """Transform parameters for given c"""
        
    def extrapolate_to_infinity(self, c_values: np.ndarray) -> Dict:
        """Richardson extrapolation c→∞"""
        
    def compute_nr_exponents(self) -> Dict:
        """Non-relativistic scaling exponents"""
```

**Output**: 
- NR fixed point parameters
- NR scaling exponents

**Visualization**:
```python
def plot_c_limit_convergence():
    """Exponents vs 1/c"""
    # Show extrapolation to c→∞
    # Error bands from fitting
```

**Tests**:
1. Parameters converge as c→∞
2. Recover approximate NS behavior
3. Causality preserved at finite c

---

### Task 5.2: Physical Predictions
**Objective**: Generate experimentally testable predictions

**Predictions to Generate**:
1. Energy spectrum: $E(k) = C_K \epsilon^{2/3} k^{-5/3+\delta}$
2. Structure functions: $S_n(r) = C_n (\epsilon r)^{\zeta_n}$
3. Intermittency corrections: $\Delta\zeta_n = \zeta_n - n/3$

**Implementation**:
```python
class TurbulencePredictions:
    def __init__(self, nr_limit: NonRelativisticLimit):
        self.exponents = nr_limit.compute_nr_exponents()
        
    def energy_spectrum(self, k: np.ndarray, epsilon: float) -> np.ndarray:
        """E(k) with corrections"""
        
    def structure_functions(self, r: np.ndarray, n: int) -> np.ndarray:
        """S_n(r) scaling"""
        
    def probability_distributions(self, r: float) -> PDF:
        """PDF of velocity increments"""
```

**Output**: 
- Scaling functions
- Universal constants
- Correction terms

**Visualization**:
```python
def plot_energy_spectrum_comparison():
    """E(k) vs k with experimental data"""
    # Log-log plot
    # Show K41 and corrected scaling
    # Include experimental points
```

**Tests**:
1. E(k) integrates to total energy
2. Structure functions satisfy inequalities
3. PDFs properly normalized

---

## PHASE 6: VALIDATION AND PRODUCTION

### Task 6.1: Comprehensive Validation Suite
**Objective**: Validate all results against known limits and data

**Validation Tests**:
1. **Gaussian limit**: g→0 recovers free theory
2. **Burgers limit**: 1D reduces to Burgers equation
3. **NS limit**: τ→0 recovers Navier-Stokes
4. **DNS comparison**: Match simulation data

**Implementation**:
```python
class ValidationSuite:
    def __init__(self, results: Dict):
        self.results = results
        
    def test_analytical_limits(self) -> TestReport:
        """Check known analytical results"""
        
    def compare_with_dns(self, dns_data: Dataset) -> Comparison:
        """Statistical comparison with DNS"""
        
    def benchmark_performance(self) -> BenchmarkReport:
        """Computational performance metrics"""
```

**Output**: 
- Validation report
- Error analysis
- Performance metrics

**Visualization**:
```python
def create_validation_dashboard():
    """Interactive dashboard of all tests"""
    # Multi-panel display
    # Pass/fail indicators
    # Quantitative comparisons
```

**Tests**:
1. All analytical limits satisfied
2. Statistical agreement with DNS
3. Performance within targets

---

### Task 6.2: Documentation and User Interface
**Objective**: Create comprehensive documentation and user tools

**Documentation Components**:
1. Theory guide
2. API reference
3. Tutorials
4. Example gallery

**Implementation**:
```python
class DocumentationBuilder:
    def generate_api_docs(self) -> None:
        """Auto-generate from docstrings"""
        
    def create_tutorials(self) -> None:
        """Jupyter notebook tutorials"""
        
    def build_theory_guide(self) -> None:
        """LaTeX → PDF theory manual"""
```

**Output**: 
- Complete documentation
- Interactive tutorials
- Theory manual

**Visualization**:
```python
def create_interactive_explorer():
    """Web app for exploring results"""
    # Plotly Dash application
    # Parameter sliders
    # Real-time visualization
```

---

### Task 6.3: Publication and Dissemination
**Objective**: Prepare results for scientific publication

**Deliverables**:
1. Research paper draft
2. Data repository
3. Reproducible analysis scripts
4. Presentation materials

**Implementation**:
```python
class PublicationPreparation:
    def generate_figures(self) -> Dict[str, Figure]:
        """Publication-quality figures"""
        
    def create_data_archive(self) -> Archive:
        """Zenodo-ready data package"""
        
    def format_tables(self) -> Dict[str, Table]:
        """LaTeX tables of results"""
```

**Output**: 
- Paper manuscript
- Figure collection
- Data archive

---

## CONTINUOUS INTEGRATION AND TESTING

### Daily CI Pipeline
```yaml
name: Daily Tests
on:
  schedule:
    - cron: '0 2 * * *'
jobs:
  test:
    - Unit tests: pytest tests/unit
    - Integration: pytest tests/integration  
    - Coverage: pytest --cov=rtrg --cov-report=html
    - Linting: flake8 rtrg/
    - Type checking: mypy rtrg/
```

### Weekly Validation
```yaml
name: Weekly Validation
on:
  schedule:
    - cron: '0 3 * * 0'
jobs:
  validate:
    - DNS comparison
    - Analytical limits
```

### Monthly Full Pipeline
```yaml
name: Monthly Full Run
jobs:
  full_analysis:
    - Complete RG calculation
    - All parameter sets
    - Generate reports
    - Update documentation
```

---

## SUCCESS METRICS

### Phase Completion Criteria

**Phase 1**: ✓ Tensor algebra working ✓ IS equations implemented ✓ Linear analysis correct

**Phase 2**: ✓ MSRJD action constructed ✓ Propagators calculated ✓ Vertices extracted

**Phase 3**: ✓ One-loop integrals converge ✓ Beta functions extracted ✓ Ward identities satisfied

**Phase 4**: ✓ Fixed points found ✓ Stability analyzed ✓ Exponents physical

**Phase 5**: ✓ NR limit converges ✓ K41 approximately recovered ✓ Predictions generated

**Phase 6**: ✓ Validation complete ✓ Documentation ready ✓ Results published

---

## RISK REGISTER AND MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical instability | Medium | High | Multiple integration methods, adaptive algorithms |
| No stable fixed point | Low | Critical | Explore extended parameter space, modified theories |
| Slow convergence | High | Medium | Parallelization, GPU acceleration, approximations |
| Unphysical results | Low | High | Extensive validation, consistency checks |
| Memory limitations | Medium | Medium | Streaming computation, cloud resources |

---

This task list provides complete specifications for implementing the relativistic RG turbulence project, with clear objectives, equations, outputs, and validation criteria for each component.