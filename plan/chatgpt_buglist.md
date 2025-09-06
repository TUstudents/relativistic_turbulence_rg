Great — I went through the new files and flagged concrete issues (logic bugs, math mistakes, API misuses, and a few design hazards). Below are the highest-impact problems and crisp fixes, grouped by file.

---

# `constants.py`

1. ❌ **Wrong GeV→cgs conversion factors for length and time (×1000 off).**

* You currently use `1 GeV⁻¹ = 1.973e-11 cm` and `ħ/GeV = 6.582e-22 s`. The accepted values are **1.97327e-14 cm** and **6.58212e-25 s**. Same mistake appears in `UnitSystem._setup_conversions`. Fix both maps.&#x20;
  **Patch:**

```python
"length": 1.97327e-14,  # cm (ħc / GeV)
"time":   6.58212e-25,  # s  (ħ / GeV)
```

2. ⚠️ **Docstring promises validation on signature length, code doesn’t.**
   `Metric` docs say it raises when signature length ≠ dimension, but your constructor silently truncates by `min(len(signature), dimension)`. Either enforce the error or fix the docstring (see tensors section for constructor).&#x20;

---

# `fields.py`

1. ❌ **SymPy indexed symbols built incorrectly.**
   You build `u`/`π`/`q` as `sp.IndexedBase(name)[tuple(index_symbols)]`, which creates a **single tuple index**, not one index per slot. Use argument unpacking.&#x20;
   **Patch:**

```python
self.symbol = sp.IndexedBase(properties.name)[*index_symbols]
```

2. ❌ **Bad import and missing helpers.**
   `validate_components` imports `is_spatial_vector` and `is_symmetric_traceless_spatial` from `..israel_stewart.constraints`, but that module (in this repo) exposes `spatial_projector` and `tt_projector` — not those helpers — so this will raise `ImportError`. Either change the import to the available helpers and rework the checks, or implement the missing functions. &#x20;

3. ⚠️ **Duplication/overlays of registry classes.**
   `FieldRegistry` appears multiple times (base and “enhanced”), redefining behavior in the same file — easy to shadow unexpectedly. Consider splitting “enhanced” types to a separate module or namespacing them clearly.&#x20;

4. ⚠️ **Traceless “fix” mixes coordinate trace with metric trace.**
   `_make_traceless` uses `tensor.trace()` which, via `LorentzTensor.trace`, defaults to `np.trace` (coordinate trace). For Lorentz tensors, the trace should be **metric-contracted** (`T^μ_μ = g_{μν} T^{μν}` for mixed/lowered). This interacts with index variance; subtracting `(trace/d) g^{μν}` only makes sense for **contravariant** rank-2 tensors. Needs a metric-aware trace (see tensors section) and variance-aware subtraction.&#x20;

5. ⚠️ **Heat-flux evolution uses time derivative where a gradient is intended.**
   `thermal_gradient = -kappa*T*diff(alpha, t)` should be the **spatial/4-gradient** `∇^μ α`. At minimum, model `∂_i α` or `∂^μ α` symbolically.&#x20;

---

# `tensors.py`

1. ❌ **Contractions ignore the metric and return an object when a scalar is expected.**

* `_build_contraction_einsum` doesn’t insert `g_{μν}` when contracting like-variance indices; it just sums axes. That’s not Lorentz-invariant.
* `contract(...)` always returns a `LorentzTensor`, even when rank → 0, and elsewhere you use it as a number (e.g., `abs(...)` in orthogonality). Make it return a Python/NumPy scalar for rank-0 results.&#x20;
  **Minimal fix idea:**

  * When a contraction pair is same variance, first raise/lower one index with `Metric` then contract.
  * After einsum, if `result.ndim == 0`, return `result.item()`.

2. ❌ **Trace is not metric-aware.**
   `trace()` uses `np.trace(self.components)` for rank-2, which is not the invariant trace unless the mixed index is already formed. Provide a metric-aware path: for `(upper,upper)` use `g_{μν} T^{μν}`, for `(lower,lower)` use `g^{μν} T_{μν}`, and for mixed use `T^μ_μ`.&#x20;

3. ❌ **Normalization routine can’t change the sign of the norm.**
   `enforce_normalization_constraint` rescales by `sqrt(|target/current|)` and tries to flip sign with `*=-1` if `current*target<0`. But the sign of a Minkowski norm is invariant under real scaling; multiplying by −1 does **not** change `u·u`. If the current vector is spacelike and `target` is timelike (−1), this procedure can’t succeed; it should **raise** or **rebuild** `u` from a spatial 3-velocity.&#x20;

4. ⚠️ **Velocity-orthogonal projectors applied regardless of variance.**
   `project_spatial` applies a covariant projector `Δ_{μν}` to every index position the same way. For an upper index you need `Δ^{μ}{}_{ν}`; for two uppers `Δ^{μν}`; etc. Use a mixed projector per index variance. The `ProjectionOperators.velocity_orthogonal_projector` has a rank-aware variant you can reuse or extend.&#x20;

5. ⚠️ **Index-compatibility too strict.**
   `TensorIndex.is_contractible_with` requires **same name** and opposite variance. In practice you often contract different labels (e.g., μ with ρ). Drop the name equality; keep only type (spacetime/spatial) and opposite variance checks. This currently blocks valid contractions.&#x20;

6. ⚠️ **`Metric` constructor/doc mismatch.**
   As noted above, docstring says it raises on signature length mismatch; the code truncates silently. Either enforce:

```python
if signature and len(signature) != dimension:
    raise ValueError(...)
```

or correct the docs.&#x20;

7. ⚠️ **Constraint helpers assume diag metric and hard-code signature.**
   `ConstrainedTensorField` uses `np.array([-1,1,1,1])` in several places instead of the provided `Metric`. Replace with `metric.g`/`metric.g_inv` to support dimension/signature changes.&#x20;

8. ⚠️ **Projector/Decomposition mix covariant/contravariant inconsistently.**
   E.g., `spatial_projector` builds `h^μν = g^{μν}+u^μu^ν` (contravariant), but in other places you use a covariant projector. Be consistent and document whether your arrays are upper or lower per axis.&#x20;

---

# `parameters.py`

1. ⚠️ **Causality check has redundant/fragile logic.**
   You compute `shear_arg = η/(ε τ_π)` then (a) compare `shear_arg > c²`, then (b) compute `v_shear = √shear_arg` and also check `v_shear > c`. The second suffices (and also handles NaNs more transparently if you guard). Consider collapsing to one check and return a detailed reason.&#x20;

2. ⚠️ **Over-confident fallbacks.**
   Several places catch broad exceptions and “assume valid” (`validate` → `True`). For safety, prefer “unknown”/warning over silently passing causality.&#x20;

3. ⚠️ **Kinetic-theory stubs can yield negative ζ.**
   `ζ = p τ (1/3 − c_s²)` becomes negative for `c_s² > 1/3`. Clamp or warn.&#x20;

(Otherwise, the data model and the dimensionless helpers look consistent.)

---

# `__init__.py` (core + package)

Looks fine: clean re-exports and no circulars detected from these edits. &#x20;

---

## Quick, concrete patches (drop-in)

* **Fix SymPy indexing in `Field`:** see patch in fields #1.&#x20;
* **Fix constants:** see constants #1 (both `to_cgs` and `UnitSystem._setup_conversions`).&#x20;
* **Metric-aware trace (sketch):**

```python
def trace(self) -> complex:
    if self.rank != 2:
        raise NotImplementedError
    cov = self.indices.types.count("covariant")
    con = self.indices.types.count("contravariant")
    if cov == 2:   # T_{μν} -> g^{μν} T_{μν}
        ginv = np.linalg.inv(self.metric.g)
        return float(np.einsum("mn,mn->", ginv, self.components))
    if con == 2:   # T^{μν} -> g_{μν} T^{μν}
        return float(np.einsum("mn,mn->", self.metric.g, self.components))
    # mixed: T^μ_μ
    return float(np.trace(self.components))
```

Wire this into `_make_traceless` accordingly.&#x20;

* **Return scalar from rank-0 contraction:**

```python
res = np.einsum(einsum_str, ...)
return res.item() if res.ndim == 0 else LorentzTensor(res, result_indices, self.metric)
```

…and insert metric when contracting same-variance pairs.&#x20;

* **Variance-aware spatial projection:** apply `Δ^{μ}{}_{ν}` to upper indices and `Δ_{μ}{}^{ν}` to lower ones, not a single covariant `Δ_{μν}` to all slots. You already have the pieces in `ProjectionOperators`.&#x20;

* **Remove name-equality from contractibility:** allow different dummy labels if index types are compatible and variances are opposite.&#x20;

Here’s a tight code review of this MSRJD batch—specific bugs + why they’re wrong + crisp fixes. Citations point to the exact spots in your files.

---

# Top correctness bugs


2. ## Noise correlators aren’t projectors (and δ⁴ is wrong)

* In `velocity_velocity_correlator` you build
  `(KroneckerDelta(mu,nu) - 1/c^2) * δ4D`. The second term must be **u^μ u^ν / c²**, not a scalar; and in a covariant setup you need **g^{μν}**, not Kronecker. Your “δ⁴” is also only two Dirac deltas of odd index pairs, not `δ(t-t') δ³(x-x')`. &#x20;
* `heat_flux_correlator` has the same mistake: `(KroneckerDelta + 1/c²)` is not `g^{μν} + u^μu^ν/c²`.&#x20;
* `shear_stress_correlator` drops the **TT** tensor structure entirely (needs the full $P^{\mu\nu\alpha\beta}_{\mathrm{TT}}$).&#x20;
  **Fix (sketch):**

```python
# proper δ⁴ with explicit coords (t,x,y,z)
delta_4d = sp.DiracDelta(t - tp) * sp.DiracDelta(x - xp) * sp.DiracDelta(y - yp) * sp.DiracDelta(z - zp)

# P_⊥^{μν} = g^{μν} + u^{μ}u^{ν}/c²   (contravariant projector)
P_perp = g_up[mu,nu] + u_up[mu]*u_up[nu]/PhysicalConstants.c**2

D_uu = 2*self.k_B*self.temperature*(self.parameters.eta/self.parameters.tau_pi)*P_perp*delta_4d
```

…and for shear, use $P^{\mu\nu\alpha\beta}_{\mathrm{TT}}=\tfrac12(\Delta^{\mu\alpha}\Delta^{\nu\beta}+\Delta^{\mu\beta}\Delta^{\nu\alpha})-\tfrac{1}{d-1}\Delta^{\mu\nu}\Delta^{\alpha\beta}$ with $\Delta^{\mu\nu}=g^{\mu\nu}+u^\mu u^\nu/c^2$.

3. ## Symbolic tensor components are constructed inconsistently

* `SymbolicTensorField.create_component(...)` calls `self(*all_args)`, which goes through `__call__` (meant for **scalar** fields) and will treat **tensor indices as coordinates**. It should use `__getitem__` logic that encodes indices in the *name* and passes coordinates as arguments. &#x20;
  **Fix:**

```python
def create_component(self, tensor_indices, coordinate_values=None):
    coordinate_values = coordinate_values or self._coordinates
    n_coords = len(self._coordinates)
    return self[tuple(tensor_indices) + tuple(coordinate_values)]
```

* `apply_constraint("normalization")` computes $\sum_\mu u^\mu u^\mu + c^2$ — no metric, wrong index placement. Should be $g_{\mu\nu}u^\mu u^\nu + c^2 = 0$. The “traceless” case also misses the metric for lower/upper index placement.&#x20;

4. ## Propagator matrix math/typing issues

* You pass **strings** as `field_basis` into `PropagatorMatrix`, but `get_component` expects `Field` objects. Any downstream call that looks up by object will fail. Either make the basis `Field` objects or change lookup to by-name.&#x20;
* `_get_tensor_field_dimensions` sets `pi: 9` with a misleading comment (“10 − 4 = 6, but we use 9”). In 3+1D Landau frame, $\pi^{\mu\nu}$ has **5** independent DOF. Using 9 breaks block sizes and projections.&#x20;
* Numeric substitution tries `complex(self.omega)` and `abs(self.k)` while both are **SymPy Symbols** — this will raise or return non-numerics and then fail casting to `complex`. Evaluate **after** substituting numeric `omega,k`, or keep expressions symbolic until the end.&#x20;

5. ## Momentum-space “transform” doesn’t match any pattern

`_transform_to_momentum_space_symbolic` replaces `Derivative(Symbol("f"), t)` with `-iω f`. Your action uses functions of spacetime coordinates, not the literal symbol `f`, so this replacement never triggers; Laplacian replacement likewise won’t match. Use a **plane-wave ansatz** (∂→ multiplication by $-i\omega$, $i\mathbf{k}$) on field symbols or differentiate actual `sympy.Function`s.&#x20;

6. ## Validation calls a non-existent API

`HydrodynamicModeAnalyzer` uses `self.calculator.compute_propagator(...)`, but neither the full nor simple calculators expose that method (simple exposes `calculate_retarded_propagator`, the full one builds a matrix). This code path will error. &#x20;
Also, the shear check compares to `η` directly, but the small-k shear mode is $ \omega=-i\,\nu k^2$ with $\nu=\eta/(\epsilon+p)$ (or a chosen mass/enthalpy scale); comparing to `η` alone is dimensionally off.&#x20;

7. ## Tensor-aware noise (tensor\_msrjd\_action.py) still uses a 1D delta and placeholders

`tensor_velocity_correlator` uses `DiracDelta(self.coordinates[0])` (time only), no spatial delta; and builds projectors from a bare `IndexedBase("g")` instead of the supplied `Metric`. This won’t contract consistently.&#x20;

---

# High-impact cleanups

* **Make δ⁴ explicit** with (t,x,y,z) symbols everywhere you need locality; avoid the `IndexedBase("x")` indexing trick (it’s mixing points).&#x20;
* **Use the metric** for every trace/contraction/orthogonality. Replace all KroneckerDeltas used as spacetime metrics with `g^{μν}`/`g_{μν}` consistently. &#x20;
* **Fix DOF counts** in propagator blocks (`pi: 5`) and apply proper TT projections before inversion.&#x20;
* **Unify the calculator API**: expose a single method (e.g., `compute_propagator(name1,name2,omega,k)`) in both “simple” and “full” calculators, delegating as needed, so `physics_validation.py` runs. &#x20;
* **Make `SymbolicTensorField` index-aware**: have `__getitem__` build components; make `create_component` call `__getitem__`; make constraints metric-aware; document index variance. &#x20;

---

# Minimal, drop-in patches (ready to paste)



### 2) Correct velocity/heat noise projectors and δ⁴

```python
# msrjd_action.py (inside NoiseCorrelator)
t,x,y,z,tp,xp,yp,zp = sp.symbols("t x y z tp xp yp zp", real=True)
delta_4d = sp.DiracDelta(t-tp)*sp.DiracDelta(x-xp)*sp.DiracDelta(y-yp)*sp.DiracDelta(z-zp)

g_up = sp.IndexedBase("g_up")  # or wire in Metric
u_up = sp.Function("u_up")  # u_up(mu) at background

P_perp = g_up[self.mu, self.nu] + u_up(self.mu)*u_up(self.nu)/PhysicalConstants.c**2

D_uu = 2*self.k_B*self.temperature*(self.parameters.eta/self.parameters.tau_pi)*P_perp*delta_4d
```

(Do the analogous change for `heat_flux_correlator` and use $P_{\mathrm{TT}}^{\mu\nu\alpha\beta}$ for shear.)&#x20;

### 3) Make `create_component` use indexing; make constraints metric-aware

```python
# symbolic_tensors.py
def create_component(self, tensor_indices, coordinate_values=None):
    coordinate_values = coordinate_values or self._coordinates
    if len(tensor_indices) != self.index_count:
        raise ValueError(...)
    return self[tuple(tensor_indices) + tuple(coordinate_values)]

def apply_constraint(self, constraint_type: str) -> sp.Expr:
    if constraint_type == "normalization" and self.field_name == "u":
        mu,nu = symbols("mu nu", integer=True)
        g_dn = sp.IndexedBase("g_dn")
        return sp.Sum(g_dn[mu,nu]*self[mu,*self._coordinates]*self[nu,*self._coordinates],
                      (mu,0,3),(nu,0,3)) + PhysicalConstants.c**2
    elif constraint_type == "traceless" and self.field_name == "pi":
        mu,nu = symbols("mu nu", integer=True)
        g_up = sp.IndexedBase("g_up")
        return sp.Sum(g_up[mu,nu]*self[mu,nu,*self._coordinates], (mu,0,3),(nu,0,3))
```

### 4) Fix propagator block sizes, types, and numeric substitution

```python
# propagators.py
def _get_tensor_field_dimensions(self) -> dict[str,int]:
    return {"rho":1, "u":3, "pi":5, "Pi":1, "q":3}  # correct DOF
# ...
# Keep expressions symbolic; substitute numeric ω,k only when the caller passes them:
def _construct_tensor_block(...):
    coeff_expr = self.quadratic_action[block_key]
    params = self.is_system.parameters
    subs = {"tau_pi": params.tau_pi, "tau_Pi": params.tau_Pi, "tau_q": params.tau_q,
            "eta": params.eta, "zeta": params.zeta, "kappa": params.kappa}
    expr = coeff_expr.subs(subs)              # still symbolic in ω,k
    return self._create_diagonal_tensor_block(field1, expr, dim1)  # keep symbolic

# When you finally need numbers, call .subs({self.omega: ωval, self.k: kval}).evalf()
```

Also pass a **name→index** basis or actual `Field` objects to `PropagatorMatrix`. &#x20;

### 5) Unify calculator API for validation

Provide a thin wrapper:

```python
# both calculators
def compute_propagator(self, f1: str, f2: str, omega, k):
    # simple: return calculate_retarded_propagator(...)
    # full:   build/invert the quadratic matrix and select [f1,f2]
```

so `PhysicsValidation` runs as written. &#x20;

### 6) Replace the ad-hoc derivative replacement with a plane-wave rule

```python
# propagators.py
def _to_momentum_space(self, expr):
    ω,kx,ky,kz = self.omega, *self.k_vec
    rules = {
        sp.Derivative(sp.Function, sp.Symbol): None  # remove dead code
    }
    # Instead, after building linearized equations E[fields], substitute
    # ∂_t→(-I*ω), ∂_i→(I*k_i) via mapping on each Derivative node using .replace
```

(Or better: build the linearized operator directly in (ω, k) from the start.)&#x20;

---

# Smaller nits (quick wins)

* `tensor_msrjd_action.TensorNoiseCorrelator`: use full δ⁴ and the provided `Metric` for $g$; avoid `IndexedBase("g")`.&#x20;
* `physics_validation` shear/bulk checks: compare to $\nu=\eta/(\epsilon+p)$ (or whatever your chosen denominator is), not to `η` itself; and don’t rely on the “>1e6 near pole” heuristic without confirming the matrix inversion actually returns a scalar component. &#x20;
* In `propagators.get_velocity_propagator_components`, you divide by `equilibrium_pressure`; if your IS parameters don’t expose that, this will break. Double-check the parameter name/source.&#x20;

---
