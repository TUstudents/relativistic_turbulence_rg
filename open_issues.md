# Open Issues and Known Bugs

This document tracks known issues, bugs, and inconsistencies that need to be addressed in future development.

## Critical Issues

### 1. Spatial Projector Non-Idempotency Bug
**File**: `rtrg/israel_stewart/constraints.py:31-52`
**Status**: Open  
**Priority**: High  
**Description**: The spatial projector `Δ_{μν} = g_{μν} + u_μ u_ν` does not satisfy the idempotency condition `Δ^2 = Δ` for non-rest frame four-velocities.

**Evidence**:
- Rest frame `u = (1,0,0,0)`: ✅ Works perfectly (`max |Δ² - Δ| = 0.0`)
- Moving frame `u = (1.005, 0.100, 0, 0)`: ❌ Fails (`max |Δ² - Δ| = 0.020`)

**Impact**:
- Affects traceless-transverse projector accuracy in moving frames
- Could impact linearized Israel-Stewart calculations for non-equilibrium states
- Numerical stability issues in dispersion relation calculations

**Root Cause**: Likely issue with covariant vs contravariant index handling or normalization

**Proposed Fix**: Investigate proper spatial projector formulation for arbitrary frames

---

### 2. Metric Dimension Extension Bug  
**File**: `rtrg/core/tensors.py` (Metric class constructor)
**Status**: Open
**Priority**: Medium
**Description**: Metric constructor doesn't properly handle spacetime dimensions > 4.

**Evidence**:
```python
# 5D metric should be diag(-1, 1, 1, 1, 1)
# Actually creates: diag(-1, 1, 1, 1, 0)  # Missing last spatial dimension
```

**Root Cause**:
```python
for i in range(min(len(self.signature), dimension)):
    self.g[i, i] = self.signature[i]
```
Only fills diagonal up to length of default signature `(-1, 1, 1, 1)`.

**Impact**:
- Breaks higher-dimensional theoretical studies
- TT projector fails for dim > 4 unless explicit signature provided
- Could affect field theory calculations in higher dimensions

**Proposed Fix**: Auto-extend signature to `(-1, 1, 1, ..., 1)` for arbitrary dimensions

---

## Minor Issues

### 3. Test Documentation Inconsistencies
**File**: `tests/unit/test_linearized_is.py`
**Status**: Partially Fixed
**Priority**: Low
**Description**: Some test docstrings don't accurately describe test behavior

**Examples**:
- Pressure source inconsistency tests (fixed in recent commit)
- Missing documentation for numerical solver assumptions

**Proposed Fix**: Comprehensive documentation review and update

---

### 4. Missing Test Coverage

#### 4.1 Dimensional Consistency Tests
**Status**: Open
**Priority**: Medium  
**Description**: No tests verify linearized IS system works correctly across different spacetime dimensions

**Missing Coverage**:
- TT projector behavior in 2D/3D/5D spacetime
- Dispersion relations for non-4D cases  
- Causality validation for arbitrary dimensions

#### 4.2 Extreme Parameter Edge Cases
**Status**: Open  
**Priority**: Low
**Description**: Limited testing of edge cases with extreme transport coefficients

**Missing Coverage**:
- Very small/large viscosities
- Near-critical relaxation times
- Temperature/density extremes

---

### 5. Numerical Stability Issues

#### 5.1 Dispersion Relation Solver Robustness
**Status**: Open
**Priority**: Medium
**Description**: Numerical dispersion relation solver may fail for certain parameter combinations

**Symptoms**:
- Occasional convergence failures
- Spurious solution selection
- Sensitivity to initial conditions

**Proposed Fix**:
- Implement more robust root-finding with multiple algorithms
- Add better spurious solution filtering
- Improve initial guess strategies

---

## Recent Fixes (Completed)

### ✅ Background Normalization Hard-codes Minkowski Metric (Fixed)
**File**: `rtrg/israel_stewart/linearized.py:94-99`, `tests/unit/test_linearized_is.py:65-100`
**Description**: Fixed BackgroundState four-velocity normalization that hard-coded Minkowski metric assumption instead of using proper metric tensor
**Evidence**:
- **Problem**: `u_norm_sq = -(self.u[0] ** 2) + sum(self.u[i] ** 2 for i in range(1, 4))` assumed signature `(-,+,+,+)`
- **Inconsistency**: Validation occurred in `BackgroundState.__post_init__` before metric was available in `LinearizedIS`
- **Impact**: Non-Minkowski metrics would pass validation but be inconsistent with rest of package
**Comprehensive Fixes**:
- **Removed hard-coded validation**: Moved from `__post_init__` to explicit method `validate_four_velocity_normalization(metric)`
- **Added proper metric calculation**: Uses `u_norm_sq = np.dot(u_array, metric.g @ u_array)` for `g_{μν} u^μ u^ν`
- **Updated LinearizedIS constructor**: Calls validation with actual metric: `self.background.validate_four_velocity_normalization(self.metric)`
- **Enhanced error messages**: More informative error reporting with scientific notation
**New API**:
- `BackgroundState.validate_four_velocity_normalization(metric=None)`: Explicit validation with any metric
- Backward compatible: Default None parameter uses Minkowski metric for standalone validation
**Updated Tests**:
- `test_invalid_four_velocity_normalization`: Now tests explicit validation method
- `test_invalid_four_velocity_in_linearized_system`: Tests integration with LinearizedIS constructor
- `test_metric_dependent_normalization`: Validates custom metric support
**Impact**: Critical fix for metric consistency - four-velocity normalization now properly uses the actual metric tensor
**Root Cause**: Architectural issue where validation happened before metric was available
**Tests**: All BackgroundState tests pass, maintains backward compatibility while enabling proper metric support

### ✅ Expensive __str__ Method with Heavy Side Effects (Fixed)
**File**: `rtrg/israel_stewart/linearized.py:632`
**Description**: Fixed `LinearizedIS.__str__` method that triggered expensive stability analysis every time the object was printed
**Evidence**:
- **Problem**: `return f"LinearizedIS(background={self.background}, stable={self.is_linearly_stable()})"`
- **Side Effects**: `is_linearly_stable()` performs O(100 × polynomial_solving) computations, can take seconds
- **Violation**: `__str__` methods should be fast and side-effect free
**Comprehensive Fixes**:
- **Removed expensive call**: `__str__` now returns simple `f"LinearizedIS(background={self.background})"`
- **Added cached property**: `@cached_property stability_status` for lazy evaluation when stability info is needed
- **Added detailed method**: `get_stability_summary()` provides comprehensive stability analysis with configurable parameters
- **Added functools import**: Required for `@cached_property` decorator
**Impact**: Critical performance fix - object printing now instant instead of potentially taking seconds during debugging
**Alternative Access**:
- Use `system.stability_status` for cached boolean result
- Use `system.get_stability_summary()` for detailed analysis with growth rates and mode information
**Root Cause**: Expensive computation embedded in string representation violated Python best practices
**Tests**: All existing functionality preserved, no tests depended on `__str__` format

### ✅ isinstance PEP-604 Union Syntax Bug (Fixed)
**Files**: `rtrg/israel_stewart/linearized.py:197`, `rtrg/field_theory/symbolic_tensors.py:185`, `rtrg/field_theory/tensor_action_expander.py:171`, `rtrg/core/fields.py:301`, `tests/unit/test_tensors.py:133`
**Description**: Fixed TypeError risk from invalid isinstance calls using PEP-604 union syntax instead of tuple syntax
**Evidence**:
- **Incorrect**: `isinstance(sol, list | tuple)` (TypeError at runtime)
- **Correct**: `isinstance(sol, (list, tuple))` (Standard tuple syntax)
**Comprehensive Fixes**:
- `linearized.py`: `isinstance(sol, (list, tuple))`
- `symbolic_tensors.py`: `isinstance(indices, (tuple, list))`
- `tensor_action_expander.py`: `isinstance(arg, (int, Symbol))`
- `fields.py`: `isinstance(trace, (int, float, complex))`
- `test_tensors.py`: `isinstance(trace, (int, float, complex))`
**Impact**: Critical fix for runtime compatibility - all isinstance calls now use proper tuple syntax as required by Python specification
**Root Cause**: PEP-604 union syntax (`type1 | type2`) cannot be used with isinstance; requires traditional tuple syntax (`(type1, type2)`)
**Tests**: All changes preserve functionality while ensuring runtime compatibility

### ✅ Linearization Variable Shape Inconsistencies (Fixed)
**Files**: `rtrg/israel_stewart/equations.py:418-490`
**Description**: Fixed critical inconsistency where linearized perturbations were defined as scalar Functions but accessed as vectors/tensors
**Evidence**:
- **Incorrect**: `delta_u = sp.Function("delta_u")(t,x,y,z)` but used as `sp.diff(delta_u, x) + sp.diff(delta_u, y)`
- **Correct**: `delta_u = sp.IndexedBase("delta_u")` used as `sp.diff(delta_u[1], x) + sp.diff(delta_u[2], y) + sp.diff(delta_u[3], z)`
**Comprehensive Fixes**:
- `delta_u`: Changed from scalar Function to IndexedBase vector field with proper spatial components [1,2,3]
- `delta_pi`: Changed from scalar Function to IndexedBase tensor field with proper tensor components [i,j]
- `delta_q`: Changed from scalar Function to IndexedBase vector field with proper spatial components [1,2,3]
- Updated all equations to use proper tensor component notation matching linearized.py implementation
**Mathematical Corrections**:
- Velocity divergence: `∇·δu = ∂_x δu¹ + ∂_y δu² + ∂_z δu³`
- Shear rate tensor: `δσⁱʲ = ½(∇ⁱδuʲ + ∇ʲδuⁱ) - ⅓δⁱʲ∇·δu`
- Vector component equations: Separate equations for each spatial component
**Impact**: Critical fix for mathematical consistency - linearized perturbations now properly treated as tensor objects
**Root Cause**: Mixed scalar/tensor approach in get_linearized_system method conflicted with proper tensor implementation
**Tests**: All 264 unit tests pass, confirming mathematical consistency restored

### ✅ Heat Flux Orthogonality Metric Bug (Fixed)
**Files**: `rtrg/israel_stewart/equations.py:250`, `rtrg/field_theory/msrjd_action.py:558`, `tests/integration/test_basic_phase2_validation.py:135`
**Description**: Fixed incorrect heat flux orthogonality constraint that computed `u^μ q^μ` instead of proper `u_μ q^μ = g_{μν} u^μ q^ν`
**Evidence**:
- **Incorrect**: `heat_orthogonality = sp.Sum(u_mu[mu] * q_mu[mu], (mu, 0, 3))`
- **Correct**: `heat_orthogonality = sp.Sum(g_munu[mu, nu] * u_mu[mu] * q_mu[nu], (mu, 0, 3), (nu, 0, 3))`
**Additional Fixes**:
- Fixed four-velocity normalization constraint in msrjd_action.py to use proper `g_{μν} u^μ u^ν`
- Corrected test contraction in test_basic_phase2_validation.py for proper tensor index summation
**Impact**: Critical fix for Lorentz covariance - all orthogonality and normalization constraints now mathematically correct
**Root Cause**: Missing metric tensor for proper index lowering in constraint calculations
**Tests**: All tests pass, including specific constraint validation tests

### ✅ Tracelessness Implementation Bug (Fixed)
**Files**: `rtrg/israel_stewart/equations.py:239`, `rtrg/field_theory/msrjd_action.py:549`  
**Description**: Fixed incorrect tracelessness constraint implementation that used non-Lorentz-invariant `∑_μ π_{μμ}` instead of proper `g^{μν} π_{μν}`
**Evidence**:
- **Incorrect**: `trace_constraint = sp.Sum(pi_munu[mu, mu], (mu, 0, 3))`
- **Correct**: `trace_constraint = sp.Sum(g_inv[mu, nu] * pi_munu[mu, nu], (mu, 0, 3), (nu, 0, 3))`
**Impact**: Critical fix for mathematical correctness - tracelessness is now properly Lorentz-invariant across all reference frames
**Root Cause**: Missing inverse metric tensor in trace calculation for covariant tensor components
**Tests**: All 264 unit tests pass, confirming no regressions introduced

### ✅ TT Projector Dimension Bug (Fixed)
**Commit**: `1a85396`  
**Description**: Fixed hard-coded `1/3` factor to dimension-dependent `1/(d-1)`
**Impact**: TT projector now works correctly for arbitrary spacetime dimensions

### ✅ Pressure Source Inconsistency (Fixed)  
**Description**: Fixed test inconsistency where some tests used `parameters.equilibrium_pressure` instead of `background.pressure`

---

## Investigation Notes

### Spatial Projector Mathematics
The correct spatial projector should satisfy:
1. **Orthogonality**: `u^μ Δ_{μν} = 0`
2. **Idempotency**: `Δ_{μρ} Δ^{ρν} = Δ_μ^ν`  
3. **Trace**: `tr(Δ) = d - 1` where `d` is spacetime dimension

Current implementation satisfies (1) and (3) but fails (2) for moving frames.

### Alternative Formulations to Investigate
- Contravariant form: `Δ^{μν} = g^{μν} + u^μ u^ν`
- Mixed index handling with proper metric contractions
- Explicit projection operator construction

---

## Contributing

When adding new issues to this file:
1. Use clear, descriptive titles
2. Include file paths and line numbers
3. Provide evidence/examples
4. Assess impact and priority
5. Suggest proposed fixes where applicable

When fixing issues:
1. Update the issue status to ✅ Fixed
2. Reference the commit hash
3. Move to "Recent Fixes" section
4. Keep the entry for historical record
