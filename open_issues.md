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
