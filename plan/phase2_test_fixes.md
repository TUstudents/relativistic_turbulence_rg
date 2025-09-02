# Fix Phase 2 Test Failures Comprehensive Plan

## Analysis of Test Failures

Based on the test execution results showing 17 failed tests, 16 passed tests, 2 skipped, and 4 errors in `test_phase2_tensor_validation.py`, the following key issues have been identified:

### Key Issues Identified

1. **SymbolicTensorField Constructor Issues**
   - Tests are passing `field_type` parameter that doesn't exist in constructor
   - UndefinedFunction errors suggest issues with SymPy Function integration
   - Missing TensorFieldProperties class in symbolic_tensors.py

2. **Missing Properties and Methods**
   - `FieldProperties` missing `field_type` attribute in phase_integration.py
   - Missing property accessors in SymbolicTensorField
   - Missing constraint application methods (`apply_constraint`)

3. **Phase Integration Problems**
   - Conversion between Phase 1 and Phase 2 field systems has attribute mismatches
   - Field registry methods missing (`get_tensor_aware_field`)

4. **TensorDerivative Issues**
   - Constructor may have parameter mismatches
   - Missing methods for derivative operations

5. **Action Construction Problems**
   - Issues with field creation during Israel-Stewart field setup
   - MSRJD action construction failing with tensor field creation

## Implementation Plan

### Phase 1: Fix SymbolicTensorField Constructor and Properties

1. **Add missing TensorFieldProperties dataclass**
   - Create dataclass with name, index_structure, coordinates, constraints, field_type
   - Add to symbolic_tensors.py imports and usage
   - Location: `rtrg/field_theory/symbolic_tensors.py`

2. **Fix SymbolicTensorField constructor**
   - Update `__new__` method to accept field_type parameter
   - Handle field_type storage in tensor properties
   - Fix SymPy Function integration issues causing UndefinedFunction errors
   - Location: `rtrg/field_theory/symbolic_tensors.py:100-140`

3. **Add missing constraint methods**
   - Implement `apply_constraint` method for normalization, traceless, orthogonal constraints
   - Add constraint validation logic for physical tensor fields
   - Location: `rtrg/field_theory/symbolic_tensors.py` (new methods)

### Phase 2: Fix Field Properties and Integration

4. **Add field_type to FieldProperties class**
   - Update FieldProperties dataclass in core/fields.py to include field_type attribute
   - Ensure backward compatibility with existing code
   - Location: `rtrg/core/fields.py` (FieldProperties class)

5. **Fix phase integration property access**
   - Update phase_integration.py to handle missing field_type attributes safely
   - Add proper error handling for attribute access
   - Location: `rtrg/field_theory/phase_integration.py:231`

6. **Add missing field registry methods**
   - Implement `get_tensor_aware_field` method in FieldRegistry
   - Update registry interfaces for Phase 1/Phase 2 compatibility
   - Location: `rtrg/core/fields.py` (FieldRegistry class)

### Phase 3: Fix TensorDerivative Operations  

7. **Fix TensorDerivative class constructor**
   - Ensure `__new__` method matches test usage patterns
   - Add proper parameter validation and storage
   - Location: `rtrg/field_theory/symbolic_tensors.py` (TensorDerivative class)

8. **Implement derivative operation methods**
   - Add missing methods for partial and covariant derivatives
   - Implement index contraction operations
   - Location: `rtrg/field_theory/symbolic_tensors.py` (TensorDerivative methods)

### Phase 4: Fix Action Construction and Field Creation

9. **Fix Israel-Stewart field creation**
   - Update `create_israel_stewart_fields` method to use correct constructor
   - Handle field_type parameter properly in field creation
   - Location: `rtrg/field_theory/symbolic_tensors.py` (IndexedFieldRegistry)

10. **Fix MSRJD action construction**
    - Resolve tensor action component creation issues
    - Ensure proper field-antifield pairing
    - Location: `rtrg/field_theory/tensor_msrjd_action.py`

### Phase 5: Update Tests and Validation

11. **Fix test parameter usage**
    - Update tests to use correct constructor parameters
    - Add proper error handling for missing methods
    - Location: `tests/integration/test_phase2_tensor_validation.py`

12. **Add missing test fixtures**
    - Ensure all test setup methods create valid objects
    - Add proper mock objects where needed
    - Location: `tests/integration/test_phase2_tensor_validation.py` (setup methods)

## Error-Specific Fixes

### UndefinedFunction.__new__() Error
- **Issue**: SymbolicTensorField extends SymPy Function incorrectly
- **Fix**: Proper SymPy Function subclassing with correct argument handling
- **File**: `rtrg/field_theory/symbolic_tensors.py:130-150`

### AttributeError: 'FieldProperties' object has no attribute 'field_type'
- **Issue**: FieldProperties missing field_type attribute
- **Fix**: Add field_type: str field to FieldProperties dataclass
- **File**: `rtrg/core/fields.py` (FieldProperties definition)

### 'FieldRegistry' object has no attribute 'get_tensor_aware_field'
- **Issue**: Missing method in FieldRegistry
- **Fix**: Implement get_tensor_aware_field method
- **File**: `rtrg/core/fields.py` (FieldRegistry class)

## Expected Outcomes

After implementing this plan:
- All 39 Phase 2 integration tests should pass
- Phase 1 functionality remains intact (233 tests still passing)
- Complete Phase 1/Phase 2 integration working correctly
- All tensor field operations validated against Israel-Stewart theory
- Full symbolic tensor system operational for MSRJD calculations

## Priority Order

1. **High Priority**: Fix SymbolicTensorField constructor (fixes ~8 tests)
2. **High Priority**: Add field_type property (fixes ~4 tests)  
3. **Medium Priority**: Fix TensorDerivative operations (fixes ~4 tests)
4. **Medium Priority**: Fix action construction (fixes ~3 tests)
5. **Low Priority**: Update test fixtures and error handling

## Validation Strategy

After each phase:
1. Run Phase 2 integration tests: `uv run pytest tests/integration/test_phase2_tensor_validation.py -v`
2. Run Phase 1 tests to ensure no regressions: `uv run pytest tests/unit/test_tensor_infrastructure.py -v`
3. Run full test suite periodically: `uv run pytest tests/unit/ -x`

This systematic approach ensures that each fix builds on the previous ones and maintains system coherence throughout the implementation process.
