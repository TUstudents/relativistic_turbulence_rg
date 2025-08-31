import numpy as np

from rtrg.core.constants import PhysicalConstants

# Test case that's failing
eta = 100.0  # Very large viscosity
zeta = 50.0
tau_pi = 0.001  # Very short relaxation time
tau_Pi = 0.001
cs = 0.1  # Small sound speed
energy_density = 0.01  # Small energy density

print(f"Parameters: eta={eta}, zeta={zeta}, tau_pi={tau_pi}, energy_density={energy_density}")
print(f"Speed of light: {PhysicalConstants.c}")
print(f"Sound speed: {cs}")


# Manually compute the new causality check
def check_causality(eta, zeta, tau_pi, tau_Pi, cs, energy_density):
    # Sound speed must be subluminal
    if cs >= PhysicalConstants.c:
        return False

    # Check if we have valid parameters for computation
    if energy_density <= 0 or tau_pi <= 0:
        # Can't compute shear speed, assume valid
        v_shear = 0.0
    else:
        # Shear mode speed: In relativistic hydrodynamics, this should be bounded by c
        # Using a more physically motivated approach
        shear_arg = max(0.0, eta / (energy_density * tau_pi))
        # Ensure the argument doesn't lead to superluminal speeds
        if shear_arg >= PhysicalConstants.c**2:
            return False
        v_shear = np.sqrt(shear_arg)

    # Bulk mode contributions
    if energy_density <= 0 or tau_Pi <= 0:
        v_bulk_contrib = 0.0
    else:
        # Bulk mode speed: Also bounded by c
        bulk_arg = max(0.0, zeta / (energy_density * tau_Pi))
        # Ensure the argument doesn't lead to superluminal speeds
        if bulk_arg >= PhysicalConstants.c**2:
            return False
        v_bulk_contrib = np.sqrt(bulk_arg)

    # Individual speeds must be subluminal (more stringent than combined)
    if v_shear >= PhysicalConstants.c or v_bulk_contrib >= PhysicalConstants.c:
        return False

    return True


result = check_causality(eta, zeta, tau_pi, tau_Pi, cs, energy_density)
print(f"Causality check result: {result}")

# Let's also check what the individual terms are:
shear_arg = eta / (energy_density * tau_pi)
bulk_arg = zeta / (energy_density * tau_Pi)
print(f"shear_arg = {shear_arg}")
print(f"bulk_arg = {bulk_arg}")
print(f"shear_arg >= c^2? {shear_arg >= PhysicalConstants.c**2}")
print(f"bulk_arg >= c^2? {bulk_arg >= PhysicalConstants.c**2}")
