import numpy as np
import scipy.linalg as la

def pure_fidelity(state1, state2):
    """Calculates the fidelity between two pure quantum states."""
    return np.abs(np.vdot(state1, state2))**2

def mixed_fidelity(rho1, rho2):
    """Calculates the fidelity between two mixed quantum states (density matrices)."""
    sqrt_rho1 = la.sqrtm(rho1)
    product = sqrt_rho1 @ rho2 @ sqrt_rho1
    return np.real(np.trace(la.sqrtm(product)))**2

def mixed_pure_fidelity(rho, psi):
    """Calculates the fidelity between a mixed state (density matrix) and a pure state."""
    return np.vdot(psi, rho @ psi).real