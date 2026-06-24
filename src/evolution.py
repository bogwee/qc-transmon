import scipy.linalg as la

def propagator(H, t):
    """Evolution of the propagator for a spin-1/2 particle for a given time."""
    return la.expm(-1j * H * t)