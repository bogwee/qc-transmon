import scipy.linalg as la

def evolution_operator(H, time):
    """Evolution of the propagator for a spin-1/2 particle for a given time."""
    return la.expm(-1j * H * time)