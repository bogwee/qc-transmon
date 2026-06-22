import hamiltonians
import scipy.linalg as la

def evolve(state, gamma, beta, delta, time):
    """Evolves the state of a spin-1/2 particle under the Hamiltonian for a given time."""
    H = hamiltonians.hamiltonian(gamma, beta, delta)
    U = la.expm(-1j * H * time)  # Time evolution operator
    return U @ state  # Evolve the state