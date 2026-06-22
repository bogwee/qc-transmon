import numpy as np

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

def hamiltonian(gamma, beta, delta):
    """Returns the Hamiltonian for a spin-1/2 particle in a magnetic field."""
    return gamma * sigma_x + beta * sigma_y + delta * sigma_z
