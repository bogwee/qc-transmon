import numpy as np

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
identity = np.eye(2, dtype=complex)

def hamiltonian(omega, beta, delta):
    """Returns the Hamiltonian matrix."""
    return 0.5 * omega * sigma_z + beta * sigma_y + delta * sigma_x
