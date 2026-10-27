import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

def gate(h, theta):
    """Returns the unitary matrix for a single-qubit rotation gate around the specified axis (X, Y, or Z) by the given angle in radians."""
    if len(h) != 3:
        raise ValueError("The axis vector h must be a 1D array of length 3.")
    return np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * (h[0] * X + h[1] * Y + h[2] * Z)

S = gate([0, 0, 1], np.pi / 2)
T = gate([0, 0, 1], np.pi / 4)
SX = gate([1, 0, 0], np.pi / 2)
SY = gate([0, 1, 0], np.pi / 2)


CNOT_gate = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]], dtype=complex)

