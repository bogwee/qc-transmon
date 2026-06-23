from hamiltonians import sigma_x, sigma_y, sigma_z, identity
import numpy as np

def test_pauli():
    """Test the Pauli matrices."""
    expected_x = np.array([[0, 1], [1, 0]])
    assert np.allclose(sigma_x, expected_x), "Pauli-X matrix is incorrect."
    expected_y = np.array([[0, -1j], [1j, 0]])
    assert np.allclose(sigma_y, expected_y), "Pauli-Y matrix is incorrect."
    expected_z = np.array([[1, 0], [0, -1]])
    assert np.allclose(sigma_z, expected_z), "Pauli-Z matrix is incorrect."
    expected_sq = identity
    assert np.allclose(sigma_x @ sigma_x, expected_sq), "Pauli-X squared is not the identity matrix."
    assert np.allclose(sigma_y @ sigma_y, expected_sq), "Pauli-Y squared is not the identity matrix."
    assert np.allclose(sigma_z @ sigma_z, expected_sq), "Pauli-Z squared is not the identity matrix."

test_pauli()