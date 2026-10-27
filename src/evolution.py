import scipy.linalg as la
import numpy as np
from scipy.interpolate import CubicSpline

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
identity = np.eye(2, dtype=complex)

def hamiltonian(omega, beta, delta):
    """Returns the Hamiltonian matrix."""
    return 0.5 * (omega * sigma_z + beta * sigma_y + delta * sigma_x)

def propagator(H, t):
    """Propagator for a time-independent Hamiltonian."""
    return la.expm(-1j * H * t)

def generate_random_drift(t_grid, n_knots=12, amplitudes=(1.0, 0.3, 0.3), detuning=1.0):
    """Generates a random drift Hamiltonian as a function of time using cubic splines."""
    rng = np.random.default_rng()

    t_knots = np.linspace(t_grid[0], t_grid[-1], n_knots)

    omega_knots = (amplitudes[0] * (detuning + 0.2 * rng.standard_normal(n_knots)))
    beta_knots = (amplitudes[1] * rng.standard_normal(n_knots))
    delta_knots = (amplitudes[2] * rng.standard_normal(n_knots))

    omega_spline = CubicSpline(t_knots, omega_knots)
    beta_spline = CubicSpline(t_knots, beta_knots)
    delta_spline = CubicSpline(t_knots, delta_knots)

    drift_hamiltonian = lambda t: hamiltonian(float(omega_spline(t)), float(beta_spline(t)), float(delta_spline(t)))

    return drift_hamiltonian, (omega_spline, beta_spline, delta_spline)

def propagate_time_dependent(H_t, t_grid):
    """Calculates the time-ordered propagator for a time-dependent Hamiltonian using the midpoint rule."""
    U0 = np.eye(2, dtype=complex)

    U_values = np.empty((len(t_grid), U0.shape[0], U0.shape[1]), dtype=complex)
    U_values[0] = U0

    for k in range(len(t_grid) - 1):
        dt = t_grid[k + 1] - t_grid[k]
        t_mid = 0.5 * (t_grid[k] + t_grid[k + 1])

        H_mid = H_t(t_mid)
        U_step = la.expm(-1j * H_mid * dt)

        U_values[k + 1] = U_step @ U_values[k]

    return U_values

def make_controlled_hamiltonian(H_drift, t_final, pulse_x, pulse_y=np.zeros(1)):
    """Constructs our systems total Hamiltonian from a given set of control pulses and a drift."""
    pulse_x = np.asarray(pulse_x, dtype=float)
    pulse_y = np.asarray(pulse_y, dtype=float)
    n_segments_x = len(pulse_x)
    n_segments_y = len(pulse_y)
    segment_duration_x = t_final / n_segments_x
    segment_duration_y = t_final / n_segments_y

    H_total = lambda t: H_drift(t) + 0.5 * (pulse_x[min(int(t / segment_duration_x), n_segments_x - 1)] * sigma_x + pulse_y[min(int(t / segment_duration_y), n_segments_y - 1)] * sigma_y)

    return H_total


def simulate_pulse(H_drift, t_grid, pulse_x, pulse_y=np.zeros(1)):
    """Simulates the evolution of a quantum system under a time-dependent Hamiltonian with a given control pulse."""
    H_total = make_controlled_hamiltonian(H_drift, t_grid[-1], pulse_x, pulse_y)

    return propagate_time_dependent(H_total, t_grid)[-1]

