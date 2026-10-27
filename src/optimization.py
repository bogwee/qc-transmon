import numpy as np
from fidelity import average_gate_fidelity
from evolution import simulate_pulse

def monte_carlo_pulse_search(H_drift, U_target, t_grid, n_segments=6, tol=1e-2, max_samples=500, u_max=5.0):
    """Random search over piecewise-constant X-control pulses. Returns the best pulse, its fidelity, and the best-fidelity history."""
    rng = np.random.default_rng()

    best_pulse = None
    best_fidelity = -np.inf
    history = []

    while best_fidelity < 1-tol:
        pulse_x = rng.uniform(-u_max, u_max, size=n_segments)

        U_simulated = simulate_pulse(H_drift, t_grid, pulse_x)

        fidelity = average_gate_fidelity(U_target, U_simulated)

        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_pulse = pulse_x

        history.append(best_fidelity)

        if len(history) >= max_samples:
            print(f"Reached max samples ({max_samples}).")
            break  

    return best_pulse, best_fidelity, np.array(history)

def grape_optimization():
    """Gradient-based optimization of piecewise-constant X- and Y-control pulses using GRAPE."""
    return 0