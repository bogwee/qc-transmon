import numpy as np
from fidelity import average_gate_fidelity, gate_fidelity
from evolution import simulate_pulse, propagator_jax
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from functools import partial


def monte_carlo_pulse_search(H_drift, U_target, t_grid, n_segments=6, tol=1e-2, max_samples=500, u_max=5.0, y_control=False):
    """Random search over piecewise-constant control pulses. Returns the best pulse, its fidelity, and the best-fidelity history."""
    rng = np.random.default_rng()

    best_pulse = None
    best_fidelity = -np.inf
    history = []

    while best_fidelity < 1-tol:
        pulse_x = rng.uniform(-u_max, u_max, size=n_segments)
        pulse_y = rng.uniform(-u_max, u_max, size=n_segments) if y_control else np.zeros(n_segments)

        U_simulated = simulate_pulse(H_drift, t_grid, pulse_x, pulse_y)

        fidelity = average_gate_fidelity(U_target, U_simulated)

        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_pulse = pulse_x

        history.append(best_fidelity)

        if len(history) >= max_samples:
            print(f"Reached max samples ({max_samples}).")
            break  

    return best_pulse, best_fidelity, np.array(history)


def constraints(n_segments, max_amplitude):
    """Constraints functions for the optimization problem."""
    amplitude_constraint = lambda controls : 1.0 - ((controls[:n_segments])**2 + (controls[n_segments:])**2) / max_amplitude**2
    amplitude_constraint_jac = lambda controls : np.concatenate((np.diag(-2.0 * (controls[:n_segments]) / max_amplitude**2), 
                                                                np.diag(-2.0 * (controls[n_segments:]) / max_amplitude**2)), axis=1)

    return {"type": "ineq","fun": amplitude_constraint,"jac": amplitude_constraint_jac}


def gate_infidelity(drift, controls, delta_t, U_target):
    """Calculates the gate infidelity for a given set of controls."""
    n_segments = controls.shape[0] // 2
    ux = controls[:n_segments]
    uy = controls[n_segments:]
    controls_2d = jnp.stack([ux, uy])

    U_actual = propagator_jax(drift, controls_2d, delta_t)

    return 1.0 - gate_fidelity(U_target, U_actual)


objective_and_gradient = jax.jit(jax.value_and_grad(gate_infidelity,argnums=1))


def scipy_objective(controls, drift, delta_t, U_target):
    """Objective function for scipy.optimize.minimize. Returns the gate infidelity and its gradient."""
    value, gradient = objective_and_gradient(drift, controls, delta_t, U_target)
    
    return float(value), np.asarray(gradient).reshape(-1)

def optimization_callback(controls, history, drift, delta_t, U_target):
    """Record fidelity and control parameters after each iteration."""

    infidelity, _ = scipy_objective(controls, drift, delta_t, U_target)

    history["fidelities"].append(1.0 - infidelity)
    history["pulses"].append(controls.copy())


def grape_optimization(drift, controls, delta_t, U_target, constraint, maxiter=500, ftol=1e-10):
    """Performs GRAPE optimization using scipy.optimize.minimize with the SLSQP method."""
    history = {
    "fidelities": [],
    "pulses": []
    }

    callback = partial(
        optimization_callback,
        history=history,
        drift=drift,
        delta_t=delta_t,
        U_target=U_target
    )
    
    result = minimize(
        scipy_objective,
        np.asarray(controls, dtype=float),
        args=(drift, delta_t, U_target),
        method="SLSQP",
        jac=True,
        constraints=constraint,
        callback=callback,
        options={
            "maxiter": maxiter,
            "ftol": ftol,
            "disp": False
        })

    return {"result": result, "iteration_fidelities": history["fidelities"], "pulse_history": history["pulses"]}


def finite_difference_gradient(drift, controls, delta_t, U_target, index, epsilon=1e-6):
    """Calculates the finite difference gradient of the objective function at a given index."""
    x_plus = controls.copy()
    x_minus = controls.copy()

    x_plus[index] += epsilon
    x_minus[index] -= epsilon

    f_plus = float(gate_infidelity(drift, x_plus, delta_t, U_target))
    f_minus = float(gate_infidelity(drift, x_minus, delta_t, U_target))

    return (f_plus - f_minus) / (2 * epsilon)