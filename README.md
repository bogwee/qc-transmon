# Quantum Optimal Control of a Transmon Qubit

## Overview

This project investigates the numerical simulation and optimal control of superconducting transmon qubits, with the goal of designing high-fidelity quantum gates while accounting for realistic hardware constraints. It combines quantum dynamics, numerical optimization and performance benchmarking to study how control pulses can improve quantum operations and reduce leakage to higher energy levels.

## Technical Development

1. **Two-Level Quantum Dynamics** — Modelled the time-dependent Schrödinger equation under drift and control Hamiltonians. Analysed state populations, Rabi oscillations and Bloch-sphere trajectories.

2. **Quantum Gate Fidelity** — Implemented target quantum gates and fidelity metrics to evaluate gate performance as a function of control parameters.

3. **GRAPE Optimal Control** — Developed gradient-based pulse optimization to synthesize quantum gates and maximize fidelity.

4. **Multi-Level Transmon Modelling** — Extended the model to an anharmonic multi-level system to investigate leakage outside the computational subspace.

5. **Derivative-Free Optimization** — Planned integration of NOMAD to optimize control pulses and benchmark its performance against GRAPE.

6. **HPC Acceleration and Benchmarking** — Planned investigation of CUDA-Q for accelerated quantum dynamics simulations, comparing computational runtime and numerical accuracy against a CPU baseline.

7. **Robustness and Validation** — Planned evaluation of gate fidelity, leakage, convergence, computational cost and sensitivity to control and model parameter variations.

## Technologies and Methods

**Languages:** Python

**Libraries:** NumPy, SciPy, QuTiP, JAX, Matplotlib

**Quantum Control:** GRAPE, derivative-free optimization with NOMAD

**Simulation and Computing:** Schrödinger equation, Hamiltonian modelling, multi-level transmon dynamics, CUDA-Q/HPC exploration

**Engineering Practices:** Numerical benchmarking, automated testing, reproducible experiments