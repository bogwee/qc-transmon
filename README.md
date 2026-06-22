# Quantum Optimal Control of a Transmon Qubit

## Overview

This project investigates the control of quantum systems through numerical optimal control techniques. The primary objective is to develop a simulation framework for single-qubit dynamics and implement pulse optimization algorithms capable of synthesizing high-fidelity quantum gates.

The project is inspired by real-world challenges encountered in quantum computing platforms such as superconducting qubits, neutral-atom processors, trapped ions, and photonic systems. In particular, it focuses on the type of control problems faced by industrial quantum computing companies such as Alice & Bob, PASQAL, Quandela, Rigetti, IonQ, IBM Quantum, and Google Quantum AI.

The project combines concepts from:

* Quantum Mechanics
* Quantum Information Theory
* Geometric Control Theory
* Numerical Optimization
* Scientific Computing
* Applied Mathematics

---

## Motivation

Modern quantum processors rely on precise control of quantum states to execute computational tasks. Physical qubits are manipulated through external control fields, such as microwave pulses, laser pulses, or electromagnetic fields.

Designing these control sequences is a challenging optimization problem due to:

* Decoherence
* Hardware imperfections
* Noise
* Limited control resources

Optimal control methods provide a framework for automatically designing pulse sequences that achieve target quantum operations with high fidelity.

The long-term objective of this project is to bridge the gap between:

* Mathematical control theory
* Numerical optimization
* Physical quantum hardware

---

## Project Goals

The project is divided into several stages.

### Stage 1 — Quantum Dynamics Simulation

Develop a simulator capable of:

* Representing quantum states
* Defining Hamiltonians
* Solving the time-dependent Schrödinger equation
* Computing unitary evolutions

Topics covered:

* Qubits
* Bloch sphere
* Pauli matrices
* Matrix exponentials
* Time evolution operators

---

### Stage 2 — Controlled Quantum Dynamics

Introduce external control fields into the Hamiltonian:

[
H(t)=H_0+\sum_i u_i(t)H_i
]

where:

* (H_0) is the drift Hamiltonian
* (H_i) are control Hamiltonians
* (u_i(t)) are time-dependent control amplitudes

Applications:

* Rabi oscillations
* State transfer
* Quantum gate generation

---

### Stage 3 — Gate Fidelity Analysis

Implement quantitative measures of control performance.

Target tasks include:

* X gate synthesis
* Y gate synthesis
* Hadamard gate synthesis

Metrics:

* Gate fidelity
* State fidelity
* Error estimation

---

### Stage 4 — Optimal Control

Formulate quantum gate synthesis as an optimization problem.

Objectives:

* Minimize gate error
* Maximize gate fidelity
* Respect physical constraints

Methods:

* Gradient descent
* Numerical optimization
* Constrained optimization

Libraries:

* SciPy
* NumPy

---

### Stage 5 — GRAPE Algorithm

Implement the Gradient Ascent Pulse Engineering (GRAPE) algorithm.

Goals:

* Compute gradients efficiently
* Optimize pulse sequences
* Achieve high-fidelity quantum gates

Expected outcomes:

* Fidelity above 99%
* Robust pulse design
* Faster convergence than finite-difference methods

---

### Stage 6 — Open Quantum Systems

Extend the simulator to include realistic noise processes.

Topics:

* Density matrices
* Lindblad master equations
* Decoherence
* Relaxation (T1)
* Dephasing (T2)

This stage introduces realistic hardware effects encountered in experimental quantum processors.

---

### Stage 7 — Transmon Qubit Modeling

Develop a simplified model of a superconducting transmon qubit.

Topics:

* Josephson junctions
* Anharmonic oscillators
* Energy level structure
* Microwave control

Applications:

* Superconducting quantum computing
* Quantum gate implementation
* Hardware-aware control design

---

## Technologies

### Programming Languages

* Python

### Scientific Computing

* NumPy
* SciPy
* Matplotlib

### Quantum Computing

* QuTiP
* Qiskit (future extension)

### Optimization

* SciPy Optimize
* JAX

### Development Tools

* Git
* GitHub
* Jupyter Notebook
* VS Code

---

## Mathematical Background

The project draws upon several mathematical disciplines:

### Linear Algebra

* Complex vector spaces
* Eigenvalue problems
* Matrix exponentials
* Unitary operators

### Differential Equations

* Schrödinger equation
* Lindblad equation

### Lie Theory

* SU(2)
* Lie algebras
* Commutators
* Reachability
* Controllability

### Optimization

* Gradient methods
* Automatic differentiation
* Optimal control

---

## Repository Structure

```text
quantum-control/

├── notebooks/
│   ├── 01_two_level_system.ipynb
│   ├── 02_rabi_oscillations.ipynb
│   ├── 03_gate_fidelity.ipynb
│   ├── 04_optimal_control.ipynb
│   └── 05_grape.ipynb
│
├── src/
│   ├── hamiltonians.py
│   ├── evolution.py
│   ├── fidelity.py
│   ├── optimization.py
│   └── grape.py
│
├── tests/
│
├── figures/
│
└── README.md
```

---

## Learning Outcomes

Upon completion, this project aims to provide practical experience in:

* Quantum dynamics simulation
* Quantum gate design
* Optimal control theory
* Scientific software development
* Numerical optimization
* Quantum hardware modeling

The resulting framework serves as a foundation for more advanced topics such as:

* Quantum error correction
* Quantum calibration
* Reinforcement learning for control
* Quantum systems engineering
* Fault-tolerant quantum computing

---

## References

1. Nielsen, M. A., & Chuang, I. L. *Quantum Computation and Quantum Information*.

2. Khaneja, N., Reiss, T., Kehlet, C., Schulte-Herbrüggen, T., & Glaser, S. J. *Optimal Control of Coupled Spin Dynamics: Design of NMR Pulse Sequences by Gradient Ascent Algorithms*.

3. D'Alessandro, D. *Introduction to Quantum Control and Dynamics*.

4. Schulte-Herbrüggen, T. et al. *Quantum Optimal Control in Quantum Technologies*.

5. Koch, C. P. *Controlling Open Quantum Systems: Tools, Achievements, and Limitations*.

---

## Author

This project is part of a personal exploration of quantum control, quantum systems engineering, and the mathematical foundations of quantum technologies, with the goal of developing expertise relevant to next-generation quantum computing platforms and industrial quantum research.
