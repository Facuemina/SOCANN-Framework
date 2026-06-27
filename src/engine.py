#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine.py

Core physics formulations, matrix initializations, and highly 
optimized JAX integration loops for CANN simulations.
"""

import numpy as np
from jax import numpy as jnp
from jax import jit, lax
from theory import learned_U_std, theoretical_distance

# ---------------------------------------------------------
# INITIALIZATIONS & TOPOLOGY
# ---------------------------------------------------------

@jit
def gaussian(x, y, std, L):
    """Periodic Gaussian function."""
    dx = jnp.minimum(jnp.abs(x - y), L - jnp.abs(x - y))
    norm = 1 / jnp.sqrt(2 * jnp.pi * std**2)
    return jnp.exp(-0.5 * (dx / std) ** 2) * norm

def create_gaussian_connectivity_matrix(Nin, std, Lin, Nout=None, Lout=None, W0=1., periodic=True):
    """Generates a Gaussian weight matrix for the network."""
    x_in = jnp.linspace(0, Lin, Nin, 1 - periodic).reshape((1, Nin))
    Nout = Nin if Nout is None else Nout
    Lout = Lin if Lout is None else Lout
    x_out = jnp.linspace(0, Lout, Nout, 1 - periodic).reshape((Nout, 1)) / Lout * Lin
    return W0 * gaussian(x_in, x_out, std, Lin * periodic)

def set_initial_conditions(N, J0, params, input_params):
    """Initializes the network state based on theoretical equilibrium."""
    STD = learned_U_std(input_params['beta'], input_params['std'])
    x = jnp.linspace(0, input_params['L'], N, False).reshape((N, 1))
    rho = N / input_params['L']
    
    U0_term = np.sqrt(params['tau'] * params['m'] / params['tauv']) + 1
    root_term = np.sqrt((rho * J0)**2 - 4 * np.sqrt(2 * np.pi) * (U0_term)**2 * params['k'] * rho * STD)
    
    U0 = (rho * J0 + root_term) / (2 * np.sqrt(np.pi) * U0_term * params['k'] * rho * STD)
    V0 = (rho * J0 + root_term) / (np.sqrt(2 * np.pi) * params['k'] * rho**2 * STD * J0)
    
    d = theoretical_distance(params['m'], params['tau'], params['tauv'], STD, v=0)
    U = U0 * gaussian(x, 0, STD, input_params['L'])
    r = jnp.maximum(U, 0)**2 / (1 + params['k'] * jnp.sum(jnp.maximum(U, 0)**2))
    V = V0 * gaussian(x, -d, STD, input_params['L'])
    
    return U, V, r

# ---------------------------------------------------------
# CORE PHYSICS ENGINE (UNJITTED)
# ---------------------------------------------------------

def _cann_dynamics(U, V, W_ff, W_rec, I0, v_curr, neural_params, input_params):
    """
    Core ODE system for the CANN. 
    This function is inlined by the JAX compiler in the simulation loops.
    """
    tau_inv, tauv_inv, m, k, g, gain = neural_params
    dt, sR, L, v_eq, R0, a_vel = input_params
    
    # Firing rate
    U_pos = jnp.maximum(U, 0)
    norm_sq = jnp.sum(U_pos ** g)
    fU = gain * (U_pos ** g) / (1 + k * norm_sq)
    
    # Dynamics
    dU = (-U - V + (W_ff @ I0) + (W_rec @ fU) + a_vel * (v_curr - v_eq)) * tau_inv
    dV = (-V + m * U) * tauv_inv
    
    return dU, dV, fU

# ---------------------------------------------------------
# SIMULATION LOOPS
# ---------------------------------------------------------

def run_basic_simulation(U0, V0, W_ff, W_rec, x0, v, neural_params, input_params, pos, T):
    """Standard simulation with fixed weights."""
    angles = jnp.linspace(0, jnp.pi * 2, len(U0), False).reshape((len(U0), 1))    
    
    @jit
    def step(carry, i):
        U, V, x_input = carry
        dt, sR, L, v_eq, R0, a_vel = input_params
        
        v0 = lax.dynamic_index_in_dim(v, i, axis=0, keepdims=False)
        x_input_next = jnp.mod(x_input + dt * v0, L)
        I0 = R0 * gaussian(pos, x_input_next, sR, L)
        
        dU, dV, _ = _cann_dynamics(U, V, W_ff, W_rec, I0, v0, neural_params, input_params)
        
        U_next = U + dt * dU
        V_next = V + dt * dV
        
        # Center of mass calculation for tracking
        U_pos = jnp.maximum(U_next, 0)
        u_posx = jnp.sum(jnp.cos(angles) * U_pos) / jnp.sum(U_pos)
        u_posy = jnp.sum(jnp.sin(angles) * U_pos) / jnp.sum(U_pos)
        u_pos = jnp.mod(jnp.atan2(u_posy, u_posx), 2 * jnp.pi) / (2 * jnp.pi) * L
        
        return (U_next, V_next, x_input_next), (x_input_next, u_pos)
    
    (Uf, Vf, xf), history = lax.scan(step, (U0, V0, x0), jnp.arange(T))
    return (Uf, Vf, xf), history


def run_learning_ff_and_rec(U0, V0, W_ff0, W_rec0, x0, v, neural_params, learning_params, pos, T, T0=0): 
    """Simulation with Hebbian plasticity on both feedforward and recurrent weights."""
    diag_idx = jnp.diag_indices(W_rec0.shape[0])           
    
    @jit
    def step(i, carry):
        U, V, W_ff, W_rec, x_input = carry
        
        # Unpack learning parameters 
        dt, betaJ, betaW, etaJ, etaW, sR, L, v_eq, av, R0, ain, arec = learning_params
        input_params = (dt, sR, L, v_eq, R0, av)
        
        v0 = lax.dynamic_index_in_dim(v, i, axis=0, keepdims=False)
        x_input_next = jnp.mod(x_input + dt * v0, L)
        I0 = R0 * gaussian(pos, x_input_next, sR, L)
        
        # Get ODE derivatives
        dU, dV, fU = _cann_dynamics(U, V, W_ff, W_rec, I0, v0, neural_params, input_params)
        
        # Hebbian plasticity steps
        dW_ff = etaJ * fU * (I0.T - ain * W_ff ** betaJ)
        dW_rec = etaW * fU * (fU.T - arec * W_rec ** betaW)
        
        # Integration
        U_next = U + dt * dU
        V_next = V + dt * dV
        W_ff_next = jnp.maximum(W_ff + dt * dW_ff, 0)
        W_rec_next = jnp.maximum(W_rec + dt * dW_rec, 0).at[diag_idx].set(0) #avoid self-connections
        
        return U_next, V_next, W_ff_next, W_rec_next, x_input_next
    
    Uf, Vf, Wf, Wrecf, xf = lax.fori_loop(T0, T, step, (U0, V0, W_ff0, W_rec0, x0))
    return Uf, Vf, Wf, Wrecf, xf