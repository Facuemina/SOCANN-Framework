#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theory.py

Theoretical formulations and analytical solutions for 
Continuous Attractor Neural Networks (CANNs) with Hebbian plasticity.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit


@jit
def gaussian(x, y, std, L):
    """Periodic Gaussian function."""
    dx = jnp.minimum(jnp.abs(x - y), L - jnp.abs(x - y))
    norm = 1 / jnp.sqrt(2 * jnp.pi * std**2)
    return jnp.exp(-0.5 * (dx / std) ** 2) * norm

corr_fact = 0.7166 #empirical correction factor
def y(m, v, G):
    """
    Normalized distance between equilibrium U and V bumps.
    """
    denominator = 2 * G * v
    numerator = -(m + 1) + G * v**2
    sqrt_term = np.sqrt(1 + 4 * v**2 * G * (G + 1) / (numerator**2))
    return (numerator / denominator) * (1 - sqrt_term)

def gamma(m, v, G):
    """
    Theoretical asymmetry coefficient.
    """
    y_val = y(m, v, G)
    denominator = (1 + G * v * y_val)
    return y_val * m / denominator - v

def theoretical_speed(m, tau, tauv, sigma):
    """
    Theoretical speed in recurrent CANN.
    """
    return corr_fact * 2 * sigma / tauv * np.sqrt(m * tauv / tau - np.sqrt(m * tauv / tau))

def learned_weight_std(beta, sigma):
    """
    Theoretical learned standard deviation in weights.
    """
    return np.sqrt(3 * beta / (2 - beta)) * sigma

def learned_U_std(beta, sigma):
    """
    Theoretical learned standard deviation in U variable.
    """
    return np.sqrt((2 * beta + 2) / (2 - beta)) * sigma

def theoretical_distance(m, tau, tauv, sigma_u, v=0):
    """
    Theoretical lag between the adaptation bump and the neural bump.
    """
    if v == 0:
        return corr_fact * np.sqrt(2) * sigma_u * np.sqrt(1 - np.sqrt(tau / (m * tauv)))
    else:
        return sigma_u**2 / (tauv * v) * (np.sqrt(1 + 2 * (tauv * v / sigma_u)**2) - 1)
    
def learned_weight_amplitude(beta, sigma, R0=30.0, alpha=1.0):
    """
    Theoretical learned weight amplitude A_J (or J_0).
    """
    std_w = learned_weight_std(beta, sigma)
    return (R0 / alpha / np.sqrt((2 * np.pi * std_w**2) ** (1 - beta) / beta)) ** (1 / beta)

def estimate_gamma_periodic(y, mu, std, L):
    """
    Project empirical bump against theoretical modes 0 and 1
    """
    x_ = np.linspace(0,L,len(y),False)
    dx = x_[1] - x_[0]
    z_ = (x_ - mu + L/2) % L - L/2
    def CANN_eigen_func0(z, std, L):
        norm = 1
        return norm * gaussian(z, 0, std, L)
    def CANN_eigen_func1(z, std, L):
        norm = 1
        return norm * gaussian(z, 0, std, L) * 2 * z / std
    
    a0 = np.sum(CANN_eigen_func0(z_, std, L) * y) * dx
    a1 = np.sum(CANN_eigen_func1(z_, std, L) * y) * dx
    
    return a0, a1