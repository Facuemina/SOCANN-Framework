#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theory.py

Theoretical formulations and analytical solutions for 
Continuous Attractor Neural Networks (CANNs) with Hebbian plasticity.
"""

import numpy as np

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
    return np.sqrt(2) * sigma / tauv * np.sqrt(m * tauv / tau - np.sqrt(m * tauv / tau))

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
    corr_fact = 0.7166 #empirical correction factor
    if v == 0:
        return corr_fact * np.sqrt(2) * sigma_u * np.sqrt(1 - np.sqrt(tau / (m * tauv)))
    else:
        return sigma_u**2 / (tauv * v) * (np.sqrt(1 + 2 * (tauv * v / sigma_u)**2) - 1)