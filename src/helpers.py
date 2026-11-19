#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
helpers.py

Data processing, matrix sorting, and noise generation utilities.
"""

import numpy as np
from jax import numpy as jnp
from jax import jit, random

def moving_average(a, n=5):
    """Computes moving average over a 1D array."""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n

def rectify_positions(x, L):
    """Unwraps periodic positions into a continuous trajectory."""
    dx = jnp.diff(x)
    true_dx = dx - L * jnp.round(dx / L)
    return jnp.concatenate([jnp.array([x[0]]), x[0] + jnp.cumsum(true_dx)])

def center_of_mass(x, y, L=1, periodic=True):
    """Calculates the periodic center of mass."""
    Xcm = jnp.sum(jnp.cos(x / L * 2 * jnp.pi) * y) / jnp.sum(y)
    Ycm = jnp.sum(jnp.sin(x / L * 2 * jnp.pi) * y) / jnp.sum(y)
    return jnp.mod(jnp.arctan2(Ycm, Xcm), 2 * jnp.pi) / (2 * jnp.pi) * L

def circ_std(M, L):
    """Calculates the periodic standard deviation along each row of a weight matrix."""
    angles = np.linspace(0, 2 * np.pi, M.shape[1], endpoint=False).reshape(1, -1)
    row_sums = np.sum(M, axis=1, keepdims=True)
    Mnorm = M / np.where(row_sums == 0, 1.0, row_sums)
    C = np.sum(np.cos(angles) * Mnorm, axis=1)
    S = np.sum(np.sin(angles) * Mnorm, axis=1)
    mu = np.mod(np.arctan2(S, C), 2 * np.pi).reshape(-1, 1)
    D = np.abs(mu - angles) / (2 * np.pi) * L
    D = np.minimum(D, L - D)
    return np.sqrt(np.sum(Mnorm * (D ** 2), axis=1))

def circ_mean(M, L):
    angles = jnp.linspace(0,2*jnp.pi,M.shape[1],False).reshape(1,-1)
    Mnorm = M / M.sum(axis=1).reshape(-1,1)
    C = np.sum(np.cos(angles) * Mnorm, axis=1)
    S = np.sum(np.sin(angles) * Mnorm, axis=1)
    return jnp.mod(jnp.arctan2(S, C), 2*jnp.pi) / (2*jnp.pi) * L

@jit
def sort_connectivity_matrix(matrix):
    """Sorts a weight matrix by its maximum values to visualize connectivity."""
    MI = jnp.argmax(matrix, axis=1)
    sorted_index = jnp.argsort(MI)
    return matrix[sorted_index, :], sorted_index

def gaussian_filter_1d(x, sigma, truncate=4.0):
    """1D Gaussian filter using JAX."""
    radius = int(truncate * sigma + 0.5)
    x_idx = jnp.arange(-radius, radius + 1)
    kernel = jnp.exp(-(x_idx ** 2) / (2 * sigma ** 2))
    kernel /= jnp.sum(kernel)
    x_padded = jnp.pad(x, (radius, radius), mode='symmetric')
    return jnp.convolve(x_padded, kernel, mode='valid')

def generate_random_smooth_curve_jax(a, b, s, T, smoothness=5, seed=0):
    """Generates a smooth random curve using JAX PRNG."""
    key = random.PRNGKey(seed)
    noise = random.normal(key, (T,))
    smooth_curve = gaussian_filter_1d(noise, sigma=smoothness)
    norm_curve = (smooth_curve - jnp.min(smooth_curve)) / (jnp.max(smooth_curve) - jnp.min(smooth_curve) + 1e-8)
    scaled_curve = a + (b - a) * norm_curve
    delta = scaled_curve - scaled_curve[0]
    return s + delta