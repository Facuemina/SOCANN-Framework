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

def rectify_positions(x, T, reverse=True):
    """Rectifies positions to remove periodic boundary jumps for smooth plotting."""
    if type(x) == list:
        x2 = np.zeros(len(x))
        x2[:] = x[::-1]
    else:
        m = np.argmax(x.shape)
        x2 = np.zeros(x.shape[m])
        if m == 0 and len(x.shape) > 1:
            x2[:] = x[:,0][::-1] if reverse else x[:,0]
        elif len(x.shape) == 1:
            x2[:] = x[::-1] if reverse else x
        else:
            x2[:] = x[0][::-1] if reverse else x[0]
    
    x_rectified = np.zeros_like(x2)
    x_rectified[0] = x2[0]
    
    for i in range(1, len(x2)):
        D = x2[i] - x2[i-1]
        if np.abs(D) > T/2:
            x_rectified[i] = -np.sign(D)*T + D + x_rectified[i-1]
        else:
            x_rectified[i] = D + x_rectified[i-1]
    
    return x_rectified[::-1] if reverse else x_rectified

def center_of_mass(x, y, L=1, periodic=True):
    """Calculates the periodic center of mass."""
    Xcm = jnp.sum(jnp.cos(x / L * 2 * jnp.pi) * y) / jnp.sum(y)
    Ycm = jnp.sum(jnp.sin(x / L * 2 * jnp.pi) * y) / jnp.sum(y)
    return jnp.mod(jnp.arctan2(Ycm, Xcm), 2 * jnp.pi) / (2 * jnp.pi) * L

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