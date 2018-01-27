import numpy as np
import math as m


def chernoff(it, N, delta, sqrt_C, log_C, range=1.):
    ci = range * np.sqrt(sqrt_C * m.log(log_C * (it + 1) / delta) / np.maximum(1, N))
    return ci


def bernstein(it, P, N, delta, log_C, alpha_1, alpha_2, scale_f=4.):
    Z = scale_f * m.log(log_C * m.log(it + 2) / delta)
    n = np.maximum(1, N)
    Va = np.sqrt(2 * P * (1 - P) * Z / n[:, :, np.newaxis])
    Vb = Z * 7 / (3 * n)
    return alpha_1 * Va + alpha_2 * Vb[:, :, np.newaxis]


def chernoff2(sqrt_value, log_value):
    ci = np.sqrt(sqrt_value * m.log(log_value))
    return ci


def bernstein2(scale_a, log_scale_a, scale_b, log_scale_b, alpha_1, alpha_2):
    A = scale_a * m.log(log_scale_a)
    B = scale_b * m.log(log_scale_b)
    return alpha_1 * np.sqrt(A) + alpha_2 * B
