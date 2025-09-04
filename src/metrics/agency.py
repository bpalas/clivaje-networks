from typing import Dict, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix


def compute_pi_opp(q: np.ndarray, edges_ij: np.ndarray) -> np.ndarray:
    """
    Smooth opposition mask per edge: pi_opp_ij = (q_i1*q_j2 + q_i2*q_j1) * (1-q_i0)*(1-q_j0)
    q: [N,3], edges_ij: [m,2]
    Returns: [m] array in [0,1].
    """
    q1 = q[:, 0]
    q2 = q[:, 1]
    q0 = q[:, 2]
    i = edges_ij[:, 0]
    j = edges_ij[:, 1]
    base = (q1[i] * q2[j] + q2[i] * q1[j])
    mask = (1.0 - q0[i]) * (1.0 - q0[j])
    return base * mask


def power_iteration_ceig(A_abs_sub: csr_matrix, iters: int = 30, eps: float = 1e-9) -> np.ndarray:
    """
    Power iteration on unsigned adjacency A_abs_sub to estimate principal eigenvector (centrality).
    Returns normalized vector (L2 norm = 1). If graph is empty, returns zeros.
    """
    n = A_abs_sub.shape[0]
    if n == 0 or A_abs_sub.nnz == 0:
        return np.zeros(n, dtype=float)
    x = np.ones(n, dtype=float) / np.sqrt(max(1, n))
    for _ in range(max(1, iters)):
        x_new = A_abs_sub @ x
        norm = np.linalg.norm(x_new) + eps
        x = x_new / norm
    return x


def compute_indf_soft(q: np.ndarray, A_pos: csr_matrix, deg: np.ndarray) -> np.ndarray:
    """
    IndF_soft(i) = [sum_{j in E+} pi_opp_ij / deg(i)] * log(1 + sum_{j in E+} pi_opp_ij)
    0 if deg(i)=0.
    """
    A = A_pos.tocoo(copy=True)
    i = A.row
    j = A.col
    pi = compute_pi_opp(q, np.stack([i, j], axis=1))
    # accumulate per i over positive edges
    sum_i = np.zeros(q.shape[0], dtype=float)
    np.add.at(sum_i, i, pi)
    denom = np.maximum(deg.astype(float), 1.0)
    frac = sum_i / denom
    return np.where(deg > 0, frac * np.log1p(sum_i), 0.0)


def compute_h_ext(q: np.ndarray, E: np.ndarray, W: np.ndarray, A_s: csr_matrix, eps: float = 1e-12) -> np.ndarray:
    """
    H_ext(i) from binary entropy of p_i^+ = sum_j pi_opp_ij * sigma(E_i^T W E_j) / sum_j pi_opp_ij
    """
    n = q.shape[0]
    A = A_s.tocoo(copy=True)
    mask = A.row < A.col
    rows = A.row[mask]; cols = A.col[mask]
    edges = np.stack([rows, cols], axis=1)
    pi = compute_pi_opp(q, edges)
    # scores per undirected edge
    Ei = E[edges[:, 0]]
    Ej = E[edges[:, 1]]
    logits = np.einsum('nd,dk,mk->n', Ei, W, Ej)
    probs = 1.0 / (1.0 + np.exp(-logits))
    # accumulate per endpoint with pi weights
    num = np.zeros(n, dtype=float)
    den = np.zeros(n, dtype=float)
    # add both directions consistently
    np.add.at(num, edges[:, 0], pi * probs)
    np.add.at(den, edges[:, 0], pi)
    np.add.at(num, edges[:, 1], pi * probs)
    np.add.at(den, edges[:, 1], pi)
    p_plus = num / (den + eps)
    p_plus = np.clip(p_plus, eps, 1 - eps)
    return -(p_plus * np.log(p_plus) + (1 - p_plus) * np.log(1 - p_plus))


def compute_sh_star(
    q: np.ndarray,
    E: np.ndarray,
    W: np.ndarray,
    A_s: csr_matrix,
    lambda_H: float = 0.5,
    lambda_C: float = 0.2,
) -> Dict[str, np.ndarray]:
    n = q.shape[0]
    # unsigned adjacency and positive-only for IndF
    A_abs = A_s.copy(); A_abs.data = np.abs(A_abs.data)
    A_pos = A_s.copy(); A_pos.data = (A_pos.data > 0).astype(float)
    deg = np.array(A_abs.getnnz(axis=1)).astype(float)

    indf = compute_indf_soft(q, A_pos, deg)
    h_ext = compute_h_ext(q, E, W, A_s)

    # nucleus mask (non-neutral)
    y = np.argmax(q, axis=1)
    nucleus = (y != 2)
    ceig = np.zeros(n, dtype=float)
    if np.any(nucleus):
        idx = np.where(nucleus)[0]
        A_sub = A_abs[idx[:, None], idx]
        x = power_iteration_ceig(A_sub, iters=30)
        ceig[idx] = np.abs(x)

    phi = (1.0 - q[:, 2]) * np.minimum(q[:, 0], q[:, 1])
    sh = phi * indf + float(lambda_H) * h_ext + float(lambda_C) * ceig
    support_mask = (deg >= 1) & (np.isfinite(indf))

    return {
        'phi': phi,
        'indf_soft': indf,
        'h_ext': h_ext,
        'ceig': ceig,
        'sh_star': sh,
        'degree': deg,
        'support_mask': support_mask,
    }

