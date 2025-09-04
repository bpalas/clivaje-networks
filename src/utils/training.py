from typing import Tuple

import numpy as np
import torch


def sample_non_edges_hard(n: int, existing_edges: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample undirected non-edges (i<j) avoiding 1-hop neighbors from the observed graph.
    existing_edges: [m,2] undirected edges (i<j)
    Returns [k,2] with k<=num_samples
    """
    if num_samples <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    exist_set = set(map(tuple, existing_edges.tolist()))
    # Build neighbor sets to avoid 1-hop
    nbr = [set() for _ in range(n)]
    for a, b in exist_set:
        nbr[a].add(b); nbr[b].add(a)
    samples = []
    trials = 0
    limit = max(20000, num_samples * 20)
    while len(samples) < num_samples and trials < limit:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            trials += 1; continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in exist_set:  # already an edge
            trials += 1; continue
        if (b in nbr[a]) or (a in nbr[b]):  # 1-hop neighbor check (redundant due to exist_set but keeps intent)
            trials += 1; continue
        exist_set.add((a, b))
        samples.append([a, b])
        trials += 1
    return np.array(samples, dtype=np.int64) if samples else np.zeros((0, 2), dtype=np.int64)


def dropedge_coo(A: torch.Tensor, drop_rate: float) -> torch.Tensor:
    """
    Randomly drop a fraction of edges from sparse COO adjacency (independently per nonzero).
    Returns a new sparse tensor. Values are preserved for kept edges.
    """
    if drop_rate <= 0.0:
        return A
    A = A.coalesce()
    idx = A.indices()
    val = A.values()
    m = val.numel()
    keep = torch.rand(m, device=val.device) > drop_rate
    if keep.sum() == m:
        return A
    if keep.sum() == 0:
        # return empty with same shape
        return torch.sparse_coo_tensor(idx[:, :0], val[:0], A.size(), device=val.device, dtype=val.dtype)
    return torch.sparse_coo_tensor(idx[:, keep], val[keep], A.size(), device=val.device, dtype=val.dtype).coalesce()

