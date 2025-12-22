# src/graph/graph_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

try:
    from scipy.spatial import Delaunay  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


@dataclass
class GraphStats:
    n_nodes: int
    n_edges: int
    avg_degree: float
    density: float
    avg_edge_len: float
    std_edge_len: float
    n_components: int
    giant_component_ratio: float


def _knn_edges(points: np.ndarray, k: int = 3) -> List[Tuple[int, int]]:
    """Fallback if scipy isn't available: connect each node to k nearest neighbors."""
    n = len(points)
    if n < 2:
        return []
    edges = set()
    for i in range(n):
        d = np.linalg.norm(points - points[i], axis=1)
        nn = np.argsort(d)[1 : min(k + 1, n)]
        for j in nn:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            edges.add((a, b))
    return sorted(list(edges))


def delaunay_edges(points: np.ndarray) -> List[Tuple[int, int]]:
    """Delaunay triangulation edges (best for "bubble neighbor graph")."""
    if len(points) < 3 or not _HAS_SCIPY:
        return _knn_edges(points, k=3)

    tri = Delaunay(points)
    edges = set()
    # Each simplex is a triangle; add its 3 edges
    for simplex in tri.simplices:
        a, b, c = simplex
        for u, v in [(a, b), (b, c), (a, c)]:
            u, v = int(u), int(v)
            if u == v:
                continue
            x, y = (u, v) if u < v else (v, u)
            edges.add((x, y))
    return sorted(list(edges))


def connected_components(n_nodes: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """Simple BFS components (no networkx dependency)."""
    adj = [[] for _ in range(n_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    seen = [False] * n_nodes
    comps = []
    for i in range(n_nodes):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = [i]
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
                    comp.append(nxt)
        comps.append(comp)
    return comps


def compute_graph_stats(points: np.ndarray, edges: List[Tuple[int, int]]) -> GraphStats:
    n = int(len(points))
    m = int(len(edges))
    if n <= 1:
        return GraphStats(
            n_nodes=n,
            n_edges=m,
            avg_degree=0.0,
            density=0.0,
            avg_edge_len=0.0,
            std_edge_len=0.0,
            n_components=n,
            giant_component_ratio=1.0 if n == 1 else 0.0,
        )

    deg = np.zeros(n, dtype=float)
    lens = []
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
        lens.append(float(np.linalg.norm(points[u] - points[v])))

    avg_degree = float(deg.mean()) if n > 0 else 0.0
    density = float((2.0 * m) / (n * (n - 1))) if n > 1 else 0.0

    if lens:
        avg_edge_len = float(np.mean(lens))
        std_edge_len = float(np.std(lens))
    else:
        avg_edge_len = 0.0
        std_edge_len = 0.0

    comps = connected_components(n, edges)
    comps_sizes = sorted([len(c) for c in comps], reverse=True)
    n_components = len(comps_sizes)
    giant_ratio = float(comps_sizes[0] / n) if comps_sizes else 0.0

    return GraphStats(
        n_nodes=n,
        n_edges=m,
        avg_degree=avg_degree,
        density=density,
        avg_edge_len=avg_edge_len,
        std_edge_len=std_edge_len,
        n_components=n_components,
        giant_component_ratio=giant_ratio,
    )

