"""Persistent homology: Vietoris-Rips complexes, Betti numbers, barcodes, cycles."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .graph import Waypoint


def _euclidean(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def vietoris_rips(
    points: List[Waypoint], epsilon: float, max_dim: int = 2
) -> List[Tuple[FrozenSet[str], float]]:
    """Build a Vietoris-Rips simplicial complex from points.

    A simplex is included if all pairwise distances are <= epsilon.

    Args:
        points: List of Waypoints with positions.
        epsilon: Distance threshold for including simplices.
        max_dim: Maximum simplex dimension (0=vertices, 1=edges, 2=triangles, etc.).

    Returns:
        List of (frozenset of waypoint IDs, filtration_value) tuples.
        Vertices have filtration 0.0; higher simplices have max pairwise distance.
    """
    if not points:
        return []

    n = len(points)
    # Precompute pairwise distances
    dist: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        if not points[i].position:
            continue
        for j in range(i + 1, n):
            if not points[j].position:
                continue
            d = _euclidean(points[i].position, points[j].position)
            dist[(i, j)] = d

    simplices: List[Tuple[FrozenSet[str], float]] = []

    # 0-simplices (vertices)
    for p in points:
        simplices.append((frozenset([p.id]), 0.0))

    if max_dim < 1:
        return simplices

    # Build edges and higher simplices
    point_ids = [p.id for p in points]
    indices = list(range(n))

    for dim in range(1, max_dim + 1):
        for combo in combinations(indices, dim + 1):
            # Check all pairwise distances
            max_dist = 0.0
            valid = True
            has_positions = all(points[i].position for i in combo)
            if not has_positions:
                continue
            for a, b in combinations(combo, 2):
                key = (min(a, b), max(a, b))
                if key not in dist:
                    valid = False
                    break
                d = dist[key]
                if d > epsilon:
                    valid = False
                    break
                max_dist = max(max_dist, d)
            if valid:
                simplex_ids = frozenset(points[i].id for i in combo)
                simplices.append((simplex_ids, max_dist))

    return simplices


def betti_numbers(
    complex_data: List[Tuple[FrozenSet[str], float]],
) -> Tuple[int, int]:
    """Compute Betti numbers (b0, b1) of a simplicial complex.

    b0 = number of connected components.
    b1 = number of independent 1-cycles (loops).

    Args:
        complex_data: Output of vietoris_rips — list of (simplex, filtration_value).

    Returns:
        Tuple of (b0, b1).
    """
    if not complex_data:
        return (0, 0)

    vertices: Set[str] = set()
    edges: List[Tuple[str, str]] = []
    triangles: Set[FrozenSet[str]] = set()

    for simplex, _ in complex_data:
        if len(simplex) == 1:
            vertices.update(simplex)
        elif len(simplex) == 2:
            edge = tuple(sorted(simplex))
            edges.append(edge)
        elif len(simplex) == 3:
            triangles.add(simplex)

    # b0: connected components via union-find
    parent: Dict[str, str] = {v: v for v in vertices}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)

    components = len(set(find(v) for v in vertices))
    b0 = max(components, 0)

    # b1: cycles = edges - vertices + components - (2-simplices contribution)
    # Using Euler: b1 = edges - vertices + components - (each triangle reduces a cycle)
    # More precisely: rank(H1) = rank(Z1) - rank(B1)
    # Z1 = cycle space dimension = edges - vertices + components
    # B1 = boundary space dimension from triangles
    # Each triangle can kill one independent cycle

    cycle_space = len(edges) - len(vertices) + b0

    # Count how many triangles are independent (not sharing all edges with previous triangles)
    # Simple approach: each triangle reduces b1 by at most 1
    triangle_contribution = 0
    used_edge_sets: List[Set[Tuple[str, str]]] = []

    for tri in triangles:
        tri_edges = set()
        tri_list = sorted(tri)
        for a, b in combinations(tri_list, 2):
            tri_edges.add((a, b))
        # Check linear independence (simplified: check if edges not all already used)
        is_new = False
        for e in tri_edges:
            already = any(e in used for used in used_edge_sets)
            if not already:
                is_new = True
                break
        if is_new and triangle_contribution < cycle_space:
            triangle_contribution += 1
            used_edge_sets.append(tri_edges)

    b1 = max(cycle_space - triangle_contribution, 0)

    return (b0, b1)


def persistence_barcodes(
    points: List[Waypoint], max_epsilon: float, steps: int = 50
) -> List[Tuple[int, float, float]]:
    """Compute persistence barcodes over a range of epsilon values.

    Args:
        points: List of Waypoints with positions.
        max_epsilon: Maximum epsilon to sweep.
        steps: Number of epsilon steps.

    Returns:
        List of (dimension, birth, death) tuples.
        death=inf means the feature persists to max_epsilon.
    """
    if not points or len(points) < 2:
        return []

    barcodes: List[Tuple[int, float, float]] = []

    # Compute all pairwise distances
    distances: List[float] = []
    pair_dist: Dict[Tuple[int, int], float] = {}
    for i in range(len(points)):
        if not points[i].position:
            continue
        for j in range(i + 1, len(points)):
            if not points[j].position:
                continue
            d = _euclidean(points[i].position, points[j].position)
            pair_dist[(i, j)] = d
            distances.append(d)

    if not distances:
        return []

    epsilons = [i * max_epsilon / steps for i in range(steps + 1)]
    prev_b0 = 0
    prev_b1 = 0

    # Track component births
    component_births: Dict[FrozenSet[str], float] = {}

    for eps in epsilons:
        cx = vietoris_rips(points, eps)
        b0, b1_val = betti_numbers(cx)

        # b0 features: components die when they merge
        # We track which components exist
        current_vertices: Set[str] = set()
        current_edges: List[Tuple[str, str]] = []
        for simplex, _ in cx:
            if len(simplex) == 1:
                current_vertices.update(simplex)
            elif len(simplex) == 2:
                current_edges.append(tuple(sorted(simplex)))

        # Union-find for current components
        par: Dict[str, str] = {v: v for v in current_vertices}

        def find2(x: str) -> str:
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        def union2(a: str, b: str) -> None:
            ra, rb = find2(a), find2(b)
            if ra != rb:
                par[ra] = rb

        for a, b in current_edges:
            union2(a, b)

        # b0: component appears at birth=0, dies when merged
        if b0 > prev_b0:
            # New components appeared (shouldn't usually decrease)
            pass
        if b0 < prev_b0:
            # Components merged - record deaths
            pass

        # b1: new loops born
        if b1_val > prev_b1:
            for _ in range(b1_val - prev_b1):
                barcodes.append((1, eps, float('inf')))
        if b1_val < prev_b1:
            # Kill some barcodes
            alive = [b for b in barcodes if b[0] == 1 and b[2] == float('inf')]
            kill_count = prev_b1 - b1_val
            for k in range(min(kill_count, len(alive))):
                idx = barcodes.index(alive[-(k + 1)])
                barcodes[idx] = (1, alive[-(k + 1)][1], eps)

        prev_b0 = b0
        prev_b1 = b1_val

    # b0 barcodes: all vertices are born at 0, components die as they merge
    # Simple version: one barcode per initial vertex, death = first connection
    singletons = len([p for p in points if p.position])
    if singletons > 0:
        # All born at 0.0, n-1 die as components merge to 1
        alive_b0 = singletons
        for eps in epsilons:
            if alive_b0 <= 1:
                break
            cx = vietoris_rips(points, eps)
            b0_now, _ = betti_numbers(cx)
            while b0_now < alive_b0 and alive_b0 > 1:
                barcodes.append((0, 0.0, eps))
                alive_b0 -= 1
        # One survives
        barcodes.append((0, 0.0, float('inf')))

    # Sort by dimension, then birth
    barcodes.sort(key=lambda b: (b[0], b[1]))
    return barcodes


def persistent_cycles(
    complex_data: List[Tuple[FrozenSet[str], float]],
) -> List[List[str]]:
    """Find representative 1-cycles in the simplicial complex.

    Uses a simple spanning tree approach: edges not in a spanning forest
    each create an independent cycle.

    Args:
        complex_data: Output of vietoris_rips.

    Returns:
        List of cycles, each a list of waypoint IDs forming a loop.
    """
    edges: List[Tuple[str, str]] = []
    vertices: Set[str] = set()
    edge_set: Set[Tuple[str, str]] = set()

    for simplex, _ in complex_data:
        if len(simplex) == 1:
            vertices.update(simplex)
        elif len(simplex) == 2:
            edge = tuple(sorted(simplex))
            edges.append(edge)
            edge_set.add(edge)

    if not edges:
        return []

    # Build spanning tree via BFS
    parent: Dict[str, str] = {v: v for v in vertices}
    tree_edges: Set[Tuple[str, str]] = set()
    visited: Set[str] = set()

    for start in vertices:
        if start in visited:
            continue
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            for a, b in edges:
                neighbor = None
                if a == node and b not in visited:
                    neighbor = b
                elif b == node and a not in visited:
                    neighbor = a
                if neighbor is not None:
                    visited.add(neighbor)
                    tree_edges.add((a, b))
                    queue.append(neighbor)

    # Non-tree edges create cycles
    cycles: List[List[str]] = []
    non_tree = [e for e in edges if e not in tree_edges]

    # Build adjacency for tree path finding
    adj: Dict[str, List[str]] = {v: [] for v in vertices}
    for a, b in tree_edges:
        adj[a].append(b)
        adj[b].append(a)

    for a, b in non_tree:
        # Find path from a to b in tree
        path = _bfs_path(adj, a, b)
        if path:
            path.append(a)  # close the cycle
            cycles.append(path)

    return cycles


def _bfs_path(adj: Dict[str, List[str]], start: str, end: str) -> Optional[List[str]]:
    """BFS shortest path in adjacency dict."""
    if start == end:
        return [start]
    visited: Set[str] = {start}
    prev: Dict[str, Optional[str]] = {start: None}
    queue: List[str] = [start]
    while queue:
        node = queue.pop(0)
        for nb in adj.get(node, []):
            if nb not in visited:
                visited.add(nb)
                prev[nb] = node
                if nb == end:
                    path: List[str] = []
                    cur: Optional[str] = end
                    while cur is not None:
                        path.append(cur)
                        cur = prev.get(cur)
                    path.reverse()
                    return path
                queue.append(nb)
    return None
