"""Community detection: convergence hubs, clustering, and modularity."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .graph import SonglineGraph


def find_convergence_hubs(graph: SonglineGraph, min_degree: int = 2) -> List[Tuple[str, int]]:
    """Find waypoints that act as convergence hubs (high in-degree + out-degree).

    Args:
        graph: The songline graph.
        min_degree: Minimum total degree (in + out) to qualify as a hub.

    Returns:
        List of (waypoint_id, total_degree) sorted by degree descending.
    """
    hubs: List[Tuple[str, int]] = []
    for wid in graph.waypoints:
        in_deg = len(graph.predecessors(wid))
        out_deg = len(graph.neighbors(wid))
        total = in_deg + out_deg
        if total >= min_degree:
            hubs.append((wid, total))
    hubs.sort(key=lambda x: x[1], reverse=True)
    return hubs


def cluster_by_shared_songlines(
    graph: SonglineGraph, threshold: float = 0.3
) -> List[Set[str]]:
    """Cluster waypoints by shared songline (edge) similarity.

    Two waypoints are similar if they share a common neighbor via verses.
    Clusters are formed by grouping waypoints whose Jaccard similarity of
    neighbor sets meets the threshold.

    Args:
        graph: The songline graph.
        threshold: Minimum Jaccard similarity to merge clusters (0.0 to 1.0).

    Returns:
        List of clusters, each a set of waypoint IDs.
    """
    waypoint_ids = list(graph.waypoints.keys())
    if not waypoint_ids:
        return []

    # Build undirected neighbor sets
    neighbor_sets: Dict[str, Set[str]] = {}
    for wid in waypoint_ids:
        neighbor_sets[wid] = set(graph.neighbors(wid)) | set(graph.predecessors(wid))
        neighbor_sets[wid].discard(wid)

    # Union-Find clustering
    parent: Dict[str, str] = {wid: wid for wid in waypoint_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, w1 in enumerate(waypoint_ids):
        for w2 in waypoint_ids[i + 1:]:
            s1, s2 = neighbor_sets[w1], neighbor_sets[w2]
            union_size = len(s1 | s2)
            if union_size == 0:
                continue
            jaccard = len(s1 & s2) / union_size
            if jaccard >= threshold:
                union(w1, w2)

    # Collect clusters
    clusters_dict: Dict[str, Set[str]] = defaultdict(set)
    for wid in waypoint_ids:
        clusters_dict[find(wid)].add(wid)

    return list(clusters_dict.values())


def modularity(graph: SonglineGraph, clusters: List[Set[str]]) -> float:
    """Compute the modularity of a given clustering.

    Modularity Q = (1/2m) * sum_ij [ A_ij - k_i*k_j/(2m) ] * delta(c_i, c_j)

    where m is the number of edges, k_i is degree, and delta is 1 if same cluster.

    Args:
        graph: The songline graph.
        clusters: List of clusters (sets of waypoint IDs).

    Returns:
        Modularity score (typically between -0.5 and 1.0).
    """
    m = graph.verse_count()
    if m == 0:
        return 0.0

    # Build cluster assignment
    cluster_of: Dict[str, int] = {}
    for idx, cluster in enumerate(clusters):
        for wid in cluster:
            cluster_of[wid] = idx

    # Compute undirected degrees (treat directed edges as undirected)
    degree: Dict[str, int] = {}
    for wid in graph.waypoints:
        degree[wid] = len(graph.neighbors(wid)) + len(graph.predecessors(wid))

    two_m = 2.0 * m

    # Standard modularity: Q = sum_c [ L_c/m - (d_c / 2m)^2 ]
    # where L_c = edges within cluster c, d_c = sum of degrees in cluster c
    q = 0.0
    for cluster in clusters:
        # Count edges with both endpoints in this cluster
        l_c = 0
        for verse in graph.verses:
            if verse.source in cluster and verse.target in cluster:
                l_c += 1
        d_c = sum(degree.get(wid, 0) for wid in cluster)
        q += l_c / m - (d_c / two_m) ** 2

    return q
