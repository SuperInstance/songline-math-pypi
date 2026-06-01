"""Navigation algorithms for songline graphs."""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple

from .graph import SonglineGraph, Waypoint


def songline_pathfind(graph: SonglineGraph, start: str, end: str) -> Optional[List[str]]:
    """Find the best path from start to end, preferring high-weight waypoints.

    Uses a modified Dijkstra's algorithm where edge cost is inversely
    proportional to the target waypoint's weight, so high-weight waypoints
    are preferred.

    Args:
        graph: The songline graph to navigate.
        start: ID of the starting waypoint.
        end: ID of the destination waypoint.

    Returns:
        List of waypoint IDs forming the path, or None if no path exists.
    """
    if start not in graph.waypoints or end not in graph.waypoints:
        return None
    if start == end:
        return [start]

    # Cost: 1/weight for traversing to a waypoint
    dist: Dict[str, float] = {start: 0.0}
    prev: Dict[str, Optional[str]] = {start: None}
    pq: List[Tuple[float, str]] = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if u == end:
            break
        if d > dist.get(u, float('inf')):
            continue
        for neighbor_id in graph.neighbors(u):
            wp = graph.get_waypoint(neighbor_id)
            weight = wp.weight if wp else 1.0
            cost = 1.0 / max(weight, 1e-9)
            new_dist = d + cost
            if new_dist < dist.get(neighbor_id, float('inf')):
                dist[neighbor_id] = new_dist
                prev[neighbor_id] = u
                heapq.heappush(pq, (new_dist, neighbor_id))

    if end not in prev:
        return None

    # Reconstruct path
    path: List[str] = []
    node: Optional[str] = end
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path


def dreamtime_navigate(graph: SonglineGraph, start: str, end: str) -> List[List[str]]:
    """Navigate between possibly-disconnected components via BFS through connectivity gaps.

    Finds paths even when start and end are in different components by
    bridging through the closest waypoints across components.

    Args:
        graph: The songline graph.
        start: Starting waypoint ID.
        end: Destination waypoint ID.

    Returns:
        List of path segments. If fully connected, returns [single_path].
        If disconnected, returns multiple segments bridging components.
    """
    if start not in graph.waypoints or end not in graph.waypoints:
        return []

    # Try direct path first
    direct = songline_pathfind(graph, start, end)
    if direct is not None:
        return [direct]

    # Find connected components via BFS
    components = _find_components(graph)

    start_comp = None
    end_comp = None
    for comp in components:
        if start in comp:
            start_comp = comp
        if end in comp:
            end_comp = comp

    if start_comp is None or end_comp is None:
        return []

    if start_comp is end_comp:
        # Same component but no directed path - try undirected
        return [_bfs_undirected(graph, start, end)]

    # Bridge across components: find closest pair between components
    best_bridge = _find_best_bridge(graph, start_comp, end_comp)
    if best_bridge is None:
        # Fall back to concatenating component-local paths
        seg1 = _bfs_undirected(graph, start, list(start_comp)[0]) if start_comp else [start]
        return [seg1]

    bridge_from, bridge_to = best_bridge

    # Path within start component to bridge point
    seg1 = _bfs_undirected(graph, start, bridge_from)
    # Path within end component from bridge point
    seg2 = _bfs_undirected(graph, bridge_to, end)

    if seg1 and seg2:
        return [seg1, seg2]
    return [seg1 or [start], seg2 or [end]]


def _bfs_undirected(graph: SonglineGraph, start: str, end: str) -> List[str]:
    """BFS on the undirected version of the graph."""
    if start == end:
        return [start]
    visited: Set[str] = {start}
    prev: Dict[str, Optional[str]] = {start: None}
    queue: List[str] = [start]

    while queue:
        node = queue.pop(0)
        # Both directions
        for neighbor in graph.neighbors(node) + graph.predecessors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                prev[neighbor] = node
                if neighbor == end:
                    path: List[str] = []
                    cur: Optional[str] = end
                    while cur is not None:
                        path.append(cur)
                        cur = prev.get(cur)
                    path.reverse()
                    return path
                queue.append(neighbor)
    return [start]


def _find_components(graph: SonglineGraph) -> List[Set[str]]:
    """Find connected components (undirected) via BFS."""
    visited: Set[str] = set()
    components: List[Set[str]] = []

    for wid in graph.waypoints:
        if wid not in visited:
            comp: Set[str] = set()
            queue = [wid]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                comp.add(node)
                for n in graph.neighbors(node) + graph.predecessors(node):
                    if n not in visited:
                        queue.append(n)
            components.append(comp)
    return components


def _find_best_bridge(
    graph: SonglineGraph, comp1: Set[str], comp2: Set[str]
) -> Optional[Tuple[str, str]]:
    """Find the closest waypoint pair between two components."""
    import math
    best_dist = float('inf')
    best_pair = None

    for wid1 in comp1:
        wp1 = graph.get_waypoint(wid1)
        if wp1 is None or not wp1.position:
            continue
        for wid2 in comp2:
            wp2 = graph.get_waypoint(wid2)
            if wp2 is None or not wp2.position:
                continue
            try:
                d = wp1.distance_to(wp2)
                if d < best_dist:
                    best_dist = d
                    best_pair = (wid1, wid2)
            except ValueError:
                continue

    return best_pair


def navigability_score(graph: SonglineGraph) -> float:
    """Compute how navigable a graph is (0.0 to 1.0).

    Based on the fraction of ordered waypoint pairs that have a directed path.

    Args:
        graph: The songline graph to score.

    Returns:
        Float between 0.0 (no connectivity) and 1.0 (fully connected).
    """
    n = graph.waypoint_count()
    if n <= 1:
        return 1.0

    waypoint_ids = list(graph.waypoints.keys())
    total_pairs = n * (n - 1)
    reachable = 0

    # BFS from each node to count reachable pairs
    for src in waypoint_ids:
        visited: Set[str] = set()
        queue = [src]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for nb in graph.neighbors(node):
                if nb not in visited:
                    queue.append(nb)
        reachable += len(visited) - 1  # exclude self

    return reachable / total_pairs
