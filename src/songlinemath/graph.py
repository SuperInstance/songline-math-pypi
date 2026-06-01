"""Core graph data structures: Waypoint, Verse, and SonglineGraph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import math


@dataclass(frozen=True)
class Waypoint:
    """A node in the knowledge graph with a position and weight.

    Attributes:
        id: Unique identifier for this waypoint.
        position: N-dimensional coordinates (for topological analysis).
        weight: Importance/centrality weight (higher = more important).
    """
    id: str
    position: Tuple[float, ...] = ()
    weight: float = 1.0

    def distance_to(self, other: Waypoint) -> float:
        """Euclidean distance to another waypoint."""
        if len(self.position) != len(other.position):
            raise ValueError("Waypoints must have same dimensionality")
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, other.position)))


@dataclass(frozen=True)
class Verse:
    """A directed edge (songline) between two waypoints.

    Attributes:
        source: ID of the origin waypoint.
        target: ID of the destination waypoint.
        traversal_count: How many times this path has been traversed.
    """
    source: str
    target: str
    traversal_count: int = 1


class SonglineGraph:
    """A navigable knowledge graph inspired by Australian Aboriginal songlines.

    Waypoints are knowledge nodes; verses are directed paths between them.
    Supports subgraph extraction, reversal, and concatenation.
    """

    def __init__(self) -> None:
        self._waypoints: Dict[str, Waypoint] = {}
        self._verses: List[Verse] = []
        self._adjacency: Dict[str, List[Verse]] = {}  # source_id -> outgoing verses
        self._reverse_adj: Dict[str, List[Verse]] = {}  # target_id -> incoming verses

    @property
    def waypoints(self) -> Dict[str, Waypoint]:
        """Return a copy of the waypoints dict."""
        return dict(self._waypoints)

    @property
    def verses(self) -> List[Verse]:
        """Return a copy of the verses list."""
        return list(self._verses)

    def add_waypoint(self, waypoint: Waypoint) -> None:
        """Add or replace a waypoint in the graph."""
        self._waypoints[waypoint.id] = waypoint
        if waypoint.id not in self._adjacency:
            self._adjacency[waypoint.id] = []
        if waypoint.id not in self._reverse_adj:
            self._reverse_adj[waypoint.id] = []

    def add_verse(self, verse: Verse) -> None:
        """Add a directed verse (edge) to the graph.

        Both source and target waypoints must exist.
        """
        if verse.source not in self._waypoints:
            raise ValueError(f"Source waypoint '{verse.source}' not in graph")
        if verse.target not in self._waypoints:
            raise ValueError(f"Target waypoint '{verse.target}' not in graph")
        self._verses.append(verse)
        self._adjacency[verse.source].append(verse)
        self._reverse_adj[verse.target].append(verse)

    def get_waypoint(self, waypoint_id: str) -> Optional[Waypoint]:
        """Get a waypoint by ID, or None if not found."""
        return self._waypoints.get(waypoint_id)

    def neighbors(self, waypoint_id: str) -> List[str]:
        """Return IDs of direct successors of a waypoint."""
        return [v.target for v in self._adjacency.get(waypoint_id, [])]

    def predecessors(self, waypoint_id: str) -> List[str]:
        """Return IDs of direct predecessors of a waypoint."""
        return [v.source for v in self._reverse_adj.get(waypoint_id, [])]

    def waypoint_count(self) -> int:
        """Number of waypoints in the graph."""
        return len(self._waypoints)

    def verse_count(self) -> int:
        """Number of verses (edges) in the graph."""
        return len(self._verses)

    def extract_subgraph(self, waypoint_ids: Set[str]) -> SonglineGraph:
        """Extract a subgraph containing only the specified waypoints and their internal verses."""
        sub = SonglineGraph()
        for wid in waypoint_ids:
            if wid in self._waypoints:
                sub.add_waypoint(self._waypoints[wid])
        for verse in self._verses:
            if verse.source in waypoint_ids and verse.target in waypoint_ids:
                sub.add_verse(verse)
        return sub

    def reverse(self) -> SonglineGraph:
        """Return a new graph with all verses reversed (source <-> target)."""
        rev = SonglineGraph()
        for wp in self._waypoints.values():
            rev.add_waypoint(wp)
        for verse in self._verses:
            rev.add_verse(Verse(source=verse.target, target=verse.source, traversal_count=verse.traversal_count))
        return rev

    def concatenate(self, other: SonglineGraph, bridge: Optional[Verse] = None) -> SonglineGraph:
        """Concatenate another graph onto this one, optionally connecting with a bridge verse.

        Args:
            other: The graph to append.
            bridge: Optional verse connecting a waypoint in self to one in other.

        Returns:
            A new combined graph.
        """
        combined = SonglineGraph()
        for wp in self._waypoints.values():
            combined.add_waypoint(wp)
        for wp in other._waypoints.values():
            combined.add_waypoint(wp)
        for verse in self._verses:
            combined.add_verse(verse)
        for verse in other._verses:
            combined.add_verse(verse)
        if bridge is not None:
            combined.add_verse(bridge)
        return combined

    def __repr__(self) -> str:
        return f"SonglineGraph(waypoints={self.waypoint_count()}, verses={self.verse_count()})"
