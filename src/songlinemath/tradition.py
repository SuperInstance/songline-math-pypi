"""Evolutionary operations on songlines: mutate, recombine, decay, fitness."""

from __future__ import annotations

import random
from typing import List, Optional

from .graph import SonglineGraph, Verse, Waypoint
from .navigation import navigability_score


def mutate(
    songline: SonglineGraph,
    add_prob: float = 0.3,
    remove_prob: float = 0.1,
    rng: Optional[random.Random] = None,
) -> SonglineGraph:
    """Randomly mutate a songline graph by adding or removing verses.

    With probability add_prob, a random verse is added between existing waypoints.
    With probability remove_prob, a random verse is removed.

    Args:
        songline: The graph to mutate (not modified in place).
        add_prob: Probability of adding a random verse.
        remove_prob: Probability of removing a random verse.
        rng: Optional random number generator for reproducibility.

    Returns:
        A new mutated SonglineGraph.
    """
    r = rng or random.Random()
    result = _copy_graph(songline)
    waypoint_ids = list(result.waypoints.keys())

    # Possibly add a verse
    if r.random() < add_prob and len(waypoint_ids) >= 2:
        src, tgt = r.sample(waypoint_ids, 2)
        result.add_verse(Verse(source=src, target=tgt, traversal_count=1))

    # Possibly remove a verse
    verses = result.verses
    if r.random() < remove_prob and verses:
        to_remove = r.choice(verses)
        result._verses.remove(to_remove)
        result._adjacency[to_remove.source] = [
            v for v in result._adjacency[to_remove.source] if v is not to_remove
        ]
        result._reverse_adj[to_remove.target] = [
            v for v in result._reverse_adj[to_remove.target] if v is not to_remove
        ]

    return result


def recombine(s1: SonglineGraph, s2: SonglineGraph) -> SonglineGraph:
    """Recombine two songline graphs at shared waypoint nodes.

    Takes all waypoints from both graphs and verses: verses from s1
    connecting shared waypoints, plus all verses from s2.

    Args:
        s1: First parent songline.
        s2: Second parent songline.

    Returns:
        A new recombined SonglineGraph.
    """
    shared = set(s1.waypoints.keys()) & set(s2.waypoints.keys())
    child = SonglineGraph()

    # Add all waypoints from both parents
    for wp in list(s1.waypoints.values()) + list(s2.waypoints.values()):
        child.add_waypoint(wp)

    # Add verses from s1 that involve shared waypoints
    for verse in s1.verses:
        if verse.source in shared or verse.target in shared:
            child.add_verse(verse)

    # Add all verses from s2
    for verse in s2.verses:
        child.add_verse(verse)

    return child


def decay(songline: SonglineGraph, time: float) -> SonglineGraph:
    """Apply temporal decay to a songline graph.

    Reduces traversal counts on verses based on time elapsed.
    Verses with traversal_count dropping to 0 are removed.

    Args:
        songline: The graph to decay.
        time: Decay factor (higher = more decay). Uses exponential decay:
              new_count = max(0, round(count * exp(-time))).

    Returns:
        A new decayed SonglineGraph.
    """
    import math
    result = SonglineGraph()
    for wp in songline.waypoints.values():
        result.add_waypoint(wp)

    decay_factor = math.exp(-time)
    for verse in songline.verses:
        new_count = max(0, round(verse.traversal_count * decay_factor))
        if new_count > 0:
            result.add_verse(Verse(source=verse.source, target=verse.target, traversal_count=new_count))

    return result


def fitness(songline: SonglineGraph) -> float:
    """Compute the fitness of a songline graph.

    Fitness is the navigability score, which measures how well-connected
    the graph is.

    Args:
        songline: The graph to evaluate.

    Returns:
        Fitness score between 0.0 and 1.0.
    """
    return navigability_score(songline)


def _copy_graph(graph: SonglineGraph) -> SonglineGraph:
    """Create a deep copy of a SonglineGraph."""
    result = SonglineGraph()
    for wp in graph.waypoints.values():
        result.add_waypoint(wp)
    for verse in graph.verses:
        result.add_verse(verse)
    return result
