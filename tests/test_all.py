"""Comprehensive test suite for songline-math."""

import math
import random
import pytest

from songlinemath import (
    Waypoint, Verse, SonglineGraph,
    songline_pathfind, dreamtime_navigate, navigability_score,
    find_convergence_hubs, cluster_by_shared_songlines, modularity,
    vietoris_rips, betti_numbers, persistence_barcodes, persistent_cycles,
    mutate, recombine, decay, fitness,
)


# ─── Helpers ──────────────────────────────────────────────────────────

def _make_triangle_graph():
    """Build a simple triangular graph: A -> B -> C -> A."""
    g = SonglineGraph()
    g.add_waypoint(Waypoint("A", (0.0, 0.0), 2.0))
    g.add_waypoint(Waypoint("B", (1.0, 0.0), 3.0))
    g.add_waypoint(Waypoint("C", (0.5, 1.0), 1.0))
    g.add_verse(Verse("A", "B", 5))
    g.add_verse(Verse("B", "C", 3))
    g.add_verse(Verse("C", "A", 2))
    return g


def _make_linear_graph():
    """Build a linear graph: A -> B -> C -> D."""
    g = SonglineGraph()
    for i, letter in enumerate("ABCD"):
        g.add_waypoint(Waypoint(letter, (float(i), 0.0), float(i + 1)))
    for i in range(3):
        g.add_verse(Verse("ABCD"[i], "ABCD"[i + 1], 1))
    return g


def _make_disconnected_graph():
    """Build a graph with two disconnected components."""
    g = SonglineGraph()
    g.add_waypoint(Waypoint("A", (0.0, 0.0)))
    g.add_waypoint(Waypoint("B", (1.0, 0.0)))
    g.add_waypoint(Waypoint("C", (10.0, 0.0)))
    g.add_waypoint(Waypoint("D", (11.0, 0.0)))
    g.add_verse(Verse("A", "B", 1))
    g.add_verse(Verse("C", "D", 1))
    return g


# ─── Waypoint Tests ───────────────────────────────────────────────────

class TestWaypoint:
    def test_creation(self):
        wp = Waypoint("x", (1.0, 2.0, 3.0), 5.0)
        assert wp.id == "x"
        assert wp.position == (1.0, 2.0, 3.0)
        assert wp.weight == 5.0

    def test_default_values(self):
        wp = Waypoint("y")
        assert wp.position == ()
        assert wp.weight == 1.0

    def test_distance_to(self):
        a = Waypoint("a", (0.0, 0.0))
        b = Waypoint("b", (3.0, 4.0))
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_distance_to_self(self):
        wp = Waypoint("z", (1.0, 2.0))
        assert wp.distance_to(wp) == pytest.approx(0.0)

    def test_distance_mismatched_dims(self):
        a = Waypoint("a", (1.0,))
        b = Waypoint("b", (1.0, 2.0))
        with pytest.raises(ValueError):
            a.distance_to(b)

    def test_frozen(self):
        wp = Waypoint("x")
        with pytest.raises(AttributeError):
            wp.id = "y"  # type: ignore


# ─── Verse Tests ──────────────────────────────────────────────────────

class TestVerse:
    def test_creation(self):
        v = Verse("A", "B", 10)
        assert v.source == "A"
        assert v.target == "B"
        assert v.traversal_count == 10

    def test_default_traversal(self):
        v = Verse("A", "B")
        assert v.traversal_count == 1

    def test_frozen(self):
        v = Verse("A", "B")
        with pytest.raises(AttributeError):
            v.source = "C"  # type: ignore


# ─── SonglineGraph Tests ─────────────────────────────────────────────

class TestSonglineGraph:
    def test_empty_graph(self):
        g = SonglineGraph()
        assert g.waypoint_count() == 0
        assert g.verse_count() == 0

    def test_add_waypoint(self):
        g = SonglineGraph()
        g.add_waypoint(Waypoint("A"))
        assert g.waypoint_count() == 1
        assert "A" in g.waypoints

    def test_add_verse(self):
        g = SonglineGraph()
        g.add_waypoint(Waypoint("A"))
        g.add_waypoint(Waypoint("B"))
        g.add_verse(Verse("A", "B", 3))
        assert g.verse_count() == 1

    def test_verse_missing_source(self):
        g = SonglineGraph()
        g.add_waypoint(Waypoint("B"))
        with pytest.raises(ValueError, match="Source"):
            g.add_verse(Verse("A", "B"))

    def test_verse_missing_target(self):
        g = SonglineGraph()
        g.add_waypoint(Waypoint("A"))
        with pytest.raises(ValueError, match="Target"):
            g.add_verse(Verse("A", "B"))

    def test_neighbors(self):
        g = _make_triangle_graph()
        assert "B" in g.neighbors("A")
        assert len(g.neighbors("A")) == 1

    def test_predecessors(self):
        g = _make_triangle_graph()
        assert "C" in g.predecessors("A")

    def test_get_waypoint(self):
        g = _make_triangle_graph()
        wp = g.get_waypoint("B")
        assert wp is not None
        assert wp.weight == 3.0

    def test_get_waypoint_missing(self):
        g = SonglineGraph()
        assert g.get_waypoint("Z") is None

    def test_extract_subgraph(self):
        g = _make_triangle_graph()
        sub = g.extract_subgraph({"A", "B"})
        assert sub.waypoint_count() == 2
        assert sub.verse_count() == 1  # A->B only
        assert "C" not in sub.waypoints

    def test_extract_subgraph_empty(self):
        g = _make_triangle_graph()
        sub = g.extract_subgraph(set())
        assert sub.waypoint_count() == 0

    def test_reverse(self):
        g = _make_triangle_graph()
        rev = g.reverse()
        assert rev.waypoint_count() == 3
        assert rev.verse_count() == 3
        # A->B reversed to B->A
        assert "A" in rev.neighbors("B")

    def test_concatenate(self):
        g1 = _make_linear_graph()
        g2 = SonglineGraph()
        g2.add_waypoint(Waypoint("E", (4.0, 0.0)))
        g2.add_waypoint(Waypoint("F", (5.0, 0.0)))
        g2.add_verse(Verse("E", "F", 1))

        combined = g1.concatenate(g2, Verse("D", "E", 1))
        assert combined.waypoint_count() == 6
        # g1 has 3 verses + g2 has 1 + bridge = 5
        assert combined.verse_count() == 5

    def test_repr(self):
        g = _make_triangle_graph()
        assert "3" in repr(g) and "3" in repr(g)


# ─── Navigation Tests ────────────────────────────────────────────────

class TestNavigation:
    def test_pathfind_direct(self):
        g = _make_linear_graph()
        path = songline_pathfind(g, "A", "D")
        assert path == ["A", "B", "C", "D"]

    def test_pathfind_short(self):
        g = _make_linear_graph()
        path = songline_pathfind(g, "A", "B")
        assert path == ["A", "B"]

    def test_pathfind_cycle(self):
        g = _make_triangle_graph()
        path = songline_pathfind(g, "A", "C")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "C"

    def test_pathfind_same_node(self):
        g = _make_triangle_graph()
        assert songline_pathfind(g, "A", "A") == ["A"]

    def test_pathfind_no_path(self):
        g = _make_disconnected_graph()
        assert songline_pathfind(g, "A", "D") is None

    def test_pathfind_missing_node(self):
        g = _make_linear_graph()
        assert songline_pathfind(g, "A", "Z") is None

    def test_pathfind_prefers_high_weight(self):
        g = SonglineGraph()
        g.add_waypoint(Waypoint("A", (0,), 1.0))
        g.add_waypoint(Waypoint("B", (1,), 10.0))  # high weight
        g.add_waypoint(Waypoint("C", (2,), 1.0))
        g.add_waypoint(Waypoint("D", (3,), 1.0))
        g.add_verse(Verse("A", "B", 1))
        g.add_verse(Verse("B", "D", 1))
        g.add_verse(Verse("A", "C", 1))
        g.add_verse(Verse("C", "D", 1))
        path = songline_pathfind(g, "A", "D")
        assert path is not None
        # Should prefer B (high weight)
        assert "B" in path

    def test_dreamtime_connected(self):
        g = _make_linear_graph()
        segments = dreamtime_navigate(g, "A", "D")
        assert len(segments) >= 1
        full_path = segments[0]
        assert full_path[0] == "A"
        assert full_path[-1] == "D"

    def test_dreamtime_disconnected(self):
        g = _make_disconnected_graph()
        segments = dreamtime_navigate(g, "A", "D")
        assert len(segments) >= 1

    def test_dreamtime_missing(self):
        g = _make_linear_graph()
        assert dreamtime_navigate(g, "A", "Z") == []

    def test_navigability_full(self):
        g = SonglineGraph()
        g.add_waypoint(Waypoint("A"))
        g.add_waypoint(Waypoint("B"))
        g.add_verse(Verse("A", "B"))
        g.add_verse(Verse("B", "A"))
        assert navigability_score(g) == 1.0

    def test_navigability_none(self):
        g = SonglineGraph()
        g.add_waypoint(Waypoint("A"))
        g.add_waypoint(Waypoint("B"))
        assert navigability_score(g) == 0.0

    def test_navigability_partial(self):
        g = _make_linear_graph()  # A->B->C->D, not fully connected
        score = navigability_score(g)
        assert 0.0 < score < 1.0

    def test_navigability_single(self):
        g = SonglineGraph()
        g.add_waypoint(Waypoint("A"))
        assert navigability_score(g) == 1.0

    def test_navigability_empty(self):
        g = SonglineGraph()
        assert navigability_score(g) == 1.0  # trivially navigable


# ─── Corroboree Tests ────────────────────────────────────────────────

class TestCorroboree:
    def test_convergence_hubs(self):
        g = SonglineGraph()
        g.add_waypoint(Waypoint("hub"))
        g.add_waypoint(Waypoint("a"))
        g.add_waypoint(Waypoint("b"))
        g.add_waypoint(Waypoint("c"))
        g.add_verse(Verse("a", "hub"))
        g.add_verse(Verse("b", "hub"))
        g.add_verse(Verse("hub", "c"))
        hubs = find_convergence_hubs(g, min_degree=2)
        hub_ids = [h[0] for h in hubs]
        assert "hub" in hub_ids

    def test_convergence_hubs_empty(self):
        g = _make_linear_graph()
        hubs = find_convergence_hubs(g, min_degree=10)
        assert hubs == []

    def test_cluster_by_shared(self):
        g = SonglineGraph()
        g.add_waypoint(Waypoint("A"))
        g.add_waypoint(Waypoint("B"))
        g.add_waypoint(Waypoint("C"))
        g.add_waypoint(Waypoint("D"))
        g.add_verse(Verse("A", "B"))
        g.add_verse(Verse("B", "C"))
        g.add_verse(Verse("C", "D"))
        clusters = cluster_by_shared_songlines(g, threshold=0.1)
        assert len(clusters) >= 1

    def test_cluster_empty_graph(self):
        g = SonglineGraph()
        assert cluster_by_shared_songlines(g) == []

    def test_modularity(self):
        g = SonglineGraph()
        for w in "ABCD":
            g.add_waypoint(Waypoint(w))
        g.add_verse(Verse("A", "B"))
        g.add_verse(Verse("B", "A"))
        g.add_verse(Verse("C", "D"))
        g.add_verse(Verse("D", "C"))
        clusters = [{"A", "B"}, {"C", "D"}]
        q = modularity(g, clusters)
        assert q > 0  # Good clustering should have positive modularity

    def test_modularity_empty(self):
        g = SonglineGraph()
        assert modularity(g, []) == 0.0


# ─── Dreamtime (Persistent Homology) Tests ───────────────────────────

class TestDreamtime:
    def _make_triangle_points(self):
        return [
            Waypoint("A", (0.0, 0.0)),
            Waypoint("B", (1.0, 0.0)),
            Waypoint("C", (0.5, 0.866)),
        ]

    def test_vietoris_rips_small_epsilon(self):
        pts = self._make_triangle_points()
        cx = vietoris_rips(pts, epsilon=0.1)
        # Only vertices, no edges (all pairwise > 0.1... well sides are ~1.0)
        vertex_count = sum(1 for s, _ in cx if len(s) == 1)
        edge_count = sum(1 for s, _ in cx if len(s) == 2)
        assert vertex_count == 3
        assert edge_count == 0

    def test_vietoris_rips_large_epsilon(self):
        pts = self._make_triangle_points()
        cx = vietoris_rips(pts, epsilon=2.0)
        vertex_count = sum(1 for s, _ in cx if len(s) == 1)
        edge_count = sum(1 for s, _ in cx if len(s) == 2)
        tri_count = sum(1 for s, _ in cx if len(s) == 3)
        assert vertex_count == 3
        assert edge_count == 3
        assert tri_count == 1

    def test_vietoris_rips_empty(self):
        assert vietoris_rips([], 1.0) == []

    def test_betti_single_point(self):
        cx = [(frozenset(["A"]), 0.0)]
        assert betti_numbers(cx) == (1, 0)

    def test_betti_disconnected(self):
        cx = [
            (frozenset(["A"]), 0.0),
            (frozenset(["B"]), 0.0),
        ]
        assert betti_numbers(cx) == (2, 0)

    def test_betti_triangle_with_loop(self):
        cx = [
            (frozenset(["A"]), 0.0),
            (frozenset(["B"]), 0.0),
            (frozenset(["C"]), 0.0),
            (frozenset(["A", "B"]), 0.5),
            (frozenset(["B", "C"]), 0.5),
            (frozenset(["C", "A"]), 0.5),
        ]
        b0, b1 = betti_numbers(cx)
        assert b0 == 1
        assert b1 == 1  # One loop

    def test_betti_filled_triangle(self):
        cx = [
            (frozenset(["A"]), 0.0),
            (frozenset(["B"]), 0.0),
            (frozenset(["C"]), 0.0),
            (frozenset(["A", "B"]), 0.5),
            (frozenset(["B", "C"]), 0.5),
            (frozenset(["C", "A"]), 0.5),
            (frozenset(["A", "B", "C"]), 0.5),
        ]
        b0, b1 = betti_numbers(cx)
        assert b0 == 1
        assert b1 == 0  # Triangle filled the loop

    def test_betti_empty(self):
        assert betti_numbers([]) == (0, 0)

    def test_persistence_barcodes_nonempty(self):
        pts = self._make_triangle_points()
        barcodes = persistence_barcodes(pts, max_epsilon=2.0, steps=20)
        assert len(barcodes) > 0
        # Should have b0 features
        b0_bars = [b for b in barcodes if b[0] == 0]
        assert len(b0_bars) >= 1

    def test_persistence_barcodes_empty(self):
        assert persistence_barcodes([], 1.0) == []

    def test_persistent_cycles_no_cycles(self):
        cx = [
            (frozenset(["A"]), 0.0),
            (frozenset(["B"]), 0.0),
            (frozenset(["A", "B"]), 0.5),
        ]
        cycles = persistent_cycles(cx)
        assert cycles == []

    def test_persistent_cycles_with_cycle(self):
        cx = [
            (frozenset(["A"]), 0.0),
            (frozenset(["B"]), 0.0),
            (frozenset(["C"]), 0.0),
            (frozenset(["A", "B"]), 0.5),
            (frozenset(["B", "C"]), 0.5),
            (frozenset(["C", "A"]), 0.5),
        ]
        cycles = persistent_cycles(cx)
        assert len(cycles) >= 1


# ─── Tradition (Evolutionary) Tests ──────────────────────────────────

class TestTradition:
    def test_mutate_add(self):
        g = _make_linear_graph()
        rng = random.Random(42)
        # Force add by setting add_prob=1.0, remove_prob=0.0
        mutated = mutate(g, add_prob=1.0, remove_prob=0.0, rng=rng)
        assert mutated.waypoint_count() == g.waypoint_count()
        # Should have at least one more verse
        assert mutated.verse_count() >= g.verse_count()

    def test_mutate_no_change(self):
        g = _make_linear_graph()
        rng = random.Random(42)
        mutated = mutate(g, add_prob=0.0, remove_prob=0.0, rng=rng)
        assert mutated.verse_count() == g.verse_count()

    def test_recombine(self):
        g1 = _make_triangle_graph()
        g2 = _make_linear_graph()
        child = recombine(g1, g2)
        # Should have all waypoints from both
        assert child.waypoint_count() >= max(g1.waypoint_count(), g2.waypoint_count())
        assert child.verse_count() > 0

    def test_recombine_shared_nodes(self):
        g1 = SonglineGraph()
        g1.add_waypoint(Waypoint("A", (0.0,)))
        g1.add_waypoint(Waypoint("B", (1.0,)))
        g1.add_verse(Verse("A", "B"))

        g2 = SonglineGraph()
        g2.add_waypoint(Waypoint("B", (1.0,)))
        g2.add_waypoint(Waypoint("C", (2.0,)))
        g2.add_verse(Verse("B", "C"))

        child = recombine(g1, g2)
        assert child.waypoint_count() == 3  # A, B, C
        assert child.verse_count() >= 2

    def test_decay_zero(self):
        g = _make_linear_graph()
        decayed = decay(g, 0.0)
        assert decayed.verse_count() == g.verse_count()

    def test_decay_full(self):
        g = SonglineGraph()
        g.add_waypoint(Waypoint("A"))
        g.add_waypoint(Waypoint("B"))
        g.add_verse(Verse("A", "B", 1))
        decayed = decay(g, 10.0)  # Heavy decay
        # traversal_count=1 * e^-10 ≈ 0, rounds to 0, verse removed
        assert decayed.verse_count() == 0

    def test_decay_moderate(self):
        g = SonglineGraph()
        g.add_waypoint(Waypoint("A"))
        g.add_waypoint(Waypoint("B"))
        g.add_verse(Verse("A", "B", 100))
        decayed = decay(g, 0.5)
        # 100 * e^-0.5 ≈ 60.65, rounds to 61
        assert decayed.verse_count() == 1
        assert decayed.verses[0].traversal_count == 61

    def test_fitness(self):
        g = _make_linear_graph()
        f = fitness(g)
        assert 0.0 <= f <= 1.0

    def test_fitness_perfect(self):
        g = SonglineGraph()
        g.add_waypoint(Waypoint("A"))
        g.add_waypoint(Waypoint("B"))
        g.add_verse(Verse("A", "B"))
        g.add_verse(Verse("B", "A"))
        assert fitness(g) == 1.0


# ─── Integration Tests ───────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline(self):
        """Build graph → navigate → compute topology → evolve."""
        # 1. Build
        g = SonglineGraph()
        for i in range(5):
            angle = 2 * math.pi * i / 5
            g.add_waypoint(Waypoint(f"P{i}", (math.cos(angle), math.sin(angle)), float(i + 1)))
        for i in range(5):
            g.add_verse(Verse(f"P{i}", f"P{(i + 1) % 5}"))
            g.add_verse(Verse(f"P{i}", f"P{(i + 2) % 5}"))

        # 2. Navigate
        path = songline_pathfind(g, "P0", "P3")
        assert path is not None
        assert path[0] == "P0"
        assert path[-1] == "P3"

        # 3. Topology
        pts = [g.get_waypoint(wid) for wid in g.waypoints]
        pts = [p for p in pts if p is not None]
        cx = vietoris_rips(pts, epsilon=2.5)
        b0, b1 = betti_numbers(cx)
        assert b0 == 1  # Single component

        # 4. Evolve
        mutated = mutate(g, add_prob=0.5, remove_prob=0.0)
        assert mutated.waypoint_count() == 5

    def test_subgraph_then_navigate(self):
        g = _make_triangle_graph()
        sub = g.extract_subgraph({"A", "B"})
        path = songline_pathfind(sub, "A", "B")
        assert path == ["A", "B"]

    def test_reverse_then_navigate(self):
        g = _make_linear_graph()
        rev = g.reverse()
        path = songline_pathfind(rev, "D", "A")
        assert path == ["D", "C", "B", "A"]

    def test_concatenate_navigability(self):
        g1 = _make_linear_graph()
        g2 = SonglineGraph()
        g2.add_waypoint(Waypoint("E", (4.0,)))
        g2.add_waypoint(Waypoint("F", (5.0,)))
        g2.add_verse(Verse("E", "F"))
        combined = g1.concatenate(g2, Verse("D", "E"))
        score = navigability_score(combined)
        assert score > 0
