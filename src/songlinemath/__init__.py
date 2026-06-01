"""songline-math: Navigable knowledge graphs with persistent homology."""

from .graph import Waypoint, Verse, SonglineGraph
from .navigation import songline_pathfind, dreamtime_navigate, navigability_score
from .corroboree import find_convergence_hubs, cluster_by_shared_songlines, modularity
from .dreamtime import vietoris_rips, betti_numbers, persistence_barcodes, persistent_cycles
from .tradition import mutate, recombine, decay, fitness

__version__ = "0.1.0"
__all__ = [
    "Waypoint", "Verse", "SonglineGraph",
    "songline_pathfind", "dreamtime_navigate", "navigability_score",
    "find_convergence_hubs", "cluster_by_shared_songlines", "modularity",
    "vietoris_rips", "betti_numbers", "persistence_barcodes", "persistent_cycles",
    "mutate", "recombine", "decay", "fitness",
]
