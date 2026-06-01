# songline-math

Navigable knowledge graphs with persistent homology for data scientists.

## Install

```bash
pip install songline-math
```

## Quick Start

```python
from songlinemath import (
    Waypoint, Verse, SonglineGraph,
    songline_pathfind, dreamtime_navigate, navigability_score,
    vietoris_rips, betti_numbers, persistence_barcodes,
    mutate, recombine, decay, fitness,
)

# Build a knowledge graph
g = SonglineGraph()
g.add_waypoint(Waypoint("origin", (0.0, 0.0), 5.0))
g.add_waypoint(Waypoint("mid", (1.0, 0.0), 3.0))
g.add_waypoint(Waypoint("dest", (2.0, 0.0), 2.0))
g.add_verse(Verse("origin", "mid"))
g.add_verse(Verse("mid", "dest"))

# Navigate
path = songline_pathfind(g, "origin", "dest")
print(path)  # ['origin', 'mid', 'dest']

# Topology
points = list(g.waypoints.values())
cx = vietoris_rips(points, epsilon=2.0)
print(betti_numbers(cx))  # connected components and loops
```

## License

MIT
