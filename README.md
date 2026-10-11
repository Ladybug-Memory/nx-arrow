# nx-arrow

Columnar graph storage for [NetworkX](https://networkx.org/) backed by
[Apache Arrow](https://arrow.apache.org/). Graphs live in `pyarrow.Table`s
(tight, typed columns instead of per-object Python dicts) while staying
usable through the standard NetworkX dispatchable API.

Derived from prototyping in the NetworkX repository; NetworkX itself is
a regular dependency here, not a fork.

## Install

```bash
pip install nx-arrow
```

For development:

```bash
pip install -e ".[test]"
```

## Use

```python
import networkx as nx

# build an arrow-backed graph through the standard constructor
G = nx.DiGraph([(0, 1), (1, 2)], backend="arrow")

# run natively on Arrow tables ...
nx.out_degree_centrality(G, backend="arrow")
list(nx.topological_sort(G, backend="arrow"))

# ... or let dispatch pick the backend automatically
# NETWORKX_BACKEND_PRIORITY=arrow python script.py

# heterogeneous node/edge types are first-class columns
G.add_node("alice", node_type="user", age=30)
G.add_edge("alice", 0, edge_type="rates")
G.nodes_of_type("user")
G.edges_of_type("rates")
```

Single-edge incremental writes stay dict-backed territory; convert with
`ArrowDiGraph.from_networkx` / `to_networkx`, or set
`NETWORKX_FALLBACK_TO_NX=True` and let dispatch convert back.

## Persist

Storage providers bulk-load per-type frames into an external store:

```python
from nxarrow import ParquetStorageProvider, persist

provider = ParquetStorageProvider("graph_store")
persist(G, provider, graph_attrs=G.graph)
restored = provider.load_graph()  # full round-trip
```

Custom stores implement `store_nodes(frames)` / `store_edges(frames)` and
register with `nxarrow.register_storage_provider`.

## Develop

```bash
pytest
python benchmarks/bench_memory.py --n 20000
```

See `benchmarks/results.md` for memory/throughput numbers.
