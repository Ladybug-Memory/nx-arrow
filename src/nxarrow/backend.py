"""``arrow`` NetworkX backend: columnar graph storage on ``pyarrow.Table``.

This module reframes ``ArrowDiGraph`` (see :mod:`nxarrow.graph`) as a
*backend graph object* instead of a ``DiGraph`` replacement. NetworkX
dispatches to it via the standard plugin protocol:

- ``convert_from_nx`` / ``convert_to_nx`` (required) translate between the
  dict-backed reference implementation and Arrow tables. Extra attributes
  are always preserved (a superset of what the caller requests), which is
  harmless for algorithm correctness.
- ``can_run`` / ``should_run`` gate partial algorithm support: the backend
  only volunteers for algorithms with native Arrow implementations.
  Anything else falls back to NetworkX when
  ``NETWORKX_FALLBACK_TO_NX=True``.
- Native implementations (``in_degree_centrality``,
  ``out_degree_centrality``, ``topological_sort``) run directly on Arrow
  tables via ``pyarrow.compute``.

Usage::

    nx.out_degree_centrality(G, backend="arrow")  # explicit, converts + caches
    NETWORKX_BACKEND_PRIORITY=arrow python script.py  # automatic

Columnar stores hook in through :func:`register_storage_provider` and
persist whole graphs with :func:`persist`, which hands over per-type
node frames first and edge frames second.

Registered under the ``networkx.backends`` entry-point name ``arrow``.
"""

import pyarrow.compute as pc

from .graph import ArrowDiGraph

__all__ = [
    "ArrowDiGraph",
    "ArrowBackendInterface",
    "backend_interface",
    "register_storage_provider",
    "unregister_storage_provider",
    "get_storage_provider",
    "list_storage_providers",
    "persist",
]


# -- storage provider registry --------------------------------------------
# A storage provider bulk-loads columnar frames into an external store.
# It is any object with two methods (extra keyword arguments are passed
# through untouched):
#
#     store_nodes(node_frames) -> dict[str, int]
#     store_edges(edge_frames) -> dict[str, int]
#
# ``node_frames`` / ``edge_frames`` map a type name (or ``None`` for
# untagged rows) to a ``pyarrow.Table`` holding the full rows. Node frames
# are always handed over before edge frames so providers can resolve
# endpoints against stored nodes; empty frames are never passed.
# Providers register under a name and are referenced by that name from
# :func:`persist`. Column-to-field mapping is the provider's job: it
# knows its own stored schema and projects/selects frame columns itself.

_storage_providers = {}


def register_storage_provider(name, provider):
    """Register a bulk-storage provider under ``name``.

    ``provider`` must define ``store_nodes`` and ``store_edges`` as
    described above. Re-registering a name replaces the old provider.
    """
    for method in ("store_nodes", "store_edges"):
        if not callable(getattr(provider, method, None)):
            raise TypeError(
                f"storage provider {name!r} must define a {method}() method"
            )
    _storage_providers[name] = provider
    return provider


def unregister_storage_provider(name):
    """Remove the provider registered under ``name`` (``KeyError`` if absent)."""
    del _storage_providers[name]


def get_storage_provider(name):
    """Return the provider registered under ``name`` (``KeyError`` if absent)."""
    return _storage_providers[name]


def list_storage_providers():
    """Return the sorted names of all registered storage providers."""
    return sorted(_storage_providers)


def persist(graph, provider, **kwargs):
    """Bulk-persist a graph through a registered storage provider.

    ``graph`` is an ``ArrowDiGraph`` (plain NetworkX graphs are converted
    first); ``provider`` is a registered name or a provider object. The
    graph is partitioned into per-type frames, node frames are stored
    before edge frames, and per-table row counts are merged into one
    ``{table_name: rows}`` dict.
    """
    if isinstance(provider, str):
        provider = get_storage_provider(provider)
    for method in ("store_nodes", "store_edges"):
        if not callable(getattr(provider, method, None)):
            raise TypeError(f"storage provider must define a {method}() method")
    if not isinstance(graph, ArrowDiGraph):
        graph = ArrowDiGraph.from_networkx(graph)
    node_frames, edge_frames = graph.partition_tables()
    return {
        **provider.store_nodes(node_frames, **kwargs),
        **provider.store_edges(edge_frames, **kwargs),
    }


class ArrowBackendInterface:
    """BackendInterface object for the ``arrow`` backend."""

    @staticmethod
    def convert_from_nx(
        graph,
        *,
        edge_attrs=None,
        node_attrs=None,
        preserve_edge_attrs=None,
        preserve_node_attrs=None,
        preserve_graph_attrs=None,
        preserve_all_attrs=False,
        name=None,
        graph_name=None,
    ):
        if isinstance(graph, ArrowDiGraph):
            return graph
        return ArrowDiGraph.from_networkx(graph)

    @staticmethod
    def convert_to_nx(result, *, name=None):
        if isinstance(result, ArrowDiGraph):
            return result.to_networkx()
        if isinstance(result, dict):
            return {
                (s.decode() if isinstance(s, bytes) else s): v
                for s, v in result.items()
            }
        if isinstance(result, (list, tuple)) and not isinstance(result, str):
            out = []
            for item in result:
                out.append(
                    item.to_networkx() if isinstance(item, ArrowDiGraph) else item
                )
            return type(result)(out) if isinstance(result, tuple) else out
        return result

    @staticmethod
    def can_run(name, args, kwargs):
        return True

    @staticmethod
    def should_run(name, args, kwargs):
        # Only volunteer for algorithms with native Arrow implementations
        # (plus graph construction); everything else stays on the NetworkX
        # path unless the user opts into backend priority + fallback.
        return name in {
            "graph__new__",
            "digraph__new__",
            "in_degree_centrality",
            "out_degree_centrality",
            "topological_sort",
        }

    # -- graph construction: nx.Graph(backend="arrow") / nx.DiGraph(...) --
    # NOTE: ArrowDiGraph is directed; nx.Graph(backend="arrow") returns it
    # as-is (directed semantics). There is no undirected Arrow class yet.

    @staticmethod
    def graph__new__(cls, incoming_graph_data=None, **attr):
        attr.pop("backend", None)  # consumed by dispatch machinery
        return ArrowDiGraph(incoming_graph_data, **attr)

    @staticmethod
    def digraph__new__(cls, incoming_graph_data=None, **attr):
        attr.pop("backend", None)  # consumed by dispatch machinery
        return ArrowDiGraph(incoming_graph_data, **attr)

    # -- native Arrow implementations (signatures mirror the nx originals) --

    @staticmethod
    def in_degree_centrality(G):
        return ArrowBackendInterface._centrality(G, "target")

    @staticmethod
    def out_degree_centrality(G):
        return ArrowBackendInterface._centrality(G, "source")

    @staticmethod
    def _centrality(G, col):
        G._sync()
        nodes = G.nodes
        n = len(nodes)
        if n <= 1:
            return {v: 0 for v in nodes}
        t = G.edges_table()
        scale = 1 / (n - 1)
        centrality = dict.fromkeys(nodes, 0.0)
        if t.num_rows == 0:
            return centrality
        grouped = t.group_by(col).aggregate([(col, "count")])
        keys = grouped.column(col).to_pylist()
        vals = grouped.column(f"{col}_count").to_pylist()
        for k, c in zip(keys, vals):
            if k in G._node_orig:
                centrality[G._node_orig[k]] = c * scale
        return centrality

    @staticmethod
    def topological_sort(G):
        G._sync()
        nodes = G.nodes
        succ = {v: [] for v in nodes}
        indeg = dict.fromkeys(nodes, 0)
        for row in G.edges_table().to_pylist():
            u, v = G._node_orig[row["source"]], G._node_orig[row["target"]]
            succ[u].append(v)
            indeg[v] += 1
        zero = sorted(
            [v for v in nodes if indeg[v] == 0],
            key=lambda v: str(v),
        )
        order = []
        while zero:
            u = zero.pop(0)
            order.append(u)
            for w in succ[u]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    zero.append(w)
        if len(order) != len(nodes):
            raise RuntimeError("Graph contains a cycle; topological sort not possible")
        return iter(order)


backend_interface = ArrowBackendInterface()
