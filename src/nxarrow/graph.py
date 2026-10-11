"""Directed graph backed by ``pyarrow.Table`` for nodes and edges.

Nodes: ``node`` key column + one typed column per homogeneous attribute.
Edges: ``source``/``target`` key columns + one typed column per attribute.
Key columns keep native Arrow types: integers use ``int64``, floats use
``float64``, booleans normalize to integers (matching NetworkX ``True ==
1`` node semantics), strings use ``string``, dates use ``date32``, times
use ``time64[us]`` and datetimes use ``timestamp[us]``. An explicit
``key_type`` (e.g. ``pa.float32()``) selects the precision up front. The
first incompatible key migrates the graph: numeric keys widen along the
``int -> float`` tower, anything else falls back to ``string`` while
preserving the old merge semantics. Reserved ``node_type``/``edge_type`` columns are materialized only when at
least one node/edge actually has a type, so single-type (or untyped)
graphs pay nothing for the feature. Missing values are null. Columns are (dictionary-)encoded by Arrow, so
large homogeneous attrs use far less memory than per-object Python dicts.
"""

import datetime

import pyarrow as pa
import pyarrow.compute as pc

__all__ = ["ArrowDiGraph"]


def _typed_column(values):
    try:
        return pa.array(values)
    except Exception:
        return pa.array([None if v is None else str(v) for v in values])


_key_category_cache = {}


def _key_category(dtype):
    """Map an Arrow type to a key category, or None if unsupported."""
    if dtype is None:
        return None
    try:
        return _key_category_cache[dtype]
    except KeyError:
        pass
    if pa.types.is_boolean(dtype) or pa.types.is_integer(dtype):
        category = "int"
    elif pa.types.is_floating(dtype):
        category = "float"
    elif pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        category = "str"
    elif pa.types.is_date(dtype):
        category = "date"
    elif pa.types.is_time(dtype):
        category = "time"
    elif pa.types.is_timestamp(dtype):
        category = "timestamp"
    else:
        category = None
    _key_category_cache[dtype] = category
    return category


def _key_type_for(node):
    """Infer the Arrow key type for the first node of an empty graph."""
    if isinstance(node, bool):
        return pa.int64()
    if isinstance(node, int):
        return pa.int64()
    if isinstance(node, float):
        return pa.float64()
    if isinstance(node, str):
        return pa.string()
    if isinstance(node, datetime.datetime):
        return pa.timestamp("us")
    if isinstance(node, datetime.date):
        return pa.date32()
    if isinstance(node, datetime.time):
        return pa.time64("us")
    return None


def _partition_by_type(table, column):
    """Split ``table`` into ``{type value: frame}`` preserving full rows."""
    if table.num_rows == 0:
        return {}
    if column not in table.column_names:
        return {None: table}
    values = table.column(column)
    frames = {}
    for type_value in pc.unique(values).to_pylist():
        if type_value is None:
            mask = pc.is_null(values)
        else:
            mask = pc.equal(values, type_value)
        frame = pc.take(table, pc.indices_nonzero(mask))
        if frame.num_rows:
            frames[type_value] = frame
    return frames


class ArrowDiGraph:
    """Minimal directed graph with nodes/edges + attrs in Arrow tables."""

    __networkx_backend__ = "arrow"

    def __init__(self, incoming_graph_data=None, key_type=None, **attr):
        attr.pop("backend", None)  # consumed by dispatch machinery
        if key_type is not None and _key_category(key_type) is None:
            raise TypeError(f"unsupported Arrow key type: {key_type!r}")
        self.graph = dict(attr)  # graph-level attributes, like nx.Graph.graph
        self._key_type = key_type  # Arrow type of key columns, or None if empty
        self._node_order = []  # canonical keys, insertion order
        self._node_pos = {}  # key -> index (O(1) lookup)
        self._node_orig = {}  # key -> original object
        self._node_cols = {}  # attr name -> list aligned with _node_order
        self._node_types = []  # reserved node_type column, aligned with _node_order
        self._edge_order = []  # (skey, tkey) tuples
        self._edge_pos = {}  # edge key -> index
        self._edge_cols = {}
        self._edge_types = []  # reserved edge_type column, aligned with _edge_order
        self._nodes_table = pa.table({"node": pa.array([], type=pa.string())})
        self._edges_table = pa.table(
            {"source": pa.array([], type=pa.string()),
             "target": pa.array([], type=pa.string())}
        )
        self._nodes_dirty = False
        self._edges_dirty = False
        self._index_built = True  # Python lookup indexes; see _ensure_index
        if isinstance(incoming_graph_data, ArrowDiGraph):
            other = incoming_graph_data
            other._ensure_index()
            self._key_type = other._key_type
            self._node_order = list(other._node_order)
            self._node_pos = dict(other._node_pos)
            self._node_orig = dict(other._node_orig)
            self._node_cols = {k: list(v) for k, v in other._node_cols.items()}
            self._node_types = list(other._node_types)
            self._edge_order = list(other._edge_order)
            self._edge_pos = dict(other._edge_pos)
            self._edge_cols = {k: list(v) for k, v in other._edge_cols.items()}
            self._edge_types = list(other._edge_types)
            self.graph.update(other.graph)
            self._nodes_dirty = self._edges_dirty = True
        elif incoming_graph_data is not None:
            if not hasattr(incoming_graph_data, "nodes"):
                # edgelists, dicts, generators...: use the standard conversion
                import networkx as nx

                incoming_graph_data = nx.convert.to_networkx_graph(
                    incoming_graph_data, create_using=nx.DiGraph
                )
            self._add_nodes_batch(incoming_graph_data.nodes(data=True))
            self._add_edges_batch(incoming_graph_data.edges(data=True))
            if hasattr(incoming_graph_data, "graph"):
                self.graph.update(incoming_graph_data.graph)

    # -- keys --
    # Canonical keys keep native Arrow scalar types: ints (bool normalizes
    # to int, matching NetworkX ``True == 1`` node semantics), floats,
    # strings, dates, times and datetimes. Numeric keys widen along the
    # ``int -> float`` tower; the first otherwise-incompatible key replays
    # the whole graph into string keys (one-time O(n) cost).
    def _canonical_key(self, node):
        if self._key_type is None:
            inferred = _key_type_for(node)
            if inferred is None:
                self._key_type = pa.string()
                return str(node)
            self._key_type = inferred
            return int(node) if isinstance(node, bool) else node
        category = _key_category(self._key_type)
        if category == "int":
            if isinstance(node, bool):
                return int(node)
            if isinstance(node, int):
                return node
            if isinstance(node, float):
                self._rekey(pa.float64())
                return float(node)
        elif category == "float":
            if isinstance(node, (bool, int, float)):
                return float(node)
        elif category == "str":
            return str(node)
        elif category == "date":
            if isinstance(node, datetime.date) and not isinstance(
                node, datetime.datetime
            ):
                return node
        elif category == "time":
            if isinstance(node, datetime.time):
                return node
        elif category == "timestamp":
            if isinstance(node, datetime.datetime):
                return node
        self._rekey(pa.string())
        return str(node)

    def _lookup_key(self, node):
        """Read-only key resolution: never triggers a migration."""
        category = _key_category(self._key_type)
        if category == "int":
            if isinstance(node, bool):
                return int(node)
            if isinstance(node, int):
                return node
            if isinstance(node, float):
                # Misses unless the graph has widened; 1.0 still hits int 1
                # per NetworkX numeric node equality.
                return float(node)
            return str(node)
        if category == "float":
            if isinstance(node, (bool, int, float)):
                return float(node)
            return str(node)
        if category == "date":
            if isinstance(node, datetime.date) and not isinstance(
                node, datetime.datetime
            ):
                return node
            return str(node)
        if category == "time":
            if isinstance(node, datetime.time):
                return node
            return str(node)
        if category == "timestamp":
            if isinstance(node, datetime.datetime):
                return node
            return str(node)
        return str(node)

    def _rekey(self, key_type):
        """Reset key structures and replay all data under ``key_type``."""
        nodes = [
            (self._node_orig[k], self._node_types[i],
             {n: c[i] for n, c in self._node_cols.items() if c[i] is not None})
            for i, k in enumerate(self._node_order)
        ]
        edges = [
            (self._node_orig[s], self._node_orig[t], self._edge_types[i],
             {n: c[i] for n, c in self._edge_cols.items() if c[i] is not None})
            for i, (s, t) in enumerate(self._edge_order)
        ]
        self._key_type = key_type
        self._node_order = []
        self._node_pos = {}
        self._node_orig = {}
        self._node_cols = {}
        self._node_types = []
        self._edge_order = []
        self._edge_pos = {}
        self._edge_cols = {}
        self._edge_types = []
        self._add_nodes_batch([
            (o, {"node_type": t, **a} if t is not None else a)
            for o, t, a in nodes
        ])
        self._add_edges_batch([
            (u, v, {"edge_type": t, **a} if t is not None else a)
            for u, v, t, a in edges
        ])

    def _ensure_index(self):
        """Build Python lookup indexes from the adopted tables, once.

        Graphs created by :meth:`from_arrow` with ``index=False`` skip
        index construction entirely: table-level methods work immediately
        and anything touching nodes, edges or attributes builds the
        indexes on first use. Mutation always materializes first, so the
        tables and the indexes can never disagree.
        """
        if self._index_built:
            return
        nodes = self._nodes_table
        if "node" in nodes.column_names:
            node_type = nodes.schema.field("node").type
            if _key_category(node_type) is None:
                order = [
                    k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
                    for k in nodes.column("node").to_pylist()
                ]
                self._key_type = pa.string()
                self._nodes_dirty = True
            else:
                order = nodes.column("node").to_pylist()
                self._key_type = node_type
            self._node_order = order
            self._node_pos = {k: i for i, k in enumerate(order)}
            self._node_orig = dict(zip(order, order))
            names = nodes.column_names
            self._node_types = (
                nodes.column("node_type").to_pylist()
                if "node_type" in names else [None] * len(order)
            )
            self._node_cols = {
                name: nodes.column(name).to_pylist()
                for name in names if name not in ("node", "node_type")
            }
        edges = self._edges_table
        if "source" in edges.column_names:
            src = edges.column("source")
            tgt = edges.column("target")
            if src.type != self._key_type:
                src = pc.cast(src, self._key_type)
                self._edges_dirty = True
            if tgt.type != self._key_type:
                tgt = pc.cast(tgt, self._key_type)
                self._edges_dirty = True
            src = src.to_pylist()
            tgt = tgt.to_pylist()
            order = list(zip(src, tgt))
            self._edge_order = order
            self._edge_pos = {k: i for i, k in enumerate(order)}
            names = edges.column_names
            self._edge_types = (
                edges.column("edge_type").to_pylist()
                if "edge_type" in names else [None] * len(order)
            )
            self._edge_cols = {
                name: edges.column(name).to_pylist()
                for name in names if name not in ("source", "target", "edge_type")
            }
        self._index_built = True

    # -- nodes --
    def add_node(self, node, node_type=None, **attrs):
        self._ensure_index()
        key = self._canonical_key(node)
        if key not in self._node_orig:
            self._node_orig[key] = node
            self._node_pos[key] = len(self._node_order)
            self._node_order.append(key)
            self._node_types.append(None)
            for name in self._node_cols:
                self._node_cols[name].append(None)
        idx = self._node_pos[key]
        if node_type is not None:
            self._node_types[idx] = node_type
        for name, val in attrs.items():
            self._node_cols.setdefault(name, [None] * len(self._node_order))
            self._node_cols[name][idx] = val
        self._nodes_dirty = True

    def node_type(self, node):
        self._ensure_index()
        return self._node_types[self._node_pos[self._lookup_key(node)]]

    @property
    def node_types(self):
        self._ensure_index()
        return sorted({t for t in self._node_types if t is not None})

    def nodes_of_type(self, node_type):
        """Nodes of a given type, filtered in Arrow (no Python scan)."""
        self._ensure_index()
        self._sync()
        if "node_type" not in self._nodes_table.column_names:
            return []
        mask = pc.equal(self._nodes_table.column("node_type"), node_type)
        keys = pc.take(
            self._nodes_table.column("node"), pc.indices_nonzero(mask)
        ).to_pylist()
        return [self._node_orig[k] for k in keys]

    def add_nodes_from(self, nodes):
        self._add_nodes_batch(nodes)

    def _add_nodes_batch(self, nodes):
        """Bulk node ingest: a few list-level passes, one table build."""
        self._ensure_index()
        norm = []  # (orig, type, attrs)
        for n in nodes:
            if isinstance(n, tuple) and len(n) == 2 and isinstance(n[1], dict):
                d = n[1]
                norm.append((n[0], d.get("node_type"), d))
            else:
                norm.append((n, None, None))
        if not norm:
            return
        kind_before = _key_category(self._key_type)
        keyed = []  # (key, type, attrs); keys computed once here ...
        for orig, t, d in norm:
            key = self._canonical_key(orig)
            if key not in self._node_orig:
                self._node_orig[key] = orig
                self._node_pos[key] = len(self._node_order)
                self._node_order.append(key)
            keyed.append((key, t, d))
        new_count = len(self._node_order) - len(self._node_types)
        if new_count:
            self._node_types.extend([None] * new_count)
            for col in self._node_cols.values():
                col.extend([None] * new_count)
        for _, _, d in keyed:
            if d:
                for name in d:
                    if name != "node_type" and name not in self._node_cols:
                        self._node_cols[name] = [None] * len(self._node_order)
        # ... unless a migration (to strings) invalidated them mid-loop.
        # (Numeric widening keeps keys valid via int == float equality.)
        if kind_before != "str" and _key_category(self._key_type) == "str":
            keyed = [
                (self._canonical_key(orig), t, d) for orig, t, d in norm
            ]
        for key, t, d in keyed:
            idx = self._node_pos[key]
            if t is not None:
                self._node_types[idx] = t
            if d:
                for k, v in d.items():
                    if k != "node_type":
                        self._node_cols[k][idx] = v
        self._nodes_dirty = True

    def has_node(self, node):
        self._ensure_index()
        return self._lookup_key(node) in self._node_orig

    def remove_node(self, node):
        self._ensure_index()
        key = self._lookup_key(node)
        if key not in self._node_orig:
            raise KeyError(node)
        idx = self._node_pos.pop(key)
        del self._node_order[idx]
        del self._node_orig[key]
        del self._node_types[idx]
        for name in self._node_cols:
            del self._node_cols[name][idx]
        for i in range(idx, len(self._node_order)):
            self._node_pos[self._node_order[i]] = i
        keep = [(s, t) for (s, t) in self._edge_order if s != key and t != key]
        drop = set(self._edge_order) - set(keep)
        if drop:
            idxs = [i for i, e in enumerate(self._edge_order) if e in drop]
            for i in sorted(idxs, reverse=True):
                del self._edge_order[i]
                del self._edge_types[i]
                for name in self._edge_cols:
                    del self._edge_cols[name][i]
            self._edge_pos = {e: i for i, e in enumerate(self._edge_order)}
        self._nodes_dirty = True
        self._edges_dirty = True

    def node_attrs(self, node):
        self._ensure_index()
        key = self._lookup_key(node)
        idx = self._node_pos[key]
        d = {n: c[idx] for n, c in self._node_cols.items() if c[idx] is not None}
        if self._node_types[idx] is not None:
            d["node_type"] = self._node_types[idx]
        return d

    @property
    def nodes(self):
        self._ensure_index()
        return [self._node_orig[k] for k in self._node_order]

    def number_of_nodes(self):
        # Table-level: works without materializing the index.
        self._sync()
        return self._nodes_table.num_rows

    # -- edges --
    def add_edge(self, u, v, edge_type=None, **attrs):
        self._ensure_index()
        self.add_node(u)
        self.add_node(v)
        key = (self._canonical_key(u), self._canonical_key(v))
        if key not in self._edge_pos:
            self._edge_pos[key] = len(self._edge_order)
            self._edge_order.append(key)
            self._edge_types.append(None)
            for name in self._edge_cols:
                self._edge_cols[name].append(None)
        idx = self._edge_pos[key]
        if edge_type is not None:
            self._edge_types[idx] = edge_type
        for name, val in attrs.items():
            self._edge_cols.setdefault(name, [None] * len(self._edge_order))
            self._edge_cols[name][idx] = val
        self._edges_dirty = True

    def edge_type(self, u, v):
        self._ensure_index()
        key = (self._lookup_key(u), self._lookup_key(v))
        return self._edge_types[self._edge_pos[key]]

    @property
    def edge_types(self):
        self._ensure_index()
        return sorted({t for t in self._edge_types if t is not None})

    def edges_of_type(self, edge_type):
        """Edges of a given type, filtered in Arrow (no Python scan)."""
        self._ensure_index()
        self._sync()
        if "edge_type" not in self._edges_table.column_names:
            return []
        mask = pc.equal(self._edges_table.column("edge_type"), edge_type)
        rows = pc.take(self._edges_table, pc.indices_nonzero(mask)).to_pylist()
        return [(self._node_orig[r["source"]], self._node_orig[r["target"]])
                for r in rows]

    def add_edges_from(self, ebunch):
        self._add_edges_batch(ebunch)

    def _add_edges_batch(self, ebunch):
        """Bulk edge ingest: endpoints ensured in one node batch, then
        edge keys/attrs merged in list-level passes, one table build."""
        self._ensure_index()
        norm = []  # (u_orig, v_orig, type, attrs)
        for e in ebunch:
            if len(e) == 2:
                norm.append((e[0], e[1], None, None))
            else:
                norm.append((e[0], e[1], e[2].get("edge_type"), e[2]))
        if not norm:
            return
        self._add_nodes_batch([u for u, _, _, _ in norm] +
                              [v for _, v, _, _ in norm])
        ekeys = [
            (self._canonical_key(u), self._canonical_key(v))
            for u, v, _, _ in norm
        ]
        for key in ekeys:
            if key not in self._edge_pos:
                self._edge_pos[key] = len(self._edge_order)
                self._edge_order.append(key)
        new_count = len(self._edge_order) - len(self._edge_types)
        if new_count:
            self._edge_types.extend([None] * new_count)
            for col in self._edge_cols.values():
                col.extend([None] * new_count)
        for _, _, _, d in norm:
            if d:
                for name in d:
                    if name != "edge_type" and name not in self._edge_cols:
                        self._edge_cols[name] = [None] * len(self._edge_order)
        for (_, _, t, d), key in zip(norm, ekeys):
            idx = self._edge_pos[key]
            if t is not None:
                self._edge_types[idx] = t
            if d:
                for k, v in d.items():
                    if k != "edge_type":
                        self._edge_cols[k][idx] = v
        self._edges_dirty = True

    def has_edge(self, u, v):
        self._ensure_index()
        return (self._lookup_key(u), self._lookup_key(v)) in self._edge_pos

    def remove_edge(self, u, v):
        self._ensure_index()
        key = (self._lookup_key(u), self._lookup_key(v))
        if key not in self._edge_pos:
            raise KeyError((u, v))
        idx = self._edge_pos.pop(key)
        del self._edge_order[idx]
        del self._edge_types[idx]
        for name in self._edge_cols:
            del self._edge_cols[name][idx]
        for i in range(idx, len(self._edge_order)):
            self._edge_pos[self._edge_order[i]] = i
        self._edges_dirty = True

    def edge_attrs(self, u, v):
        self._ensure_index()
        idx = self._edge_pos[(self._lookup_key(u), self._lookup_key(v))]
        d = {n: c[idx] for n, c in self._edge_cols.items() if c[idx] is not None}
        if self._edge_types[idx] is not None:
            d["edge_type"] = self._edge_types[idx]
        return d

    def successors(self, node):
        self._ensure_index()
        key = self._lookup_key(node)
        return [self._node_orig[t] for (s, t) in self._edge_order if s == key]

    def predecessors(self, node):
        self._ensure_index()
        key = self._lookup_key(node)
        return [self._node_orig[s] for (s, t) in self._edge_order if t == key]

    @property
    def edges(self):
        self._ensure_index()
        return [(self._node_orig[s], self._node_orig[t]) for (s, t) in self._edge_order]

    def number_of_edges(self):
        # Table-level: works without materializing the index.
        self._sync()
        return self._edges_table.num_rows

    # -- arrow-native --
    def _sync(self):
        if self._nodes_dirty:
            self._rebuild_nodes_table()
            self._nodes_dirty = False
        if self._edges_dirty:
            self._rebuild_edges_table()
            self._edges_dirty = False

    def nodes_table(self):
        self._sync()
        return self._nodes_table

    def edges_table(self):
        self._sync()
        return self._edges_table

    def partition_tables(self):
        """Split node/edge tables into ``{type: frame}`` dicts for bulk storage.

        Each distinct ``node_type``/``edge_type`` gets its own frame holding
        the full rows (key columns included, so endpoints stay resolvable).
        Rows without a type are grouped under the ``None`` key; tables with
        no type column at all yield a single ``None``-keyed frame. Empty
        frames are omitted. Column-to-field mapping is left to the storage
        provider, which knows its own schema.
        """
        self._sync()
        return (
            _partition_by_type(self._nodes_table, "node_type"),
            _partition_by_type(self._edges_table, "edge_type"),
        )

    def as_writeable(self):
        """Fall back to a dict-backed ``DiGraph`` for write-heavy work.

        Use when doing many incremental adds/removes (O(1) per edit vs
        O(n) table rebuilds here). Convert back with ``from_networkx``.
        """
        return self.to_networkx()

    @classmethod
    def from_networkx(cls, g):
        """Bulk-load from a dict-backed graph (the write-heavy fallback)."""
        return cls(g)

    @classmethod
    def from_arrow(cls, nodes=None, edges=None, *, index=True):
        """Bulk-adopt Arrow tables: no per-row Python loop.

        The input tables become the backing tables directly. With the
        default ``index=True`` the Python lookup indexes are derived with
        vectorized ``to_pylist()`` calls. With ``index=False`` even that
        is deferred: only the key type is read from the schema, and the
        indexes are built on first Python-level access. Table-level
        methods (``nodes_table``, ``edges_table``, ``partition_tables``,
        ``out_degree_table``, counts) work immediately without building
        them. The key column type is preserved (int, float, string, date,
        time, timestamp); anything else falls back to string keys on first
        index build. Node objects are the adopted keys themselves. Edge
        endpoint columns are cast to the node key type when they differ."""
        g = cls()
        if nodes is not None:
            node_type = nodes.schema.field("node").type
            g._key_type = (
                node_type if _key_category(node_type) is not None else pa.string()
            )
            g._nodes_table = nodes
            g._index_built = False
        if edges is not None:
            if nodes is None:
                endpoint_type = edges.schema.field("source").type
                g._key_type = (
                    endpoint_type
                    if _key_category(endpoint_type) is not None
                    else pa.string()
                )
            g._edges_table = edges
            g._index_built = False
        if index:
            g._ensure_index()
        return g

    def out_degree_table(self):
        self._sync()
        if self._edges_table.num_rows == 0:
            return pa.table(
                {"node": pa.array([], type=pa.string()),
                 "out_degree": pa.array([], type=pa.int64())}
            )
        grouped = self._edges_table.group_by("source").aggregate([("source", "count")])
        return pa.table(
            {
                "node": grouped.column("source"),
                "out_degree": pc.fill_null(grouped.column("source_count"), 0),
            }
        )

    def to_networkx(self):
        import networkx as nx

        self._ensure_index()
        g = nx.DiGraph()
        g.graph.update(self.graph)
        for k in self._node_order:
            g.add_node(self._node_orig[k], **self.node_attrs(self._node_orig[k]))
        for s, t in self._edge_order:
            g.add_edge(
                self._node_orig[s],
                self._node_orig[t],
                **self.edge_attrs(self._node_orig[s], self._node_orig[t]),
            )
        return g

    def _rebuild_nodes_table(self):
        key_type = self._key_type if self._key_type is not None else pa.string()
        cols = {"node": pa.array(self._node_order, type=key_type)}
        if any(t is not None for t in self._node_types):
            cols["node_type"] = pa.array(self._node_types, type=pa.string())
        for name, vals in self._node_cols.items():
            cols[name] = _typed_column(list(vals))
        self._nodes_table = pa.table(cols)

    def _rebuild_edges_table(self):
        key_type = self._key_type if self._key_type is not None else pa.string()
        cols = {
            "source": pa.array(
                [s for s, _ in self._edge_order], type=key_type
            ),
            "target": pa.array(
                [t for _, t in self._edge_order], type=key_type
            ),
        }
        if any(t is not None for t in self._edge_types):
            cols["edge_type"] = pa.array(self._edge_types, type=pa.string())
        for name, vals in self._edge_cols.items():
            cols[name] = _typed_column(list(vals))
        self._edges_table = pa.table(cols)

    def is_directed(self):
        return True

    def is_multigraph(self):
        return False

    def __len__(self):
        return self.number_of_nodes()

    def __contains__(self, node):
        return self.has_node(node)
