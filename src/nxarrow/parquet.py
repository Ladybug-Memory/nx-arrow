"""Parquet storage provider for arrow-backed graphs.

Persists the per-type ``pyarrow.Table`` frames handed over by
 :func:`nxarrow.backend.persist` as one Parquet file per
table, plus a ``manifest.json`` recording the type-to-file mapping and the
graph-level attributes::

    <root>/
        manifest.json
        nodes/nodes-000-user.parquet
        nodes/nodes-001-post.parquet
        edges/edges-000-authored.parquet
        ...

Round-trip::

    provider = ParquetStorageProvider(root)
    persist(graph, provider, graph_attrs=graph.graph)
    restored = provider.load_graph()

``load_graph`` concatenates the per-type files back into single node and
edge tables (unifying heterogeneous attribute columns with nulls) and
rebuilds an ``ArrowDiGraph`` via :meth:`ArrowDiGraph.from_arrow`, so the
restored graph compares equal to the persisted one at the table level.

Two fidelity notes. First, key columns keep their Arrow types, so integer,
float, date, time and boolean keys round-trip natively; anything else
stringifies on adopt (see :meth:`ArrowDiGraph.from_arrow`). Second,
``graph_attrs`` must be JSON-serializable; it is forwarded through
:func:`persist` keyword arguments, which providers receive untouched.
"""

import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .graph import ArrowDiGraph

__all__ = ["ParquetStorageProvider"]

MANIFEST_NAME = "manifest.json"
UNTAGGED_SLUG = "untagged"


def _slug(type_value, index, kind):
    if type_value is None:
        base = UNTAGGED_SLUG
    else:
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", str(type_value)).strip("._") or "type"
    return f"{kind}-{index:03d}-{base}.parquet"


def _unify(tables):
    """Concatenate tables with heterogeneous columns, filling gaps with nulls."""
    if len(tables) == 1:
        return tables[0]
    schema = pa.unify_schemas([t.schema for t in tables])
    return pa.concat_tables([t.cast(schema) for t in tables])


class ParquetStorageProvider:
    """Store per-type Arrow frames as Parquet files under a root directory."""

    def __init__(self, root):
        self.root = Path(root)

    # -- storage provider protocol (see nxarrow.backend.persist) --

    def store_nodes(self, frames, graph_attrs=None, root=None, **kwargs):
        """Write one Parquet file per node-type frame. Returns ``{file: rows}``."""
        return self._store("nodes", frames, root=root, graph_attrs=graph_attrs)

    def store_edges(self, frames, graph_attrs=None, root=None, **kwargs):
        """Write one Parquet file per edge-type frame. Returns ``{file: rows}``."""
        return self._store("edges", frames, root=root, graph_attrs=graph_attrs)

    # -- loading --

    def load_graph(self, root=None):
        """Rebuild the persisted graph (tables, types, attrs, graph attrs)."""
        root = Path(root) if root is not None else self.root
        manifest = json.loads((root / MANIFEST_NAME).read_text())
        node_tables = [pq.read_table(root / e["file"]) for e in manifest["nodes"]]
        edge_tables = [pq.read_table(root / e["file"]) for e in manifest["edges"]]
        graph = ArrowDiGraph.from_arrow(
            _unify(node_tables) if node_tables else None,
            _unify(edge_tables) if edge_tables else None,
        )
        graph.graph.update(manifest.get("graph", {}))
        return graph

    @classmethod
    def load(cls, root):
        """Rebuild the graph persisted under ``root``."""
        return cls(root).load_graph()

    # -- internals --

    def _store(self, kind, frames, root=None, graph_attrs=None):
        root = Path(root) if root is not None else self.root
        kind_dir = root / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        for stale in sorted(kind_dir.glob("*.parquet")):
            stale.unlink()
        entries, counts = [], {}
        manifest = self._read_manifest(root)
        for index, (type_value, table) in enumerate(frames.items()):
            if table.num_rows == 0:
                continue
            filename = _slug(type_value, index, kind)
            pq.write_table(table, kind_dir / filename)
            rel = f"{kind}/{filename}"
            entries.append({"type": type_value, "file": rel, "rows": table.num_rows})
            counts[rel] = table.num_rows
        manifest[kind] = entries
        if graph_attrs is not None:
            manifest["graph"] = dict(graph_attrs)
        self._write_manifest(root, manifest)
        return counts

    @staticmethod
    def _read_manifest(root):
        path = Path(root) / MANIFEST_NAME
        if path.exists():
            manifest = json.loads(path.read_text())
            manifest.setdefault("nodes", [])
            manifest.setdefault("edges", [])
            manifest.setdefault("graph", {})
            return manifest
        return {"version": 1, "graph": {}, "nodes": [], "edges": []}

    @staticmethod
    def _write_manifest(root, manifest):
        manifest["version"] = 1
        (Path(root) / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
