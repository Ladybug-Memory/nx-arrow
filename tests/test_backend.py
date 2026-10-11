import datetime

import pytest

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")

import networkx as nx
from nxarrow.backend import (
    backend_interface,
    get_storage_provider,
    list_storage_providers,
    persist,
    register_storage_provider,
    unregister_storage_provider,
)
from nxarrow.graph import ArrowDiGraph
from nxarrow.parquet import ParquetStorageProvider


@pytest.fixture
def typed_graph():
    g = ArrowDiGraph()
    g.add_node("alice", node_type="user", age=30)
    g.add_node("bob", node_type="user", age=25)
    g.add_node("post1", node_type="post", text="hello")
    g.add_edge("alice", "post1", edge_type="authored")
    g.add_edge("bob", "post1", edge_type="liked", weight=0.9)
    g.add_edge("alice", "bob", edge_type="follows")
    return g


def test_type_columns_and_filters(typed_graph):
    assert typed_graph.node_types == ["post", "user"]
    assert typed_graph.edge_types == ["authored", "follows", "liked"]
    assert typed_graph.nodes_of_type("user") == ["alice", "bob"]
    assert typed_graph.edges_of_type("liked") == [("bob", "post1")]
    assert typed_graph.node_type("alice") == "user"
    assert typed_graph.edge_type("bob", "post1") == "liked"
    assert typed_graph.nodes_table().schema.field("node_type").type == pa.string()
    assert typed_graph.edges_table().schema.field("edge_type").type == pa.string()


def test_heterogeneous_attrs_keep_types(typed_graph):
    schema = typed_graph.nodes_table().schema
    assert schema.field("age").type == pa.int64()
    assert schema.field("text").type == pa.string()


def test_removal_keeps_type_columns_aligned(typed_graph):
    typed_graph.remove_edge("alice", "bob")
    typed_graph.remove_node("bob")
    assert typed_graph.node_types == ["post", "user"]
    assert typed_graph.edge_types == ["authored"]
    assert typed_graph.nodes_table().num_rows == 2
    assert typed_graph.edges_table().num_rows == 1


def test_networkx_roundtrip_preserves_types(typed_graph):
    rt = ArrowDiGraph.from_networkx(typed_graph.to_networkx())
    assert rt.node_types == ["post", "user"]
    assert rt.edge_types == ["authored", "follows", "liked"]
    assert rt.nodes_of_type("post") == ["post1"]
    assert rt.node_attrs("alice")["node_type"] == "user"


def test_arrow_table_roundtrip_preserves_types(typed_graph):
    rt = ArrowDiGraph.from_arrow(typed_graph.nodes_table(), typed_graph.edges_table())
    assert rt.node_types == ["post", "user"]
    assert rt.edges_of_type("follows") == [("alice", "bob")]


def test_bulk_ingest_matches_incremental():
    edges = [(0, 1), (1, 2), (0, 2), (2, 0)]
    attrs = {0: {"color": "red"}, 1: {"color": "blue"}, 2: {}}

    incr = ArrowDiGraph()
    for n, d in attrs.items():
        incr.add_node(n, **d)
    for u, v in edges:
        incr.add_edge(u, v, weight=1.0)

    bulk = ArrowDiGraph()
    bulk.add_nodes_from(list(attrs.items()))
    bulk.add_edges_from([(u, v, {"weight": 1.0}) for u, v in edges])
    assert bulk.nodes_table().equals(incr.nodes_table())
    assert bulk.edges_table().equals(incr.edges_table())

    from_nx = ArrowDiGraph(nx.DiGraph(incr.to_networkx()))
    assert {r["node"] for r in from_nx.nodes_table().to_pylist()} == {
        r["node"] for r in incr.nodes_table().to_pylist()
    }
    assert {(r["source"], r["target"]) for r in from_nx.edges_table().to_pylist()} == {
        (r["source"], r["target"]) for r in incr.edges_table().to_pylist()
    }
    assert from_nx.node_attrs(0) == incr.node_attrs(0)


def test_type_columns_omitted_when_unused():
    g = ArrowDiGraph()
    g.add_node("a", color="red")
    g.add_edge("a", "b", weight=1.0)
    assert "node_type" not in g.nodes_table().column_names
    assert "edge_type" not in g.edges_table().column_names
    assert g.nodes_of_type("user") == []
    assert g.edges_of_type("likes") == []
    assert g.node_types == [] and g.edge_types == []
    # setting a type later materializes the column ...
    g.add_node("a", node_type="user")
    g.add_edge("a", "b", edge_type="likes")
    assert g.nodes_table().column("node_type").to_pylist() == ["user", None]
    assert g.edges_table().column("edge_type").to_pylist() == ["likes"]
    # ... and removing every typed entry drops it again
    g.remove_edge("a", "b")
    g.remove_node("a")
    assert "node_type" not in g.nodes_table().column_names
    assert "edge_type" not in g.edges_table().column_names


def test_storage_provider_hook(typed_graph):
    calls = []

    class RecordingProvider:
        def store_nodes(self, frames, **kwargs):
            calls.append(("nodes", sorted(frames), kwargs))
            return {k: t.num_rows for k, t in frames.items()}

        def store_edges(self, frames, **kwargs):
            calls.append(("edges", sorted(frames), kwargs))
            return {k: t.num_rows for k, t in frames.items()}

    register_storage_provider("recording", RecordingProvider())
    try:
        assert "recording" in list_storage_providers()
        assert get_storage_provider("recording") is not None
        counts = persist(typed_graph, "recording", tag="t")
        assert counts == {
            "user": 2,
            "post": 1,
            "authored": 1,
            "liked": 1,
            "follows": 1,
        }
        # nodes are handed over before edges, kwargs pass through
        assert [c[0] for c in calls] == ["nodes", "edges"]
        assert calls[0][1] == ["post", "user"]
        assert calls[1][1] == ["authored", "follows", "liked"]
        assert calls[0][2] == {"tag": "t"}
        # plain NetworkX graphs are converted first
        assert persist(typed_graph.to_networkx(), "recording") == counts
    finally:
        unregister_storage_provider("recording")
    assert "recording" not in list_storage_providers()


def test_storage_provider_hook_rejects_bad_providers():
    with pytest.raises(TypeError):
        register_storage_provider("bad", object())
    with pytest.raises(KeyError):
        get_storage_provider("no-such-provider")
    with pytest.raises(KeyError):
        unregister_storage_provider("no-such-provider")
    with pytest.raises(TypeError):
        persist(ArrowDiGraph(), object())


def test_storage_provider_hook_skips_empty_frames():
    seen = {}

    class RecordingProvider:
        def store_nodes(self, frames, **kwargs):
            seen["nodes"] = frames
            return {}

        def store_edges(self, frames, **kwargs):
            seen["edges"] = frames
            return {}

    persist(ArrowDiGraph(), RecordingProvider())
    assert seen == {"nodes": {}, "edges": {}}

    g = ArrowDiGraph()
    g.add_node("a", node_type="user")
    persist(g, RecordingProvider())
    assert list(seen["nodes"]) == ["user"]
    assert seen["edges"] == {}


def test_partition_tables_untyped():
    g = ArrowDiGraph()
    g.add_edge("a", "b")
    nodes, edges = g.partition_tables()
    assert list(nodes) == [None] and nodes[None].num_rows == 2
    assert list(edges) == [None] and edges[None].num_rows == 1
    assert "node" in nodes[None].column_names
    assert "source" in edges[None].column_names
    assert "target" in edges[None].column_names


def _assert_graphs_equal(original, restored):
    assert restored.node_types == original.node_types
    assert restored.edge_types == original.edge_types
    assert restored.nodes_table().num_rows == original.nodes_table().num_rows
    assert restored.edges_table().num_rows == original.edges_table().num_rows
    for node in original.nodes:
        assert restored.has_node(node)
        assert restored.node_attrs(node) == original.node_attrs(node)
    for u, v in original.edges:
        assert restored.has_edge(u, v)
        assert restored.edge_attrs(u, v) == original.edge_attrs(u, v)
    assert restored.graph == original.graph


def test_parquet_provider_roundtrip(tmp_path, typed_graph):
    typed_graph.graph["name"] = "roundtrip"
    provider = ParquetStorageProvider(tmp_path / "store")
    counts = persist(typed_graph, provider, graph_attrs=typed_graph.graph)
    assert sum(counts.values()) == 6
    assert (tmp_path / "store" / "manifest.json").exists()
    assert sorted((tmp_path / "store" / "nodes").glob("*.parquet")) != []

    restored = provider.load_graph()
    _assert_graphs_equal(typed_graph, restored)
    # per-type files: 2 node types + 3 edge types
    assert len(list((tmp_path / "store" / "nodes").glob("*.parquet"))) == 2
    assert len(list((tmp_path / "store" / "edges").glob("*.parquet"))) == 3


def test_parquet_provider_roundtrip_untyped_and_empty(tmp_path):
    provider = ParquetStorageProvider(tmp_path / "store")
    g = ArrowDiGraph()
    g.add_edge("a", "b", weight=1.5)
    persist(g, provider)
    restored = provider.load_graph()
    _assert_graphs_equal(g, restored)
    assert restored.nodes_table().column_names == ["node"]

    persist(ArrowDiGraph(), provider)
    empty = provider.load_graph()
    assert empty.number_of_nodes() == 0
    assert empty.number_of_edges() == 0


def test_parquet_provider_repersist_drops_stale_files(tmp_path, typed_graph):
    provider = ParquetStorageProvider(tmp_path / "store")
    persist(typed_graph, provider)
    assert len(list((tmp_path / "store" / "edges").glob("*.parquet"))) == 3

    typed_graph.remove_edge("alice", "bob")  # drops the only "follows" edge
    typed_graph.remove_edge("bob", "post1")  # drops the only "liked" edge
    persist(typed_graph, provider)
    assert len(list((tmp_path / "store" / "edges").glob("*.parquet"))) == 1
    _assert_graphs_equal(typed_graph, provider.load_graph())


def test_native_key_types():
    cases = [
        ([0, 1, 2], pa.int64()),
        ([0.5, 1.5, 2.5], pa.float64()),
        (["a", "b", "c"], pa.string()),
        (
            [
                datetime.date(2024, 1, 1),
                datetime.date(2024, 1, 2),
                datetime.date(2024, 1, 3),
            ],
            pa.date32(),
        ),
        (
            [datetime.time(1, 2), datetime.time(3, 4), datetime.time(5, 6)],
            pa.time64("us"),
        ),
        (
            [
                datetime.datetime(2024, 1, 1, 1, 2),
                datetime.datetime(2024, 1, 2),
                datetime.datetime(2024, 1, 3, 4, 5),
            ],
            pa.timestamp("us"),
        ),
    ]
    for keys, expected_type in cases:
        g = ArrowDiGraph()
        g.add_edge(keys[0], keys[1])
        g.add_edge(keys[1], keys[2])
        assert g.nodes_table().schema.field("node").type == expected_type
        assert g.edges_table().schema.field("source").type == expected_type
        assert g.nodes == keys
        assert g.has_node(keys[2])
        assert g.successors(keys[0]) == [keys[1]]
        assert g.has_edge(keys[1], keys[2])


def test_bool_keys_normalize_to_int():
    g = ArrowDiGraph()
    g.add_edge(True, False)
    assert g.nodes_table().schema.field("node").type == pa.int64()
    assert g.has_node(True) and g.has_node(1)  # NetworkX True == 1 semantics
    assert g.has_edge(True, False)


def test_explicit_key_type():
    g = ArrowDiGraph(key_type=pa.float32())
    g.add_edge(0.5, 1.5)
    assert g.nodes_table().schema.field("node").type == pa.float32()
    assert g.has_edge(0.5, 1.5)
    with pytest.raises(TypeError):
        ArrowDiGraph(key_type=pa.binary())


def test_mixed_keys_widen_or_fallback_to_string():
    g = ArrowDiGraph()
    g.add_node(1)
    g.add_edge(1, 2)
    g.add_edge(2.5, 3)  # int widens to float, preserving nodes
    assert g.nodes_table().schema.field("node").type == pa.float64()
    assert g.has_node(2) and g.has_node(2.0)
    assert g.nodes == [1, 2, 2.5, 3]

    g = ArrowDiGraph()
    g.add_node(1)
    g.add_node("1")  # old merge semantics: 1 and "1" collide
    assert g.nodes_table().schema.field("node").type == pa.string()
    assert g.number_of_nodes() == 1


def test_from_arrow_preserves_key_types():
    keys = {
        pa.int32(): [0, 1],
        pa.float32(): [0.5, 1.5],
        pa.date64(): [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
        pa.string(): ["a", "b"],
    }
    for arrow_type, values in keys.items():
        nodes = pa.table({"node": pa.array(values, type=arrow_type)})
        edges = pa.table(
            {
                "source": pa.array(values[:1], type=arrow_type),
                "target": pa.array(values[1:], type=arrow_type),
            }
        )
        g = ArrowDiGraph.from_arrow(nodes, edges)
        assert g.nodes_table().schema.field("node").type == arrow_type
        assert g.has_edge(values[0], values[1])
        assert g.nodes == values
    # binary key columns (no native key kind) decode to utf-8 strings
    nodes = pa.table({"node": pa.array([b"a", b"b"])})
    g = ArrowDiGraph.from_arrow(nodes, None)
    assert g.nodes_table().schema.field("node").type == pa.string()
    assert g.nodes == ["a", "b"]


def test_parquet_roundtrip_preserves_int_keys(tmp_path):
    g = ArrowDiGraph()
    g.add_edge(0, 1, weight=2.5)
    g.add_edge(1, 2)
    provider = ParquetStorageProvider(tmp_path / "store")
    persist(g, provider)
    restored = provider.load_graph()
    assert restored.nodes == [0, 1, 2]
    assert restored.edges == [(0, 1), (1, 2)]
    assert restored.edge_attrs(0, 1) == {"weight": 2.5}
    assert restored.nodes_table().schema.field("node").type == pa.int64()


def test_from_arrow_lazy_defers_index(typed_graph):
    nodes, edges = typed_graph.nodes_table(), typed_graph.edges_table()
    lazy = ArrowDiGraph.from_arrow(nodes, edges, index=False)
    assert lazy._index_built is False
    # table-level surface works without building the index ...
    assert lazy.number_of_nodes() == 3
    assert lazy.number_of_edges() == 3
    assert lazy.nodes_table().equals(nodes)
    assert lazy.edges_table().equals(edges)
    assert lazy.out_degree_table().num_rows == 2
    node_frames, edge_frames = lazy.partition_tables()
    assert sorted(node_frames) == ["post", "user"]
    assert sorted(edge_frames) == ["authored", "follows", "liked"]
    assert lazy._index_built is False
    # ... first Python-level touch builds it, with identical content
    assert lazy.nodes == ["alice", "bob", "post1"]
    assert lazy._index_built is True
    assert lazy.nodes_table().equals(nodes)
    assert lazy.node_attrs("alice") == typed_graph.node_attrs("alice")
    assert lazy.edge_attrs("bob", "post1") == typed_graph.edge_attrs(
        "bob", "post1"
    )


def test_lazy_graph_mutation_and_copy(typed_graph):
    nodes, edges = typed_graph.nodes_table(), typed_graph.edges_table()
    lazy = ArrowDiGraph.from_arrow(nodes, edges, index=False)
    lazy.add_node("carol", node_type="user", age=41)
    assert lazy._index_built is True
    assert lazy.has_node("carol")
    assert lazy.nodes_of_type("user") == ["alice", "bob", "carol"]
    clone = ArrowDiGraph(lazy)
    assert clone.nodes == lazy.nodes
    assert clone.edges_table().equals(lazy.edges_table())


def test_lazy_persist_roundtrip(tmp_path, typed_graph):
    nodes, edges = typed_graph.nodes_table(), typed_graph.edges_table()
    lazy = ArrowDiGraph.from_arrow(nodes, edges, index=False)
    provider = ParquetStorageProvider(tmp_path / "store")
    counts = persist(lazy, provider)
    assert lazy._index_built is False
    assert sum(counts.values()) == 6
    _assert_graphs_equal(typed_graph, provider.load_graph())


def test_backend_constructors():
    g = nx.Graph(backend="arrow")
    d = nx.DiGraph(backend="arrow")
    assert isinstance(g, ArrowDiGraph)
    assert isinstance(d, ArrowDiGraph)
    assert g.__networkx_backend__ == "arrow"

    e = nx.DiGraph([(1, 2), (2, 3)], backend="arrow")
    assert e.number_of_nodes() == 3
    assert e.number_of_edges() == 2

    f = nx.DiGraph(backend="arrow", name="foo")
    assert f.graph == {"name": "foo"}


def test_native_dispatch_matches_networkx(typed_graph):
    expected = typed_graph.to_networkx()
    assert nx.out_degree_centrality(typed_graph, backend="arrow") == (
        nx.out_degree_centrality(expected)
    )
    assert nx.in_degree_centrality(typed_graph, backend="arrow") == (
        nx.in_degree_centrality(expected)
    )
    assert list(nx.topological_sort(typed_graph, backend="arrow")) == list(
        nx.topological_sort(expected)
    )


def test_backend_interface_conversions(typed_graph):
    expected = typed_graph.to_networkx()
    converted = backend_interface.convert_from_nx(expected)
    assert isinstance(converted, ArrowDiGraph)
    assert converted.node_types == ["post", "user"]
    assert converted.to_networkx().nodes["alice"]["node_type"] == "user"
    assert backend_interface.convert_to_nx(converted).nodes["post1"]["node_type"] == (
        "post"
    )
