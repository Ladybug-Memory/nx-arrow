"""Benchmark: ArrowDiGraph (columnar) vs DiGraph (dict-backed).

Run:  python benchmarks/bench_arrow_memory.py [--n 20000 --seed 0]

Compares resident columnar bytes (Arrow ``Table.nbytes``) against an
estimate of the dict-backed ``DiGraph`` footprint, plus single-edge write
throughput. Shows where Arrow wins (memory + bulk reads) and where the
dict fallback wins (incremental writes).
"""

import argparse
import sys
import time
import tracemalloc

import networkx as nx
from nxarrow.graph import ArrowDiGraph


def build_pair(n, seed=0):
    import random

    rng = random.Random(seed)
    edges = [(rng.randrange(n), (i * 7 + 1) % n) for i in range(n * 2)]
    colors = ["red", "green", "blue", "yellow"]

    t0 = time.perf_counter()
    tracemalloc.start()
    a = ArrowDiGraph()
    for i in range(n):
        a.add_node(i, color=colors[i % 4], size=i % 100)
    for u, v in edges:
        a.add_edge(u, v, weight=float(u % 10))
    _, peak_arrow = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t_arrow = time.perf_counter() - t0

    t0 = time.perf_counter()
    tracemalloc.start()
    d = nx.DiGraph()
    for i in range(n):
        d.add_node(i, color=colors[i % 4], size=i % 100)
    for u, v in edges:
        d.add_edge(u, v, weight=float(u % 10))
    _, peak_dict = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t_dict = time.perf_counter() - t0

    t0 = time.perf_counter()
    bulk = ArrowDiGraph(d)
    t_bulk = time.perf_counter() - t0
    assert bulk.number_of_nodes() == a.number_of_nodes()
    assert bulk.number_of_edges() == a.number_of_edges()

    arrow_bytes = a.nodes_table().nbytes + a.edges_table().nbytes
    dict_bytes = sys.getsizeof(d._node) + sys.getsizeof(d._succ) + sys.getsizeof(
        d._pred
    ) + sum(
        sys.getsizeof(v) for v in list(d._node.values()) + list(d._succ.values())
    )
    return a, d, {
        "arrow_table_bytes": arrow_bytes,
        "dict_struct_bytes": dict_bytes,
        "peak_arrow_build": peak_arrow,
        "peak_dict_build": peak_dict,
        "build_arrow_s": t_arrow,
        "build_dict_s": t_dict,
        "bulk_arrow_s": t_bulk,
    }


def bench_writes(g_arrow, g_dict, n, n_ops=2000):
    # Incremental single-edge writes with type-consistent int keys, so the
    # Arrow graph stays on its fast path (no key-type migration).
    base = 10 * n
    t0 = time.perf_counter()
    for i in range(n_ops):
        g_arrow.add_edge(base + i, base + i + 1, weight=1.0)
    t_arrow = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(n_ops):
        g_dict.add_edge(base + i, base + i + 1, weight=1.0)
    t_dict = time.perf_counter() - t0
    return t_arrow, t_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a, d, m = build_pair(args.n, args.seed)
    ratio = m["dict_struct_bytes"] / max(m["arrow_table_bytes"], 1)
    print(f"nodes={a.number_of_nodes()} edges={a.number_of_edges()}")
    print(f"Arrow tables : {m['arrow_table_bytes'] / 1e6:.2f} MB (columnar)")
    print(f"DiGraph dicts: {m['dict_struct_bytes'] / 1e6:.2f} MB (struct estimate)")
    print(f"~{ratio:.1f}x smaller columnar footprint")
    print(f"bulk build peak: arrow {m['peak_arrow_build'] / 1e6:.1f}MB "
          f"vs dict {m['peak_dict_build'] / 1e6:.1f}MB")
    print(f"bulk build time: arrow {m['build_arrow_s']:.2f}s vs dict {m['build_dict_s']:.2f}s")
    print(f"bulk load ArrowDiGraph(nx_graph): {m['bulk_arrow_s']:.2f}s "
          f"({m['build_arrow_s'] / max(m['bulk_arrow_s'], 1e-9):.1f}x faster than incremental)")
    tw_arrow, tw_dict = bench_writes(a, d, args.n)
    print(f"single-edge writes (2000): arrow {tw_arrow:.3f}s vs dict {tw_dict:.3f}s")
    if tw_arrow > tw_dict:
        print("-> write-heavy? fall back: g_dict = g_arrow.as_writeable(); "
              "..mutate..; ArrowDiGraph.from_networkx(g_dict)")


if __name__ == "__main__":
    main()
