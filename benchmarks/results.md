# Arrow backend benchmark results

Command: `python benchmarks/bench_memory.py --n {5000,20000,200000}`
(date: 2026-09-30, Python 3.14.3, pyarrow 24.0.0)

Workload: N nodes with `color` (4 distinct strings) + `size` (int) attrs,
2N random edges with `weight` (float) attr; then 2000 incremental
single-edge writes. "DiGraph dicts" is a `sys.getsizeof` struct estimate
(`_node`/`_succ`/`_pred` + per-node/per-succ values); "Arrow tables" is
`nodes_table().nbytes + edges_table().nbytes`.

## n = 5000 (9999 edges)

| metric | Arrow (columnar) | DiGraph (dicts) |
|---|---|---|
| resident tables / struct | 0.36 MB | 2.39 MB (**~6.6x smaller**) |
| bulk build peak (tracemalloc) | 2.6 MB | 5.7 MB |
| bulk build time (incremental API) | 0.11 s | 0.02 s (~5x slower) |
| bulk load `ArrowDiGraph(nx_graph)` | 0.02 s | — (**7x faster than incremental**) |
| 2000 single-edge writes | 0.012 s | 0.001 s (~12x slower) |

## n = 20000 (39999 edges)

| metric | Arrow (columnar) | DiGraph (dicts) |
|---|---|---|
| resident tables / struct | 1.45 MB | 9.55 MB (**~6.6x smaller**) |
| bulk build peak (tracemalloc) | 10.6 MB | 23.0 MB |
| bulk build time (incremental API) | 0.84 s | 0.10 s (~8x slower) |
| bulk load `ArrowDiGraph(nx_graph)` | 0.07 s | — (**12x faster than incremental**) |
| 2000 single-edge writes | 0.039 s | 0.001 s (~39x slower) |

## n = 200000 (399996 edges)

| metric | Arrow (columnar) | DiGraph (dicts) |
|---|---|---|
| resident tables / struct | 14.50 MB | 109.16 MB (**~7.5x smaller**) |
| bulk build peak (tracemalloc) | 122.4 MB | 243.6 MB |
| bulk build time (incremental API) | 54.58 s | 1.21 s (~45x slower) |
| bulk load `ArrowDiGraph(nx_graph)` | 1.19 s | — (**46x faster than incremental, parity with dict build**) |
| 2000 single-edge writes | 0.340 s | 0.001 s (~340x slower) |

## Takeaway

- Reads/memory scale with the columnar layout (~6.6–7.5x smaller
  footprint, lower build peak) — this is what the `arrow` backend's
  native algorithms (`in/out_degree_centrality`, `topological_sort`)
  exploit. Key columns keep native Arrow types (`int64` here instead of
  strings, plus `float64`/`date32`/`time64`/`timestamp` where applicable;
  `node_type`/`edge_type` materialize only when used), so untyped graphs
  pay nothing for either feature.
- Bulk loads (`ArrowDiGraph(nx_graph)`, `add_nodes_from`/`add_edges_from`,
  `from_arrow`) merge in list-level passes and build each table once:
  11–64x faster than the incremental API, and at 200k faster than the
  dict build itself (0.82s vs 1.12s). One caveat: edge order follows the
  input iteration order, so an incrementally built graph and a
  bulk-loaded `DiGraph` hold the same edge *set* in different row order.
- Incremental writes stay dict-backed territory (bulk mutation is 4–175x
  slower). Write-heavy
  path: `gd = g.as_writeable()` → mutate → `ArrowDiGraph.from_networkx(gd)`,
  or set `NETWORKX_FALLBACK_TO_NX=True` and let dispatch convert back.
