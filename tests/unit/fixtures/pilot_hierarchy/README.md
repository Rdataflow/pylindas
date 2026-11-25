# pilot_hierarchy fixtures

Frozen copy of what the notebook `Varia/showcase_cube_v1_hierarchy.ipynb`
(Base-URI `https://environment.ld.admin.ch/foen`, identifier
`wood-showcase-hierarchy`, version 1) produced. The tests never read `Varia/`.

| File | Source in the notebook |
| --- | --- |
| `description.yaml` | cell 11 (`DESCRIPTION`, tokens substituted). Differences: hierarchy `root` is the concept ID `total` instead of the absolute IRI (Req. 4.3 resolves it); `Concepts.<dim>.parent-field: parent` added (Req. 3.x). |
| `assortment.csv`, `origin.csv` | cell 7 (`CONCEPTS`): `id`, labels de/fr/it/en, plus a `parent` column taken from each entry's `parent` (empty = root). |
| `data.csv` | cell 9 shape (`year, assortment, origin, quantity`). Synthetic: the original `table-data.csv` is not in `Varia/`; one year, every assortment x origin combination, constant quantity. Values do not influence hierarchy triples. |
| `expected_edges.ttl` | cell 13: the manual `cube._graph.add((parent, SCHEMA.hasPart, child))` loop; the cell output reports 6 triples, and cell 16 output shows the `forest-wood -> hardwood, softwood` links. |
| `expected_roots.json` | cell 11 output ("Hierarchy roots") and cell 14 output (roots `assortment/total`, `origin/total`). |
