# Warnaar Knowledge Graph

Knowledge graph built from the Warnaar Conjecture 2.7 literature corpus (cylindric partitions, q-series, positivity).

## How it works

`build_graph.py` reads chunk summaries from the literature corpus, sends them to Claude for entity/relation extraction, assembles a unified graph, and writes `graph.json`. `index.html` renders the result as an interactive visualization.

```bash
python build_graph.py            # full build
python build_graph.py --dry-run  # show what would be sent, no API calls
python build_graph.py --resume   # skip papers already extracted
open index.html
```

## License

MIT — see [LICENSE](LICENSE).
