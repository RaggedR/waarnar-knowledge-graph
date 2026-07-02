#!/usr/bin/env python3
"""Build a knowledge graph from the waarnar literature corpus.

Reads chunk summaries from literature/chunks/, sends them to Claude
for entity extraction, assembles a unified graph, and writes graph.json.

Usage:
    python build_graph.py                # full build
    python build_graph.py --dry-run      # show what would be sent, no API calls
    python build_graph.py --resume       # skip papers already in raw_extractions/
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import anthropic

# ── Paths ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
CHUNKS_DIR = ROOT.parent / "literature" / "chunks"
RAW_DIR = ROOT / "raw_extractions"
GRAPH_FILE = ROOT / "graph.json"

SUMMARY_RE = re.compile(r"^\[Summary:\s*(.+)\]$", re.MULTILINE)
SECTION_RE = re.compile(r"^% Section:\s*(.+)$", re.MULTILINE)

# ── Extraction prompt ────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a mathematical knowledge graph extractor. Given summaries of chunks \
from a mathematics paper, extract structured entities and relationships.

Return valid JSON with exactly this shape:
{
  "concepts": ["cylindric partitions", "Hall-Littlewood polynomials", ...],
  "results": ["Andrews-Gordon identities", "Conjecture 2.7", ...],
  "authors": ["Warnaar", "Corteel", ...],
  "edges": [
    {"source": "...", "target": "...", "relation": "..."},
    ...
  ]
}

Guidelines:
- **concepts**: mathematical objects, structures, techniques (e.g. "Macdonald polynomials", \
"cylindric partitions", "Rogers-Ramanujan identities", "virtual Koornwinder integrals"). \
Use lowercase except for proper names. Normalise: prefer the most standard name.
- **results**: named theorems, conjectures, identities, lemmas that are STATED or PROVED \
in this paper (not just referenced in passing). Include the name as commonly cited.
- **authors**: surnames of the paper's authors only (not cited authors).
- **edges**: relationships between ANY entities above. The relation field should be one of: \
"proves", "conjectures", "generalises", "uses", "specialises_to", "is_instance_of", \
"related_to". Source and target must be exact strings from the concepts/results lists.

Be precise. Only include entities that are genuinely central to the paper, not every \
term mentioned in passing. Aim for 5-30 concepts and 2-15 results per paper."""

USER_TEMPLATE = """\
Paper: {paper_name}

Here are the section-by-section summaries of all chunks in this paper:

{summaries}

Extract the knowledge graph entities and relationships."""


# ── Summary collection ───────────────────────────────────────────────

def collect_paper_summaries() -> dict[str, str]:
    """Collect summaries from all chunk files, grouped by paper.

    Returns {paper_name: formatted_summary_text}.
    """
    papers: dict[str, list[str]] = {}

    for paper_dir in sorted(CHUNKS_DIR.iterdir()):
        if not paper_dir.is_dir():
            continue

        lines = []
        for chunk_file in sorted(paper_dir.glob("chunk_*.tex")):
            text = chunk_file.read_text(errors="replace")

            section = ""
            summary = ""
            for m in SECTION_RE.finditer(text):
                section = m.group(1).strip()
            for m in SUMMARY_RE.finditer(text):
                summary = m.group(1).strip()

            if summary:
                prefix = f"[{section}] " if section else ""
                lines.append(f"- {prefix}{summary}")

        if lines:
            papers[paper_dir.name] = "\n".join(lines)

    return papers


# ── API calls ────────────────────────────────────────────────────────

async def extract_one(
    client: anthropic.AsyncAnthropic,
    paper_name: str,
    summaries: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Extract entities from one paper's summaries."""
    async with semaphore:
        user_msg = USER_TEMPLATE.format(
            paper_name=paper_name, summaries=summaries
        )
        try:
            resp = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = resp.content[0].text
            # Extract JSON from response (may be wrapped in ```json ... ```)
            text = re.sub(r"^```json\s*", "", text.strip())
            text = re.sub(r"\s*```$", "", text.strip())
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"  WARN {paper_name}: bad JSON: {e}")
            # Save raw response for debugging
            (RAW_DIR / f"{paper_name}.raw.txt").write_text(text)
            return None
        except Exception as e:
            print(f"  ERROR {paper_name}: {e}")
            return None


async def extract_all(
    papers: dict[str, str],
    skip_existing: bool = False,
    concurrency: int = 5,
) -> dict[str, dict]:
    """Extract entities from all papers concurrently."""
    RAW_DIR.mkdir(exist_ok=True)
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, dict] = {}

    # Load existing extractions if resuming
    if skip_existing:
        for f in RAW_DIR.glob("*.json"):
            name = f.stem
            results[name] = json.loads(f.read_text())
            print(f"  CACHED {name}")

    # Filter to papers not yet extracted
    todo = {k: v for k, v in papers.items() if k not in results}
    print(f"\nExtracting {len(todo)} papers ({len(results)} cached)...")

    async def do_one(name: str, sums: str):
        result = await extract_one(client, name, sums, semaphore)
        if result:
            results[name] = result
            (RAW_DIR / f"{name}.json").write_text(
                json.dumps(result, indent=2)
            )
            n_concepts = len(result.get("concepts", []))
            n_results = len(result.get("results", []))
            print(f"  OK {name}: {n_concepts} concepts, {n_results} results")

    tasks = [do_one(name, sums) for name, sums in todo.items()]
    await asyncio.gather(*tasks)

    return results


# ── Graph assembly ───────────────────────────────────────────────────

def assemble_graph(extractions: dict[str, dict]) -> dict:
    """Merge per-paper extractions into a unified knowledge graph."""
    nodes: dict[str, dict] = {}  # id -> {id, label, type, papers: [...]}
    edges: list[dict] = []

    def node_id(label: str, node_type: str) -> str:
        """Normalise a label into a stable node ID."""
        return f"{node_type}:{label.lower().strip()}"

    def ensure_node(label: str, node_type: str, paper: str) -> str:
        nid = node_id(label, node_type)
        if nid not in nodes:
            nodes[nid] = {
                "id": nid,
                "label": label,
                "type": node_type,
                "papers": [],
            }
        if paper not in nodes[nid]["papers"]:
            nodes[nid]["papers"].append(paper)
        return nid

    for paper_name, ext in extractions.items():
        # Paper node
        paper_nid = ensure_node(paper_name, "paper", paper_name)

        # Author nodes + edges
        for author in ext.get("authors", []):
            author_nid = ensure_node(author, "author", paper_name)
            edges.append({
                "source": author_nid,
                "target": paper_nid,
                "relation": "authored",
            })

        # Concept nodes + edges
        for concept in ext.get("concepts", []):
            concept_nid = ensure_node(concept, "concept", paper_name)
            edges.append({
                "source": paper_nid,
                "target": concept_nid,
                "relation": "discusses",
            })

        # Result nodes + edges
        for result in ext.get("results", []):
            result_nid = ensure_node(result, "result", paper_name)
            edges.append({
                "source": paper_nid,
                "target": result_nid,
                "relation": "establishes",
            })

        # Inter-entity edges from extraction
        for edge in ext.get("edges", []):
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            rel = edge.get("relation", "related_to")

            # Find which type the source/target belongs to
            src_nid = None
            tgt_nid = None
            for node_type in ["concept", "result"]:
                if node_id(src, node_type) in nodes:
                    src_nid = node_id(src, node_type)
                if node_id(tgt, node_type) in nodes:
                    tgt_nid = node_id(tgt, node_type)

            if src_nid and tgt_nid:
                edges.append({
                    "source": src_nid,
                    "target": tgt_nid,
                    "relation": rel,
                })

    # Deduplicate edges
    seen = set()
    unique_edges = []
    for e in edges:
        key = (e["source"], e["target"], e["relation"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    graph = {
        "nodes": list(nodes.values()),
        "edges": unique_edges,
        "meta": {
            "papers": len(extractions),
            "total_nodes": len(nodes),
            "total_edges": len(unique_edges),
        },
    }
    return graph


# ── CLI ──────────────────────────────────────────────────────────────

async def main_async():
    parser = argparse.ArgumentParser(description="Build waarnar knowledge graph")
    parser.add_argument("--dry-run", action="store_true", help="No API calls")
    parser.add_argument("--resume", action="store_true", help="Skip cached papers")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    print("Collecting summaries from chunks...")
    papers = collect_paper_summaries()
    print(f"Found {len(papers)} papers\n")

    if args.dry_run:
        for name, sums in papers.items():
            n_lines = sums.count("\n") + 1
            n_chars = len(sums)
            print(f"  {name}: {n_lines} summaries, {n_chars:,} chars")
        total = sum(len(s) for s in papers.values())
        print(f"\nTotal: {total:,} chars across {len(papers)} papers")
        return

    extractions = await extract_all(
        papers, skip_existing=args.resume, concurrency=args.concurrency
    )

    print(f"\nAssembling graph from {len(extractions)} papers...")
    graph = assemble_graph(extractions)
    GRAPH_FILE.write_text(json.dumps(graph, indent=2))
    print(
        f"Written {GRAPH_FILE}: "
        f"{graph['meta']['total_nodes']} nodes, "
        f"{graph['meta']['total_edges']} edges"
    )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
