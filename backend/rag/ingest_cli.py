#!/usr/bin/env python3
"""
rag/ingest_cli.py
=================
CLI tool to ingest documents into the global knowledge base.

Examples:
    # Ingest a single PDF
    python -m rag.ingest_cli --file docs/manual.pdf

    # Ingest an entire directory
    python -m rag.ingest_cli --dir data/knowledge/

    # Ingest a web page (replace with a real URL)
    python -m rag.ingest_cli --url https://example.org/faq

    # Ingest inline text
    python -m rag.ingest_cli --text "Our return policy is 30 days." --source "return_policy"

    # Show knowledge base stats
    python -m rag.ingest_cli --stats

    # Clear the knowledge base (DESTRUCTIVE)
    python -m rag.ingest_cli --clear
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the backend package root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.knowledge_base import KnowledgeBase
from rag.ingestion import DocumentIngestionPipeline


def _print_banner():
    print("\n" + "═" * 55)
    print("   📚  RAG Knowledge Base Ingestion Tool")
    print("═" * 55)


def cmd_stats(kb: KnowledgeBase) -> None:
    stats = kb.stats()
    print("\n📊  Knowledge Base Stats")
    print(f"   Total vectors  : {stats['total_vectors']}")
    print(f"   Total sources  : {stats['total_sources']}")
    print(f"   Index type     : {stats['index_type']}")
    if stats["ingested_sources"]:
        print("\n   Ingested sources:")
        for src in stats["ingested_sources"]:
            print(f"     • {src}")
    else:
        print("   (empty — no documents ingested yet)")
    print()


def cmd_ingest_file(kb: KnowledgeBase, pipeline: DocumentIngestionPipeline, path: str) -> None:
    p = Path(path)
    source = p.name

    if kb.is_ingested(source):
        print(f"⚠️   {source!r} already ingested (use --force to re-ingest).")
        return

    print(f"📄  Ingesting file: {p}")
    chunks = pipeline.ingest_file(p)
    added  = kb.add_chunks(chunks)
    kb.register_source(source)
    print(f"✅  Added {added} chunks from {source!r}")


def cmd_ingest_dir(kb: KnowledgeBase, pipeline: DocumentIngestionPipeline, directory: str) -> None:
    d = Path(directory)
    print(f"📁  Ingesting directory: {d}")
    results = pipeline.ingest_directory(d)
    total_added = 0
    for filename, chunks in results.items():
        if kb.is_ingested(filename):
            print(f"   ⏭   Skipping {filename!r} (already ingested)")
            continue
        added = kb.add_chunks(chunks)
        kb.register_source(filename)
        total_added += added
        print(f"   ✅  {filename}: {added} chunks")
    print(f"\n📊  Total chunks added: {total_added}")


def cmd_ingest_url(kb: KnowledgeBase, pipeline: DocumentIngestionPipeline, url: str) -> None:
    if kb.is_ingested(url):
        print(f"⚠️   {url!r} already ingested.")
        return
    print(f"🌐  Fetching URL: {url}")
    chunks = pipeline.ingest_url(url)
    added  = kb.add_chunks(chunks)
    kb.register_source(url)
    print(f"✅  Added {added} chunks from {url!r}")


def cmd_ingest_text(kb: KnowledgeBase, pipeline: DocumentIngestionPipeline, text: str, source: str) -> None:
    chunks = pipeline.ingest_text(text, source=source)
    added  = kb.add_chunks(chunks)
    kb.register_source(source)
    print(f"✅  Added {added} chunks from inline text (source={source!r})")


def cmd_clear(kb: KnowledgeBase) -> None:
    confirm = input("⚠️   This will DELETE all vectors. Type 'yes' to confirm: ").strip()
    if confirm.lower() == "yes":
        kb.clear()
        print("✅  Knowledge base cleared.")
    else:
        print("Aborted.")


def main() -> None:
    _print_banner()

    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file",   metavar="PATH", help="Ingest a single file")
    parser.add_argument("--dir",    metavar="DIR",  help="Ingest all files in a directory")
    parser.add_argument("--url",    metavar="URL",  help="Ingest a web page")
    parser.add_argument("--text",   metavar="TEXT", help="Ingest inline text")
    parser.add_argument("--source", metavar="NAME", default="inline", help="Source label for --text")
    parser.add_argument("--stats",  action="store_true", help="Show knowledge base stats")
    parser.add_argument("--clear",  action="store_true", help="Clear all vectors (DESTRUCTIVE)")
    parser.add_argument(
        "--chunk-size", type=int, default=400, metavar="N",
        help="Target chunk size in characters (default: 400)"
    )
    parser.add_argument(
        "--overlap", type=int, default=80, metavar="N",
        help="Chunk overlap in characters (default: 80)"
    )

    args = parser.parse_args()

    if not any([args.file, args.dir, args.url, args.text, args.stats, args.clear]):
        parser.print_help()
        return

    print("🔄  Loading knowledge base...")
    kb       = KnowledgeBase()
    pipeline = DocumentIngestionPipeline(chunk_size=args.chunk_size, overlap=args.overlap)

    if args.stats:
        cmd_stats(kb)

    if args.clear:
        cmd_clear(kb)
        return

    if args.file:
        cmd_ingest_file(kb, pipeline, args.file)

    if args.dir:
        cmd_ingest_dir(kb, pipeline, args.dir)

    if args.url:
        cmd_ingest_url(kb, pipeline, args.url)

    if args.text:
        cmd_ingest_text(kb, pipeline, args.text, args.source)

    # Always show updated stats after ingestion
    if any([args.file, args.dir, args.url, args.text]):
        print()
        cmd_stats(kb)

    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()
