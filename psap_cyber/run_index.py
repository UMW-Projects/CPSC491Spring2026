#!/usr/bin/env python3
"""
runs root scripts/03_index_pinecone.py with PINECONE_INDEX taken from PINECONE_INDEX_PSAP
so you do not have to edit the original indexer or your review-bot .env.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    indexer = root / "scripts" / "03_index_pinecone.py"
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chunks",
        type=Path,
        default=Path("doc/911-sources/extracted/chunks.jsonl"),
    )
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--namespace", default="")
    args = ap.parse_args()

    env = os.environ.copy()
    psap = env.get("PINECONE_INDEX_PSAP", "").strip()
    if psap:
        env["PINECONE_INDEX"] = psap
    elif not env.get("PINECONE_INDEX", "").strip():
        print("set PINECONE_INDEX_PSAP or PINECONE_INDEX in .env", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        str(indexer),
        "--chunks",
        str(args.chunks),
    ]
    if args.reset:
        cmd.append("--reset")
    if args.namespace:
        cmd.extend(["--namespace", args.namespace])

    subprocess.run(cmd, cwd=str(root), env=env, check=True)


if __name__ == "__main__":
    main()
