#!/usr/bin/env python3
"""
documents.jsonl (from ingest_csv.py) -> chunks.jsonl shaped for scripts/03_index_pinecone.py
without editing that script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import tiktoken
from dotenv import load_dotenv

load_dotenv()


def chunk_tokens(text: str, max_tokens: int, overlap: int, enc) -> List[str]:
    toks = enc.encode(text or "")
    chunks: List[str] = []
    start = 0
    while start < len(toks):
        end = min(start + max_tokens, len(toks))
        chunk = enc.decode(toks[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(toks):
            break
        start = max(0, end - overlap)
    return chunks


def main(inp: Path, out: Path, max_tokens: int, overlap: int) -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    n = 0
    with open(inp, "r", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = str(doc.get("doc_id") or "").strip() or "unknown"
            ref = (doc.get("reference") or "").strip()
            body = (doc.get("text") or "").strip()
            if not body:
                continue

            pieces = chunk_tokens(body, max_tokens=max_tokens, overlap=overlap, enc=enc)
            for j, ch in enumerate(pieces):
                rid = f"psap-{doc_id}"
                rec: Dict[str, Any] = {
                    "id": f"{rid}-c{j:03d}",
                    "review_id": rid,
                    "chunk_id": j,
                    "text": ch,
                    "token_count": len(enc.encode(ch)),
                    "char_count": len(ch),
                    "rating": None,
                    "date": (doc.get("year") or "") or "",
                    "author": (doc.get("organization") or "") or "",
                    "is_owner_response": False,
                    "is_owner_response_text_detected": False,
                    "parent_review_id": None,
                    "business_name": doc.get("title") or "",
                    "business_location": None,
                    "source": ref[:512] if ref else "",
                    "themes": [],
                    "theme_hits": [],
                    "sentiment": {"compound": 0.0, "label": "neutral"},
                    "sentiment_label": "neutral",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1

    print(f"chunks written: {n} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="inp",
        type=Path,
        default=Path("doc/911-sources/extracted/documents.jsonl"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("doc/911-sources/extracted/chunks.jsonl"),
    )
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=60)
    args = ap.parse_args()
    main(args.inp, args.out, args.max_tokens, args.overlap)
