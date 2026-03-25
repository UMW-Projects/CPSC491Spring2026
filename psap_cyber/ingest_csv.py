#!/usr/bin/env python3
"""
read brown source csv -> one jsonl document per row with extracted text.
- refs starting with http: fetch (pdf bytes -> pypdf; else html -> visible text)
- otherwise: treat ref as filename under --pdf-dir

does not modify any other package in this repo.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

DEFAULT_UA = (
    "Mozilla/5.0 (compatible; PSAP-Corpus-Bot/1.0; +https://example.local; university research)"
)


def _norm_url(url: str) -> str:
    u = (url or "").strip()
    if "%25" in u:
        u = unquote(u)
    return u


def _is_probably_pdf_url(url: str) -> bool:
    u = url.lower().split("?", 1)[0]
    return u.endswith(".pdf")


def _extract_pdf_text(data: bytes) -> str:
    from io import BytesIO

    reader = PdfReader(BytesIO(data))
    parts: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts).strip()


def _extract_html_text(data: bytes, encoding: str = "utf-8") -> str:
    raw = data.decode(encoding, errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def fetch_reference(ref: str, session: requests.Session, timeout: int) -> tuple[str, Optional[str]]:
    """
    returns (text, error_message). text may be empty on failure.
    """
    url = _norm_url(ref)
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").lower()
        data = r.content
        if "pdf" in ctype or _is_probably_pdf_url(url) or data[:5] == b"%PDF-":
            try:
                return _extract_pdf_text(data), None
            except Exception as e:
                return "", f"pdf_parse_error: {e}"
        enc = r.encoding or "utf-8"
        return _extract_html_text(data, encoding=enc), None
    except Exception as e:
        return "", f"fetch_error: {e}"


def load_local_pdf(path: Path) -> tuple[str, Optional[str]]:
    if not path.is_file():
        return "", f"missing_file: {path}"
    try:
        data = path.read_bytes()
        return _extract_pdf_text(data), None
    except Exception as e:
        return "", f"pdf_read_error: {e}"


def parse_sources_row(row: List[str], fallback_id: str) -> Optional[Dict[str, str]]:
    if not row or len(row) < 3:
        return None
    rid = (row[0] or "").strip() or fallback_id
    title = (row[1] or "").strip()
    ref = (row[2] or "").strip()
    year = (row[3] or "").strip() if len(row) > 3 else ""
    org = (row[4] or "").strip() if len(row) > 4 else ""
    desc = (row[5] or "").strip() if len(row) > 5 else ""
    if not ref and not title:
        return None
    return {
        "doc_id": rid,
        "title": title,
        "reference": ref,
        "year": year,
        "organization": org,
        "description": desc,
    }


def run(csv_path: Path, pdf_dir: Path, out_jsonl: Path, delay_s: float, timeout: int) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_UA})

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    written = 0
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for i, row in enumerate(rows, start=1):
            meta = parse_sources_row(row, fallback_id=str(i))
            if not meta:
                continue
            ref = meta["reference"]
            text = ""
            err: Optional[str] = None

            if ref.lower().startswith("http://") or ref.lower().startswith("https://"):
                text, err = fetch_reference(ref, session, timeout=timeout)
                time.sleep(delay_s)
            else:
                text, err = load_local_pdf(pdf_dir / ref)

            header_bits = [
                meta["title"],
                meta["organization"],
                meta["year"],
                meta["description"],
            ]
            preamble = "\n".join(b for b in header_bits if b).strip()
            full_text = (preamble + "\n\n" + text).strip() if text else preamble

            rec: Dict[str, Any] = {
                **meta,
                "text": full_text if not err else preamble,
                "ok": bool(text and not err),
                "error": err,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} records -> {out_jsonl}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("doc/911-sources/Sources from Brown(Sheet1).csv"),
    )
    ap.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("doc/911-sources/pdfs"),
        help="directory containing pdf files named like the csv reference column",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("doc/911-sources/extracted/documents.jsonl"),
    )
    ap.add_argument("--delay", type=float, default=0.6, help="seconds between http requests")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()
    run(args.csv, args.pdf_dir, args.out, delay_s=args.delay, timeout=args.timeout)


if __name__ == "__main__":
    main()
