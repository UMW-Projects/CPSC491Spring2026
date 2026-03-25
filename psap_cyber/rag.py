"""
psap corpus rag: strict grounding only. separate from app/rag.py.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

EMBED_MODEL_DEFAULT = "text-embedding-3-small"


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"missing env var: {name}")
    return v


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    out: Dict[str, Any] = {}
    for k in ("id", "score", "metadata"):
        if hasattr(obj, k):
            out[k] = getattr(obj, k)
    return out


def _extract_json_object(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    first = t.find("{")
    last = t.rfind("}")
    if first != -1 and last != -1 and last > first:
        t = t[first : last + 1].strip()
    return t


def embed_text(client: OpenAI, text: str, embed_model: str) -> List[float]:
    return client.embeddings.create(model=embed_model, input=[text]).data[0].embedding


def retrieve(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    openai_key = _env("OPENAI_API_KEY")
    pinecone_key = _env("PINECONE_API_KEY")
    index_name = _env("PINECONE_INDEX_PSAP")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", EMBED_MODEL_DEFAULT)

    client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)
    qvec = embed_text(client, query, embed_model)

    res = index.query(
        vector=qvec,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
    )

    matches = getattr(res, "matches", None) or []
    out: List[Dict[str, Any]] = []
    for m in matches:
        mdict = _as_dict(m)
        md = mdict.get("metadata", {}) or {}
        out.append(
            {
                "id": mdict.get("id"),
                "score": float(mdict.get("score", 0.0)),
                "text": md.get("text", ""),
                "source": md.get("source", ""),
                "author": md.get("author", ""),
                "date": md.get("date", ""),
                "business_name": md.get("business_name", ""),
            }
        )
    return out


def build_messages(
    query: str,
    contexts: List[Dict[str, Any]],
    max_chunk_chars: int = 900,
) -> List[Dict[str, str]]:
    lines = []
    for c in contexts:
        txt = (c.get("text") or "").replace("\n", " ").strip()
        if len(txt) > max_chunk_chars:
            txt = txt[:max_chunk_chars].rstrip() + "…"
        meta = []
        if c.get("business_name"):
            meta.append(f"title={c['business_name']}")
        if c.get("author"):
            meta.append(f"publisher={c['author']}")
        if c.get("date"):
            meta.append(f"year={c['date']}")
        if c.get("source"):
            meta.append(f"ref={c['source'][:200]}")
        head = ", ".join(meta)
        lines.append(f"- [chunk_id={c['id']}] ({head}) score={c['score']:.3f} :: {txt}")

    ctx_block = "\n".join(lines)

    system = (
        "you are a research assistant for public safety answering point (psap) staff. "
        "you must answer using only the excerpt blocks labeled with chunk_id. "
        "you must not use outside knowledge, common sense security advice, or uncited inference. "
        "if the excerpts do not contain enough information, say so explicitly. "
        "respond with a single json object (no markdown fences)."
    )

    user = f"""
user question:
{query}

corpus excerpts (only valid evidence):
{ctx_block}

return only valid json with this schema:
{{
  "answer_summary": "string",
  "insufficient_evidence": true or false,
  "bullets": [
    {{"text": "string", "chunk_ids": ["string"]}}
  ],
  "citations": [
    {{"chunk_id": "string", "excerpt": "string"}}
  ]
}}

rules:
- every factual claim must cite at least one chunk_id in bullets or citations.
- excerpts in citations must be copied or tightly shortened from the excerpt text (<= 25 words).
- if you cannot ground the answer, set insufficient_evidence true, answer_summary must say the indexed corpus does not contain enough information, and bullets must be [].
- do not exceed 6 bullets.
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_grounded_response(
    query: str,
    top_k: int = 10,
    include_debug: bool = False,
) -> Dict[str, Any]:
    openai_key = _env("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=openai_key)

    contexts = retrieve(query, top_k=top_k)
    messages = build_messages(query, contexts)

    resp = client.responses.create(
        model=model,
        input=messages,
        text={"format": {"type": "json_object"}},
        temperature=0,
    )

    raw = (resp.output_text or "").strip()
    cooked = _extract_json_object(raw)

    try:
        out = json.loads(cooked)
    except json.JSONDecodeError as e:
        return {
            "error": "model did not return valid json",
            "raw_output": raw[:4000],
            "exception": str(e),
        }

    if include_debug:
        out["debug"] = {
            "retrieved": [
                {"id": c.get("id"), "score": c.get("score"), "source": c.get("source")}
                for c in contexts
            ],
            "top_k": top_k,
        }

    return out
