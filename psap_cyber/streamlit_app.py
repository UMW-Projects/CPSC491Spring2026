# psap / 911 cybersecurity corpus chat — separate from root streamlit_app.py
import json
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from psap_cyber.rag import generate_grounded_response

st.set_page_config(
    page_title="PSAP cyber corpus assistant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_state() -> None:
    if "psap_conversations" not in st.session_state:
        st.session_state.psap_conversations = {}
    if "psap_active_chat_id" not in st.session_state:
        st.session_state.psap_active_chat_id = None
    if "psap_settings" not in st.session_state:
        st.session_state.psap_settings = {"top_k": 12, "debug_on": False}


def _new_chat() -> None:
    chat_id = str(uuid.uuid4())
    st.session_state.psap_conversations[chat_id] = {
        "title": "new chat",
        "messages": [],
        "created": time.time(),
    }
    st.session_state.psap_active_chat_id = chat_id


def _ensure_active_chat() -> None:
    if not st.session_state.psap_conversations:
        _new_chat()
        return
    aid = st.session_state.psap_active_chat_id
    if aid not in st.session_state.psap_conversations:
        newest = max(
            st.session_state.psap_conversations.items(),
            key=lambda kv: kv[1]["created"],
        )[0]
        st.session_state.psap_active_chat_id = newest


def _set_title_from_first_user_message(chat_id: str) -> None:
    convo = st.session_state.psap_conversations[chat_id]
    if convo["title"] != "new chat":
        return
    for m in convo["messages"]:
        if m["role"] == "user" and m["content"].strip():
            t = m["content"].strip()
            convo["title"] = t[:40] + ("…" if len(t) > 40 else "")
            return


def _delete_chat(chat_id: str) -> None:
    st.session_state.psap_conversations.pop(chat_id, None)
    if not st.session_state.psap_conversations:
        _new_chat()
    else:
        newest = max(
            st.session_state.psap_conversations.items(),
            key=lambda kv: kv[1]["created"],
        )[0]
        st.session_state.psap_active_chat_id = newest


def _sorted_chat_ids_newest_first():
    items = sorted(
        st.session_state.psap_conversations.items(),
        key=lambda kv: kv[1]["created"],
        reverse=True,
    )
    return [cid for cid, _ in items]


_init_state()
if st.session_state.psap_active_chat_id is None:
    _new_chat()
_ensure_active_chat()

with st.sidebar:
    st.header("conversations")
    if st.button("new chat", use_container_width=True):
        _new_chat()
        st.rerun()
    st.divider()
    chat_ids = _sorted_chat_ids_newest_first()
    active = st.session_state.psap_active_chat_id
    selected = st.radio(
        "history",
        options=chat_ids,
        index=chat_ids.index(active) if active in chat_ids else 0,
        format_func=lambda cid: st.session_state.psap_conversations[cid]["title"],
        label_visibility="collapsed",
    )
    st.session_state.psap_active_chat_id = selected
    _ensure_active_chat()
    st.divider()
    convo = st.session_state.psap_conversations[st.session_state.psap_active_chat_id]
    with st.expander("rename / delete", expanded=False):
        new_title = st.text_input("title", value=convo["title"])
        c1, c2 = st.columns(2)
        with c1:
            if st.button("save", use_container_width=True):
                convo["title"] = new_title.strip() or "untitled"
                st.rerun()
        with c2:
            if st.button("delete", use_container_width=True):
                _delete_chat(st.session_state.psap_active_chat_id)
                st.rerun()

st.title("psap cybersecurity corpus assistant")
st.caption(
    "answers use only text indexed from your psap source list. "
    "not legal or operational dispatch advice."
)

cid = st.session_state.psap_active_chat_id
convo = st.session_state.psap_conversations[cid]

with st.expander("settings", expanded=False):
    st.session_state.psap_settings["top_k"] = st.slider(
        "retrieved chunks (top_k)",
        min_value=3,
        max_value=40,
        value=int(st.session_state.psap_settings["top_k"]),
        step=1,
    )
    st.session_state.psap_settings["debug_on"] = st.toggle(
        "debug json",
        value=bool(st.session_state.psap_settings["debug_on"]),
    )

for msg in convo["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def _render_psap_details(out: dict) -> None:
    with st.expander("grounding details", expanded=False):
        flag = out.get("insufficient_evidence")
        st.markdown(f"**insufficient evidence (per model):** `{flag}`")
        st.subheader("bullets")
        for b in out.get("bullets") or []:
            st.markdown(f"- {b.get('text', '')}")
            ids = b.get("chunk_ids") or []
            if ids:
                st.caption("chunk_ids: " + ", ".join(ids))
        st.subheader("citations")
        for c in out.get("citations") or []:
            st.markdown(f"- `{c.get('chunk_id','')}` — {c.get('excerpt','')}")


prompt = st.chat_input("ask about the indexed psap cybersecurity corpus…")

if prompt:
    convo["messages"].append({"role": "user", "content": prompt})
    _set_title_from_first_user_message(cid)
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("retrieving + generating…"):
            out = generate_grounded_response(
                query=prompt,
                top_k=int(st.session_state.psap_settings["top_k"]),
                include_debug=bool(st.session_state.psap_settings["debug_on"]),
            )
        if not isinstance(out, dict):
            st.error("unexpected backend output")
            st.write(out)
        elif out.get("error"):
            st.error(out.get("error"))
            st.code((out.get("raw_output") or "")[:2000])
        else:
            answer = (out.get("answer_summary") or "").strip() or "no summary returned."
            st.markdown(answer)
            _render_psap_details(out)
            if st.session_state.psap_settings["debug_on"]:
                with st.expander("debug", expanded=False):
                    st.code(json.dumps(out, ensure_ascii=False, indent=2), language="json")
            convo["messages"].append({"role": "assistant", "content": answer})
