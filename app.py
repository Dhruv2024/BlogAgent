from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
import re

# -----------------------------
# Load the backend workflow runner
# -----------------------------
def load_backend_runner():
    try:
        import backend
    except Exception as exc:
        raise RuntimeError("Could not import backend.py") from exc

    runner = getattr(backend, "run", None)
    if not callable(runner):
        raise RuntimeError("Could not find run() in backend.py")

    return runner


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"



def normalize_latex_delimiters(md: str) -> str:
    # """Convert \[ ... \] / \( ... \) — and bare [ ... ] that look like LaTeX — into $$ / $ so KaTeX picks them up."""
    # Standard LaTeX display/inline delimiters
    md = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", md, flags=re.DOTALL)
    md = re.sub(r"\\\((.+?)\\\)", r"$\1$", md, flags=re.DOTALL)

    # Fallback: bare [ ... ] that clearly contains LaTeX commands (backslash-word),
    # e.g. "[ T(n) = \Theta(...) ]" with the outer backslash missing.
    def _looks_like_latex(match: re.Match) -> str:
        inner = match.group(1)
        if re.search(r"\\[a-zA-Z]+", inner):  # contains \Theta, \log, \bigl, etc.
            return f"$${inner}$$"
        return match.group(0)  # leave ordinary bracketed text alone

    md = re.sub(r"(?<!\\)\[\s*(\\[a-zA-Z].+?)\s*\](?!\()", _looks_like_latex, md, flags=re.DOTALL)
    return md

def render_markdown(md: str):
    st.markdown(normalize_latex_delimiters(md), unsafe_allow_html=False)


# -----------------------------
# ✅ NEW: Past blogs helpers
# -----------------------------
def list_past_blogs() -> List[Path]:
    """
    Returns .md files in current working directory, newest first.
    Filters out obvious non-blog markdown files if needed.
    """
    cwd = Path("./streamlit_files")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    """
    Use first '# ' heading as title if present.
    """
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")

st.title("Blog Writing Agent")

with st.sidebar:
    st.header("Generate New Blog")
    topic = st.text_area(
        "Topic",
        height=120,
    )
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary")

    # ✅ NEW: Past blogs list (keeps everything else intact)
    st.divider()
    st.subheader("Past blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md in current folder).")
        selected_md_file = None
    else:
        # Build labels from file name + (optional) parsed title
        options: List[str] = []
        file_by_label: dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load selected blog"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                # Load into session_state as if it were a run output
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "final": md_text,
                }
                # also update the topic input to the title (best-effort) without changing UI
                st.session_state["topic_prefill"] = extract_title_from_md(md_text, selected_md_file.stem)
        if st.button("🗑️ Delete selected blog"):
            if selected_md_file and selected_md_file.exists():
                selected_md_file.unlink()
                st.success(f"Deleted {selected_md_file.name}")
                st.rerun()

    

# Keep your topic input as-is; optionally prefill for next run after loading a blog
if "topic_prefill" in st.session_state and isinstance(st.session_state["topic_prefill"], str):
    # Do not mutate widgets; just keep as a hint.
    pass

# Storage for latest run
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# Layout
tab_plan, tab_evidence, tab_preview, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🧾 Logs"]
)

logs: List[str] = []


def log(msg: str):
    logs.append(msg)


# if run_btn:
#     if not topic.strip():
#         st.warning("Please enter a topic.")
#         st.stop()

#     status = st.status("Running blog generation…", expanded=True)
#     progress_area = st.empty()

#     try:
#         runner = load_backend_runner()
#         out = runner(topic=topic.strip(), thread_id="streamlit")
#         st.session_state["last_out"] = out

#         # ✅ NEW: persist the generated blog to disk so it shows up in "Past blogs"
#         final_md = out.get("final") or ""
#         if final_md:
#             plan_obj = out.get("plan")
#             if hasattr(plan_obj, "blog_title"):
#                 blog_title = plan_obj.blog_title
#             elif isinstance(plan_obj, dict):
#                 blog_title = plan_obj.get("blog_title", topic.strip())
#             else:
#                 blog_title = topic.strip()

#             save_path = Path(f"{safe_slug(blog_title)}.md")
#             # avoid overwriting an existing file with the same slug
#             counter = 1
#             while save_path.exists():
#                 save_path = Path(f"{safe_slug(blog_title)}_{counter}.md")
#                 counter += 1

#             save_path.write_text(final_md, encoding="utf-8")
#             log(f"[run] saved blog to {save_path}")

#         status.update(label="✅ Done", state="complete", expanded=False)
#         progress_area.success("Blog generation completed.")
#         log("[run] completed")
#     except Exception as exc:
#         status.update(label="❌ Failed", state="error", expanded=False)
#         progress_area.error(str(exc))
#         st.exception(exc)
#         log(f"[run] error: {exc}")

def load_backend_stream_runner():
    try:
        import backend
    except Exception as exc:
        raise RuntimeError("Could not import backend.py") from exc
    runner = getattr(backend, "run_stream", None)
    if not callable(runner):
        raise RuntimeError("Could not find run_stream() in backend.py")
    return runner


def render_plan(plan_dict: dict):
    st.write("**Title:**", plan_dict.get("blog_title"))
    cols = st.columns(3)
    cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
    cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
    cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))

    tasks = plan_dict.get("tasks", [])
    if tasks:
        df = pd.DataFrame(
            [
                {
                    "id": t.get("id"),
                    "title": t.get("title"),
                    "target_words": t.get("target_words"),
                    "requires_research": t.get("requires_research"),
                    "requires_citations": t.get("requires_citations"),
                    "requires_code": t.get("requires_code"),
                    "tags": ", ".join(t.get("tags") or []),
                }
                for t in tasks
            ]
        ).sort_values("id")
        st.dataframe(df, use_container_width=True, hide_index=True)
        with st.expander("Task details"):
            st.json(tasks)


def render_evidence(evidence: list):
    if not evidence:
        st.info("No evidence returned (maybe closed_book mode or no Tavily key/results).")
        return
    rows = []
    for e in evidence:
        if hasattr(e, "model_dump"):
            e = e.model_dump()
        rows.append(
            {"title": e.get("title"), "published_at": e.get("published_at"),
             "source": e.get("source"), "url": e.get("url")}
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_sections_progress(plan_dict: dict, sections: list):
    """Shows section-by-section progress: done sections rendered, pending ones as placeholders."""
    done_ids = {tid for tid, _ in sections}
    section_by_id = {tid: md for tid, md in sections}
    tasks = plan_dict.get("tasks", []) if plan_dict else []

    if not tasks:
        st.info("Waiting for plan…")
        return

    for t in sorted(tasks, key=lambda x: x.get("id")):
        tid = t.get("id")
        if tid in done_ids:
            render_markdown(section_by_id[tid])
        else:
            st.markdown(f"##  {t.get('title')}")
            st.caption("⏳ Writing…")


if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    plan_slot = tab_plan.empty()
    evidence_slot = tab_evidence.empty()
    preview_slot = tab_preview.empty()
    status = st.status("Running blog generation…", expanded=True)

    try:
        runner_stream = load_backend_stream_runner()
        final_state = {}

        for node_name, node_update, state in runner_stream(topic=topic.strip(), thread_id="streamlit"):
            final_state = state

            if node_name == "router":
                status.write(f"🧭 Routing decided: mode = `{state.get('mode')}`")

            elif node_name == "research":
                status.write(f"🔎 Research done — {len(state.get('evidence', []))} sources")
                with evidence_slot.container():
                    render_evidence(state.get("evidence", []))

            elif node_name == "orchestrator":
                status.write("🧩 Plan ready")
                plan_obj = state.get("plan")
                plan_dict = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else plan_obj
                with plan_slot.container():
                    render_plan(plan_dict)

            elif node_name == "worker":
                sections = state.get("sections", [])
                status.write(f"✍️ Section {len(sections)} written")
                plan_obj = state.get("plan")
                plan_dict = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else plan_obj
                with preview_slot.container():
                    render_sections_progress(plan_dict, sections)

            elif node_name == "reducer":
                status.write("📄 Assembling final markdown")
                with preview_slot.container():
                    render_markdown(state.get("final", ""))

        st.session_state["last_out"] = final_state
        status.update(label="✅ Done", state="complete", expanded=False)

    except Exception as exc:
        status.update(label="❌ Failed", state="error", expanded=False)
        st.exception(exc)

# Render last result (if any)
out = st.session_state.get("last_out")
if out:
    # --- Plan tab ---
    with tab_plan:
        st.subheader("Plan")
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan found in output.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.write("**Title:**", plan_dict.get("blog_title"))
            cols = st.columns(3)
            cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
            cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
            cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": t.get("requires_research"),
                            "requires_citations": t.get("requires_citations"),
                            "requires_code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Task details"):
                    st.json(tasks)

    # --- Evidence tab ---
    with tab_evidence:
        st.subheader("Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned (maybe closed_book mode or no Tavily key/results).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "title": e.get("title"),
                        "published_at": e.get("published_at"),
                        "source": e.get("source"),
                        "url": e.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Preview tab ---
    with tab_preview:
        st.subheader("Markdown Preview")
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            render_markdown(final_md)

            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                # fallback: parse from markdown title
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"
            st.download_button(
                "⬇️ Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
            )

    # --- Logs tab ---
    with tab_logs:
        st.subheader("Logs")
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)

        st.text_area("Event log", value="\n\n".join(st.session_state["logs"][-80:]), height=520)
else:
    st.info("Enter a topic and click **Generate Blog**.")