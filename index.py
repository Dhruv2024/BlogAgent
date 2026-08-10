from __future__ import annotations

import os
import re
import json
import unicodedata
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from uuid import uuid4
from streamlit_js_eval import streamlit_js_eval
from typing import List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime, timedelta, timezone


load_dotenv()

# -----------------------------
# Supabase setup
# -----------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error(
        "Missing SUPABASE_URL / SUPABASE_ANON_KEY. "
        "Add them to your .env file (see setup instructions)."
    )
    st.stop()


@st.cache_resource
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


supabase = get_supabase()

BUCKET = "blog-files"


# -----------------------------
# Backend loader
# -----------------------------
def load_backend_stream_runner():
    try:
        import backend
    except Exception as exc:
        raise RuntimeError("Could not import backend.py") from exc
    runner = getattr(backend, "run_stream", None)
    if not callable(runner):
        raise RuntimeError("Could not find run_stream() in backend.py")
    return runner


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    name = unicodedata.normalize("NFKD", title or "blog")
    name = name.encode("ascii", "ignore").decode()
    name = re.sub(r"[^A-Za-z0-9 _-]", "", name).strip().lower().replace(" ", "_")
    return name or "blog"


def normalize_latex_delimiters(md: str) -> str:
    """Convert \\[ \\] / \\( \\) — and bare [ ] that look like LaTeX — into $$ / $."""
    md = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", md, flags=re.DOTALL)
    md = re.sub(r"\\\((.+?)\\\)", r"$\1$", md, flags=re.DOTALL)

    def _looks_like_latex(match: re.Match) -> str:
        inner = match.group(1)
        if re.search(r"\\[a-zA-Z]+", inner):
            return f"$${inner}$$"
        return match.group(0)

    md = re.sub(r"(?<!\\)\[\s*(\\[a-zA-Z].+?)\s*\](?!\()", _looks_like_latex, md, flags=re.DOTALL)
    return md


def render_markdown(md: str):
    st.markdown(normalize_latex_delimiters(md), unsafe_allow_html=False)


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback

# -----------------------------
# Rate limiting (per-user daily blog cap)
# -----------------------------
DAILY_BLOG_LIMIT = 4


def count_blogs_today_utc(user_id: str) -> int:
    now_utc = datetime.now(timezone.utc)
    start_of_day_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    res = (
        supabase.table("blogs")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .gte("created_at", start_of_day_utc.isoformat())
        # no deleted_at filter — a hidden row created today still counts
        .execute()
    )
    return res.count or 0


def next_reset_local(user_tz_name: str) -> datetime:
    now_utc = datetime.now(timezone.utc)
    next_utc_midnight = (now_utc + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    try:
        tz = ZoneInfo(user_tz_name)
    except Exception:
        tz = timezone.utc
    return next_utc_midnight.astimezone(tz)

def time_until_reset_str(user_tz_name: str) -> str:
    now_utc = datetime.now(timezone.utc)
    next_utc_midnight = (now_utc + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    delta = next_utc_midnight - now_utc
    total_minutes = int(delta.total_seconds() // 60)
    hours, minutes = divmod(total_minutes, 60)

    if hours > 0 and minutes > 0:
        return f"{hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h"
    else:
        return f"{minutes}m"

# -----------------------------
# Cloud storage helpers (per-user)
# -----------------------------
def list_past_blogs(user_id: str) -> List[dict]:
    res = (
        supabase.table("blogs")
        .select("*")
        .eq("user_id", user_id)
        .is_("deleted_at", "null")   # hidden ones don't show in sidebar
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []


def read_md_file(storage_path: str) -> str:
    data = supabase.storage.from_(BUCKET).download(storage_path)
    return data.decode("utf-8")


def save_blog(user_id: str, title: str, content: str) -> dict:
    slug = safe_slug(title)
    storage_path = f"{user_id}/{slug}-{uuid4().hex[:8]}.md"

    supabase.storage.from_(BUCKET).upload(
        storage_path,
        content.encode("utf-8"),
        {"content-type": "text/markdown"},
    )

    row = (
        supabase.table("blogs")
        .insert(
            {
                "user_id": user_id,
                "title": title,
                "slug": slug,
                "storage_path": storage_path,
            }
        )
        .execute()
    )
    return row.data[0]


def delete_blog(blog_id: str, storage_path: str) -> None:
    """
    Soft-deletes a blog: hides it from the user first, then removes the
    underlying file. Ordered this way so that if something fails between
    the two steps, the failure mode is an orphaned file (invisible, just
    wastes storage) rather than a visible-but-broken row (bad UX).
    """
    # ① Hide the row FIRST — this is what the user actually sees/clicks
    supabase.table("blogs").update(
        {"deleted_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", blog_id).execute()

    # ② Remove the file — if this fails, the row is already safely hidden;
    #    the file becomes an orphan that the cleanup job below will catch.
    try:
        supabase.storage.from_(BUCKET).remove([storage_path])
    except Exception as exc:
        # Don't crash the delete flow over a storage hiccup — the row is
        # already hidden, which is the user-facing correctness that matters.
        # Log it so orphan_cleanup can pick it up later.
        st.session_state.setdefault("logs", []).append(
            f"[delete_blog] storage remove failed for {storage_path}: {exc}"
        )

def find_orphaned_storage_files(user_id: str) -> List[str]:
    """
    Returns storage paths under this user's folder that have no matching
    non-deleted row in the blogs table — i.e. files left behind by a
    partially-failed delete.
    """
    # All files actually in storage for this user
    storage_files = supabase.storage.from_(BUCKET).list(user_id)
    storage_paths = {f"{user_id}/{f['name']}" for f in storage_files}

    # All storage_paths this user's *visible* rows currently claim
    res = (
        supabase.table("blogs")
        .select("storage_path")
        .eq("user_id", user_id)
        .is_("deleted_at", "null")
        .execute()
    )
    claimed_paths = {row["storage_path"] for row in (res.data or [])}

    return sorted(storage_paths - claimed_paths)


def cleanup_orphaned_storage_files(user_id: str) -> int:
    """Deletes orphaned files found by find_orphaned_storage_files. Returns count removed."""
    orphans = find_orphaned_storage_files(user_id)
    if orphans:
        supabase.storage.from_(BUCKET).remove(orphans)
    return len(orphans)


# -----------------------------
# Rendering helpers for streamed state
# -----------------------------
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
        st.dataframe(df, width='stretch', hide_index=True)
        with st.expander("Task details"):
            st.json(tasks)


def render_evidence(evidence: list):
    if not evidence:
        st.info("No evidence returned (maybe closed_book mode or no Tavily results).")
        return
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
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def render_sections_progress(plan_dict: Optional[dict], sections: list):
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
            st.markdown(f"## {t.get('title')}")
            st.caption("⏳ Writing…")

# -----------------------------
# Auth gate
# -----------------------------
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")

if "user" not in st.session_state:
    st.session_state["user"] = None

if not st.session_state["user"]:
    st.title("🔐 Sign in to Blog Writing Agent")
    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    if "login_in_progress" not in st.session_state:
        st.session_state["login_in_progress"] = False
    if "signup_in_progress" not in st.session_state:
        st.session_state["signup_in_progress"] = False

    with tab_login:
        email = st.text_input("Email", key="login_email", disabled=st.session_state["login_in_progress"])
        password = st.text_input(
            "Password", type="password", key="login_pw",
            disabled=st.session_state["login_in_progress"],
        )

        if st.session_state["login_in_progress"]:
            st.button("Log in", type="primary", key="login_btn", disabled=True)
            with st.spinner("Logging in…"):
                try:
                    res = supabase.auth.sign_in_with_password(
                        {"email": email, "password": password}
                    )
                    st.session_state["user"] = res.user
                    st.session_state["session"] = res.session
                except Exception as e:
                    st.session_state["login_in_progress"] = False
                    st.error(f"Login failed: {e}")
                    st.stop()
            st.session_state["login_in_progress"] = False
            st.rerun()
        else:
            if st.button("Log in", type="primary", key="login_btn"):
                if not email or not password:
                    st.warning("Please enter both email and password.")
                else:
                    st.session_state["login_in_progress"] = True
                    st.rerun()

    with tab_signup:
        email2 = st.text_input("Email", key="signup_email", disabled=st.session_state["signup_in_progress"])
        password2 = st.text_input(
            "Password", type="password", key="signup_pw",
            disabled=st.session_state["signup_in_progress"],
        )

        if st.session_state["signup_in_progress"]:
            st.button("Sign up", key="signup_btn", disabled=True)
            with st.spinner("Creating your account…"):
                try:
                    supabase.auth.sign_up({"email": email2, "password": password2})
                    signup_ok = True
                except Exception as e:
                    signup_ok = False
                    st.session_state["signup_in_progress"] = False
                    st.error(f"Signup failed: {e}")
                    st.stop()
            st.session_state["signup_in_progress"] = False
            if signup_ok:
                st.success("Account created. Check your email if confirmation is required, then log in.")
        else:
            if st.button("Sign up", key="signup_btn"):
                if not email2 or not password2:
                    st.warning("Please enter both email and password.")
                else:
                    st.session_state["signup_in_progress"] = True
                    st.rerun()

    st.stop()

user = st.session_state["user"]
user_id = user.id

# ✅ NEW: detect user's browser timezone (cached per session)
if "user_timezone" not in st.session_state:
    detected_tz = streamlit_js_eval(
        js_expressions="Intl.DateTimeFormat().resolvedOptions().timeZone",
        key="user_tz",
    )
    st.session_state["user_timezone"] = detected_tz or "UTC"

user_timezone = st.session_state["user_timezone"]

# -----------------------------
# Main app (only reached when logged in)
# -----------------------------
st.title("Blog Writing Agent")

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "confirm_delete" not in st.session_state:
    st.session_state["confirm_delete"] = None

if "topic_key_counter" not in st.session_state:
    st.session_state["topic_key_counter"] = 0

with st.sidebar:
    st.caption(f"Signed in as **{user.email}**")
    if st.button("Log out"):
        supabase.auth.sign_out()
        st.session_state["user"] = None
        st.session_state["last_out"] = None
        st.rerun()
    with st.expander("⚙️ Maintenance"):
        if st.button("🧹 Clean up orphaned files"):
            removed = cleanup_orphaned_storage_files(user_id)
            if removed:
                st.success(f"Removed {removed} orphaned file(s).")
            else:
                st.info("No orphaned files found.")

    st.divider()

    # ✅ NEW: New Blog button
    if st.button("🆕 New Blog", width='stretch'):
        st.session_state["last_out"] = None
        st.session_state["confirm_delete"] = None
        st.session_state["topic_key_counter"] += 1  # forces a fresh, empty topic widget
        st.rerun()

    st.header("Generate New Blog")

    # ✅ NEW: rate limit display
    blogs_today = count_blogs_today_utc(user_id)
    remaining = max(0, DAILY_BLOG_LIMIT - blogs_today)
    reset_local = next_reset_local(user_timezone)

    if remaining > 0:
        st.caption(f"📊 {blogs_today} of {DAILY_BLOG_LIMIT} blogs used today")
    else:
        st.error(
            f"⛔ Daily limit reached ({DAILY_BLOG_LIMIT}/{DAILY_BLOG_LIMIT}). "
            f"Resets in **{time_until_reset_str(user_timezone)}**."
        )

    topic = st.text_area(
        "Topic",
        height=120,
        key=f"topic_input_{st.session_state['topic_key_counter']}",
    )
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary", disabled=(remaining <= 0))

    st.divider()
    st.divider()
    st.subheader("📚 Past blogs")

    past_blogs = list_past_blogs(user_id)

    if not past_blogs:
        st.caption("No saved blogs yet — generate your first one above!")
    else:
        search_query = st.text_input(
            "Search",
            key="blog_search",
            placeholder="🔍 Search past blogs…",
            label_visibility="collapsed",
        )

        if search_query:
            filtered_blogs = [
                b for b in past_blogs if search_query.lower() in b["title"].lower()
            ]
        else:
            filtered_blogs = past_blogs

        if not filtered_blogs:
            st.caption(f"No blogs matching '{search_query}'")

        # Track which blog (by id) is currently loaded, for highlighting
        currently_loaded_id = st.session_state.get("loaded_blog_id")

        for b in filtered_blogs:
            is_active = b["id"] == currently_loaded_id

            with st.container(border=True):
                title_line = f"**{'🟢 ' if is_active else ''}{b['title']}**"
                st.markdown(title_line)
                st.caption(f"🗓️ {b['created_at'][:10]}")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("📂 Load", key=f"load_{b['id']}", width='stretch'):
                        md_text = read_md_file(b["storage_path"])
                        st.session_state["last_out"] = {
                            "plan": None,
                            "evidence": [],
                            "final": md_text,
                        }
                        st.session_state["loaded_blog_id"] = b["id"]
                        st.rerun()

                with c2:
                    if st.button("🗑️ Delete", key=f"del_{b['id']}", width='stretch'):
                        st.session_state["confirm_delete"] = b

                if st.session_state.get("confirm_delete") == b:
                    st.warning(f"Delete **{b['title']}**? This can't be undone.")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        if st.button("✅ Yes, delete", key=f"confirm_del_{b['id']}", width='stretch'):
                            delete_blog(b["id"], b["storage_path"])
                            st.session_state["confirm_delete"] = None
                            if st.session_state.get("loaded_blog_id") == b["id"]:
                                st.session_state["last_out"] = None
                                st.session_state["loaded_blog_id"] = None
                            st.rerun()
                    with cc2:
                        if st.button("❌ Cancel", key=f"cancel_del_{b['id']}", width='stretch'):
                            st.session_state["confirm_delete"] = None
                            st.rerun()

# Layout
tab_plan, tab_evidence, tab_preview, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🧾 Logs"]
)

# -----------------------------
# Run generation (streaming)
# -----------------------------
if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    # ✅ NEW: re-check limit server-side at click time
    current_count = count_blogs_today_utc(user_id)
    if current_count >= DAILY_BLOG_LIMIT:
        reset_local = next_reset_local(user_timezone)
        st.error(
            f"⛔ Daily limit reached. Resets in {time_until_reset_str(user_timezone)}."
        )
        st.stop()

    plan_slot = tab_plan.empty()
    evidence_slot = tab_evidence.empty()
    preview_slot = tab_preview.empty()
    status = st.status("Running blog generation…", expanded=True)

    logs: List[str] = []

    try:
        runner_stream = load_backend_stream_runner()
        final_state = {}

        for node_name, node_update, state in runner_stream(
            topic=topic.strip(), thread_id=f"{user_id}:{uuid4().hex[:8]}"
        ):
            final_state = state

            if node_name == "router":
                msg = f"🧭 Routing decided: mode = `{state.get('mode')}`"
                status.write(msg)
                logs.append(msg)

            elif node_name == "research":
                msg = f"🔎 Research done — {len(state.get('evidence', []))} sources"
                status.write(msg)
                logs.append(msg)
                with evidence_slot.container():
                    render_evidence(state.get("evidence", []))

            elif node_name == "orchestrator":
                status.write("🧩 Plan ready")
                logs.append("[orchestrator] plan ready")
                plan_obj = state.get("plan")
                plan_dict = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else plan_obj
                with plan_slot.container():
                    render_plan(plan_dict)

            elif node_name == "worker":
                sections = state.get("sections", [])
                msg = f"✍️ Section {len(sections)} written"
                status.write(msg)
                logs.append(msg)
                plan_obj = state.get("plan")
                plan_dict = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else plan_obj
                with preview_slot.container():
                    render_sections_progress(plan_dict, sections)

            elif node_name == "reducer":
                status.write("📄 Assembling final markdown")
                logs.append("[reducer] final markdown assembled")
                with preview_slot.container():
                    render_markdown(state.get("final", ""))

        # Save to Supabase (cloud), scoped to this user
        final_md = final_state.get("final", "")
        if final_md:
            plan_obj = final_state.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", topic.strip())
            else:
                blog_title = topic.strip()

            saved_row = save_blog(user_id, blog_title, final_md)
            logs.append(f"[save] stored as {saved_row['storage_path'].split("/")[1]}")

        st.session_state["last_out"] = final_state
        st.session_state["logs"] = st.session_state.get("logs", []) + logs
        status.update(label="✅ Done", state="complete", expanded=False)
        st.rerun()  # refresh sidebar "Past blogs" list immediately

    except Exception as exc:
        status.update(label="❌ Failed", state="error", expanded=False)
        st.exception(exc)
        st.session_state["logs"] = st.session_state.get("logs", []) + logs + [f"[run] error: {exc}"]

# -----------------------------
# Render last result (persists across reruns, e.g. after loading a past blog)
# -----------------------------
out = st.session_state.get("last_out")
if out and not run_btn:
    with tab_plan:
        st.subheader("Plan")
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan available (this blog may have been loaded from storage).")
        else:
            plan_dict = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else plan_obj
            render_plan(plan_dict)

    with tab_evidence:
        st.subheader("Evidence")
        render_evidence(out.get("evidence") or [])

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
            elif isinstance(plan_obj, dict) and plan_obj:
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                blog_title = extract_title_from_md(final_md, "blog")

            st.download_button(
                "⬇️ Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=f"{safe_slug(blog_title)}.md",
                mime="text/markdown",
            )

    with tab_logs:
        st.subheader("Logs")
        st.text_area(
            "Event log",
            value="\n\n".join(st.session_state.get("logs", [])[-80:]),
            height=520,
            disabled=True,
        )
else:
    with tab_preview:
        st.info("Enter a topic and click **Generate Blog**, or load a past blog from the sidebar.")