from __future__ import annotations

import operator
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated
from uuid import uuid4

from pydantic import BaseModel, Field

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from google import genai
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os 
from langgraph.types import RetryPolicy
import time

load_dotenv() 
# TPM limit -> 8K
llm=ChatGroq(model="openai/gpt-oss-120b")
# client = genai.Client()
# GEMINI_MODEL="gemini-3.6-flash"

# TPM Limit->70K but no tool calling-> useless in this case
# llm=ChatGroq(model="groq/compound")

tool=TavilySearch(max_results=3)

class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # keep if Tavily provides; DO NOT rely on it
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)
    final: str

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

def router_node(state: State) -> dict:
    
    topic = state["topic"]
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}"),
        ]
    )
    # response = client.models.generate_content(
    #     model=GEMINI_MODEL,
    #     contents=f"Topic: {topic}",
    #     config={
    #         "system_instruction": ROUTER_SYSTEM,
    #         "response_mime_type": "application/json",
    #         "response_schema": RouterDecision,
    #     },
    # )
    # decision = response.parsed  # or RouterDecision.model_validate_json(response.text)

    # return {
    #     "needs_research": decision.needs_research,
    #     "mode": decision.mode,
    #     "queries": decision.queries,
    # }
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearch(max_results=max_results)
    results = tool.invoke({"query": query})

    if isinstance(results, dict):
        items = results.get("results", [])
    else:
        items = results or []

    normalized: List[dict] = []
    for r in items:
        if not isinstance(r, dict):
            continue
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:

    # take the first 10 queries from state
    queries = (state.get("queries", []) or [])
    max_results = 6

    raw_results: List[dict] = []

    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

    compact_results = []
    for r in raw_results[:12]:
        compact_results.append(
            {
                "title": r.get("title", "")[:120],
                "url": r.get("url", ""),
                "snippet": r.get("snippet", "")[:240],
                "published_at": r.get("published_at"),
                "source": r.get("source"),
            }
        )
    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw results:\n{compact_results}"),
        ]
    )
    # extractor = client.models.generate_content(
    #     model=GEMINI_MODEL,
    #     contents=f"Raw results:\n{raw_results}",
    #     config={
    #         "system_instruction": RESEARCH_SYSTEM,
    #         "response_mime_type": "application/json",
    #         "response_schema": EvidencePack,
    #     },
    # )

    # pack = extractor.parsed
    # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120–550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    # plan_agent = client.models.generate_content(
    #     model=GEMINI_MODEL,
    #     contents=(
    #                 f"Topic: {state['topic']}\n"
    #                 f"Mode: {mode}\n\n"
    #                 f"Evidence (ONLY use for fresh claims; may be empty):\n"
    #                 f"{[e.model_dump() for e in evidence][:16]}"
    #             ),
    #     config={
    #         "system_instruction": ORCH_SYSTEM,
    #         "response_mime_type": "application/json",
    #         "response_schema": Plan,
    #     },
    # )

    # plan = plan_agent.parsed
    return {"plan": plan}

def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    # response = client.models.generate_content(
    #     model=GEMINI_MODEL,
    #     contents=(
    #                 f"Blog title: {plan.blog_title}\n"
    #                 f"Audience: {plan.audience}\n"
    #                 f"Tone: {plan.tone}\n"
    #                 f"Blog kind: {plan.blog_kind}\n"
    #                 f"Constraints: {plan.constraints}\n"
    #                 f"Topic: {topic}\n"
    #                 f"Mode: {mode}\n\n"
    #                 f"Section title: {task.title}\n"
    #                 f"Goal: {task.goal}\n"
    #                 f"Target words: {task.target_words}\n"
    #                 f"Tags: {task.tags}\n"
    #                 f"requires_research: {task.requires_research}\n"
    #                 f"requires_citations: {task.requires_citations}\n"
    #                 f"requires_code: {task.requires_code}\n"
    #                 f"Bullets:{bullets_text}\n\n"
    #                 f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
    #             ),
    #     config={
    #         "system_instruction": WORKER_SYSTEM,
    #         "response_mime_type": "application/json",
    #     },
    # )
    # section_md = response.parsed

    return {"sections": [(task.id, section_md)]}

from pathlib import Path
import re
import unicodedata

def _safe_filename(title: str) -> str:
    name = unicodedata.normalize("NFKD", title or "output")
    name = name.encode("ascii", "ignore").decode()
    name = re.sub(r"[^A-Za-z0-9 _-]", "", name).strip().lower().replace(" ", "_")
    if not name:
        name = "output"
    return f"{name}.md"

def save_markdown(content: str, title: str, out_dir: str = "outputs") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(title)
    path = out / filename
    if not content or not content.strip():
        raise ValueError("Empty content — not writing file.")
    path.write_text(content, encoding="utf-8")
    return path

def reducer_node(state: State) -> dict:

    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    final_md = f"# {plan.blog_title}\n\n{body}\n"
    # debug info
    print("--- Reducer debug ---")
    # print("CWD:", Path.cwd())
    print("Title:", plan.blog_title)
    print("final_md length:", len(final_md))
    # try:
    #     out_path = save_markdown(final_md, plan.blog_title, out_dir="outputs")
    #     print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")
    # except ValueError as e:
    #     print("Not writing file:", e)

    return {"final": final_md}

checkpointer = MemorySaver()

def _is_retryable_429(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    message = str(exc).lower()
    return status_code == 429 or "429" in message or "rate limit" in message or "too many requests" in message

g = StateGraph(State)
g.add_node("router", router_node,retry=RetryPolicy(max_attempts=4, initial_interval=5.0, backoff_factor=2.0, retry_on=_is_retryable_429))

g.add_node("research", research_node,retry=RetryPolicy(max_attempts=4, initial_interval=5.0, backoff_factor=2.0, retry_on=_is_retryable_429))

g.add_node("orchestrator", orchestrator_node,retry=RetryPolicy(max_attempts=4, initial_interval=5.0, backoff_factor=2.0, retry_on=_is_retryable_429))

g.add_node("worker", worker_node,retry=RetryPolicy(max_attempts=5, initial_interval=10.0, backoff_factor=2.0, retry_on=_is_retryable_429))

g.add_node("reducer", reducer_node)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile(checkpointer=checkpointer)
app



def run(topic: str, thread_id: Optional[str] = None, max_attempts: int = 5):
    thread_id = thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    input_state = {
        "topic": topic,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "sections": [],
        "final": "",
    }

    for attempt in range(max_attempts):
        try:
            return app.invoke(input_state if attempt == 0 else None, config=config)
        except Exception as exc:
            if not _is_retryable_429(exc) or attempt + 1 >= max_attempts:
                raise
            print("Retrying...")
            time.sleep(60)

from typing import Iterator, Tuple, Dict, Any

def run_stream(
    topic: str,
    thread_id: Optional[str] = None,
    max_attempts: int = 2,
) -> Iterator[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """
    Streams (node_name, node_update, accumulated_state) as the graph executes.

    node_name: which node just finished ("router", "research", "orchestrator", "worker", "reducer")
    node_update: the raw dict that node returned
    accumulated_state: merged state so far, safe to hand straight to the UI
    """
    thread_id = thread_id or str(uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": 3,
        }
    input_state = {
        "topic": topic,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "sections": [],
        "final": "",
    }

    accumulated: Dict[str, Any] = dict(input_state)

    attempt = 0
    while attempt < max_attempts:
        try:
            stream_input = input_state if attempt == 0 else None
            for update in app.stream(stream_input, config=config, stream_mode="updates"):
                # `update` looks like {"node_name": {...node's returned dict...}}
                for node_name, node_update in update.items():
                    if node_name == "worker":
                        # sections use operator.add — append rather than overwrite
                        accumulated["sections"] = accumulated.get("sections", []) + node_update.get("sections", [])
                    else:
                        accumulated.update(node_update)
                    yield node_name, node_update, dict(accumulated)
            return  # graph finished normally
        except Exception as exc:
            if not _is_retryable_429(exc) or attempt + 1 >= max_attempts:
                raise
            print("Retrying...")
            time.sleep(60)
            attempt += 1