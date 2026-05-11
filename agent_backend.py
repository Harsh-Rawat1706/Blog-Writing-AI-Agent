# ============================================================
# ULTIMATE PRODUCTION BLOG WRITING AI AGENT
# LangGraph + Groq + Tavily + Gemini Images
# ============================================================

from __future__ import annotations

import operator
import os
import re
import time
import warnings
from pathlib import Path
from typing import TypedDict, List, Optional, Annotated

from dotenv import load_dotenv
from groq import RateLimitError
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from langchain_core._api.deprecation import (
    LangChainPendingDeprecationWarning,
)

from langchain_groq import ChatGroq

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.types import Send

# ============================================================
# WARNINGS
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
)

# ============================================================
# ENV
# ============================================================

load_dotenv(".env")

required_keys = [
    "GROQ_API_KEY",
    "TAVILY_API_KEY",
    "GOOGLE_API_KEY",
]

for key in required_keys:

    if not os.getenv(key):

        raise ValueError(
            f"Missing environment variable: {key}"
        )

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

ENABLE_IMAGES = True

# ============================================================
# SCHEMAS
# ============================================================

class Task(BaseModel):
    id: int
    title: str
    goal: str
    bullets: List[str]
    target_words: int = 450          # raised from 350
    section_type: str = "body"       # "intro" | "body" | "conclusion"
    tone_note: str = ""              # per-section tone hint from planner
    
class Plan(BaseModel):
    blog_title: str = "Untitled Blog"
    audience: str = "developers"
    tone: str = "professional"
    blog_kind: str = "technical"
    constraints: List[str] = Field(default_factory=list)
    seo_keywords: List[str] = Field(default_factory=list)   # NEW
    tasks: List[Task] = Field(default_factory=list)
    
class EvidenceItem(BaseModel):

    title: str

    url: str

    snippet: Optional[str] = None

    source: Optional[str] = None


class RouterDecision(BaseModel):

    needs_research: bool = True

    mode: str = "hybrid"

    reason: str = ""

    queries: List[str] = Field(
        default_factory=list
    )


class ImageSpec(BaseModel):

    placeholder: str

    filename: str

    alt: str

    caption: str

    prompt: str


class State(TypedDict):

    topic: str

    mode: str

    needs_research: bool

    queries: List[str]

    evidence: List[EvidenceItem]

    plan: Optional[Plan]

    sections: Annotated[
        List[tuple[int, str]],
        operator.add,
    ]

    merged_md: str

    md_with_placeholders: str

    image_specs: List[dict]

    final: str

# ============================================================
# MODELS
# ============================================================

router_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=250,
    groq_api_key=GROQ_API_KEY,
)

writer_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.6,
    max_tokens=2000,          # was 1200 — needs room to breathe
    groq_api_key=GROQ_API_KEY,
)

planner_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=1800,          # was 1200 — plan JSON was getting cut
    groq_api_key=GROQ_API_KEY,
)


# ============================================================
# SAFE INVOKE
# ============================================================

def safe_invoke(chain, messages, retries=5):

    last_error = None

    for attempt in range(retries):

        try:

            return chain.invoke(messages)

        except RateLimitError as e:

            last_error = e

            wait_time = 2 ** attempt

            print(
                f"Rate limit hit. "
                f"Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

        except OutputParserException as e:

            last_error = e

            print("\n========== PARSER ERROR ==========")
            print(e)
            print("==================================\n")

            time.sleep(1)

        except Exception as e:

            last_error = e

            print("\n========== UNKNOWN ERROR ==========")
            print(e)
            print("===================================\n")

            time.sleep(1)

    raise Exception(
        f"Max retries exceeded. "
        f"Last error: {last_error}"
    )

# ============================================================
# ROUTER
# ============================================================

ROUTER_SYSTEM = """
You are a routing agent for a professional AI blog system.

Return valid JSON only.

You MUST return this schema:

{
  "needs_research": true,
  "mode": "hybrid",
  "reason": "short explanation",
  "queries": [
    "query1",
    "query2"
  ]
}

Rules:
- max 3 concise queries
- no markdown
- no explanations
"""

def router_node(state: State):

    decider = router_llm.with_structured_output(
        RouterDecision,
        method="json_mode",
    )

    decision = safe_invoke(
        decider,
        [
            SystemMessage(content=ROUTER_SYSTEM),

            HumanMessage(
                content=f"""
Topic: {state['topic']}

Generate focused search queries.
"""
            ),
        ],
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries[:3],
    }

def route_next(state: State):

    if state["needs_research"]:

        return "research"

    return "orchestrator"

# ============================================================
# RESEARCH
# ============================================================
def research_node(state: State):
    try:
        from tavily import TavilyClient

        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        evidence = []

        for query in state["queries"][:3]:
            response = tavily.search(
                query=query,
                max_results=3,          # was 2
                search_depth="advanced", # NEW — deeper results
            )
            for r in response.get("results", []):
                evidence.append(
                    EvidenceItem(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("content", "")[:600],  # was 500
                        source="tavily",
                    )
                )

        return {"evidence": evidence[:8]}   # was 5

    except Exception as e:
        print(f"Research failed: {e}")
        return {"evidence": []}


# ============================================================
# ORCHESTRATOR
# ============================================================

ORCH_SYSTEM = """
You are a senior technical blog architect writing for a developer audience.

Return valid JSON only. No markdown. No extra fields.

Schema:

{
  "blog_title": "Compelling, SEO-friendly title",
  "audience": "senior developers",
  "tone": "professional yet approachable",
  "blog_kind": "technical deep-dive",
  "constraints": [
    "no vague claims without evidence",
    "every section must have at least one concrete example"
  ],
  "seo_keywords": ["keyword1", "keyword2", "keyword3"],
  "tasks": [
    {
      "id": 1,
      "title": "section title",
      "section_type": "intro",
      "goal": "what this section must achieve for the reader",
      "tone_note": "hook the reader with a real-world pain point",
      "bullets": [
        "specific point to cover",
        "another specific point"
      ],
      "target_words": 450
    }
  ]
}

Rules:
- Generate exactly 5 sections: intro, 3 body sections, conclusion
- Section 1 must be section_type "intro" — hook + problem statement
- Sections 2-4 must be section_type "body" — meaty technical content
- Section 5 must be section_type "conclusion" — recap + call to action
- bullets must be specific and actionable, not vague labels
- seo_keywords: 3–5 terms the blog should rank for
- no markdown, no extra fields
"""


def orchestrator_node(state: State):

    planner = planner_llm.with_structured_output(
        Plan,
        method="json_mode",
    )

    evidence_text = "\n".join([
        f"- {e.title}: {e.snippet}"
        for e in state.get("evidence", [])
    ])

    try:

        plan = safe_invoke(
            planner,
            [
                SystemMessage(content=ORCH_SYSTEM),

                HumanMessage(
                    content=f"""
Topic: {state['topic']}

Research:
{evidence_text}

Create a professional technical blog outline.
"""
                ),
            ],
        )

    except Exception as e:

        print(f"Planner failed: {e}")

        plan = Plan(
            blog_title=state["topic"].title(),

            audience="developers",

            tone="professional",

            tasks=[
                Task(
                    id=1,
                    title="Introduction",
                    goal="Introduce the topic",
                    bullets=[
                        "Overview",
                        "Importance",
                    ],
                ),

                Task(
                    id=2,
                    title="Core Concepts",
                    goal="Explain key concepts",
                    bullets=[
                        "Architecture",
                        "Mechanism",
                    ],
                ),

                Task(
                    id=3,
                    title="Implementation",
                    goal="Implementation details",
                    bullets=[
                        "Code examples",
                        "Optimization",
                    ],
                ),

                Task(
                    id=4,
                    title="Applications",
                    goal="Discuss applications",
                    bullets=[
                        "Use cases",
                        "Best practices",
                    ],
                ),
            ],
        )

    return {
        "plan": plan
    }

# ============================================================
# FANOUT
# ============================================================

def fanout(state: State):
    assert state["plan"] is not None

    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "plan": state["plan"].model_dump(),
                "evidence": [
                    e.model_dump()
                    for e in state.get("evidence", [])
                ][:6],              # was :3 — workers need more context
            },
        )
        for task in state["plan"].tasks
    ]

# ============================================================
# WORKER
# ============================================================

WORKER_SYSTEM = """
You are an elite technical blog writer publishing on a top engineering blog.

Your writing must be:
- Deeply technical yet readable by intermediate developers
- Grounded in real examples, not abstract theory
- Structured with clear H2/H3 headings
- Honest: no hype, no vague claims
- Concise: every sentence earns its place

Markdown rules:
- Use ## for the main section heading
- Use ### for sub-topics within the section
- Use code blocks (```language) for any code
- Use **bold** for key terms on first use
- Use bullet lists only for genuinely list-like content
- Use tables for comparisons

Quality checklist (you must satisfy all):
[ ] Opens with a sentence that immediately delivers value — no "In this section..."
[ ] Contains at least one concrete, runnable code example OR a specific real-world scenario
[ ] Every claim is specific (numbers, names, versions) not vague ("it's fast", "it's popular")
[ ] Closes with a sentence that bridges to the next idea or lands the key takeaway
"""


def worker_node(payload: dict):

    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = payload.get("evidence", [])

    bullets_text = "\n".join(
        f"  - {b}" for b in task.bullets
    )

    evidence_text = (
        "\n".join(
            f"- [{e['title']}]({e['url']}): {e['snippet']}"
            for e in evidence
        )
        if evidence
        else "No external research available — rely on your training knowledge."
    )

    constraints_text = "\n".join(
        f"- {c}" for c in plan.constraints
    ) or "None"

    keywords_text = ", ".join(
        getattr(plan, "seo_keywords", [])
    ) or "None"

    prompt = f"""
Blog title: {plan.blog_title}
Target audience: {plan.audience}
Overall tone: {plan.tone}
SEO keywords to weave in naturally: {keywords_text}
Global constraints: 
{constraints_text}

---

Section #{task.id} of 5
Type: {task.section_type}
Title: {task.title}
Goal: {task.goal}
Tone note: {task.tone_note}
Target length: ~{task.target_words} words

Points to cover:
{bullets_text}

Supporting research:
{evidence_text}

---

Write this section now. Start directly with the ## heading.
Do NOT include a preamble or meta-commentary.
"""

    try:
        section_md = safe_invoke(
            writer_llm,
            [
                SystemMessage(content=WORKER_SYSTEM),
                HumanMessage(content=prompt),
            ],
        ).content.strip()

        return {"sections": [(task.id, section_md)]}

    except Exception as e:
        return {
            "sections": [
                (
                    task.id,
                    f"## {task.title}\n\n*Generation failed: {e}*",
                )
            ]
        }


# ============================================================
# MERGE
# ============================================================
def merge_content(state: State):
    plan = state["plan"]
    if plan is None:
        raise ValueError("Plan missing")

    ordered_sections = [
        md
        for _, md in sorted(state["sections"], key=lambda x: x[0])
    ]

    body = "\n\n---\n\n".join(ordered_sections)   # visual separator between sections

    # Build a metadata header for the rendered blog
    keywords = ", ".join(getattr(plan, "seo_keywords", []))
    header = (
        f"# {plan.blog_title}\n\n"
        f"> **Audience:** {plan.audience} &nbsp;|&nbsp; "
        f"**Type:** {plan.blog_kind} &nbsp;|&nbsp; "
        f"**Keywords:** {keywords}\n\n"
    )

    return {"merged_md": header + body}


# ============================================================
# IMAGE PLANNER
# ============================================================

IMAGE_SYSTEM = """
You are an AI image planner.

Return valid JSON only.

Generate ONE professional technical diagram.

Schema:

{
  "placeholder": "[IMAGE_1]",
  "filename": "diagram.png",
  "alt": "description",
  "caption": "caption",
  "prompt": "detailed image prompt"
}

Rules:
- technical illustration style
- professional
- educational
- no extra fields
"""

def decide_images(state: State):

    if not ENABLE_IMAGES:

        return {
            "md_with_placeholders": state["merged_md"],
            "image_specs": [],
        }

    planner = planner_llm.with_structured_output(
        ImageSpec,
        method="json_mode",
    )

    try:

        image_plan = safe_invoke(
            planner,
            [
                SystemMessage(content=IMAGE_SYSTEM),

                HumanMessage(
    content=f"""
Blog title: {state['plan'].blog_title if state.get('plan') else 'Technical Blog'}
Topic: {state['topic']}

Blog markdown (first 3000 chars):
{state['merged_md'][:3000]}

Design one technical architecture diagram that visually explains the core concept.
Make the prompt detailed: include style (clean vector, white background, labeled components).
"""
)
,
            ],
        )

        placeholder = (
            f"\n\n{image_plan.placeholder}\n\n"
        )

        md = (
            state["merged_md"]
            + placeholder
        )

        return {
            "md_with_placeholders": md,

            "image_specs": [
                image_plan.model_dump()
            ],
        }

    except Exception as e:

        print(f"Image planning failed: {e}")

        return {
            "md_with_placeholders": state["merged_md"],
            "image_specs": [],
        }

# ============================================================
# GEMINI IMAGE GENERATION
# ============================================================

def generate_image(prompt: str):

    client = genai.Client(
        api_key=GOOGLE_API_KEY
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",

        contents=prompt,

        config=types.GenerateContentConfig(
            response_modalities=[
                "TEXT",
                "IMAGE",
            ]
        ),
    )

    for part in response.candidates[0].content.parts:

        if part.inline_data:

            return part.inline_data.data

    return None

# ============================================================
# GENERATE & INSERT IMAGES
# ============================================================

def generate_and_place_images(state: State):

    md = state["md_with_placeholders"]

    image_specs = state.get(
        "image_specs",
        [],
    )

    images_dir = Path("images")

    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:

        try:

            image_bytes = generate_image(
                spec["prompt"]
            )

            if image_bytes:

                image_path = (
                    images_dir
                    / spec["filename"]
                )

                image_path.write_bytes(
                    image_bytes
                )

                image_md = f"""
![{spec['alt']}](images/{spec['filename']})

*{spec['caption']}*
"""

                md = md.replace(
                    spec["placeholder"],
                    image_md,
                )

        except Exception as e:

            print(
                f"Image generation failed: {e}"
            )

    filename = (
        state["topic"]
        .lower()
        .replace(" ", "_")
    )

    filename = re.sub(
        r"[^a-z0-9_]+",
        "_",
        filename,
    )

    Path(
        f"{filename}.md"
    ).write_text(
        md,
        encoding="utf-8",
    )

    return {
        "final": md
    }

# ============================================================
# REDUCER SUBGRAPH
# ============================================================

reducer_graph = StateGraph(State)

reducer_graph.add_node(
    "merge_content",
    merge_content,
)

reducer_graph.add_node(
    "decide_images",
    decide_images,
)

reducer_graph.add_node(
    "generate_and_place_images",
    generate_and_place_images,
)

reducer_graph.add_edge(
    START,
    "merge_content",
)

reducer_graph.add_edge(
    "merge_content",
    "decide_images",
)

reducer_graph.add_edge(
    "decide_images",
    "generate_and_place_images",
)

reducer_graph.add_edge(
    "generate_and_place_images",
    END,
)

reducer_subgraph = reducer_graph.compile()

# ============================================================
# MAIN GRAPH
# ============================================================

g = StateGraph(State)

g.add_node("router", router_node)

g.add_node("research", research_node)

g.add_node("orchestrator", orchestrator_node)

g.add_node("worker", worker_node)

g.add_node("reducer", reducer_subgraph)

g.add_edge(
    START,
    "router",
)

g.add_conditional_edges(
    "router",
    route_next,
    {
        "research": "research",
        "orchestrator": "orchestrator",
    },
)

g.add_edge(
    "research",
    "orchestrator",
)

g.add_conditional_edges(
    "orchestrator",
    fanout,
    ["worker"],
)

g.add_edge(
    "worker",
    "reducer",
)

g.add_edge(
    "reducer",
    END,
)

app = g.compile()

print(
    "Ultimate multimodal "
    "AI blog agent loaded successfully"
)