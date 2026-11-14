from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_slug(title: str) -> str:

    slug = title.strip().lower()

    slug = re.sub(
        r"[^a-z0-9 _-]+",
        "",
        slug
    )

    slug = re.sub(
        r"\s+",
        "_",
        slug
    ).strip("_")

    return slug or "blog"


def list_past_blogs():

    files = [
        path
        for path in Path(".").glob("*.md")
        if path.is_file()
    ]

    files.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    return files


def read_md_file(path: Path):

    return path.read_text(
        encoding="utf-8",
        errors="replace"
    )


def extract_title_from_md(
    markdown: str,
    fallback: str
):

    for line in markdown.splitlines():

        if line.startswith("# "):

            return line[2:].strip() or fallback

    return fallback


def get_plan_dict(plan):

    if hasattr(plan, "model_dump"):
        return plan.model_dump()

    if isinstance(plan, dict):
        return plan

    return {}


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Blog Writing Agent",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ AI Blog Writing Agent")

st.caption(
    "Generate audience-aware blogs using LangGraph, "
    "Groq, and Tavily."
)


# ============================================================
# SESSION STATE
# ============================================================

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

if "logs" not in st.session_state:
    st.session_state["logs"] = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Generate New Blog")

    topic = st.text_area(
        "Enter Blog Topic",
        height=120,
        placeholder=(
            "Example: Benefits of investing "
            "in mutual funds"
        )
    )

    as_of = st.date_input(
        "As-of date",
        value=date.today()
    )

    st.divider()

    # --------------------------------------------------------
    # AUDIENCE SELECTION
    # --------------------------------------------------------

    st.subheader("🎯 Target Audience")

    audience_mode = st.radio(
        "How should we select your audience?",
        [
            "Automatic (AI decides)",
            "I'll choose the audience"
        ],
        index=0
    )

    audience = ""

    if audience_mode == "I'll choose the audience":

        audience_choice = st.selectbox(

            "Select your target audience",

            [
                "General readers",
                "Students and beginners",
                "Business owners and founders",
                "Researchers and academics",
                "Content creators and marketers",
                "Software developers and technical learners",
                "Working professionals",
                "Custom audience"
            ]
        )

        if audience_choice == "Custom audience":

            audience = st.text_input(
                "Describe your target audience",
                placeholder=(
                    "Example: First-time home buyers "
                    "in India"
                )
            )

        else:

            audience = audience_choice

    else:

        st.info(
            "AI will analyse your topic and select "
            "the most relevant target audience."
        )

    st.divider()

    run_btn = st.button(
        "🚀 Generate Blog",
        type="primary",
        use_container_width=True
    )


    # --------------------------------------------------------
    # PAST BLOGS
    # --------------------------------------------------------

    st.divider()

    st.subheader("📚 Past Blogs")

    past_files = list_past_blogs()

    if not past_files:

        st.caption(
            "No saved blogs found."
        )

    else:

        file_options = {}

        for path in past_files[:50]:

            try:

                markdown = read_md_file(path)

                title = extract_title_from_md(
                    markdown,
                    path.stem
                )

            except Exception:

                title = path.stem

            label = f"{title} · {path.name}"

            file_options[label] = path

        selected = st.selectbox(
            "Select a previous blog",
            list(file_options.keys())
        )

        if st.button("📂 Load Selected Blog"):

            selected_path = file_options[selected]

            markdown = read_md_file(
                selected_path
            )

            st.session_state["last_out"] = {

                "plan": None,

                "evidence": [],
                "final": markdown,
                "review": None,
                "generation_logs": ["Loaded a previously generated blog."]
            }

            st.rerun()


# ============================================================
# MAIN TABS
# ============================================================

tab_plan, tab_evidence, tab_preview, tab_logs = st.tabs(
    [
        "🧩 Blog Plan",
        "🔎 Research Evidence",
        "📝 Blog Preview",
        "🧾 Review & Logs"
    ]
)


# ============================================================
# BLOG GENERATION
# ============================================================

if run_btn:

    # Validate topic
    if not topic.strip():

        st.warning(
            "Please enter a blog topic."
        )

        st.stop()


    # Validate custom audience
    if (
        audience_mode == "I'll choose the audience"
        and not audience.strip()
    ):

        st.warning(
            "Please specify your target audience."
        )

        st.stop()


    # --------------------------------------------------------
    # LANGGRAPH INPUT STATE
    # --------------------------------------------------------

    inputs = {
        "topic": topic.strip(),
        "audience_preference": audience.strip() or "AUTO",
        "tone_preference": "AUTO",
        "as_of": as_of.isoformat(),
    }


    # --------------------------------------------------------
    # RUN GRAPH EXACTLY ONCE
    # --------------------------------------------------------

    status = st.status(
        "Generating your blog...",
        expanded=True
    )

    try:

        status.write("Preparing the generation workflow...")

        with st.spinner(
            "Researching and generating your blog..."
        ):

            from agent_backend import app

            output = app.invoke(inputs)

        st.session_state["last_out"] = output

        status.update(
            label="Blog generation completed!",
            state="complete",
            expanded=False
        )

        st.success("Your blog has been generated and reviewed.")

    except Exception as error:

        status.update(
            label="Blog generation blocked or failed",
            state="error",
            expanded=True
        )

        st.error(str(error))

        st.session_state["logs"].append(
            f"ERROR: {str(error)}"
        )


# ============================================================
# DISPLAY GENERATED OUTPUT
# ============================================================

output = st.session_state.get("last_out")


if output:

    # ========================================================
    # TAB 1: BLOG PLAN
    # ========================================================

    with tab_plan:

        st.subheader("Blog Generation Plan")

        plan = output.get("plan")

        if not plan:

            st.info(
                "No plan available for this saved blog."
            )

        else:

            plan_dict = get_plan_dict(plan)

            st.write(
                "**Blog Title:**",
                plan_dict.get("blog_title")
            )

            col1, col2, col3 = st.columns(3)

            col1.write(
                "**Audience:** "
                + str(plan_dict.get("audience"))
            )

            col2.write(
                "**Tone:** "
                + str(plan_dict.get("tone"))
            )

            col3.write(
                "**Blog Type:** "
                + str(plan_dict.get("blog_kind"))
            )

            st.divider()

            st.subheader("Section Breakdown")

            tasks = plan_dict.get(
                "tasks", []
            )

            if tasks:

                df = pd.DataFrame(

                    [
                        {
                            "Section ID": task.get("id"),

                            "Title": task.get("title"),

                            "Type": task.get("section_type"),

                            "Target Words": task.get(
                                "target_words"
                            )
                        }

                        for task in tasks
                    ]
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                with st.expander(
                    "View Complete Plan"
                ):

                    st.json(plan_dict)


    # ========================================================
    # TAB 2: RESEARCH EVIDENCE
    # ========================================================

    with tab_evidence:

        st.subheader(
            "Research Evidence"
        )

        evidence = output.get(
            "evidence", []
        )

        if not evidence:

            st.warning(
                "No external research evidence returned."
            )

        else:

            for index, item in enumerate(
                evidence,
                start=1
            ):

                if hasattr(item, "model_dump"):
                    item = item.model_dump()

                with st.expander(
                    f"{index}. {item.get('title', 'Untitled')}"
                ):

                    st.write(
                        item.get("snippet", "")
                    )

                    st.caption(
                        "Source: "
                        + str(item.get("source", "Unknown"))
                    )

                    url = item.get("url")

                    if url:
                        st.markdown(
                            f"[Open Source]({url})"
                        )


    # ========================================================
    # TAB 3: BLOG PREVIEW
    # ========================================================

    with tab_preview:

        st.subheader(
            "Generated Blog"
        )

        final_markdown = output.get(
            "final", ""
        )

        if not final_markdown:

            st.warning(
                "No final blog content available."
            )

        else:

            st.markdown(
                final_markdown,
                unsafe_allow_html=False
            )

            plan_dict = get_plan_dict(
                output.get("plan")
            )

            blog_title = plan_dict.get(
                "blog_title",
                extract_title_from_md(
                    final_markdown,
                    "blog"
                )
            )

            filename = (
                safe_slug(blog_title) + ".md"
            )

            st.divider()

            st.download_button(

                label="⬇️ Download Blog (Markdown)",

                data=final_markdown.encode(
                    "utf-8"
                ),

                file_name=filename,

                mime="text/markdown",

                use_container_width=True
            )


    # ========================================================
    # TAB 4: REVIEW AND LOGS
    # ========================================================

    with tab_logs:

        st.subheader("Factuality Review")

        review = output.get("review")
        if review:
            if review.get("misleading"):
                st.error("The review identified potentially misleading content.")
            else:
                st.success("The blog passed the misleading-content review.")

            for label, key in (
                ("Unsupported claims", "unsupported_claims"),
                ("Issues", "issues"),
                ("Revision instructions", "revision_instructions"),
            ):
                items = review.get(key, [])
                if items:
                    st.markdown(f"**{label}**")
                    st.write(items)
        else:
            st.info("No factuality review is available for this saved blog.")

        st.divider()

        st.subheader(
            "Generation Logs"
        )

        generation_logs = output.get("generation_logs", [])
        if generation_logs:
            st.text_area(
                "Workflow",
                value="\n".join(generation_logs),
                height=220
            )

        if st.session_state["logs"]:

            st.text_area(
                "Errors",
                value="\n\n".join(
                    st.session_state["logs"][-50:]
                ),
                height=400
            )

        elif not generation_logs:

            st.info(
                "No workflow or error logs recorded."
            )


else:

    st.info(
        "Enter your blog topic, select your target "
        "audience and click Generate Blog."
    )