---
title: AI Blog Writing Agent
emoji: ✍️
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: agent_frontend.py
python_version: "3.11"
---

# BLOG WRITING AI AGENT

An advanced AI-powered multi-agent blog generation system built using LangGraph, Groq LLMs, Gemini image generation, and Tavily research integration.

This project automatically:
- Researches a topic
- Plans the blog structure
- Generates professional technical blog sections
- Creates AI-generated architecture diagrams
- Produces a polished Markdown blog file

The system uses a production-style multi-agent workflow with routing, planning, research, parallel writing, merging, and image generation pipelines.

---

# Features

## AI Research Pipeline
- Automatic topic analysis
- Smart research query generation
- Tavily-powered web research
- Evidence collection and summarization

## Multi-Agent Architecture
The system contains specialized AI agents for:
- Routing
- Research
- Blog planning
- Parallel section writing
- Content merging
- Image planning
- Image generation

## Professional Blog Writing
- SEO-aware blog generation
- Structured technical writing
- Code examples
- Tables and markdown formatting
- Real-world examples
- Production-style documentation

## AI Image Generation
- Automatic technical diagram creation
- Gemini-powered image generation
- Architecture illustrations
- Markdown image embedding

## LangGraph Workflow
Built with a graph-based orchestration pipeline using:
- StateGraph
- Parallel fanout workers
- Reducer subgraphs
- Structured outputs

---

# Architecture

```text
User Topic
    │
    ▼
Router Agent
    │
    ├── Research Needed?
    │        │
    │        ▼
    │   Research Agent
    │
    ▼
Orchestrator Agent
    │
    ▼
Parallel Worker Agents
    │
    ▼
Reducer Subgraph
    │
    ├── Merge Content
    ├── Plan Images
    └── Generate Images
    │
    ▼
Final Markdown Blog
```
---
# Tech Stack
## AI & LLMs
- LangGraph
- LangChain
- Groq LLMs
- Gemini Image Generation
- Tavily Search API
## Backend
- Python
- Pydantic
- dotenv
## Workflow & Orchestration
- StateGraph
- Parallel Agent Fanout
- Reducer Subgraphs
---
# Project Structure
```text
BLOG_WRITING_AI_AGENT/
│
├── agent_backend.py
├── requirements.txt
├── .env
├── images/
├── generated_blog.md
├── README.md
│
└── chatu/                 # Virtual Environment
```
---
# Workflow Overview
## 1. Router Agent

### Determines:

- Whether research is required
- Research strategy
- Search queries
## 2. Research Agent

### Uses Tavily API to:

- Search the web
- Collect evidence
- Extract useful snippets
## 3. Orchestrator Agent

### Creates:

- SEO-friendly blog structure
- Audience targeting
- Writing constraints
- Section-level planning
## 4. Worker Agents

### Parallel AI writers generate:

- Introduction
- Technical sections
- Examples
- Conclusion
## 5. Reducer Pipeline

### Responsible for:

- Merging sections
- Planning images
- Generating diagrams
- Exporting final markdown
---
# Installation
## Clone Repository
```text
git clone https://github.com/your-username/Blog-Writing-Agent.git
cd Blog-Writing-Agent
```
## Create Virtual Environment
```text
python -m venv chatu
```
## Windows
```text
chatu\\Scripts\\activate
```
## Linux/Mac
```text
source chatu/bin/activate
```
## Install Dependencies
```text
pip install -r requirements.txt
```
---
# Environment Variables
Create a .env file:
```text
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_API_KEY=your_google_api_key
```

`GROQ_API_KEY` is required. `TAVILY_API_KEY` is optional; without it, blogs are
generated without web research.

## Deployment

### Hugging Face Spaces

Create a Streamlit Space and add `GROQ_API_KEY` under **Settings → Variables
and secrets** as a secret. Add `TAVILY_API_KEY` there if you want web research.
The Space configuration at the top of this README selects `agent_frontend.py`
as the app entry point.

### Render

Deploy this repository as a Blueprint using `render.yaml`, then enter your
`GROQ_API_KEY` when prompted. Add `TAVILY_API_KEY` in the Render service
environment if you want web research.
The Blueprint pins Python to 3.11.9 and disables Streamlit's development file
watcher.

The project does not install or use Guardrails AI. It uses a Groq-powered
factuality review after generation.
---
# Running the Project
```text
python agent_backend.py
```
---
# Example Usage
```text
result = app.invoke({
    "topic": "How Retrieval-Augmented Generation Works",
    "sections": []
})
```
### The system will:

- Research the topic
- Generate a professional blog
- Create technical diagrams
- Export markdown output
---
# Generated Output

### The system generates:

- Markdown blog file
- AI-generated technical images
- SEO-aware content
- Structured engineering-style documentation
---
# Learning Outcomes

### This project demonstrates:

- AI Agent Systems
- Workflow orchestration
- LLM pipelines
- Parallel execution
- AI research systems
- Multimodal AI applications
- Production AI engineering concepts
---
# Author

Harsh Rawat
---
# License

This project is open-source and available under the MIT License.
