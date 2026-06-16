# Local Jarvis

## Description

This project implements a locall assitant called "Jarvis" running entirely on CPU infrastructure using quantized LLMs via Ollama. The system integrates:

- Local LLM inference (Qwen 2.5 7B)
- Retrieval-Augmented Generation (RAG)
- KV cache scaling analysis
- Tool usage via MCP (Google Workspace via Goose)
- Automated evaluation pipeline

---

## Architecture

The system consists of:

- **Ollama**: runs quantized models
- **ChromaDB**: used for RAG retrieval
- **Goose MCP Agent**: tool orchestration layer
- **Google Workspace MCP Server**: calendar/email tools
- **Benchmarking pipeline**: KV cache + quantization experiments
- **Evaluation framework**: automated scoring of 20+ prompts

---

## Repository Structure

```bash
config/ # Configuration files (Ollama YAML settings)
data/
├── corpus/ # Raw text corpus for RAG
├── prompts/ # test_set.json (Part E evaluation)
├── outputs/ # All generated CSV/JSON results
└── chroma_db/ # Vector database storage

docs/ # Plots and report figures
report/ # Final report (PDF)
scripts/ # Optional execution scripts (run_all)
src/ # Core system implementation
```

## Installation

```bash
pip install -r requirements.txt
```

For Ollama:

```bash

curl -fsSL https://ollama.com/install.sh | sh

ollama pull qwen2.5:7b-instruct-q3_K_M
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:7b-instruct-q8_0
ollama pull nomic-embed-text
ollama pull qwen3:8b # Used only for MCP tool evaluation
```

One-command execution:

```bash
bash scripts/run_all.sh
```

This script executes:
- Part A: Quantization benchmark
- Part B: KV cache scaling
- Part C: RAG evaluation
- Part E: Automated evaluation
- Plot generation

## Experimental

### Part A - Quantization Study

Measures:
- model size
- tokens/sec
- RAM usage
- quality score (0–3 rubric)

```markdown
data/outputs/measurements.csv
docs/quantization_study.png
```

### Part B - KV Cache Scaling

Evaluates:
- context lengths: 512 → 16384
- RAM growth
- latency increase

```markdown
data/outputs/kv_cache_measurements.csv
docs/kv_cache_study.png
```

### Part C — RAG Pipeline

Main description:
- corpus: local text (La Máquina del Tiempo)
- embedding model: nomic-embed-text
- vector DB: ChromaDB

Comparison:
With RAG vs Without RAG

### Part D — MCP Tool Integration

Tools enabled in Google Cloud Platform for MCP Server:
- Google Calendar
- Gmail

> [!NOTE]
> Google authentication is required.
> Interactive permission prompts may appear during execution.
> This is expected behavior.

### Part E — Evaluation

Automated evaluation over 20 prompts:

Categories:

pure chat
RAG-required
tool-required
multi-step
adversarial

```bash
data/outputs/prompts_results.csv
data/outputs/prompts_summary.json
```

## Demo Video

[Youtube Link](https://youtu.be/2H6synI3vzM)

## AI Disclosure

This project was built for an academic assignment. All outputs are reproducible via scripts included in this repository.

AI assistants were used for:
- code debugging
- documentation assistance
- plotting refinement