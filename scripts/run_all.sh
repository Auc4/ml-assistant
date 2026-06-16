#!/bin/bash

set -e

echo "Running Part A - Quantization"
python src/benchmark_runner.py

echo "Running Part B - KV Cache"
python src/kv_cache_benchmark.py

echo "Running Part C - RAG Evaluation"
python src/evaluate_rag.py

echo "Running Part E - Evaluation"
python src/evaluation_prompts.py

echo "Generating plots"
python src/generate_plots.py
python src/generate_kv_plots.py

echo "ALL EXPERIMENTS COMPLETED"