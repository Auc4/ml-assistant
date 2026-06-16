import json
import csv
import time
import subprocess
import requests
import os
from collections import defaultdict

from rag_pipeline import query_rag

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"

TEST_SET_FILE = "test_set.json"

DOCS_DIR = "docs"
RESULTS_FILE = os.path.join(DOCS_DIR, "prompts_results.csv")
SUMMARY_FILE = os.path.join(DOCS_DIR, "prompts_summary.json")

# Ajusta este comando si usas Goose CLI de otra forma
GOOSE_COMMAND = ["goose", "session"]

def estimate_tokens(text):
    return max(1, len(text) // 4)


def run_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_gpu": 0,
            "temperature": 0.1,
            "num_predict": 300
        }
    }

    start = time.time()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    latency = time.time() - start
    response.raise_for_status()

    data = response.json()

    output = data.get("response", "")
    prompt_tokens = data.get("prompt_eval_count", estimate_tokens(prompt))
    output_tokens = data.get("eval_count", estimate_tokens(output))

    return output, latency, prompt_tokens + output_tokens


def run_rag(prompt):
    start = time.time()
    output = query_rag(prompt, use_rag=True, top_k=8)
    latency = time.time() - start

    total_tokens = estimate_tokens(prompt + output)

    return output, latency, total_tokens


def run_goose(prompt):
    """
    Ejecuta Goose en modo sesión para permitir tools + multi-step reasoning.
    """

    start = time.time()

    try:
        result = subprocess.run(
            GOOSE_COMMAND,
            input=prompt + "\nexit\n",   # importante: fuerza cierre de sesión
            capture_output=True,
            text=True,
            timeout=600
        )

        latency = time.time() - start

        output = (
            result.stdout.strip()
            + "\n"
            + result.stderr.strip()
        )

        total_tokens = estimate_tokens(prompt + output)

        return output, latency, total_tokens

    except Exception as e:
        latency = time.time() - start
        return f"ERROR_RUNNING_GOOSE: {e}", latency, estimate_tokens(prompt)


def score_response(output, expected_keywords):
    output_lower = output.lower()

    hits = 0

    for keyword in expected_keywords:
        if keyword.lower() in output_lower:
            hits += 1

    if hits == len(expected_keywords):
        return "success"

    if hits > 0:
        return "partial"

    return "fail"


def choose_runner(category):
    if category == "rag_required":
        return run_rag

    if category in ["tool_required", "multi_step"]:
        return run_goose

    return run_ollama


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)

    with open(TEST_SET_FILE, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    rows = []

    for item in test_set:
        test_id = item["id"]
        category = item["category"]
        prompt = item["prompt"]
        expected_keywords = item.get("expected_keywords", [])

        print(f"\nRunning {test_id} | {category}")
        print("-" * 80)

        runner = choose_runner(category)

        output, latency, total_tokens = runner(prompt)

        result = score_response(output, expected_keywords)

        print(f"Result: {result}")
        print(f"Latency: {latency:.2f}s")
        print(f"Tokens: {total_tokens}")
        print(output[:500])

        rows.append({
            "id": test_id,
            "category": category,
            "prompt": prompt,
            "expected_outcome_type": item["expected_outcome_type"],
            "expected_keywords": "; ".join(expected_keywords),
            "result": result,
            "latency_sec": round(latency, 2),
            "total_tokens": total_tokens,
            "output": output.replace("\n", " ")
        })

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id",
            "category",
            "prompt",
            "expected_outcome_type",
            "expected_keywords",
            "result",
            "latency_sec",
            "total_tokens",
            "output"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = compute_summary(rows)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nEvaluation completed.")
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")


def compute_summary(rows):
    total = len(rows)

    success_count = sum(1 for r in rows if r["result"] == "success")
    partial_count = sum(1 for r in rows if r["result"] == "partial")
    fail_count = sum(1 for r in rows if r["result"] == "fail")

    by_category = defaultdict(list)

    for row in rows:
        by_category[row["category"]].append(row)

    category_stats = {}

    for category, items in by_category.items():
        n = len(items)
        successes = sum(1 for r in items if r["result"] == "success")
        partials = sum(1 for r in items if r["result"] == "partial")
        fails = sum(1 for r in items if r["result"] == "fail")

        avg_latency = sum(float(r["latency_sec"]) for r in items) / n
        avg_tokens = sum(int(r["total_tokens"]) for r in items) / n

        category_stats[category] = {
            "total": n,
            "success": successes,
            "partial": partials,
            "fail": fails,
            "success_rate": round(successes / n, 3),
            "average_latency_sec": round(avg_latency, 2),
            "average_tokens": round(avg_tokens, 2)
        }

    summary = {
        "total_tests": total,
        "overall_success": success_count,
        "overall_partial": partial_count,
        "overall_fail": fail_count,
        "overall_success_rate": round(success_count / total, 3),
        "category_stats": category_stats
    }

    return summary


if __name__ == "__main__":
    main()