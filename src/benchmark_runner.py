import time
import csv
import os
import threading
import requests
import psutil
from pathlib import Path

# ─────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
TAGS_URL = "http://localhost:11434/api/tags"
PS_URL = "http://localhost:11434/api/ps"

# ─────────────────────────────────────────────
# Project paths (REPRODUCIBLE)
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "outputs" / "kv_cache_measurements.csv"

# ─────────────────────────────────────────────
# Global monitoring state
# ─────────────────────────────────────────────

monitoring = False
peak_ram_used = 0


# ─────────────────────────────────────────────
# Model management
# ─────────────────────────────────────────────

def unload_model(model_name):
    try:
        requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "keep_alive": 0
            },
            timeout=10
        )

        print(f"🧹 Descargando modelo: {model_name}")
        time.sleep(4)

    except Exception as e:
        print(f"⚠️ Error descargando modelo: {e}")


# ─────────────────────────────────────────────
# RAM monitoring
# ─────────────────────────────────────────────

def monitor_ollama_ram():
    global monitoring, peak_ram_used
    peak_ram_used = 0

    while monitoring:
        current_total_rss = 0

        for proc in psutil.process_iter(["pid", "name"]):
            try:
                name = proc.info.get("name")

                if name and "ollama" in name.lower():
                    p = psutil.Process(proc.info["pid"])
                    current_total_rss += p.memory_info().rss / (1024 ** 3)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        peak_ram_used = max(peak_ram_used, current_total_rss)
        time.sleep(0.05)


# ─────────────────────────────────────────────
# Model metadata
# ─────────────────────────────────────────────

def get_model_file_size(model_name):
    try:
        models = requests.get(TAGS_URL, timeout=10).json().get("models", [])

        for model in models:
            if model["name"] == model_name:
                return model["size"] / (1024 ** 3)

    except Exception as e:
        print(f"⚠️ Error obteniendo tamaño del modelo: {e}")

    return 0


def get_vram_usage(model_name):
    try:
        data = requests.get(PS_URL, timeout=10).json()

        for model in data.get("models", []):
            if model["name"] == model_name:
                return model.get("size_vram", 0) / (1024 ** 3)

    except Exception as e:
        print(f"⚠️ Error leyendo VRAM: {e}")

    return 0


# ─────────────────────────────────────────────
# Benchmark execution
# ─────────────────────────────────────────────

def run_benchmark(model_name, quantization, context_length, prompt):
    global monitoring, peak_ram_used

    unload_model(model_name)

    print(f"\n🚀 Evaluando {model_name}")
    print(f"📏 Contexto: {context_length}")
    print("🧠 Modo forzado: CPU")

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": context_length,
            "temperature": 0.0,
            "num_gpu": 0
        }
    }

    monitoring = True
    ram_thread = threading.Thread(target=monitor_ollama_ram)
    ram_thread.start()

    start_time = time.time()

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=600
        )

        response.raise_for_status()
        response_data = response.json()

    except Exception as e:
        monitoring = False
        ram_thread.join()
        print(f"❌ Error ejecutando benchmark: {e}")
        return

    end_time = time.time()

    monitoring = False
    ram_thread.join()

    duration = end_time - start_time
    eval_count = response_data.get("eval_count", 0)

    tokens_per_second = eval_count / duration if duration > 0 else 0

    peak_ram = peak_ram_used
    peak_vram = get_vram_usage(model_name)
    file_size_gb = get_model_file_size(model_name)

    processor_detected = "CPU" if peak_vram < 0.1 else "GPU/PARCIAL"

    # ─────────────────────────────────────────────
    # Output path handling
    # ─────────────────────────────────────────────

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = OUTPUT_FILE.exists()

    with open(OUTPUT_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "model",
                "quantization",
                "context_length",
                "file_size_gb",
                "peak_ram_gb",
                "peak_vram_gb",
                "processor_detected",
                "tokens_per_second"
            ])

        writer.writerow([
            model_name,
            quantization,
            context_length,
            f"{file_size_gb:.2f}",
            f"{peak_ram:.2f}",
            f"{peak_vram:.2f}",
            processor_detected,
            f"{tokens_per_second:.2f}"
        ])

    print(
        f"✅ Completado -> "
        f"Size: {file_size_gb:.2f} GB | "
        f"RAM: {peak_ram:.2f} GB | "
        f"VRAM: {peak_vram:.2f} GB | "
        f"Mode: {processor_detected} | "
        f"Speed: {tokens_per_second:.2f} tok/s"
    )

    unload_model(model_name)


# ─────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────

if __name__ == "__main__":

    prompt_200_tokens = (
        "Escribe una función en Python que implemente el algoritmo "
        "Quicksort de forma iterativa. Explica brevemente la ventaja "
        "de usar la versión iterativa sobre la recursiva para mitigar "
        "el desbordamiento de pila (stack overflow) y detalla su "
        "complejidad temporal en el peor caso."
    )

    modelos = [
        {
            "name": "qwen2.5:7b-instruct-q3_K_M",
            "tag": "Q3_K_M"
        },
        {
            "name": "qwen2.5:7b-instruct-q4_K_M",
            "tag": "Q4_K_M"
        },
        {
            "name": "qwen2.5:7b-instruct-q8_0",
            "tag": "Q8_0"
        }
    ]

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    for model in modelos:
        run_benchmark(
            model_name=model["name"],
            quantization=model["tag"],
            context_length=2048,
            prompt=prompt_200_tokens
        )

    print("\n🎉 Benchmark finalizado")
    print(f"📄 Resultados guardados en: {OUTPUT_FILE}")