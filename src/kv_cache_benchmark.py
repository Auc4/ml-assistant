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
PS_URL = "http://localhost:11434/api/ps"

# ─────────────────────────────────────────────
# Project paths (UPDATED)
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CSV_FILE_B = DATA_DIR / "outputs" / "kv_cache_measurements.csv"

MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"

# ─────────────────────────────────────────────
# Global monitoring state
# ─────────────────────────────────────────────

monitoring = False
peak_ram_used = 0

def unload_model():
    try:
        requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "keep_alive": 0
            },
            timeout=10
        )
        print("🧹 Vaciando caché de memoria en Ollama...")
        time.sleep(5)

    except Exception as e:
        print(f"⚠️ Error descargando modelo: {e}")


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


def get_vram_usage():
    try:
        data = requests.get(PS_URL, timeout=10).json()

        for model in data.get("models", []):
            if model["name"] == MODEL_NAME:
                return model.get("size_vram", 0) / (1024 ** 3)

    except Exception as e:
        print(f"⚠️ Error leyendo VRAM: {e}")

    return 0


def get_cpu_temperature():
    try:
        temps = psutil.sensors_temperatures()

        if "coretemp" in temps:
            for entry in temps["coretemp"]:
                if entry.label == "Package id 0":
                    return entry.current

    except Exception:
        pass

    return None


def run_kv_benchmark(context_length):
    global monitoring, peak_ram_used

    unload_model()

    print(f"\n🔥 Evaluando KV Cache -> Contexto: {context_length}")

    cpu_temp_before = get_cpu_temperature()

    palabras_relleno = int(context_length * 1.3)

    prompt_contexto = (
        "data " * palabras_relleno
        + "\n\nResponde en exactamente 50 palabras: "
        "¿Qué es la ciberseguridad y por qué es vital en sistemas operativos Linux?"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_contexto,
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
            timeout=1200
        )

        response.raise_for_status()
        response_data = response.json()

    except Exception as e:
        monitoring = False
        ram_thread.join()
        print(f"❌ Error en inferencia: {e}")
        return

    end_time = time.time()

    monitoring = False
    ram_thread.join()

    total_duration_sec = end_time - start_time

    eval_count = response_data.get("eval_count", 0)
    prompt_eval_count = response_data.get("prompt_eval_count", 0)

    eval_duration_sec = response_data.get("eval_duration", 0) / 1e9
    prompt_eval_duration_sec = response_data.get("prompt_eval_duration", 0) / 1e9

    tokens_per_second = (
        eval_count / eval_duration_sec if eval_duration_sec > 0 else 0
    )

    prompt_tokens_per_second = (
        prompt_eval_count / prompt_eval_duration_sec if prompt_eval_duration_sec > 0 else 0
    )

    peak_vram = get_vram_usage()
    processor_detected = "CPU" if peak_vram < 0.1 else "GPU/PARCIAL"

    cpu_temp_after = get_cpu_temperature()

    # ─────────────────────────────────────────────
    # Output handling (UPDATED)
    # ─────────────────────────────────────────────

    CSV_FILE_B.parent.mkdir(parents=True, exist_ok=True)
    file_exists = CSV_FILE_B.exists()

    with open(CSV_FILE_B, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "context_length",
                "prompt_eval_count",
                "eval_count",
                "peak_ram_gb",
                "peak_vram_gb",
                "processor_detected",
                "tokens_per_second",
                "prompt_tokens_per_second",
                "prompt_eval_latency_sec",
                "total_duration_sec",
                "cpu_temp_before_c",
                "cpu_temp_after_c"
            ])

        writer.writerow([
            context_length,
            prompt_eval_count,
            eval_count,
            f"{peak_ram_used:.2f}",
            f"{peak_vram:.2f}",
            processor_detected,
            f"{tokens_per_second:.2f}",
            f"{prompt_tokens_per_second:.2f}",
            f"{prompt_eval_duration_sec:.2f}",
            f"{total_duration_sec:.2f}",
            f"{cpu_temp_before:.1f}" if cpu_temp_before else "N/A",
            f"{cpu_temp_after:.1f}" if cpu_temp_after else "N/A"
        ])

    print(
        f"✅ Contexto {context_length} -> "
        f"RAM: {peak_ram_used:.2f} GB | "
        f"VRAM: {peak_vram:.2f} GB | "
        f"Modo: {processor_detected} | "
        f"tok/s: {tokens_per_second:.2f}"
    )

    unload_model()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":

    contextos = [512, 2048, 8192, 16384]

    if CSV_FILE_B.exists():
        CSV_FILE_B.unlink()

    for ctx in contextos:
        run_kv_benchmark(ctx)

    print(f"\n🎉 Experimento finalizado -> {CSV_FILE_B}")