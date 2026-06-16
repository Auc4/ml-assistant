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
# Project paths (REPO-CORRECT)
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CSV_FILE_B = DATA_DIR / "outputs" / "kv_cache_measurements.csv"

# ─────────────────────────────────────────────
# Model config
# ─────────────────────────────────────────────

MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"

# ─────────────────────────────────────────────
# Global state
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

if __name__ == "__main__":

    contextos = [512, 2048, 8192, 16384]

    if CSV_FILE_B.exists():
        CSV_FILE_B.unlink()

    for ctx in contextos:
        run_kv_benchmark(ctx)

    print(f"\n🎉 Experimento finalizado -> {CSV_FILE_B}")