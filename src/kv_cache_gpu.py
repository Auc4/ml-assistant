import time
import csv
import os
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
PS_URL = "http://localhost:11434/api/ps"
CSV_FILE_GPU = "data/kv_cache_gpu_q8.csv"
MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"

def unload_model():
    """Descarga el modelo para limpiar la VRAM antes de empezar"""
    try:
        requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "keep_alive": 0}, timeout=10)
        print("🧹 Limpiando la VRAM de la GPU...")
        time.sleep(5)
    except Exception as e:
        print(f"⚠️ Error al descargar: {e}")

def get_vram_usage():
    """Detecta cuánta VRAM está usando el modelo en caliente"""
    try:
        data = requests.get(PS_URL, timeout=10).json()
        for model in data.get("models", []):
            if model["name"] == MODEL_NAME:
                return model.get("size_vram", 0) / (1024 ** 3)
    except Exception as e:
        print(f"⚠️ Error leyendo /api/ps: {e}")
    return 0

def run_gpu_q8_benchmark(context_length):
    unload_model()
    print(f"\n🚀 Lanzando Inferencia en GPU NVIDIA GTX 1070")
    print(f"📏 Contexto Objetivo: {context_length} tokens")
    print(f"🔒 Caché KV Forzada a: Q8_0")

    # Replicamos el prompt exacto de la Parte B
    palabras_relleno = int(context_length * 1.3)
    prompt_contexto = "data " * palabras_relleno + "\n\nResponde en exactamente 50 palabras: ¿Qué es la ciberseguridad y por qué es vital en sistemas operativos Linux?"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_contexto,
        "stream": False,
        "options": {
            "num_ctx": context_length,
            "temperature": 0.0,
            # 1. Quitamos el bloqueo de CPU dejando que Ollama use la GPU de Arch por defecto
            # 2. Inyectamos la cuantización experimental Q8 para la caché KV
            "kv_overrides": [{"key": "kv_cache_type", "type": "string", "value": "q8_0"}]
        }
    }

    start_time = time.time()
    try:
        # Timeout holgado, aunque en GPU debería tardar poquísimo
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        response.raise_for_status()
        response_data = response.json()
    except Exception as e:
        print(f"❌ Error en la inferencia en GPU: {e}")
        return

    end_time = time.time()
    total_duration_sec = end_time - start_time

    # Extraer métricas nativas de Ollama
    eval_count = response_data.get("eval_count", 0)
    eval_duration_sec = response_data.get("eval_duration", 0) / 1e9
    prompt_eval_duration_sec = response_data.get("prompt_eval_duration", 0) / 1e9

    tokens_per_second = eval_count / eval_duration_sec if eval_duration_sec > 0 else 0
    
    # Capturar la VRAM exacta en caliente antes de descargar
    peak_vram = get_vram_usage()

    # Guardar los resultados en un CSV independiente para la parte B.4
    file_exists = os.path.isfile(CSV_FILE_GPU)
    os.makedirs(os.path.dirname(CSV_FILE_GPU), exist_ok=True)

    with open(CSV_FILE_GPU, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["context_length", "peak_vram_gb", "tokens_per_second", "prompt_eval_latency_sec", "total_duration_sec"])
        writer.writerow([context_length, f"{peak_vram:.2f}", f"{tokens_per_second:.2f}", f"{prompt_eval_duration_sec:.2f}", f"{total_duration_sec:.2f}"])

    print(f"\n✅ ¡Prueba Completada con Éxito!")
    print(f"📊 VRAM Ocupada por el Modelo + Caché Q8: {peak_vram:.2f} GB")
    print(f"⏱️ Latencia de Ingesta (Pre-fill): {prompt_eval_duration_sec:.2f} s")
    print(f"⚡ Velocidad de Generación: {tokens_per_second:.2f} tok/s")
    print(f"⏳ Duración Total: {total_duration_sec:.2f} s")
    
    unload_model()

if __name__ == "__main__":
    # La rúbrica pide repetir UNA configuración, usaremos la crítica de 8k
    run_gpu_q8_benchmark(8192)