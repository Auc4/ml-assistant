import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────
# Project paths (UPDATED)
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"

CSV_FILE_B = DATA_DIR / "outputs" / "kv_cache_measurements.csv"
PLOT_OUTPUT_B = DOCS_DIR / "kv_cache_study.png"

# ─────────────────────────────────────────────
# Safety check
# ─────────────────────────────────────────────

if not CSV_FILE_B.exists():
    raise FileNotFoundError(f"❌ No se encontró el archivo {CSV_FILE_B}")

DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────

df = pd.read_csv(CSV_FILE_B)

# ─────────────────────────────────────────────
# Figure setup
# ─────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ─────────────────────────────────────────────
# PANEL IZQUIERDO: Latencia de ingestión
# ─────────────────────────────────────────────

ax1.plot(
    df['context_length'],
    df['prompt_eval_latency_sec'],
    color='forestgreen',
    marker='o',
    linewidth=2.5,
    markersize=8,
    label='Latencia de Ingesta'
)

ax1.set_xscale('log', base=2)
ax1.set_xticks([512, 2048, 8192, 16384])
ax1.set_xticklabels(['512', '2048', '8192', '16384 (Timeout)'])

ax1.set_xlabel('Longitud del Contexto (Tokens)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Tiempo de Ingesta del Prompt (Segundos)',
               color='forestgreen', fontsize=11, fontweight='bold')

ax1.set_title('Impacto de Contexto en Latencia (Pre-fill)',
              fontsize=12, fontweight='bold')

ax1.grid(True, linestyle=':', alpha=0.6)

# Anotaciones
for _, row in df.iterrows():
    ax1.annotate(
        f"{row['prompt_eval_latency_sec']:.1f} s\n"
        f"({row['prompt_eval_latency_sec']/60:.1f} min)",
        (row['context_length'], row['prompt_eval_latency_sec']),
        textcoords="offset points",
        xytext=(0, 12),
        ha='center',
        fontweight='bold',
        color='darkgreen',
        fontsize=9
    )

# Timeout marker
ax1.plot(16384, 0, color='crimson', marker='X', markersize=10)
ax1.annotate(
    "TIMEOUT\n(> 20 min)",
    (16384, 0),
    textcoords="offset points",
    xytext=(0, 15),
    ha='center',
    fontweight='bold',
    color='crimson',
    fontsize=9
)

# ─────────────────────────────────────────────
# PANEL DERECHO: RAM + CPU temperature
# ─────────────────────────────────────────────

color_ram = 'darkorange'

ax2.plot(
    df['context_length'],
    df['peak_ram_gb'],
    color=color_ram,
    marker='s',
    linewidth=2.5,
    markersize=8,
    label='Pico de RAM (GB)'
)

ax2.set_xscale('log', base=2)
ax2.set_xticks([512, 2048, 8192, 16384])
ax2.set_xticklabels(['512', '2048', '8192', '16384'])

ax2.set_xlabel('Longitud del Contexto (Tokens)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Consumo de RAM del Sistema (GB)',
               color=color_ram, fontsize=11, fontweight='bold')

ax2.tick_params(axis='y', labelcolor=color_ram)
ax2.set_ylim(4.5, 6.5)

# RAM annotations
for i, txt in enumerate(df['peak_ram_gb']):
    ax2.annotate(
        f"{txt:.2f} GB",
        (df['context_length'].iloc[i], df['peak_ram_gb'].iloc[i]),
        textcoords="offset points",
        xytext=(-15, -15),
        ha='center',
        fontweight='bold',
        color='chocolate',
        fontsize=9
    )

# CPU temperature axis
ax3 = ax2.twinx()

color_temp = 'crimson'

ax3.plot(
    df['context_length'],
    df['cpu_temp_after_c'],
    color=color_temp,
    marker='^',
    linestyle='--',
    linewidth=2,
    markersize=8,
    label='Temp. Final CPU (°C)'
)

ax3.set_ylabel('Temperatura del Paquete CPU (°C)',
               color=color_temp,
               fontsize=11,
               fontweight='bold')

ax3.tick_params(axis='y', labelcolor=color_temp)
ax3.set_ylim(35, 60)

# Temperature annotations
for i, txt in enumerate(df['cpu_temp_after_c']):
    ax3.annotate(
        f"{txt:.1f} °C",
        (df['context_length'].iloc[i], df['cpu_temp_after_c'].iloc[i]),
        textcoords="offset points",
        xytext=(15, 10),
        ha='center',
        fontweight='bold',
        color='darkred',
        fontsize=9
    )

ax2.set_title('Escalamiento de Memoria RAM y Estrés Térmico',
              fontsize=12,
              fontweight='bold')

ax2.grid(True, linestyle=':', alpha=0.6)

# Legends
lineas2, etiquetas2 = ax2.get_legend_handles_labels()
lineas3, etiquetas3 = ax3.get_legend_handles_labels()

ax2.legend(lineas2 + lineas3, etiquetas2 + etiquetas3, loc='upper left')

# ─────────────────────────────────────────────
# Save figure
# ─────────────────────────────────────────────

plt.suptitle(
    'Análisis del Comportamiento de la Caché KV (Qwen2.5 7B Q4_K_M en CPU Pura)',
    fontsize=14,
    fontweight='bold',
    y=1.02
)

plt.tight_layout()

plt.savefig(PLOT_OUTPUT_B, bbox_inches='tight', dpi=300)

print(f"📊 Gráfica exportada en: {PLOT_OUTPUT_B}")