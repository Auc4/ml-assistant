import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE_B = "data/kv_cache_measurements.csv"
PLOT_OUTPUT_B = "docs/kv_cache_study.png"

if not os.path.exists(CSV_FILE_B):
    raise FileNotFoundError(f"❌ No se encontró el archivo {CSV_FILE_B}.")

# 1. Leer los datos reales de tu telemetría
df = pd.read_csv(CSV_FILE_B)

# Crear la figura con dos subgráficas (Panel Izquierdo y Panel Derecho)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- PANEL IZQUIERDO: Contexto vs Latencia de Ingesta (Segundos) ---
ax1.plot(df['context_length'], df['prompt_eval_latency_sec'], 
         color='forestgreen', marker='o', linewidth=2.5, markersize=8, label='Latencia de Ingesta')
ax1.set_xscale('log', base=2)
ax1.set_xticks([512, 2048, 8192, 16384])
ax1.set_xticklabels(['512', '2048', '8192', '16384 (Timeout)'])
ax1.set_xlabel('Longitud del Contexto (Tokens)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Tiempo de Ingesta del Prompt (Segundos)', color='forestgreen', fontsize=11, fontweight='bold')
ax1.set_title('Impacto de Contexto en Latencia (Pre-fill)', fontsize=12, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.6)

# Anotaciones de tiempo en el eje izquierdo
for i, row in df.iterrows():
    ax1.annotate(f"{row['prompt_eval_latency_sec']:.1f} s\n({row['prompt_eval_latency_sec']/60:.1f} min)", 
                 (row['context_length'], row['prompt_eval_latency_sec']),
                 textcoords="offset points", xytext=(0,12), ha='center', fontweight='bold', color='darkgreen', fontsize=9)

# Dibujar el punto del colapso/timeout de forma manual en los 16k
ax1.plot(16384, 0, color='crimson', marker='X', markersize=10)
ax1.annotate("TIMEOUT\n(> 20 min)", (16384, 0),
             textcoords="offset points", xytext=(0,15), ha='center', fontweight='bold', color='crimson', fontsize=9)


# --- PANEL DERECHO: Contexto vs RAM y Temperatura del Silicio ---
color_ram = 'darkorange'
ax2.plot(df['context_length'], df['peak_ram_gb'], 
         color=color_ram, marker='s', linewidth=2.5, markersize=8, label='Pico de RAM (GB)')
ax2.set_xscale('log', base=2)
ax2.set_xticks([512, 2048, 8192, 16384])
ax2.set_xticklabels(['512', '2048', '8192', '16384'])
ax2.set_xlabel('Longitud del Contexto (Tokens)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Consumo de RAM del Sistema (GB)', color=color_ram, fontsize=11, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_ram)
ax2.set_ylim(4.5, 6.5)

# Anotaciones de los gigabytes de RAM
for i, txt in enumerate(df['peak_ram_gb']):
    ax2.annotate(f"{txt:.2f} GB", (df['context_length'].iloc[i], df['peak_ram_gb'].iloc[i]),
                 textcoords="offset points", xytext=(-15,-15), ha='center', fontweight='bold', color='chocolate', fontsize=9)

# Crear el segundo eje Y para la temperatura de la CPU
ax3 = ax2.twinx()
color_temp = 'crimson'
ax3.plot(df['context_length'], df['cpu_temp_after_c'], 
         color=color_temp, marker='^', linestyle='--', linewidth=2, markersize=8, label='Temp. Final CPU (°C)')
ax3.set_ylabel('Temperatura del Paquete CPU (°C)', color=color_temp, fontsize=11, fontweight='bold')
ax3.tick_params(axis='y', labelcolor=color_temp)
ax3.set_ylim(35, 60)

# Anotaciones de la temperatura
for i, txt in enumerate(df['cpu_temp_after_c']):
    ax3.annotate(f"{txt:.1f} °C", (df['context_length'].iloc[i], df['cpu_temp_after_c'].iloc[i]),
                 textcoords="offset points", xytext=(15,10), ha='center', fontweight='bold', color='darkred', fontsize=9)

ax2.set_title('Escalamiento de Memoria RAM y Estrés Térmico', fontsize=12, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.6)

# Unificar etiquetas de leyendas del panel derecho
lineas2, etiquetas2 = ax2.get_legend_handles_labels()
lineas3, etiquetas3 = ax3.get_legend_handles_labels()
ax2.legend(lineas2 + lineas3, etiquetas2 + etiquetas3, loc='upper left')

plt.suptitle('Análisis del Comportamiento de la Caché KV (Qwen2.5 7B Q4_K_M en CPU Pura)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Guardar la gráfica
os.makedirs(os.path.dirname(PLOT_OUTPUT_B), exist_ok=True)
plt.savefig(PLOT_OUTPUT_B, bbox_inches='tight', dpi=300)
print(f"📊 ¡Espectacular! Gráfica de la Parte B exportada en: {PLOT_OUTPUT_B}")