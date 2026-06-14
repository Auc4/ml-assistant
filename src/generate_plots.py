import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "data/measurements.csv"
PLOT_OUTPUT = "docs/quantization_study.png"

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"❌ No se encontró el archivo {CSV_FILE}. Primero corre el benchmark.")

# 1. Leer los datos reales de tu nuevo CSV balanceado
df = pd.read_csv(CSV_FILE)

# Filtrar solo los datos con contexto de 2048 (Parte A)
df_part_a = df[df['context_length'] == 2048].copy()

# ====================================================================
# ⚠️ TUS PROMEDIOS REALES DE CALIDAD (0-3)
# ====================================================================
puntajes_calidad = [2.55, 2.45, 2.7] 
df_part_a['quality_score'] = puntajes_calidad

# 2. Configurar el gráfico de doble eje Y
fig, ax1 = plt.subplots(figsize=(10, 6.5))

# Eje Izquierdo: Velocidad (Barras Celestes)
color_barras = 'skyblue'
ax1.set_xlabel('Variantes de Cuantización (Qwen 2.5 7B)', fontsize=11, fontweight='bold', labelpad=10)
ax1.set_ylabel('Velocidad de Inferencia (Tokens/segundo)', color='steelblue', fontsize=11, fontweight='bold')

# 🔥 SOLUCIÓN: Reducimos el ancho de las barras (width) de 0.35 a 0.25 para que se vean más estilizadas
barras = ax1.bar(df_part_a['quantization'], df_part_a['tokens_per_second'], 
                 color=color_barras, alpha=0.7, edgecolor='dodgerblue', width=0.25, label='Velocidad (tok/s)')
ax1.tick_params(axis='y', labelcolor='steelblue')

# 🔥 SOLUCIÓN EXTRA: Subimos el límite del eje Y izquierdo para que el título no choque con el texto flotante
ax1.set_ylim(0, max(df_part_a['tokens_per_second']) * 1.25)

# Colocar etiquetas de Velocidad, RAM y VRAM sobre las barras
for i, barra in enumerate(barras):
    yval = barra.get_height()
    
    ram_val = float(df_part_a['peak_ram_gb'].iloc[i])
    vram_val = float(df_part_a['peak_vram_gb'].iloc[i])
    proc_mode = df_part_a['processor_detected'].iloc[i]
    
    texto_barra = f"{yval:.2f} tok/s\nRAM: {ram_val:.2f} GB\nVRAM: {vram_val:.2f} GB ({proc_mode})"
    
    # Bajamos ligeramente el offset vertical (+ 0.2) para ajustar el texto
    ax1.text(barra.get_x() + barra.get_width()/2.0, yval + 0.3, 
             texto_barra, ha='center', va='bottom', color='darkblue', fontsize=8.5, fontweight='bold')
    
# Eje Derecho: Calidad (Línea Roja)
ax2 = ax1.twinx()  
ax2.set_ylabel('Calidad Promedio (Escala 0 - 3)', color='crimson', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 3.6) 

# Dibujamos la línea de calidad
linea_calidad = ax2.plot(df_part_a['quantization'], df_part_a['quality_score'], 
                         color='crimson', marker='o', linewidth=2.5, markersize=8, label='Calidad Promedio (0-3)')
ax2.tick_params(axis='y', labelcolor='crimson')

# Colocar el valor de la nota encima de cada hito
for i, txt in enumerate(df_part_a['quality_score']):
    ax2.annotate(f"{txt:.2f}/3", (df_part_a['quantization'].iloc[i], df_part_a['quality_score'].iloc[i]),
                 textcoords="offset points", xytext=(0,12), ha='center', color='crimson', fontweight='bold')

# Leyendas unificadas
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Añadimos un padding (pad=25) al título para darle una separación extra por seguridad
plt.title('Estudio de Impacto de Cuantización (Hardware: i7-10700K CPU - Modo Aislado)', fontsize=13, fontweight='bold', pad=25)
ax1.grid(True, linestyle=':', alpha=0.5)
fig.tight_layout()

# Exportación
os.makedirs(os.path.dirname(PLOT_OUTPUT), exist_ok=True)
plt.savefig(PLOT_OUTPUT, bbox_inches='tight', dpi=300)

print(f"📊 Gráfica optimizada exportada con éxito en: {PLOT_OUTPUT}")