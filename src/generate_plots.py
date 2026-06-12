import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "data/measurements.csv"
PLOT_OUTPUT = "docs/quantization_study.png"

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"❌ No se encontró el archivo {CSV_FILE}. Primero corre el benchmark.")

# 1. Leer los datos reales de tu CSV
df = pd.read_csv(CSV_FILE)

# Filtrar solo los datos con contexto de 2048 (Parte A)
df_part_a = df[df['context_length'] == 2048].copy()

# ====================================================================
# ⚠️ TUS PROMEDIOS REALES DE CALIDAD (0-3)
# ====================================================================
puntajes_calidad = [2.55, 2.45, 2.7] 
df_part_a['quality_score'] = puntajes_calidad

# 2. Configurar el gráfico de doble eje Y
fig, ax1 = plt.subplots(figsize=(10, 6))

# Eje Izquierdo: Velocidad (Barras)
color_barras = 'skyblue'
ax1.set_xlabel('Variantes de Cuantización (Qwen 2.5 7B)', fontsize=11, fontweight='bold', labelpad=10)
ax1.set_ylabel('Velocidad de Inferencia (Tokens/segundo)', color='steelblue', fontsize=11, fontweight='bold')

barras = ax1.bar(df_part_a['quantization'], df_part_a['tokens_per_second'], 
                 color=color_barras, alpha=0.7, edgecolor='dodgerblue', width=0.35, label='Velocidad (tok/s)')
ax1.tick_params(axis='y', labelcolor='steelblue')

# 🔥 SOLUCIÓN: Colocar etiquetas de Velocidad Y RAM sobre las barras
for i, barra in enumerate(barras):
    yval = barra.get_height()
    # Extraemos el valor real de RAM guardado en tu CSV para esa fila
    ram_val = df_part_a['peak_ram_gb'].iloc[i]
    
    # Texto combinado: muestra tokens/s y justo abajo el consumo de RAM
    texto_barra = f"{yval:.2f} tok/s\n(RAM: {ram_val:.2f} GB)"
    
    ax1.text(barra.get_x() + barra.get_width()/2.0, yval + 0.5, 
             texto_barra, ha='center', va='bottom', color='darkblue', fontsize=9, fontweight='bold')
    
# Eje Derecho: Calidad (Línea)
ax2 = ax1.twinx()  
ax2.set_ylabel('Calidad Promedio (Escala 0 - 3)', color='crimson', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 3.5) # Un poco de margen superior para los textos

# Dibujamos la línea de calidad
linea_calidad = ax2.plot(df_part_a['quantization'], df_part_a['quality_score'], 
                         color='crimson', marker='o', linewidth=2.5, markersize=8, label='Calidad Promedio (0-3)')
ax2.tick_params(axis='y', labelcolor='crimson')

# Colocar el valor de la nota encima de cada punto de la línea (con 2 decimales exactos)
for i, txt in enumerate(df_part_a['quality_score']):
    ax2.annotate(f"{txt:.2f}/3", (df_part_a['quantization'].iloc[i], df_part_a['quality_score'].iloc[i]),
                 textcoords="offset points", xytext=(0,12), ha='center', color='crimson', fontweight='bold')

# Leyendas unificadas
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Estudio de Impacto de Cuantización (Hardware: i7-10700K CPU)', fontsize=13, fontweight='bold', pad=15)
ax1.grid(True, linestyle=':', alpha=0.5)
fig.tight_layout()

plt.savefig(PLOT_OUTPUT, bbox_inches='tight', dpi=300)
print(f"📊 Gráfica exportada con éxito incluyendo métricas de RAM en: {PLOT_OUTPUT}")