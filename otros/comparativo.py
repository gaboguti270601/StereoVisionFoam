import pandas as pd

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

EXCEL_PATH = r'D:\MDT\Pruebas\II28P\1\altura_estimada_ml.xlsx'

# Horas que quieres comparar
HORAS_OBJETIVO = [
    "2026-04-28 13:34:00",
    "2026-04-28 13:39:00",
    "2026-04-28 13:59:00",
    "2026-04-28 14:08:00",
    "2026-04-28 14:22:00",
    "2026-04-28 14:30:00",
    "2026-04-28 14:40:00",  
    "2026-04-28 14:52:00",
    "2026-04-28 15:03:00",
    "2026-04-28 15:15:00",
    "2026-04-28 15:20:00",
    "2026-04-28 15:28:00",
    "2026-04-28 15:30:00",
]

# Valores medidos manualmente
ALTURAS_REALES = [
    2.0,
    2.0,
    6.0,
    5.9,
    5.9,
    5.9,
    6.1,
    7.1,
    7.4,
    7.45,
    8.0,
    8.3,
    9.0
]

# ==========================================================
# CARGAR EXCEL
# ==========================================================

df = pd.read_excel(EXCEL_PATH)

# Convertir timestamps
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# ==========================================================
# BUSCAR VALORES MÁS CERCANOS
# ==========================================================

resultados = []

for hora_str, altura_real in zip(HORAS_OBJETIVO, ALTURAS_REALES):

    hora_obj = pd.to_datetime(hora_str)

    # Diferencia absoluta de tiempo
    diferencias = (df['Timestamp'] - hora_obj).abs()

    # Índice más cercano
    idx = diferencias.idxmin()

    fila = df.loc[idx]

    timestamp_excel = fila['Timestamp']
    altura_excel = fila['Altura_filtrada_cm']

    error = altura_excel - altura_real

    resultados.append({
        'Hora objetivo': hora_obj,
        'Hora encontrada': timestamp_excel,
        'Altura real (cm)': altura_real,
        'Altura estimada (cm)': round(altura_excel, 2),
        'Error (cm)': round(error, 2)
    })

# ==========================================================
# MOSTRAR RESULTADOS
# ==========================================================

resultado_df = pd.DataFrame(resultados)

print("\n===================================================")
print("COMPARACIÓN ALTURAS")
print("===================================================\n")

print(resultado_df.to_string(index=False))

# ==========================================================
# ERROR PROMEDIO
# ==========================================================

mae = resultado_df['Error (cm)'].abs().mean()
std = resultado_df['Error (cm)'].std()


print("\n===================================================")
print(f"ERROR ABSOLUTO MEDIO: {mae:.2f} cm")
print("===================================================\n")

print("\n===================================================")
print(f"Desviación estándar: {std:.2f} cm")
print("===================================================\n")