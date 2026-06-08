import pandas as pd
import numpy as np

# =============================
# CONFIGURACIÓN
# =============================

ARCHIVO = r"D:\MDT\Pruebas\II28P\1\dataset_features.csv"

SALIDA = r"D:\MDT\Pruebas\II28P\1\dataset_aumentado.csv"

NUEVAS_MUESTRAS = 300

RUIDO = 0.03

# =============================
# CARGAR DATASET
# =============================

print("Cargando dataset...")

df = pd.read_csv(ARCHIVO)

print()
print(df.head())

# =============================
# COLUMNAS A NO MODIFICAR
# =============================

columnas_fijas = [
    "timestamp",
    "temperatura",
    "experiencia"
]

# =============================
# FEATURES
# =============================

features = [
    c for c in df.columns
    if c not in columnas_fijas
]

# =============================
# GENERAR MUESTRAS SINTÉTICAS
# =============================

muestras = []

for _ in range(NUEVAS_MUESTRAS):

    # seleccionar dos muestras reales
    a = df.sample(1).iloc[0]
    b = df.sample(1).iloc[0]

    alpha = np.random.rand()

    nueva = {}

    for col in df.columns:

        # -------------------------------------------------
        # columnas fijas
        # -------------------------------------------------

        if col in columnas_fijas:

            nueva[col] = a[col]
            continue

        # -------------------------------------------------
        # interpolación
        # -------------------------------------------------

        va = a[col]
        vb = b[col]

        valor = (1 - alpha) * va + alpha * vb

        # -------------------------------------------------
        # ruido gaussiano
        # -------------------------------------------------

        sigma = abs(valor) * RUIDO

        valor += np.random.normal(0, sigma)

        nueva[col] = valor

    muestras.append(nueva)

# =============================
# DATAFRAME NUEVO
# =============================

df_sint = pd.DataFrame(muestras)

df_total = pd.concat(
    [df, df_sint],
    ignore_index=True
)

# =============================
# GUARDAR
# =============================

print()
print("===================================")
print("RESULTADOS")
print("===================================")

print()
print("Muestras originales :", len(df))
print("Muestras sintéticas :", len(df_sint))
print("Total :", len(df_total))

print()
print("Guardando dataset aumentado...")

df_total.to_csv(SALIDA, index=False)

print()
print("Dataset guardado en:")
print(SALIDA)

# =============================
# ESTADÍSTICAS
# =============================

print()
print("===================================")
print("ESTADÍSTICAS")
print("===================================")

print()
print(df_total.describe())