import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# =========================================================
# CONFIGURACIÓN
# =========================================================

CSV_DATASET = r"D:\MDT\Pruebas\II28P\1\dataset_features.csv"


# =========================================================
# LEER DATASET
# =========================================================

df = pd.read_csv(CSV_DATASET)

print(df.head())


# =========================================================
# LIMPIEZA
# =========================================================

# eliminar timestamp
if "timestamp" in df.columns:
    df = df.drop(columns=["timestamp"])

# eliminar filas malas/corruptas
df = df[df["std"] > 1]

# reset index
df = df.reset_index(drop=True)

print("\n===================================")
print("DATASET LIMPIO")
print("===================================\n")

print(df)


# =========================================================
# DATA AUGMENTATION
# =========================================================

dataset_aug = []

for _, row in df.iterrows():

    # guardar muestra original
    dataset_aug.append(row.copy())

    # crear muestras sintéticas
    for i in range(20):

        nueva = row.copy()

        # -------------------------------------------------
        # ruido leve en features
        # -------------------------------------------------

        nueva["mean"] += (
            nueva["mean"]
            * 0.02
            * (2 * np.random.rand() - 1)
        )

        nueva["std"] += (
            nueva["std"]
            * 0.03
            * (2 * np.random.rand() - 1)
        )

        nueva["min"] += (
            nueva["min"]
            * 0.03
            * (2 * np.random.rand() - 1)
        )

        nueva["max"] += (
            nueva["max"]
            * 0.03
            * (2 * np.random.rand() - 1)
        )

        nueva["lap_var"] += (
            nueva["lap_var"]
            * 0.05
            * (2 * np.random.rand() - 1)
        )

        nueva["fft_mean"] += (
            nueva["fft_mean"]
            * 0.03
            * (2 * np.random.rand() - 1)
        )

        nueva["fft_std"] += (
            nueva["fft_std"]
            * 0.03
            * (2 * np.random.rand() - 1)
        )

        nueva["perfil_std"] += (
            nueva["perfil_std"]
            * 0.05
            * (2 * np.random.rand() - 1)
        )

        nueva["perfil_grad"] += (
            nueva["perfil_grad"]
            * 0.05
            * (2 * np.random.rand() - 1)
        )

        # -------------------------------------------------
        # ruido pequeño en caudal
        # -------------------------------------------------

        nueva["caudal"] += (
            nueva["caudal"]
            * 0.02
            * (2 * np.random.rand() - 1)
        )

        # -------------------------------------------------
        # ruido pequeño en altura
        # -------------------------------------------------

        nueva["altura"] += (
            0.15
            * (2 * np.random.rand() - 1)
        )

        dataset_aug.append(nueva)

# =========================================================
# DATAFRAME AUMENTADO
# =========================================================

df_aug = pd.DataFrame(dataset_aug)

print("\n===================================")
print("DATASET AUMENTADO")
print("===================================\n")

print(df_aug.head())

print(f"\nCantidad muestras: {len(df_aug)}")


# =========================================================
# MACHINE LEARNING
# =========================================================

X = df_aug.drop(columns=["altura"])

y = df_aug["altura"]

# =========================================================
# TRAIN / TEST
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# =========================================================
# MODELO
# =========================================================

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# PREDICCIÓN
# =========================================================

pred = model.predict(X_test)

# =========================================================
# RESULTADOS
# =========================================================

mae = mean_absolute_error(
    y_test,
    pred
)

print("\n===================================")
print(f"MAE: {mae:.2f} cm")
print("===================================\n")

resultados = pd.DataFrame({
    "Real": y_test.values,
    "Predicho": pred
})

resultados["Error"] = (
    resultados["Predicho"]
    - resultados["Real"]
)

print(resultados.head(20))


# =========================================================
# IMPORTANCIA FEATURES
# =========================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importancia": model.feature_importances_
})

importance = importance.sort_values(
    by="Importancia",
    ascending=False
)

print("\n===================================")
print("IMPORTANCIA FEATURES")
print("===================================\n")

print(importance)


# =========================================================
# GUARDAR MODELO
# =========================================================

modelo_salida = r"D:\MDT\Pruebas\II28P\1\modelo_espuma.pkl"

joblib.dump(model, modelo_salida)

print("\nModelo guardado en:")
print(modelo_salida)


# =========================================================
# GUARDAR DATASET AUMENTADO
# =========================================================

dataset_salida = r"D:\MDT\Pruebas\II28P\1\dataset_aumentado.csv"

df_aug.to_csv(dataset_salida, index=False)

print("\nDataset aumentado guardado en:")
print(dataset_salida)