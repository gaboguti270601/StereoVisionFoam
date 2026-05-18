import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================================
# CONFIGURACIÓN
# =========================================================

CSV_REAL = r"D:\MDT\Pruebas\II28P\1\dataset_features.csv"
CSV_AUMENTADO = r"D:\MDT\Pruebas\II28P\1\dataset_aumentado.csv"

MODELO_SALIDA = r"D:\MDT\Pruebas\II28P\1\modelo_espuma_validado.pkl"

# =========================================================
# CARGAR DATASETS
# =========================================================

df_real = pd.read_csv(CSV_REAL)
df_aug = pd.read_csv(CSV_AUMENTADO)

# =========================================================
# LIMPIEZA
# =========================================================

if "timestamp" in df_real.columns:
    df_real = df_real.drop(columns=["timestamp"])

if "timestamp" in df_aug.columns:
    df_aug = df_aug.drop(columns=["timestamp"])

# eliminar muestra corrupta
df_real = df_real[df_real["std"] > 1].reset_index(drop=True)
df_aug = df_aug[df_aug["std"] > 1].reset_index(drop=True)

# =========================================================
# SEPARAR FEATURES / TARGET
# =========================================================

X_train = df_aug.drop(columns=["altura"])
y_train = df_aug["altura"]

X_test = df_real.drop(columns=["altura"])
y_test = df_real["altura"]

# asegurar mismo orden de columnas
X_test = X_test[X_train.columns]

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
# VALIDACIÓN REAL
# =========================================================

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

resultados = pd.DataFrame({
    "Real_cm": y_test.values,
    "Predicho_cm": pred,
})

resultados["Error_cm"] = resultados["Predicho_cm"] - resultados["Real_cm"]
resultados["Error_abs_cm"] = resultados["Error_cm"].abs()

print("\n===================================")
print("VALIDACIÓN SOLO CON DATOS REALES")
print("===================================\n")

print(resultados)

print("\n===================================")
print(f"MAE real: {mae:.2f} cm")
print(f"R2 real : {r2:.3f}")
print("===================================\n")

# =========================================================
# IMPORTANCIA FEATURES
# =========================================================

importance = pd.DataFrame({
    "Feature": X_train.columns,
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

joblib.dump(model, MODELO_SALIDA)

print("\nModelo guardado en:")
print(MODELO_SALIDA)