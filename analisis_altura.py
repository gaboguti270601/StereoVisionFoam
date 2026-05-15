import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_excel(r'D:\MDT\Pruebas\II29P\1\altura_estimada_ml.xlsx')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # convertir a datetime

# --- Gráfico de líneas ---
plt.figure(figsize=(10,4))
#plt.plot(df['Timestamp'], df['Altura_cm'], color='blue', marker='o', linestyle='-')
plt.plot(df['Timestamp'], df['Altura_filtrada_cm'], color='blue', marker='o', linestyle='-')
plt.title("Altura vs Tiempo")
plt.xlabel("Timestamp")
plt.ylabel("Altura (cm)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()