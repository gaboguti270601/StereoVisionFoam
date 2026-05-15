import cv2
import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime, timedelta

# =========================================================
# CONFIGURACIÓN II29P
# =========================================================

CARPETA_VIDEOS = r"D:\MDT\Pruebas\II29P\1"

CSV_DATOS = r"D:\MDT\II29P.xlsx"

SALIDA_DATASET = r"D:\MDT\Pruebas\II29P\1\dataset_features.csv"

# ROI fijo
# Puedes cambiarlo si II29P tiene otro encuadre
ROI = {
    "x": 435,
    "y": 736,
    "w": 533,
    "h": 398
}

TEMPERATURA_C = 1350
EXPERIENCIA = 29

# =========================================================
# FUNCIONES
# =========================================================

def obtener_timestamp_video(video_path):
    base = os.path.basename(video_path)
    fecha_str = base.split("_")[1].replace(".avi", "")

    return datetime.strptime(
        fecha_str,
        "%Y-%m-%dT%H-%M-%S.%f"
    )


def buscar_video_y_frame(videos, tiempo_objetivo):

    for video in videos:

        inicio = obtener_timestamp_video(video)

        cap = cv2.VideoCapture(video)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            cap.release()
            continue

        duracion = total_frames / fps
        fin = inicio + timedelta(seconds=duracion)

        if inicio <= tiempo_objetivo <= fin:

            delta = (tiempo_objetivo - inicio).total_seconds()
            frame_no = int(delta * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

            ret, frame = cap.read()

            cap.release()

            if ret:
                return frame, video, frame_no

    return None, None, None


def preprocesar_roi(roi):

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray


def extraer_features(gray):

    features = {}

    features["mean"] = np.mean(gray)
    features["std"] = np.std(gray)

    features["min"] = np.min(gray)
    features["max"] = np.max(gray)

    features["p10"] = np.percentile(gray, 10)
    features["p25"] = np.percentile(gray, 25)
    features["p50"] = np.percentile(gray, 50)
    features["p75"] = np.percentile(gray, 75)
    features["p90"] = np.percentile(gray, 90)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    features["lap_var"] = lap.var()

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    features["sobelx_mean"] = np.mean(np.abs(sobelx))
    features["sobely_mean"] = np.mean(np.abs(sobely))

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    features["fft_mean"] = np.mean(magnitude)
    features["fft_std"] = np.std(magnitude)

    perfil = np.mean(gray, axis=1)

    features["perfil_std"] = np.std(perfil)
    features["perfil_grad"] = np.mean(np.abs(np.gradient(perfil)))

    return features


# =========================================================
# MAIN
# =========================================================

videos = sorted(
    glob.glob(os.path.join(CARPETA_VIDEOS, "*.avi"))
)

print("\nVideos encontrados:\n")

for v in videos:
    print(os.path.basename(v))

if len(videos) == 0:
    print("No se encontraron videos.")
    exit()

df_datos = pd.read_excel(CSV_DATOS)

print("\nDatos cargados:")
print(df_datos)

dataset = []

for _, row in df_datos.iterrows():

    timestamp = pd.to_datetime(row["Hora"]).to_pydatetime()
    caudal = float(row["Caudal_LPM"])
    altura = float(row["Altura de Espuma_cm"])

    print(f"\nBuscando frame para: {timestamp}")

    frame, video_usado, frame_no = buscar_video_y_frame(
        videos,
        timestamp
    )

    if frame is None:
        print("No encontrado")
        continue

    x = ROI["x"]
    y = ROI["y"]
    w = ROI["w"]
    h = ROI["h"]

    roi = frame[y:y+h, x:x+w]

    if roi.size == 0:
        print("ROI inválido")
        continue

    gray = preprocesar_roi(roi)

    # descartar frames corruptos/negros
    if np.std(gray) < 1:
        print("Frame descartado por baja variación")
        continue

    features = extraer_features(gray)

    features["caudal"] = caudal
    features["temperatura"] = TEMPERATURA_C
    features["experiencia"] = EXPERIENCIA
    features["altura"] = altura
    features["timestamp"] = timestamp

    dataset.append(features)

    print(
        f"OK | Video: {os.path.basename(video_usado)} | "
        f"Frame: {frame_no} | Altura: {altura:.2f} cm"
    )

    # visualización rápida del ROI
    vis = roi.copy()

    cv2.putText(
        vis,
        f"{timestamp.strftime('%H:%M:%S')} | {altura:.2f} cm",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ROI usado", vis)
    cv2.waitKey(300)

cv2.destroyAllWindows()

df_features = pd.DataFrame(dataset)

print("\n===================================")
print("DATASET FEATURES")
print("===================================\n")

print(df_features)

df_features.to_csv(SALIDA_DATASET, index=False)

print("\nDataset guardado en:")
print(SALIDA_DATASET)