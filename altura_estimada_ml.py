import cv2
import numpy as np
import pandas as pd
import os
import glob
import joblib
from datetime import datetime, timedelta

# =========================================================
# CONFIGURACIÓN II28P
# =========================================================

CARPETA_VIDEOS = r"D:\MDT\Pruebas\II28P\1"

MODELO_PATH = r"D:\MDT\Pruebas\II28P\1\modelo_espuma_validado.pkl"

SALIDA_EXCEL = r"D:\MDT\Pruebas\II28P\1\altura_estimada_ml.xlsx"

ROI = {
    "x": 435,
    "y": 736,
    "w": 533,
    "h": 398
}

FRAME_SKIP = 30

ALPHA = 0.85

TEMPERATURA_C = 1250
EXPERIENCIA = 28

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


def obtener_caudal(timestamp):
    hora = timestamp.time()

    if hora < datetime.strptime("13:52", "%H:%M").time():
        return 0.00
    elif hora < datetime.strptime("14:22", "%H:%M").time():
        return 0.05
    elif hora < datetime.strptime("14:40", "%H:%M").time():
        return 0.07
    elif hora < datetime.strptime("14:52", "%H:%M").time():
        return 0.10
    elif hora < datetime.strptime("15:03", "%H:%M").time():
        return 0.15
    elif hora < datetime.strptime("15:20", "%H:%M").time():
        return 0.20
    elif hora < datetime.strptime("15:28", "%H:%M").time():
        return 0.25
    elif hora < datetime.strptime("15:30", "%H:%M").time():
        return 0.28
    elif hora < datetime.strptime("15:37", "%H:%M").time():
        return 0.30
    else:
        return 0.10


def put_text(img, text, org, scale=1.0, color=(0, 255, 255)):
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        5,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA
    )


# =========================================================
# MAIN
# =========================================================

modelo = joblib.load(MODELO_PATH)

videos = sorted(
    glob.glob(os.path.join(CARPETA_VIDEOS, "*.avi"))
)

resultados = []

altura_filtrada = None

cv2.namedWindow("Estimacion altura espuma II28P", cv2.WINDOW_NORMAL)

for video_path in videos:

    print("\nProcesando:")
    print(os.path.basename(video_path))

    inicio = obtener_timestamp_video(video_path)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % FRAME_SKIP != 0:
            frame_id += 1
            continue

        timestamp = inicio + timedelta(seconds=frame_id / fps)

        x = ROI["x"]
        y = ROI["y"]
        w = ROI["w"]
        h = ROI["h"]

        roi = frame[y:y+h, x:x+w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        gray = clahe.apply(gray)

        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if np.std(gray) < 1:
            frame_id += 1
            continue

        features = extraer_features(gray)

        caudal = obtener_caudal(timestamp)

        features["caudal"] = caudal
        features["temperatura"] = TEMPERATURA_C
        features["experiencia"] = EXPERIENCIA

        X = pd.DataFrame([features])

        altura_pred = modelo.predict(X)[0]

        altura_pred = np.clip(altura_pred, 2.0, 9.0)

        if altura_filtrada is None:
            altura_filtrada = altura_pred
        else:
            altura_filtrada = (
                ALPHA * altura_filtrada
                + (1 - ALPHA) * altura_pred
            )

        resultados.append({
            "Timestamp": timestamp,
            "Frame": frame_id,
            "Caudal_LPM": caudal,
            "Temperatura_C": TEMPERATURA_C,
            "Experiencia": EXPERIENCIA,
            "Altura_predicha_cm": altura_pred,
            "Altura_filtrada_cm": altura_filtrada
        })

        # =================================================
        # VISUALIZACIÓN
        # =================================================

        vis = frame.copy()

        cv2.rectangle(
            vis,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        hora_texto = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        put_text(
            vis,
            f"Hora: {hora_texto}",
            (40, 60),
            1.0,
            (255, 255, 0)
        )

        put_text(
            vis,
            f"Altura ML: {altura_filtrada:.2f} cm",
            (40, 120),
            1.2,
            (0, 255, 0)
        )

        put_text(
            vis,
            f"Caudal: {caudal:.2f} LPM",
            (40, 180),
            1.0,
            (0, 255, 255)
        )

        put_text(
            vis,
            f"T: {TEMPERATURA_C} C | Exp: II{EXPERIENCIA}P",
            (40, 240),
            0.9,
            (255, 255, 255)
        )

        put_text(
            vis,
            f"Frame: {frame_id}/{total_frames}",
            (40, 300),
            0.9,
            (255, 255, 255)
        )

        h_img, w_img = vis.shape[:2]

        escala = min(1280 / w_img, 720 / h_img)

        if escala < 1:
            vis_show = cv2.resize(
                vis,
                (
                    int(w_img * escala),
                    int(h_img * escala)
                )
            )
        else:
            vis_show = vis

        cv2.imshow("Estimacion altura espuma II28P", vis_show)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()

            df = pd.DataFrame(resultados)
            df.to_excel(SALIDA_EXCEL, index=False)

            print("\nProceso detenido.")
            print("Excel guardado:")
            print(SALIDA_EXCEL)

            exit()

        if frame_id % 300 == 0:
            print(
                f"Frame {frame_id}/{total_frames} | "
                f"{timestamp.strftime('%H:%M:%S')} | "
                f"{altura_filtrada:.2f} cm | "
                f"Caudal {caudal:.2f} LPM"
            )

        frame_id += 1

    cap.release()

cv2.destroyAllWindows()

df = pd.DataFrame(resultados)

df.to_excel(SALIDA_EXCEL, index=False)

print("\n===================================")
print("Excel guardado:")
print(SALIDA_EXCEL)
print("===================================")