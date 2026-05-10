import cv2
import numpy as np
import tkinter as tk
import pandas as pd
from datetime import datetime, timedelta
import os
import re
from glob import glob

# CONFIGURACIÓN
CARPETA_VIDEOS = r'D:\MDT\Pruebas\II28P\1'

PATRON_VIDEOS = '*.avi'

ALTURA_CRISOL_CM = 9.0
ALTURA_BASE_CM = 2.0

# Frame usado para calibrar nivel de referencia
FRAME_REFERENCIA = 1800

# Suavizado temporal
SUAVIZADO = 0.03

# Límite físico realista
ALTURA_MIN_CM = 1.5
ALTURA_MAX_CM = 9.0

# Cambio máximo permitido entre frames
MAX_DELTA_CM = 0.05      # aumentado para capturar cambios rápidos al inicio

# Umbral para excluir píxeles oscuros
UMBRAL_LANZA = 30

# FUNCIONES
def get_scale(width, height):
    root = tk.Tk()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.destroy()
    escala = min(sw / width, sh / height)
    return escala if escala < 1 else 1


def seleccionar_roi(frame, escala):
    h, w = frame.shape[:2]
    frame_small = cv2.resize(frame, (int(w * escala), int(h * escala)))
    roi_small = cv2.selectROI("Seleccionar ROI", frame_small, False)
    cv2.destroyAllWindows()

    x_s, y_s, w_s, h_s = roi_small
    if w_s == 0 or h_s == 0:
        return None

    return (
        int(x_s / escala),
        int(y_s / escala),
        int(w_s / escala),
        int(h_s / escala)
    )


def detectar_altura_roi(roi, umbral_lanza=UMBRAL_LANZA):
    """
    Calcula el nivel de la superficie ignorando la lanza.
    Para cada fila del ROI, calcula el promedio solo de los
    píxeles brillantes (por encima del umbral), excluyendo
    automáticamente la zona oscura de la lanza.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # Calcular perfil excluyendo píxeles oscuros (lanza)
    perfil = np.zeros(blur.shape[0])
    for i in range(blur.shape[0]):
        fila = blur[i, :]
        pixeles_validos = fila[fila > umbral_lanza]
        if len(pixeles_validos) > 0:
            perfil[i] = np.mean(pixeles_validos)
        else:
            perfil[i] = 0

    perfil = cv2.GaussianBlur(
        perfil.reshape(-1, 1),
        (81, 1),
        0
    ).flatten()

    grad = np.gradient(perfil)

    zona_superior = int(len(grad) * 0.15)
    zona_inferior = int(len(grad) * 0.95)

    grad_util = grad[zona_superior:zona_inferior]
    idx = np.argmax(np.abs(grad_util))
    nivel = idx + zona_superior

    return nivel


def parse_timestamp_from_filename(video_path):
    base = os.path.basename(video_path)
    name_no_ext = os.path.splitext(base)[0]
    parts = name_no_ext.split('_')

    if len(parts) < 2:
        raise ValueError("El nombre del archivo no contiene timestamp")

    timestamp_str = parts[-1]

    if 'T' in timestamp_str:
        date_part, time_part = timestamp_str.split('T')
        time_part_corrected = re.sub(r'-', ':', time_part)
        datetime_str = f"{date_part}T{time_part_corrected}"
    else:
        datetime_str = timestamp_str

    return datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f")


def main():
    lista_videos = sorted(glob(os.path.join(CARPETA_VIDEOS, PATRON_VIDEOS)))

    if len(lista_videos) == 0:
        print("No se encontraron videos")
        return

    print("\nVideos encontrados:\n")
    for v in lista_videos:
        print(os.path.basename(v))

    # ROI SOLO UNA VEZ
    primer_video = lista_videos[0]
    cap = cv2.VideoCapture(primer_video)
    ret, frame = cap.read()

    if not ret:
        print("Error leyendo primer video")
        return

    h, w = frame.shape[:2]
    escala = get_scale(w, h)
    roi_data = seleccionar_roi(frame, escala)

    if roi_data is None:
        return

    x, y, rw, rh = roi_data
    cap.release()

    # VARIABLES GLOBALES
    timestamps_totales = []
    alturas_totales    = []
    altura_suavizada   = ALTURA_BASE_CM

    # PROCESAR TODOS LOS VIDEOS
    for VIDEO_PATH in lista_videos:

        print("\n================================================")
        print("Procesando:")
        print(os.path.basename(VIDEO_PATH))
        print("================================================")

        cap = cv2.VideoCapture(VIDEO_PATH)

        if not cap.isOpened():
            print("No se pudo abrir")
            continue

        # CALIBRACIÓN
        cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_REFERENCIA)
        ret, frame_ref = cap.read()

        if not ret:
            print("Error frame referencia")
            continue

        roi_ref   = frame_ref[y:y+rh, x:x+rw]
        nivel_ref = detectar_altura_roi(roi_ref)
        print(f"Nivel referencia: {nivel_ref}")

        # Volver al inicio
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        print(f"FPS: {fps:.2f}")

        timestamp_inicial = parse_timestamp_from_filename(VIDEO_PATH)
        print(f"Inicio: {timestamp_inicial}")

        frame_id = 0

        # LOOP VIDEO
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            roi = frame[y:y+rh, x:x+rw]
            if roi.size == 0:
                break

            # Detectar altura filtrando la lanza
            nivel = detectar_altura_roi(roi)

            delta_pix    = nivel_ref - nivel
            altura_medida = (
                ALTURA_BASE_CM
                + (delta_pix / rh) * ALTURA_CRISOL_CM
            )

            # Limitar rango físico real
            altura_medida = np.clip(
                altura_medida,
                ALTURA_MIN_CM,
                ALTURA_MAX_CM
            )

            # Limitar cambios bruscos
            delta = np.clip(
                altura_medida - altura_suavizada,
                -MAX_DELTA_CM,
                MAX_DELTA_CM
            )
            altura_actual = altura_suavizada + delta

            # Suavizado
            altura_suavizada = (
                (1 - SUAVIZADO) * altura_suavizada
                + SUAVIZADO * altura_actual
            )

            # Timestamp
            tiempo_transcurrido = frame_id / fps
            timestamp_actual    = (
                timestamp_inicial
                + timedelta(seconds=tiempo_transcurrido)
            )

            timestamps_totales.append(timestamp_actual)
            alturas_totales.append(altura_suavizada)

            # VISUALIZACIÓN
            vis = frame.copy()

            # ROI
            cv2.rectangle(vis, (x, y), (x + rw, y + rh), (0, 255, 0), 2)

            # Línea de nivel detectado
            y_line = y + nivel
            cv2.line(vis, (x, y_line), (x + rw, y_line), (0, 0, 255), 2)

            # Texto altura
            cv2.putText(
                vis,
                f"{altura_suavizada:.2f} cm",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2
            )

            # Texto frame
            cv2.putText(
                vis,
                f"Frame: {frame_id}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0), 2
            )

            # Mostrar redimensionado
            if escala != 1:
                vis_mostrar = cv2.resize(
                    vis,
                    (int(w * escala), int(h * escala))
                )
            else:
                vis_mostrar = vis

            cv2.imshow("Altura espuma", vis_mostrar)

            if frame_id % 300 == 0:
                print(
                    f"Frame {frame_id} | "
                    f"{altura_suavizada:.2f} cm | "
                    f"{timestamp_actual}"
                )

            frame_id += 1

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        cap.release()

    # FINAL
    cv2.destroyAllWindows()

    # GUARDAR EXCEL
    if len(timestamps_totales) > 0:
        df = pd.DataFrame({
            'Timestamp': timestamps_totales,
            'Altura_cm': alturas_totales
        })

        excel_path = os.path.join(
            CARPETA_VIDEOS,
            'alturas_todos_los_videos.xlsx'
        )

        df.to_excel(excel_path, index=False)

        print("\n====================================")
        print("Excel guardado:")
        print(excel_path)
        print("====================================")

    else:
        print("No se guardaron datos")


if __name__ == "__main__":
    main()