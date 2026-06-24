import os
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ============================================================
# CONFIGURACIÓN
# ============================================================

VIDEO_L = r"D:\MDT\Stereovision\II38P_cam1.mp4"
VIDEO_R = r"D:\MDT\Stereovision\II38P_cam2.mp4"
CSV_REGISTRO = r"D:\MDT\Stereovision\II38P_captura.csv"
ARCHIVO_CALIBRACION = r"D:\MDT\Stereovision\calibracion_opencv.mat"

CARPETA_SALIDA = r"D:\MDT\Stereovision\calibracion_empirica_II38P_calibrada"
NOMBRE_EXPERIENCIA = "II38P"

ARCHIVO_CONFIG_ESTEREO = os.path.join(
    CARPETA_SALIDA,
    f"{NOMBRE_EXPERIENCIA}_config_estereo.npz",
)

# Horas anotadas sin segundos.
MEDICIONES_MANUALES = [
    {"timestamp": "2026-06-17 19:40:00.000", "altura_mm": 14.0, "uso": "calibracion"},
    {"timestamp": "2026-06-17 19:43:00.000", "altura_mm": 12.0, "uso": "calibracion"},
    {"timestamp": "2026-06-17 19:52:00.000", "altura_mm": 29.0, "uso": "calibracion"},
    {"timestamp": "2026-06-17 20:10:00.000", "altura_mm": 37.0, "uso": "calibracion"},
    {"timestamp": "2026-06-17 20:20:00.000", "altura_mm": 38.0, "uso": "calibracion"},
    {"timestamp": "2026-06-17 20:34:00.000", "altura_mm": 45.0, "uso": "validacion"},
    {"timestamp": "2026-06-17 20:50:00.000", "altura_mm": 50.0, "uso": "validacion"},
]

VENTANA_TEMPORAL_S = 30.0

# Uno de cada 15 frames: aproximadamente 2 muestras por segundo a 30 fps.
# Cambia a 3 si quieres muchas más muestras, pero tardará bastante más.
SALTO_FRAMES_VENTANA = 15

# ------------------------------------------------------------
# ORIENTACIÓN
# ------------------------------------------------------------

VIDEOS_YA_ROTADOS_ONLINE = True

if VIDEOS_YA_ROTADOS_ONLINE:
    ROTACION_L = None
    ROTACION_R = None
else:
    ROTACION_L = cv2.ROTATE_90_CLOCKWISE
    ROTACION_R = cv2.ROTATE_90_COUNTERCLOCKWISE

# ------------------------------------------------------------
# RECTIFICACIÓN CALIBRADA
# ------------------------------------------------------------

# 0.0 maximiza el área válida. No se recorta usando roi1/roi2,
# porque en tu calibración OpenCV roi2 puede aparecer como (0,0,0,0).
ALPHA_RECTIFICACION = 0.0

# Si True, siempre solicita puntos para estimar el rango de disparidad.
# Si False, reutiliza ARCHIVO_CONFIG_ESTEREO cuando exista.
RECALCULAR_RANGO_DISPARIDAD = False

MIN_PARES_RANGO = 3
MAX_PARES_RANGO = 12
MARGEN_RANGO_DISPARIDAD_PX = 96
MAX_NUM_DISPARITIES = 768

# ------------------------------------------------------------
# CLAHE + CANNY
# ------------------------------------------------------------

CLAHE_CLIP_LIMIT = 1.2
CLAHE_GRID_SIZE = (8, 8)

BILATERAL_DIAMETER = 9
BILATERAL_SIGMA_COLOR = 45
BILATERAL_SIGMA_SPACE = 45

LOW_PERCENTILE = 70
HIGH_PERCENTILE = 90
CANNY_APERTURE_SIZE = 3

CANNY_DILATATION_KERNEL = 7
CANNY_DILATATION_ITERATIONS = 1

GAMMA_CORRECCION = 1.5

# ------------------------------------------------------------
# STEREO SGBM
# ------------------------------------------------------------

BLOCK_SIZE = 7
MARGEN_LIMITE_DISPARIDAD = 3.0
MIN_PIXELES_VALIDOS_FRAME = 80

# Se valida intensidad solo en la cámara izquierda. No se debe comparar
# la misma coordenada x en ambas cámaras porque existe disparidad horizontal.
INTENSIDAD_MIN_VALIDA = 8
INTENSIDAD_MAX_VALIDA = 245

PERCENTIL_DISPARIDAD_ROI = 50.0
IQR_FACTOR = 1.5

# ------------------------------------------------------------
# ROI
# ------------------------------------------------------------

SELECCIONAR_ROI = True
ROI_FIJA = (0, 0, 300, 300)


# ============================================================
# UTILIDADES
# ============================================================


def comprobar_archivo(ruta, nombre):
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró {nombre}:\n{ruta}")


def aplicar_rotacion(frame, lado):
    if lado == "L":
        rotacion = ROTACION_L
    elif lado == "R":
        rotacion = ROTACION_R
    else:
        raise ValueError("lado debe ser 'L' o 'R'.")

    if rotacion is None:
        return frame.copy()

    return cv2.rotate(frame, rotacion)


def put_text_outline(img, text, org, scale=0.6, color=(255, 255, 255), thickness=2):
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 4,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def aplicar_gamma(gray, gamma):
    if gamma <= 0:
        raise ValueError("gamma debe ser mayor que cero.")

    tabla = np.array(
        [((i / 255.0) ** gamma) * 255.0 for i in range(256)],
        dtype=np.float32,
    )

    return cv2.LUT(gray, np.clip(tabla, 0, 255).astype(np.uint8))


def dibujar_lineas_epipolares(img, separacion=40):
    salida = img.copy()

    for y in range(0, salida.shape[0], separacion):
        cv2.line(
            salida,
            (0, y),
            (salida.shape[1] - 1, y),
            (0, 255, 0),
            1,
        )

    return salida


def reducir_para_pantalla(imagen, ancho_max=1600, alto_max=900):
    alto, ancho = imagen.shape[:2]
    escala = min(ancho_max / ancho, alto_max / alto, 1.0)

    if escala >= 1.0:
        return imagen

    return cv2.resize(
        imagen,
        (int(round(ancho * escala)), int(round(alto * escala))),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# CALIBRACIÓN MATLAB -> OPENCV
# ============================================================


def cargar_calibracion_opencv(ruta_mat, ancho_video, alto_video):
    datos = loadmat(ruta_mat, squeeze_me=True)

    necesarias = {"K1", "D1", "K2", "D2", "R", "T", "imageSize"}
    faltantes = necesarias - set(datos.keys())

    if faltantes:
        raise RuntimeError(
            f"Faltan variables en calibracion_opencv.mat: {sorted(faltantes)}"
        )

    K1 = np.asarray(datos["K1"], dtype=np.float64)
    D1 = np.asarray(datos["D1"], dtype=np.float64).reshape(-1)
    K2 = np.asarray(datos["K2"], dtype=np.float64)
    D2 = np.asarray(datos["D2"], dtype=np.float64).reshape(-1)
    R = np.asarray(datos["R"], dtype=np.float64)
    T = np.asarray(datos["T"], dtype=np.float64).reshape(3, 1)
    image_size = np.asarray(datos["imageSize"], dtype=np.float64).reshape(-1)

    if K1.shape != (3, 3) or K2.shape != (3, 3):
        raise RuntimeError("K1 y K2 deben ser matrices 3x3.")

    if R.shape != (3, 3):
        raise RuntimeError("R debe ser una matriz 3x3.")

    # MATLAB almacena ImageSize como [filas, columnas].
    alto_cal = float(image_size[0])
    ancho_cal = float(image_size[1])

    sx = ancho_video / ancho_cal
    sy = alto_video / alto_cal

    print("\n====================================")
    print("CALIBRACIÓN ESTÉREO")
    print("====================================")
    print(f"Calibración: {int(ancho_cal)} x {int(alto_cal)}")
    print(f"Video:       {ancho_video} x {alto_video}")
    print(f"Escala X: {sx:.12f}")
    print(f"Escala Y: {sy:.12f}")

    if abs(sx - sy) > 1e-6:
        raise RuntimeError(
            "La relación de aspecto del video no coincide con la calibración."
        )

    K1v = K1.copy()
    K2v = K2.copy()

    for K in (K1v, K2v):
        K[0, 0] *= sx
        K[0, 1] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    tamano_video = (int(ancho_video), int(alto_video))

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=K1v,
        distCoeffs1=D1,
        cameraMatrix2=K2v,
        distCoeffs2=D2,
        imageSize=tamano_video,
        R=R,
        T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=ALPHA_RECTIFICACION,
        newImageSize=tamano_video,
    )

    map1_l, map2_l = cv2.initUndistortRectifyMap(
        K1v,
        D1,
        R1,
        P1,
        tamano_video,
        cv2.CV_32FC1,
    )

    map1_r, map2_r = cv2.initUndistortRectifyMap(
        K2v,
        D2,
        R2,
        P2,
        tamano_video,
        cv2.CV_32FC1,
    )

    print("T [mm]:", T.ravel())
    print("Baseline [mm]:", float(np.linalg.norm(T)))
    print("ROI válida cámara 1:", tuple(map(int, roi1)))
    print("ROI válida cámara 2:", tuple(map(int, roi2)))
    print(
        "Nota: no se recortará usando roi2, aunque aparezca como (0,0,0,0)."
    )

    return {
        "K1": K1v,
        "D1": D1,
        "K2": K2v,
        "D2": D2,
        "R": R,
        "T": T,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "map1_l": map1_l,
        "map2_l": map2_l,
        "map1_r": map1_r,
        "map2_r": map2_r,
        "tamano_video": tamano_video,
    }


def rectificar_par(frame_l, frame_r, calibracion):
    rect_l = cv2.remap(
        frame_l,
        calibracion["map1_l"],
        calibracion["map2_l"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    rect_r = cv2.remap(
        frame_r,
        calibracion["map1_r"],
        calibracion["map2_r"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rect_l, rect_r


def leer_par_frames(cap_l, cap_r, frame_idx, calibracion):
    cap_l.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))

    ok_l, frame_l = cap_l.read()
    ok_r, frame_r = cap_r.read()

    if not ok_l or not ok_r:
        return None, None

    frame_l = aplicar_rotacion(frame_l, "L")
    frame_r = aplicar_rotacion(frame_r, "R")

    return rectificar_par(frame_l, frame_r, calibracion)


def verificar_rectificacion(frame_l, frame_r):
    # Añadir líneas y rótulos antes de formar el montaje.
    vista_l = dibujar_lineas_epipolares(frame_l, 40)
    vista_r = dibujar_lineas_epipolares(frame_r, 40)

    put_text_outline(
        vista_l,
        "CAMARA 1 RECTIFICADA",
        (20, 35),
        scale=0.75,
        color=(0, 255, 255),
        thickness=2,
    )

    put_text_outline(
        vista_r,
        "CAMARA 2 RECTIFICADA",
        (20, 35),
        scale=0.75,
        color=(0, 255, 255),
        thickness=2,
    )

    # Montaje lado a lado y reducción real de la imagen para que ambas
    # cámaras entren completas en la ventana.
    montaje = np.hstack([vista_l, vista_r])
    vista = reducir_para_pantalla(
        montaje,
        ancho_max=1500,
        alto_max=800,
    )

    nombre = "Verificar rectificacion - A aceptar, Q cancelar"
    cv2.namedWindow(
        nombre,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
    )

    # Forzar una ventana horizontal suficientemente grande.
    alto_vista, ancho_vista = vista.shape[:2]
    cv2.resizeWindow(nombre, ancho_vista, alto_vista)
    cv2.imshow(nombre, vista)

    print("\nDeben aparecer las DOS cámaras lado a lado.")
    print("Comprueba el mismo detalle físico en la misma línea horizontal.")
    print("A: aceptar | Q: cancelar")

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("a"), ord("A")):
            break

        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(nombre)
            raise RuntimeError("Rectificación cancelada por el usuario.")

    cv2.destroyWindow(nombre)


# ============================================================
# SELECCIÓN DEL RANGO DE DISPARIDAD
# ============================================================


class SelectorCorrespondencias:
    def __init__(self, frame_l, frame_r):
        if frame_l.shape[:2] != frame_r.shape[:2]:
            raise ValueError("Los frames deben tener el mismo tamaño.")

        self.frame_l = frame_l
        self.frame_r = frame_r
        self.alto, self.ancho = frame_l.shape[:2]
        self.puntos_l = []
        self.puntos_r = []
        self.pendiente_l = None
        self.cancelado = False
        self.ventana = "Rango disparidad: punto L y luego mismo punto R"

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < self.ancho:
                if self.pendiente_l is None:
                    self.pendiente_l = (float(x), float(y))
                    print(f"Punto L: ({x}, {y}). Selecciona el mismo detalle en R.")
                else:
                    print("Completa primero el punto derecho pendiente.")
            else:
                if self.pendiente_l is None:
                    print("Primero selecciona el punto en la imagen izquierda.")
                    return

                pr = (float(x - self.ancho), float(y))
                self.puntos_l.append(self.pendiente_l)
                self.puntos_r.append(pr)
                d = self.pendiente_l[0] - pr[0]
                print(
                    f"Par {len(self.puntos_l)}: L={self.pendiente_l}, "
                    f"R={pr}, disparidad={d:.2f} px"
                )
                self.pendiente_l = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.pendiente_l is not None:
                self.pendiente_l = None
                print("Punto pendiente cancelado.")
            elif self.puntos_l:
                self.puntos_l.pop()
                self.puntos_r.pop()
                print("Último par eliminado.")

    def dibujar(self):
        izquierda = self.frame_l.copy()
        derecha = self.frame_r.copy()

        for i, (pl, pr) in enumerate(zip(self.puntos_l, self.puntos_r), start=1):
            xl, yl = map(int, pl)
            xr, yr = map(int, pr)

            cv2.circle(izquierda, (xl, yl), 6, (0, 255, 0), -1)
            cv2.circle(derecha, (xr, yr), 6, (0, 255, 0), -1)

            put_text_outline(izquierda, str(i), (xl + 8, yl - 8), 0.55, (0, 255, 0), 1)
            put_text_outline(derecha, str(i), (xr + 8, yr - 8), 0.55, (0, 255, 0), 1)

        if self.pendiente_l is not None:
            x, y = map(int, self.pendiente_l)
            cv2.circle(izquierda, (x, y), 7, (0, 165, 255), -1)

        put_text_outline(
            izquierda,
            f"IZQUIERDA - pares: {len(self.puntos_l)}",
            (20, 30),
            0.65,
            (255, 255, 0),
        )
        put_text_outline(derecha, "DERECHA", (20, 30), 0.65, (255, 255, 0))

        return reducir_para_pantalla(np.hstack([izquierda, derecha]))

    def ejecutar(self):
        cv2.namedWindow(self.ventana, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.ventana, self.callback)

        print("\n====================================")
        print("SELECCIÓN DEL RANGO DE DISPARIDAD")
        print("====================================")
        print(f"Selecciona al menos {MIN_PARES_RANGO} correspondencias.")
        print("Clic en detalle de la izquierda y luego el mismo detalle en la derecha.")
        print("ENTER: terminar | botón derecho: deshacer | Q: cancelar")

        while True:
            cv2.imshow(self.ventana, self.dibujar())
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):
                if self.pendiente_l is not None:
                    print("Completa o cancela el punto pendiente.")
                    continue

                if len(self.puntos_l) < MIN_PARES_RANGO:
                    print(
                        f"Necesitas {MIN_PARES_RANGO} pares; tienes {len(self.puntos_l)}."
                    )
                    continue

                break

            if key in (ord("q"), ord("Q"), 27):
                self.cancelado = True
                break

            if len(self.puntos_l) >= MAX_PARES_RANGO:
                break

        cv2.destroyWindow(self.ventana)

        if self.cancelado:
            raise RuntimeError("Selección del rango cancelada.")

        return (
            np.asarray(self.puntos_l, dtype=np.float64),
            np.asarray(self.puntos_r, dtype=np.float64),
        )


def calcular_rango_disparidad_manual(frame_l, frame_r):
    selector = SelectorCorrespondencias(frame_l, frame_r)
    puntos_l, puntos_r = selector.ejecutar()

    errores_verticales = np.abs(puntos_l[:, 1] - puntos_r[:, 1])
    disparidades = puntos_l[:, 0] - puntos_r[:, 0]

    print("Error vertical mediano:", float(np.median(errores_verticales)), "px")
    print("Disparidades seleccionadas:", disparidades)

    p05, mediana, p95 = np.percentile(disparidades, [5, 50, 95])

    minimo_deseado = int(math.floor(p05 - MARGEN_RANGO_DISPARIDAD_PX))
    maximo_deseado = int(math.ceil(p95 + MARGEN_RANGO_DISPARIDAD_PX))

    ancho_rango = maximo_deseado - minimo_deseado + 1
    num_disparities = int(math.ceil(ancho_rango / 16.0) * 16)
    num_disparities = max(16, num_disparities)

    if num_disparities > MAX_NUM_DISPARITIES:
        raise RuntimeError(
            f"Rango demasiado grande: {num_disparities} px. "
            "Selecciona correspondencias más precisas."
        )

    print("\nRango calculado:")
    print(f"P5={p05:.2f}, mediana={mediana:.2f}, P95={p95:.2f} px")
    print("minDisparity:", minimo_deseado)
    print("numDisparities:", num_disparities)
    print(
        "Rango SGBM:",
        minimo_deseado,
        "a",
        minimo_deseado + num_disparities - 1,
        "px",
    )

    return minimo_deseado, num_disparities, disparidades


# ============================================================
# CLAHE + CANNY
# ============================================================


def automatic_canny_from_gradient(gray_img, low_percentile=70, high_percentile=90):
    sobel_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    valid_gradients = magnitude[magnitude > 0]

    if valid_gradients.size == 0:
        return np.zeros_like(gray_img), 0, 1

    threshold_low = int(np.percentile(valid_gradients, low_percentile))
    threshold_high = int(np.percentile(valid_gradients, high_percentile))

    threshold_low = max(1, min(threshold_low, 254))
    threshold_high = max(threshold_low + 1, min(threshold_high, 255))

    edges = cv2.Canny(
        gray_img,
        threshold1=threshold_low,
        threshold2=threshold_high,
        apertureSize=CANNY_APERTURE_SIZE,
        L2gradient=True,
    )

    return edges, threshold_low, threshold_high


def preprocesar_frame(frame):
    gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_gamma = aplicar_gamma(gray_original, GAMMA_CORRECCION)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_GRID_SIZE,
    )

    gray_clahe = clahe.apply(gray_gamma)

    gray_filtered = cv2.bilateralFilter(
        gray_clahe,
        d=BILATERAL_DIAMETER,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
    )

    edges, threshold_low, threshold_high = automatic_canny_from_gradient(
        gray_filtered,
        LOW_PERCENTILE,
        HIGH_PERCENTILE,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (CANNY_DILATATION_KERNEL, CANNY_DILATATION_KERNEL),
    )

    edge_mask = cv2.dilate(
        edges,
        kernel,
        iterations=CANNY_DILATATION_ITERATIONS,
    )

    return {
        "gray_original": gray_original,
        "gray_filtered": gray_filtered,
        "edges": edges,
        "edge_mask": edge_mask,
        "canny_low": threshold_low,
        "canny_high": threshold_high,
    }


# ============================================================
# STEREO SGBM
# ============================================================


def crear_stereo_sgbm(min_disparity, num_disparities):
    if num_disparities % 16 != 0:
        raise ValueError("numDisparities debe ser múltiplo de 16.")

    if BLOCK_SIZE % 2 == 0:
        raise ValueError("BLOCK_SIZE debe ser impar.")

    return cv2.StereoSGBM_create(
        minDisparity=int(min_disparity),
        numDisparities=int(num_disparities),
        blockSize=BLOCK_SIZE,
        P1=8 * BLOCK_SIZE ** 2,
        P2=32 * BLOCK_SIZE ** 2,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=8,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def calcular_disparidad(stereo, frame_l, frame_r, min_disparity, num_disparities):
    proc_l = preprocesar_frame(frame_l)
    proc_r = preprocesar_frame(frame_r)

    disparity = stereo.compute(
        proc_l["gray_filtered"],
        proc_r["gray_filtered"],
    ).astype(np.float32) / 16.0

    limite_inferior = float(min_disparity)
    limite_superior = float(min_disparity + num_disparities - 1)
    valor_invalido = float(min_disparity - 1)

    intensidad_valida = (
        (proc_l["gray_original"] >= INTENSIDAD_MIN_VALIDA)
        & (proc_l["gray_original"] <= INTENSIDAD_MAX_VALIDA)
    )

    invalidos = (
        ~np.isfinite(disparity)
        | (disparity <= valor_invalido + 0.25)
        | (disparity <= limite_inferior + MARGEN_LIMITE_DISPARIDAD)
        | (disparity >= limite_superior - MARGEN_LIMITE_DISPARIDAD)
        | (~intensidad_valida)
    )

    disparity[invalidos] = np.nan

    return disparity, proc_l["edge_mask"], proc_l["edges"]


def obtener_disparidad_superficie(disparity, edge_mask, roi):
    x, y, w, h = roi

    roi_disp = disparity[y:y + h, x:x + w]
    roi_edges = edge_mask[y:y + h, x:x + w]

    if roi_disp.size == 0:
        return None, 0

    validos = np.isfinite(roi_disp) & (roi_edges > 0)
    cantidad = int(np.count_nonzero(validos))

    if cantidad < MIN_PIXELES_VALIDOS_FRAME:
        return None, cantidad

    valores = roi_disp[validos].astype(np.float32)

    q1, q3 = np.percentile(valores, [25, 75])
    iqr = q3 - q1

    if iqr > 0:
        limite_bajo = q1 - IQR_FACTOR * iqr
        limite_alto = q3 + IQR_FACTOR * iqr
        valores = valores[(valores >= limite_bajo) & (valores <= limite_alto)]

    if len(valores) < MIN_PIXELES_VALIDOS_FRAME:
        return None, len(valores)

    d_superficie = float(np.percentile(valores, PERCENTIL_DISPARIDAD_ROI))

    return d_superficie, len(valores)


# ============================================================
# CSV Y TIEMPOS
# ============================================================


def buscar_frame_mas_cercano(df_registro, timestamp_medicion):
    diferencia = (df_registro["timestamp"] - timestamp_medicion).abs()
    indice = diferencia.idxmin()
    fila = df_registro.loc[indice]

    diferencia_s = abs(
        (fila["timestamp"] - timestamp_medicion).total_seconds()
    )

    return {
        "frame_grabado": int(fila["frame_grabado"]),
        "timestamp_encontrado": fila["timestamp"],
        "diferencia_s": float(diferencia_s),
    }


def obtener_frames_ventana(df_registro, timestamp_central, ventana_s):
    inicio = timestamp_central - pd.Timedelta(seconds=ventana_s)
    fin = timestamp_central + pd.Timedelta(seconds=ventana_s)

    filas = df_registro[
        (df_registro["timestamp"] >= inicio)
        & (df_registro["timestamp"] <= fin)
    ]

    frames = filas["frame_grabado"].astype(int).tolist()
    return frames[::SALTO_FRAMES_VENTANA]


# ============================================================
# ROI
# ============================================================


def seleccionar_roi(cap_l, cap_r, frame_idx, calibracion):
    frame_l, frame_r = leer_par_frames(
        cap_l,
        cap_r,
        frame_idx,
        calibracion,
    )

    if frame_l is None:
        raise RuntimeError("No se pudo leer el frame para seleccionar ROI.")

    print("\nSelecciona en la cámara izquierda la ROI de espuma.")
    print("Evita zonas negras y, en lo posible, la lanza.")

    roi = cv2.selectROI(
        "Seleccionar ROI de espuma en camara izquierda",
        frame_l,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow("Seleccionar ROI de espuma en camara izquierda")

    x, y, w, h = map(int, roi)

    if w <= 0 or h <= 0:
        raise RuntimeError("No se seleccionó una ROI válida.")

    return x, y, w, h


# ============================================================
# PROCESAMIENTO DE CADA MEDICIÓN
# ============================================================


def procesar_medicion(
    medicion,
    df_registro,
    cap_l,
    cap_r,
    stereo,
    calibracion,
    roi,
    min_disparity,
    num_disparities,
):
    timestamp_solicitado = pd.to_datetime(medicion["timestamp"])

    coincidencia = buscar_frame_mas_cercano(
        df_registro,
        timestamp_solicitado,
    )

    frames_ventana = obtener_frames_ventana(
        df_registro,
        timestamp_solicitado,
        VENTANA_TEMPORAL_S,
    )

    disparidades = []
    pixeles_validos_lista = []

    for frame_idx in frames_ventana:
        frame_l, frame_r = leer_par_frames(
            cap_l,
            cap_r,
            frame_idx,
            calibracion,
        )

        if frame_l is None or frame_r is None:
            continue

        disparity, edge_mask, _ = calcular_disparidad(
            stereo,
            frame_l,
            frame_r,
            min_disparity,
            num_disparities,
        )

        d_superficie, pixeles_validos = obtener_disparidad_superficie(
            disparity,
            edge_mask,
            roi,
        )

        if d_superficie is not None:
            disparidades.append(d_superficie)
            pixeles_validos_lista.append(pixeles_validos)

    if not disparidades:
        d_mediana = np.nan
        d_desviacion = np.nan
        pixeles_mediana = 0
    else:
        valores = np.asarray(disparidades, dtype=np.float64)

        q1, q3 = np.percentile(valores, [25, 75])
        iqr = q3 - q1

        if iqr > 0:
            limite_bajo = q1 - IQR_FACTOR * iqr
            limite_alto = q3 + IQR_FACTOR * iqr
            valores_filtrados = valores[
                (valores >= limite_bajo) & (valores <= limite_alto)
            ]
        else:
            valores_filtrados = valores

        if valores_filtrados.size == 0:
            valores_filtrados = valores

        d_mediana = float(np.median(valores_filtrados))
        d_desviacion = float(np.std(valores_filtrados))
        pixeles_mediana = int(np.median(pixeles_validos_lista))

    return {
        "timestamp_solicitado": timestamp_solicitado,
        "timestamp_encontrado": coincidencia["timestamp_encontrado"],
        "diferencia_temporal_s": coincidencia["diferencia_s"],
        "frame_central": coincidencia["frame_grabado"],
        "altura_mm": float(medicion["altura_mm"]),
        "uso": str(medicion["uso"]).lower(),
        "disparidad_mediana_px": d_mediana,
        "desviacion_disparidad_px": d_desviacion,
        "frames_analizados": len(frames_ventana),
        "frames_validos": len(disparidades),
        "pixeles_validos_mediana": pixeles_mediana,
    }


# ============================================================
# MODELOS
# ============================================================


def calcular_metricas(real, predicho):
    error = predicho - real
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return mae, rmse


def predecir_modelo(modelo, disparidad):
    disparidad = np.asarray(disparidad, dtype=np.float64)

    if modelo["tipo"] == "polinomio":
        return np.polyval(modelo["coeficientes"], disparidad)

    if modelo["tipo"] == "inverso":
        if np.any(np.abs(disparidad) < 1e-9):
            raise ValueError("El modelo inverso recibió disparidad cero.")

        return np.polyval(modelo["coeficientes"], 1.0 / disparidad)

    raise ValueError("Tipo de modelo desconocido.")


def ajustar_modelos(df_resultados):
    calibracion_df = df_resultados[
        (df_resultados["uso"] == "calibracion")
        & df_resultados["disparidad_mediana_px"].notna()
    ].copy()

    validacion_df = df_resultados[
        (df_resultados["uso"] == "validacion")
        & df_resultados["disparidad_mediana_px"].notna()
    ].copy()

    if len(calibracion_df) < 3:
        raise RuntimeError("Se necesitan al menos 3 puntos válidos de calibración.")

    x_cal = calibracion_df["disparidad_mediana_px"].to_numpy(dtype=np.float64)
    y_cal = calibracion_df["altura_mm"].to_numpy(dtype=np.float64)

    rango_x = float(np.max(x_cal) - np.min(x_cal))

    if rango_x < 0.5:
        raise RuntimeError(
            "Las disparidades casi no cambian "
            f"(rango={rango_x:.3f} px). No se ajustará un modelo falso."
        )

    modelos = {
        "lineal": {
            "tipo": "polinomio",
            "grado": 1,
            "coeficientes": np.polyfit(x_cal, y_cal, deg=1),
        }
    }

    if len(calibracion_df) >= 4:
        modelos["cuadratico"] = {
            "tipo": "polinomio",
            "grado": 2,
            "coeficientes": np.polyfit(x_cal, y_cal, deg=2),
        }

    if np.all(np.abs(x_cal) > 1e-6):
        modelos["inverso"] = {
            "tipo": "inverso",
            "grado": 1,
            "coeficientes": np.polyfit(1.0 / x_cal, y_cal, deg=1),
        }

    for modelo in modelos.values():
        pred_cal = predecir_modelo(modelo, x_cal)
        mae_cal, rmse_cal = calcular_metricas(y_cal, pred_cal)

        modelo["mae_calibracion_mm"] = mae_cal
        modelo["rmse_calibracion_mm"] = rmse_cal

        if len(validacion_df) > 0:
            x_val = validacion_df["disparidad_mediana_px"].to_numpy(dtype=np.float64)
            y_val = validacion_df["altura_mm"].to_numpy(dtype=np.float64)
            pred_val = predecir_modelo(modelo, x_val)
            mae_val, rmse_val = calcular_metricas(y_val, pred_val)
            modelo["mae_validacion_mm"] = mae_val
            modelo["rmse_validacion_mm"] = rmse_val
        else:
            modelo["mae_validacion_mm"] = np.nan
            modelo["rmse_validacion_mm"] = np.nan

    if len(validacion_df) > 0:
        mejor_nombre = min(
            modelos,
            key=lambda nombre: modelos[nombre]["rmse_validacion_mm"],
        )
    else:
        mejor_nombre = min(
            modelos,
            key=lambda nombre: modelos[nombre]["rmse_calibracion_mm"],
        )

    return modelos, mejor_nombre


def guardar_grafico(df_resultados, modelos, mejor_nombre, ruta):
    validos = df_resultados[
        df_resultados["disparidad_mediana_px"].notna()
    ].copy()

    if validos.empty:
        return

    x = validos["disparidad_mediana_px"].to_numpy(dtype=np.float64)
    x_curva = np.linspace(np.min(x), np.max(x), 300)

    fig, ax = plt.subplots()

    calibracion_df = validos[validos["uso"] == "calibracion"]
    validacion_df = validos[validos["uso"] == "validacion"]

    ax.scatter(
        calibracion_df["disparidad_mediana_px"],
        calibracion_df["altura_mm"],
        label="Calibración",
    )

    if not validacion_df.empty:
        ax.scatter(
            validacion_df["disparidad_mediana_px"],
            validacion_df["altura_mm"],
            label="Validación",
        )

    modelo = modelos[mejor_nombre]

    ax.plot(
        x_curva,
        predecir_modelo(modelo, x_curva),
        label=f"Modelo {mejor_nombre}",
    )

    ax.set_xlabel("Disparidad representativa [px]")
    ax.set_ylabel("Altura conocida [mm]")
    ax.set_title("Calibración empírica disparidad-altura")
    ax.grid(True)
    ax.legend()

    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================


def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    comprobar_archivo(VIDEO_L, "el video izquierdo")
    comprobar_archivo(VIDEO_R, "el video derecho")
    comprobar_archivo(CSV_REGISTRO, "el CSV de registro")
    comprobar_archivo(ARCHIVO_CALIBRACION, "la calibración OpenCV")

    df_registro = pd.read_csv(CSV_REGISTRO)

    columnas_necesarias = {"frame_grabado", "timestamp"}
    faltantes = columnas_necesarias - set(df_registro.columns)

    if faltantes:
        raise RuntimeError(f"Faltan columnas en el CSV: {faltantes}")

    df_registro["timestamp"] = pd.to_datetime(
        df_registro["timestamp"],
        errors="coerce",
    )

    df_registro = (
        df_registro
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if df_registro.empty:
        raise RuntimeError("El CSV no contiene timestamps válidos.")

    cap_l = cv2.VideoCapture(VIDEO_L)
    cap_r = cv2.VideoCapture(VIDEO_R)

    if not cap_l.isOpened():
        raise RuntimeError("No se pudo abrir el video izquierdo.")

    if not cap_r.isOpened():
        raise RuntimeError("No se pudo abrir el video derecho.")

    frames_l = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho_l = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_l = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ancho_r = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_r = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\n====================================")
    print("VIDEOS")
    print("====================================")
    print("Frames cámara izquierda:", frames_l)
    print("Frames cámara derecha:", frames_r)
    print("Resolución izquierda:", ancho_l, "x", alto_l)
    print("Resolución derecha:", ancho_r, "x", alto_r)

    if (ancho_l, alto_l) != (ancho_r, alto_r):
        raise RuntimeError("Los dos videos tienen resoluciones diferentes.")

    calibracion = cargar_calibracion_opencv(
        ARCHIVO_CALIBRACION,
        ancho_l,
        alto_l,
    )

    primera_hora = pd.to_datetime(MEDICIONES_MANUALES[0]["timestamp"])
    coincidencia_inicial = buscar_frame_mas_cercano(df_registro, primera_hora)
    frame_referencia = coincidencia_inicial["frame_grabado"]

    rect_l_ref, rect_r_ref = leer_par_frames(
        cap_l,
        cap_r,
        frame_referencia,
        calibracion,
    )

    if rect_l_ref is None or rect_r_ref is None:
        raise RuntimeError("No se pudo leer el frame de referencia.")

    verificar_rectificacion(rect_l_ref, rect_r_ref)

    if (
        os.path.exists(ARCHIVO_CONFIG_ESTEREO)
        and not RECALCULAR_RANGO_DISPARIDAD
    ):
        datos_config = np.load(ARCHIVO_CONFIG_ESTEREO)
        min_disparity = int(datos_config["min_disparity"])
        num_disparities = int(datos_config["num_disparities"])

        print("\nRango de disparidad cargado:")
        print("minDisparity:", min_disparity)
        print("numDisparities:", num_disparities)
    else:
        (
            min_disparity,
            num_disparities,
            disparidades_seleccionadas,
        ) = calcular_rango_disparidad_manual(
            rect_l_ref,
            rect_r_ref,
        )

        np.savez(
            ARCHIVO_CONFIG_ESTEREO,
            min_disparity=np.asarray(min_disparity, dtype=np.int32),
            num_disparities=np.asarray(num_disparities, dtype=np.int32),
            disparidades_seleccionadas=disparidades_seleccionadas,
        )

        print("Configuración estéreo guardada en:")
        print(ARCHIVO_CONFIG_ESTEREO)

    stereo = crear_stereo_sgbm(min_disparity, num_disparities)

    if SELECCIONAR_ROI:
        roi = seleccionar_roi(
            cap_l,
            cap_r,
            frame_referencia,
            calibracion,
        )
    else:
        roi = ROI_FIJA

    print("\nROI utilizada:", roi)

    resultados = []

    for numero, medicion in enumerate(MEDICIONES_MANUALES, start=1):
        print("\n====================================")
        print(f"Procesando medición {numero}/{len(MEDICIONES_MANUALES)}")
        print("====================================")
        print("Hora:", medicion["timestamp"])
        print("Altura:", medicion["altura_mm"], "mm")

        resultado = procesar_medicion(
            medicion,
            df_registro,
            cap_l,
            cap_r,
            stereo,
            calibracion,
            roi,
            min_disparity,
            num_disparities,
        )

        resultados.append(resultado)

        print("Frame central:", resultado["frame_central"])
        print("Timestamp encontrado:", resultado["timestamp_encontrado"])
        print(
            "Diferencia temporal:",
            f"{resultado['diferencia_temporal_s']:.4f}",
            "s",
        )
        print(
            "Frames válidos:",
            resultado["frames_validos"],
            "/",
            resultado["frames_analizados"],
        )
        print("Disparidad mediana:", resultado["disparidad_mediana_px"], "px")

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

    df_resultados = pd.DataFrame(resultados)

    ruta_resultados = os.path.join(
        CARPETA_SALIDA,
        "pares_disparidad_altura_calibrados.csv",
    )

    df_resultados.to_csv(
        ruta_resultados,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nTabla guardada:")
    print(ruta_resultados)

    print("\nPares obtenidos:")
    print(
        df_resultados[
            [
                "timestamp_solicitado",
                "frame_central",
                "altura_mm",
                "uso",
                "disparidad_mediana_px",
                "desviacion_disparidad_px",
                "frames_validos",
            ]
        ]
    )

    modelos, mejor_nombre = ajustar_modelos(df_resultados)

    print("\n====================================")
    print("RESULTADOS DE LOS MODELOS")
    print("====================================")

    for nombre, modelo in modelos.items():
        print(f"\nModelo: {nombre}")
        print("Coeficientes:", modelo["coeficientes"])
        print(
            "RMSE calibración:",
            f"{modelo['rmse_calibracion_mm']:.3f} mm",
        )

        if np.isfinite(modelo["rmse_validacion_mm"]):
            print(
                "RMSE validación:",
                f"{modelo['rmse_validacion_mm']:.3f} mm",
            )

    print("\nMejor modelo:", mejor_nombre)
    mejor_modelo = modelos[mejor_nombre]

    ruta_modelo = os.path.join(
        CARPETA_SALIDA,
        "modelo_calibracion_altura_estereo_calibrado.npz",
    )

    np.savez(
        ruta_modelo,
        nombre_modelo=np.asarray(mejor_nombre),
        tipo_modelo=np.asarray(mejor_modelo["tipo"]),
        grado=np.asarray(mejor_modelo["grado"]),
        coeficientes=np.asarray(mejor_modelo["coeficientes"], dtype=np.float64),
        roi=np.asarray(roi, dtype=np.int32),
        min_disparity=np.asarray(min_disparity, dtype=np.int32),
        num_disparities=np.asarray(num_disparities, dtype=np.int32),
        K1=calibracion["K1"],
        D1=calibracion["D1"],
        K2=calibracion["K2"],
        D2=calibracion["D2"],
        R=calibracion["R"],
        T=calibracion["T"],
        R1=calibracion["R1"],
        R2=calibracion["R2"],
        P1=calibracion["P1"],
        P2=calibracion["P2"],
        Q=calibracion["Q"],
        ventana_temporal_s=np.asarray(VENTANA_TEMPORAL_S, dtype=np.float64),
        rmse_calibracion_mm=np.asarray(
            mejor_modelo["rmse_calibracion_mm"],
            dtype=np.float64,
        ),
        rmse_validacion_mm=np.asarray(
            mejor_modelo["rmse_validacion_mm"],
            dtype=np.float64,
        ),
    )

    ruta_grafico = os.path.join(
        CARPETA_SALIDA,
        "grafico_calibracion_estereo_calibrada.png",
    )

    guardar_grafico(
        df_resultados,
        modelos,
        mejor_nombre,
        ruta_grafico,
    )

    print("\nModelo guardado:")
    print(ruta_modelo)

    print("\nGráfico guardado:")
    print(ruta_grafico)

    print("\n====================================")
    print("CALIBRACIÓN FINALIZADA")
    print("====================================")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        cv2.destroyAllWindows()
        print("\n====================================")
        print("ERROR")
        print("====================================")
        print(error)
        raise
