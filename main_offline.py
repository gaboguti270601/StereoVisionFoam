import os
import cv2
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# =========================================================
# CONFIGURACIÓN
# =========================================================

VIDEO_L = (
    r"D:\MDT\Stereovision\videos"
    r"\experiencia_cam1.mp4"
)

VIDEO_R = (
    r"D:\MDT\Stereovision\videos"
    r"\experiencia_cam2.mp4"
)

CALIB_FILE = (
    r"D:\MDT\Stereovision\calibration"
    r"\stereo_calibration.npz"
)

CARPETA_SALIDA = (
    r"D:\MDT\Stereovision\resultados_offline"
)

NOMBRE_EXPERIENCIA = "prueba_offline"

# Frame desde el cual comienza la reproducción.
FRAME_INICIAL = 0

# Procesar un frame de cada N.
# 1 = todos los frames.
# 2 = uno de cada dos frames.
SALTO_FRAMES = 1

# Si True, intenta determinar la orientación adecuada.
SELECCIONAR_ROTACION_AUTOMATICA = True

# Solo se usa si la selección automática está desactivada.
ROTACION_MANUAL = cv2.ROTATE_90_COUNTERCLOCKWISE

USAR_RECTIFICACION = True
USAR_ROI = True

# Líneas horizontales para revisar alineación epipolar.
DIBUJAR_LINEAS_EPIPOLARES = True
SEPARACION_LINEAS_PX = 100

# =========================================================
# PARÁMETROS ESTÉREO
# =========================================================

NUM_DISPARIDADES = 64
BLOCK_SIZE = 11

BASELINE_FISICO_MM = 75.0
BASELINE_MATLAB_MM = 125.2247934179211

FACTOR_CORRECCION = (
    BASELINE_FISICO_MM / BASELINE_MATLAB_MM
)

ANGULO_GRADOS = 3.47

PROFUNDIDAD_MINIMA_MM = 100
PROFUNDIDAD_MAXIMA_MM = 3000

MINIMO_PIXELES_VALIDOS = 50
MINIMA_FRACCION_IMAGEN_UTIL = 0.01

VIDEO_CODEC = "mp4v"


# =========================================================
# ESTADO
# =========================================================

roi_data = {
    "seleccionado": False,
    "x": 0,
    "y": 0,
    "w": 0,
    "h": 0
}

estado = {
    "pausado": False,
    "avance_automatico": True,
    "grabando": False,
    "z_ref": None,
    "frame_referencia": None,
    "rectificacion_ok": False,
    "rotacion_nombre": "sin determinar"
}

resultados_tiempo = []
resultados_altura = []


# =========================================================
# UTILIDADES
# =========================================================

def put_text_outline(
    img,
    text,
    org,
    scale,
    color,
    thickness=2
):
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 4,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def dibujar_lineas_horizontales(
    img,
    separacion=100
):
    salida = img.copy()
    h, w = salida.shape[:2]

    for y in range(0, h, separacion):
        cv2.line(
            salida,
            (0, y),
            (w - 1, y),
            (0, 255, 0),
            1
        )

    return salida


def redimensionar_visualizacion(
    img,
    ancho_maximo=700,
    alto_maximo=800
):
    h, w = img.shape[:2]

    escala = min(
        ancho_maximo / float(w),
        alto_maximo / float(h),
        1.0
    )

    nuevo_w = int(round(w * escala))
    nuevo_h = int(round(h * escala))

    if nuevo_w % 2 != 0:
        nuevo_w += 1

    if nuevo_h % 2 != 0:
        nuevo_h += 1

    return cv2.resize(
        img,
        (nuevo_w, nuevo_h)
    )


def normalizar_nombre(texto):
    texto = texto.strip().replace(" ", "_")

    caracteres_validos = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_-"
    )

    texto = "".join(
        caracter
        for caracter in texto
        if caracter in caracteres_validos
    )

    return texto or "procesamiento_offline"


def imprimir_estadisticas(nombre, img):
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    fraccion_util = (
        np.count_nonzero(gray > 2)
        / gray.size
    )

    print(f"\n{nombre}")
    print("shape:", img.shape)
    print("min:", int(img.min()))
    print("max:", int(img.max()))
    print("mean:", float(np.mean(img)))
    print(
        "píxeles no negros:",
        f"{100 * fraccion_util:.2f}%"
    )


def imagen_valida(img):
    if img is None or img.size == 0:
        return False

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    fraccion_util = (
        np.count_nonzero(gray > 2)
        / gray.size
    )

    return (
        img.max() >= 5
        and np.mean(img) > 0.1
        and fraccion_util >= MINIMA_FRACCION_IMAGEN_UTIL
    )


# =========================================================
# CALIBRACIÓN
# =========================================================

def cargar_calibracion():
    if not os.path.isfile(CALIB_FILE):
        raise FileNotFoundError(
            f"No existe:\n{CALIB_FILE}"
        )

    calibracion = np.load(CALIB_FILE)

    requeridas = [
        "K_l",
        "D_l",
        "K_r",
        "D_r",
        "R",
        "T",
        "Q",
        "map_l1",
        "map_l2",
        "map_r1",
        "map_r2"
    ]

    faltantes = [
        variable
        for variable in requeridas
        if variable not in calibracion.files
    ]

    if faltantes:
        raise KeyError(
            "Faltan variables en el NPZ: "
            + ", ".join(faltantes)
        )

    baseline = float(
        np.linalg.norm(calibracion["T"])
    )

    print("\n===================================")
    print("CALIBRACIÓN")
    print("===================================")

    print("Archivo:", CALIB_FILE)
    print("Baseline NPZ:", baseline, "mm")
    print(
        "Factor de corrección:",
        FACTOR_CORRECCION
    )

    print(
        "map_l1:",
        calibracion["map_l1"].shape
    )

    print(
        "map_r1:",
        calibracion["map_r1"].shape
    )

    if "image_size" in calibracion.files:
        print(
            "image_size:",
            calibracion["image_size"]
        )

    return calibracion


# =========================================================
# VIDEOS
# =========================================================

def abrir_videos():
    cap_l = cv2.VideoCapture(VIDEO_L)
    cap_r = cv2.VideoCapture(VIDEO_R)

    if not cap_l.isOpened():
        raise RuntimeError(
            f"No se pudo abrir:\n{VIDEO_L}"
        )

    if not cap_r.isOpened():
        cap_l.release()

        raise RuntimeError(
            f"No se pudo abrir:\n{VIDEO_R}"
        )

    frames_l = int(
        cap_l.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frames_r = int(
        cap_r.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps_l = float(
        cap_l.get(cv2.CAP_PROP_FPS)
    )

    fps_r = float(
        cap_r.get(cv2.CAP_PROP_FPS)
    )

    width_l = int(
        cap_l.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height_l = int(
        cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    width_r = int(
        cap_r.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height_r = int(
        cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = fps_l if fps_l > 0 else fps_r

    if fps <= 0:
        fps = 30.0

    total_frames = min(frames_l, frames_r)

    print("\n===================================")
    print("VIDEOS")
    print("===================================")

    print("Video izquierdo:", VIDEO_L)
    print("Resolución:", width_l, "x", height_l)
    print("Frames:", frames_l)
    print("FPS:", fps_l)

    print("\nVideo derecho:", VIDEO_R)
    print("Resolución:", width_r, "x", height_r)
    print("Frames:", frames_r)
    print("FPS:", fps_r)

    if abs(fps_l - fps_r) > 0.1:
        print(
            "\nADVERTENCIA: los FPS de los "
            "videos son diferentes."
        )

    if frames_l != frames_r:
        print(
            "\nADVERTENCIA: los videos tienen "
            "distinto número de frames."
        )

    cap_l.set(
        cv2.CAP_PROP_POS_FRAMES,
        FRAME_INICIAL
    )

    cap_r.set(
        cv2.CAP_PROP_POS_FRAMES,
        FRAME_INICIAL
    )

    return (
        cap_l,
        cap_r,
        fps,
        total_frames
    )


# =========================================================
# ROTACIÓN Y RECTIFICACIÓN
# =========================================================

def aplicar_rotacion(img, rotacion):
    if rotacion is None:
        return img.copy()

    return cv2.rotate(img, rotacion)


def rectificar_par(
    img_l,
    img_r,
    rotacion,
    map_l1,
    map_l2,
    map_r1,
    map_r2
):
    img_l_rot = aplicar_rotacion(
        img_l,
        rotacion
    )

    img_r_rot = aplicar_rotacion(
        img_r,
        rotacion
    )

    if img_l_rot.shape[:2] != map_l1.shape[:2]:
        return (
            img_l_rot,
            img_r_rot,
            None,
            None,
            False
        )

    rect_l = cv2.remap(
        img_l_rot,
        map_l1,
        map_l2,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    rect_r = cv2.remap(
        img_r_rot,
        map_r1,
        map_r2,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    rectificacion_ok = (
        imagen_valida(rect_l)
        and imagen_valida(rect_r)
    )

    return (
        img_l_rot,
        img_r_rot,
        rect_l,
        rect_r,
        rectificacion_ok
    )


def seleccionar_mejor_rotacion(
    frame_l,
    frame_r,
    map_l1,
    map_l2,
    map_r1,
    map_r2
):
    candidatos = [
        ("sin_rotar", None),
        (
            "90_antihorario",
            cv2.ROTATE_90_COUNTERCLOCKWISE
        ),
        (
            "90_horario",
            cv2.ROTATE_90_CLOCKWISE
        ),
        (
            "180_grados",
            cv2.ROTATE_180
        )
    ]

    mejor = None

    print("\n===================================")
    print("PRUEBA DE ROTACIÓN")
    print("===================================")

    for nombre, rotacion in candidatos:
        (
            img_l_rot,
            img_r_rot,
            rect_l,
            rect_r,
            ok
        ) = rectificar_par(
            frame_l,
            frame_r,
            rotacion,
            map_l1,
            map_l2,
            map_r1,
            map_r2
        )

        print(f"\nRotación: {nombre}")
        print(
            "Frame rotado:",
            img_l_rot.shape[:2]
        )

        print(
            "Mapa:",
            map_l1.shape[:2]
        )

        if rect_l is None or rect_r is None:
            print(
                "Resolución incompatible."
            )
            continue

        gray_l = cv2.cvtColor(
            rect_l,
            cv2.COLOR_BGR2GRAY
        )

        gray_r = cv2.cvtColor(
            rect_r,
            cv2.COLOR_BGR2GRAY
        )

        proporcion_l = (
            np.count_nonzero(gray_l > 2)
            / gray_l.size
        )

        proporcion_r = (
            np.count_nonzero(gray_r > 2)
            / gray_r.size
        )

        score = (
            proporcion_l
            + proporcion_r
            + np.mean(gray_l) / 255.0
            + np.mean(gray_r) / 255.0
        )

        print(
            "Área útil izquierda:",
            f"{100 * proporcion_l:.2f}%"
        )

        print(
            "Área útil derecha:",
            f"{100 * proporcion_r:.2f}%"
        )

        print("Score:", float(score))
        print("Imagen válida:", ok)

        if mejor is None or score > mejor["score"]:
            mejor = {
                "nombre": nombre,
                "rotacion": rotacion,
                "score": score,
                "ok": ok
            }

    if mejor is None:
        raise RuntimeError(
            "Ninguna orientación tiene "
            "resolución compatible con los mapas."
        )

    print("\nOrientación seleccionada:")
    print(mejor["nombre"])
    print("Rectificación válida:", mejor["ok"])

    return (
        mejor["rotacion"],
        mejor["nombre"]
    )


# =========================================================
# ROI
# =========================================================

def seleccionar_roi(img):
    ventana = (
        "Seleccionar ROI de superficie/espuma"
    )

    roi = cv2.selectROI(
        ventana,
        img,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(ventana)

    x, y, w, h = roi

    if w <= 0 or h <= 0:
        print("\nROI no seleccionado.")
        return False

    roi_data["x"] = int(x)
    roi_data["y"] = int(y)
    roi_data["w"] = int(w)
    roi_data["h"] = int(h)
    roi_data["seleccionado"] = True

    print("\nROI:")
    print(
        roi_data["x"],
        roi_data["y"],
        roi_data["w"],
        roi_data["h"]
    )

    return True


# =========================================================
# DISPARIDAD Y PROFUNDIDAD
# =========================================================

def crear_matcher_estereo():
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISPARIDADES,
        blockSize=BLOCK_SIZE,
        P1=8 * BLOCK_SIZE ** 2,
        P2=32 * BLOCK_SIZE ** 2,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )


def calcular_disparidad(
    stereo,
    img_l,
    img_r
):
    gray_l = cv2.cvtColor(
        img_l,
        cv2.COLOR_BGR2GRAY
    )

    gray_r = cv2.cvtColor(
        img_r,
        cv2.COLOR_BGR2GRAY
    )

    disparidad = stereo.compute(
        gray_l,
        gray_r
    ).astype(np.float32) / 16.0

    return disparidad


def disparidad_a_profundidad(
    disparidad,
    Q
):
    puntos_3d = cv2.reprojectImageTo3D(
        disparidad,
        Q
    )

    depth = puntos_3d[:, :, 2].astype(
        np.float32
    )

    invalidos = (
        ~np.isfinite(depth)
        | (disparidad <= 0)
        | (depth < PROFUNDIDAD_MINIMA_MM)
        | (depth > PROFUNDIDAD_MAXIMA_MM)
    )

    depth[invalidos] = np.nan

    return depth


# =========================================================
# SUPERFICIE Y ALTURA
# =========================================================

def detectar_superficie(gray):
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def obtener_profundidad_superficie(
    depth_roi,
    mask
):
    validos = (
        (mask > 0)
        & np.isfinite(depth_roi)
    )

    cantidad = int(
        np.count_nonzero(validos)
    )

    if cantidad < MINIMO_PIXELES_VALIDOS:
        return None, cantidad

    z_superficie = float(
        np.nanmedian(depth_roi[validos])
    )

    return z_superficie, cantidad


def calcular_altura(
    z_ref,
    z_superficie
):
    altura_mm = (
        abs(z_ref - z_superficie)
        * np.cos(
            np.radians(ANGULO_GRADOS)
        )
        * FACTOR_CORRECCION
    )

    return float(altura_mm)


# =========================================================
# VISUALIZACIONES
# =========================================================

def visualizar_depth(depth):
    if depth is None:
        return np.zeros(
            (480, 640, 3),
            dtype=np.uint8
        )

    validos = np.isfinite(depth)

    if not np.any(validos):
        return np.zeros(
            (
                depth.shape[0],
                depth.shape[1],
                3
            ),
            dtype=np.uint8
        )

    minimo = np.percentile(
        depth[validos],
        5
    )

    maximo = np.percentile(
        depth[validos],
        95
    )

    if maximo <= minimo:
        normalizada = np.zeros(
            depth.shape,
            dtype=np.uint8
        )
    else:
        limitada = np.clip(
            depth,
            minimo,
            maximo
        )

        normalizada_float = (
            limitada - minimo
        ) / (
            maximo - minimo
        )

        normalizada_float = np.nan_to_num(
            normalizada_float,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        normalizada = (
            normalizada_float * 255
        ).astype(np.uint8)

    return cv2.applyColorMap(
        normalizada,
        cv2.COLORMAP_JET
    )


def guardar_grafico(
    tiempos,
    alturas,
    ruta
):
    if len(alturas) == 0:
        print(
            "\nNo hay datos para crear gráfico."
        )
        return

    fig, ax = plt.subplots()

    ax.plot(
        tiempos,
        alturas,
        label="Altura estimada"
    )

    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Altura [mm]")
    ax.set_title(
        "Altura de espuma mediante estereovisión"
    )

    ax.grid(True)
    ax.legend()

    fig.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nGráfico guardado:")
    print(ruta)


# =========================================================
# PROCESAMIENTO
# =========================================================

def main():
    os.makedirs(
        CARPETA_SALIDA,
        exist_ok=True
    )

    nombre = normalizar_nombre(
        NOMBRE_EXPERIENCIA
    )

    csv_path = os.path.join(
        CARPETA_SALIDA,
        f"{nombre}_altura.csv"
    )

    video_l_path = os.path.join(
        CARPETA_SALIDA,
        f"{nombre}_cam1_procesado.mp4"
    )

    video_r_path = os.path.join(
        CARPETA_SALIDA,
        f"{nombre}_cam2_procesado.mp4"
    )

    depth_path = os.path.join(
        CARPETA_SALIDA,
        f"{nombre}_profundidad.mp4"
    )

    grafico_path = os.path.join(
        CARPETA_SALIDA,
        f"{nombre}_grafico_altura.png"
    )

    calibracion = cargar_calibracion()

    Q = calibracion["Q"]

    map_l1 = calibracion["map_l1"]
    map_l2 = calibracion["map_l2"]
    map_r1 = calibracion["map_r1"]
    map_r2 = calibracion["map_r2"]

    (
        cap_l,
        cap_r,
        fps,
        total_frames
    ) = abrir_videos()

    ret_l, frame_l = cap_l.read()
    ret_r, frame_r = cap_r.read()

    if not ret_l or not ret_r:
        cap_l.release()
        cap_r.release()

        raise RuntimeError(
            "No fue posible leer el primer par."
        )

    if SELECCIONAR_ROTACION_AUTOMATICA:
        (
            rotacion,
            nombre_rotacion
        ) = seleccionar_mejor_rotacion(
            frame_l,
            frame_r,
            map_l1,
            map_l2,
            map_r1,
            map_r2
        )
    else:
        rotacion = ROTACION_MANUAL
        nombre_rotacion = "manual"

    estado["rotacion_nombre"] = (
        nombre_rotacion
    )

    # Volver al frame inicial.
    cap_l.set(
        cv2.CAP_PROP_POS_FRAMES,
        FRAME_INICIAL
    )

    cap_r.set(
        cv2.CAP_PROP_POS_FRAMES,
        FRAME_INICIAL
    )

    stereo = crear_matcher_estereo()

    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    writer = csv.writer(csv_file)

    writer.writerow([
        "frame",
        "tiempo_s",
        "timestamp_video",
        "altura_mm",
        "altura_cm",
        "z_ref_mm",
        "z_superficie_mm",
        "pixeles_validos",
        "rectificacion_ok",
        "rotacion"
    ])

    writer_video_l = None
    writer_video_r = None
    writer_depth = None

    frame_actual = FRAME_INICIAL
    ultimo_par = None

    cv2.namedWindow(
        "Estereo offline",
        cv2.WINDOW_NORMAL
    )

    cv2.namedWindow(
        "Mapa de profundidad",
        cv2.WINDOW_NORMAL
    )

    print("\n===================================")
    print("CONTROLES")
    print("===================================")
    print("ESPACIO: pausar/reanudar")
    print("R: capturar referencia")
    print("G: comenzar guardado")
    print("S: detener guardado")
    print("P: seleccionar nuevo ROI")
    print("A: avance automático")
    print("Q: salir")

    try:
        while frame_actual < total_frames:
            if (
                not estado["pausado"]
                or ultimo_par is None
            ):
                ret_l, frame_l = cap_l.read()
                ret_r, frame_r = cap_r.read()

                if not ret_l or not ret_r:
                    print(
                        "\nFin de uno de los videos."
                    )
                    break

                ultimo_par = (
                    frame_l.copy(),
                    frame_r.copy()
                )
            else:
                frame_l, frame_r = ultimo_par

            (
                img_l_rot,
                img_r_rot,
                rect_l,
                rect_r,
                rectificacion_ok
            ) = rectificar_par(
                frame_l,
                frame_r,
                rotacion,
                map_l1,
                map_l2,
                map_r1,
                map_r2
            )

            estado["rectificacion_ok"] = (
                rectificacion_ok
            )

            if (
                USAR_RECTIFICACION
                and rectificacion_ok
            ):
                procesada_l = rect_l
                procesada_r = rect_r
                modo = "RECTIFICADA"
            else:
                procesada_l = img_l_rot
                procesada_r = img_r_rot

                modo = (
                    "RAW - RECTIFICACION FALLIDA"
                    if USAR_RECTIFICACION
                    else "SIN RECTIFICAR"
                )

            # Inicializar los videos de salida.
            if writer_video_l is None:
                h_salida, w_salida = (
                    procesada_l.shape[:2]
                )

                fourcc = (
                    cv2.VideoWriter_fourcc(
                        *VIDEO_CODEC
                    )
                )

                writer_video_l = cv2.VideoWriter(
                    video_l_path,
                    fourcc,
                    fps / SALTO_FRAMES,
                    (w_salida, h_salida)
                )

                writer_video_r = cv2.VideoWriter(
                    video_r_path,
                    fourcc,
                    fps / SALTO_FRAMES,
                    (w_salida, h_salida)
                )

                writer_depth = cv2.VideoWriter(
                    depth_path,
                    fourcc,
                    fps / SALTO_FRAMES,
                    (w_salida, h_salida)
                )

            # Selección inicial de ROI.
            if (
                USAR_ROI
                and not roi_data["seleccionado"]
                and rectificacion_ok
            ):
                estado["pausado"] = True

                seleccionar_roi(procesada_l)

            disparidad = None
            depth = None
            roi_depth = None
            gray_roi = None

            if rectificacion_ok:
                disparidad = calcular_disparidad(
                    stereo,
                    procesada_l,
                    procesada_r
                )

                depth = disparidad_a_profundidad(
                    disparidad,
                    Q
                )

            display_l = procesada_l.copy()
            display_r = procesada_r.copy()

            if (
                rectificacion_ok
                and depth is not None
            ):
                if (
                    USAR_ROI
                    and roi_data["seleccionado"]
                ):
                    x = roi_data["x"]
                    y = roi_data["y"]
                    w = roi_data["w"]
                    h = roi_data["h"]

                    roi_l = procesada_l[
                        y:y + h,
                        x:x + w
                    ]

                    roi_depth = depth[
                        y:y + h,
                        x:x + w
                    ]

                    if (
                        roi_l.size > 0
                        and roi_depth.size > 0
                    ):
                        gray_roi = cv2.cvtColor(
                            roi_l,
                            cv2.COLOR_BGR2GRAY
                        )

                    cv2.rectangle(
                        display_l,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        3
                    )

                    cv2.rectangle(
                        display_r,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        3
                    )

                else:
                    gray_roi = cv2.cvtColor(
                        procesada_l,
                        cv2.COLOR_BGR2GRAY
                    )

                    roi_depth = depth

            z_superficie = None
            altura_mm = None
            cantidad_validos = 0
            mask = None

            if (
                gray_roi is not None
                and roi_depth is not None
            ):
                mask = detectar_superficie(
                    gray_roi
                )

                (
                    z_superficie,
                    cantidad_validos
                ) = obtener_profundidad_superficie(
                    roi_depth,
                    mask
                )

                if (
                    estado["z_ref"] is not None
                    and z_superficie is not None
                ):
                    altura_mm = calcular_altura(
                        estado["z_ref"],
                        z_superficie
                    )

            tiempo_s = frame_actual / fps

            texto_altura = (
                f"Altura: {altura_mm:.2f} mm"
                if altura_mm is not None
                else "Altura: ---"
            )

            put_text_outline(
                display_l,
                f"Camara 1 - {modo}",
                (35, 55),
                0.8,
                (255, 255, 0)
            )

            put_text_outline(
                display_r,
                f"Camara 2 - {modo}",
                (35, 55),
                0.8,
                (255, 255, 0)
            )

            put_text_outline(
                display_l,
                f"Frame: {frame_actual}/{total_frames}",
                (35, 105),
                0.7,
                (255, 255, 255)
            )

            put_text_outline(
                display_l,
                f"Tiempo: {tiempo_s:.2f} s",
                (35, 150),
                0.7,
                (255, 255, 255)
            )

            put_text_outline(
                display_l,
                texto_altura,
                (35, 200),
                0.8,
                (
                    (0, 255, 0)
                    if altura_mm is not None
                    else (0, 165, 255)
                )
            )

            put_text_outline(
                display_l,
                (
                    "Rectificacion: OK"
                    if rectificacion_ok
                    else "Rectificacion: FALLIDA"
                ),
                (35, 250),
                0.7,
                (
                    (0, 255, 0)
                    if rectificacion_ok
                    else (0, 0, 255)
                )
            )

            if estado["z_ref"] is not None:
                put_text_outline(
                    display_l,
                    f"Z ref: {estado['z_ref']:.2f} mm",
                    (35, 300),
                    0.65,
                    (220, 220, 220)
                )

            if estado["pausado"]:
                put_text_outline(
                    display_l,
                    "PAUSA",
                    (35, 350),
                    0.9,
                    (0, 165, 255)
                )

            if estado["grabando"]:
                put_text_outline(
                    display_l,
                    "GUARDANDO",
                    (35, 400),
                    0.9,
                    (0, 0, 255)
                )

            put_text_outline(
                display_l,
                (
                    "ESPACIO: pausa | R: referencia | "
                    "G/S: guardar | P: ROI | Q: salir"
                ),
                (35, 450),
                0.5,
                (230, 230, 230)
            )

            if DIBUJAR_LINEAS_EPIPOLARES:
                display_l = (
                    dibujar_lineas_horizontales(
                        display_l,
                        SEPARACION_LINEAS_PX
                    )
                )

                display_r = (
                    dibujar_lineas_horizontales(
                        display_r,
                        SEPARACION_LINEAS_PX
                    )
                )

            vista_l = redimensionar_visualizacion(
                display_l
            )

            vista_r = redimensionar_visualizacion(
                display_r
            )

            if vista_l.shape[:2] != vista_r.shape[:2]:
                vista_r = cv2.resize(
                    vista_r,
                    (
                        vista_l.shape[1],
                        vista_l.shape[0]
                    )
                )

            combined = np.hstack(
                [vista_l, vista_r]
            )

            cv2.imshow(
                "Estereo offline",
                combined
            )

            depth_color = visualizar_depth(
                depth
            )

            cv2.imshow(
                "Mapa de profundidad",
                depth_color
            )

            if (
                estado["grabando"]
                and not estado["pausado"]
            ):
                writer_video_l.write(
                    display_l
                )

                writer_video_r.write(
                    display_r
                )

                if depth_color.shape[:2] != (
                    display_l.shape[:2]
                ):
                    depth_guardado = cv2.resize(
                        depth_color,
                        (
                            display_l.shape[1],
                            display_l.shape[0]
                        )
                    )
                else:
                    depth_guardado = depth_color

                writer_depth.write(
                    depth_guardado
                )

                timestamp_video = str(
                    datetime.utcfromtimestamp(
                        tiempo_s
                    ).strftime("%H:%M:%S.%f")[:-3]
                )

                writer.writerow([
                    frame_actual,
                    tiempo_s,
                    timestamp_video,
                    (
                        altura_mm
                        if altura_mm is not None
                        else ""
                    ),
                    (
                        altura_mm / 10.0
                        if altura_mm is not None
                        else ""
                    ),
                    (
                        estado["z_ref"]
                        if estado["z_ref"] is not None
                        else ""
                    ),
                    (
                        z_superficie
                        if z_superficie is not None
                        else ""
                    ),
                    cantidad_validos,
                    rectificacion_ok,
                    estado["rotacion_nombre"]
                ])

                csv_file.flush()

                if altura_mm is not None:
                    resultados_tiempo.append(
                        tiempo_s
                    )

                    resultados_altura.append(
                        altura_mm
                    )

            # En pausa, waitKey(0) espera una tecla.
            # En reproducción, se respeta aproximadamente
            # el FPS del video.
            if estado["pausado"]:
                delay = 0
            else:
                delay = max(
                    1,
                    int(
                        1000
                        * SALTO_FRAMES
                        / fps
                    )
                )

            key = cv2.waitKey(delay) & 0xFF

            if key == ord(" "):
                estado["pausado"] = (
                    not estado["pausado"]
                )

            elif key in (ord("r"), ord("R")):
                if (
                    not rectificacion_ok
                    or z_superficie is None
                ):
                    print(
                        "\nNo se puede capturar "
                        "la referencia."
                    )

                    print(
                        "Revise rectificación, ROI "
                        "y píxeles de profundidad."
                    )
                else:
                    estado["z_ref"] = z_superficie
                    estado["frame_referencia"] = (
                        frame_actual
                    )

                    print("\nReferencia capturada:")
                    print(
                        "Frame:",
                        frame_actual
                    )

                    print(
                        "Tiempo:",
                        tiempo_s,
                        "s"
                    )

                    print(
                        "Z referencia:",
                        estado["z_ref"],
                        "mm"
                    )

                    print(
                        "Píxeles válidos:",
                        cantidad_validos
                    )

            elif key in (ord("g"), ord("G")):
                estado["grabando"] = True

                print(
                    "\nGuardado de resultados iniciado."
                )

            elif key in (ord("s"), ord("S")):
                estado["grabando"] = False

                print(
                    "\nGuardado de resultados detenido."
                )

            elif key in (ord("p"), ord("P")):
                if rectificacion_ok:
                    estado["pausado"] = True
                    estado["z_ref"] = None

                    seleccionar_roi(
                        procesada_l
                    )
                else:
                    print(
                        "\nNo se puede seleccionar ROI "
                        "sin imagen válida."
                    )

            elif key in (ord("a"), ord("A")):
                estado["avance_automatico"] = (
                    not estado["avance_automatico"]
                )

                print(
                    "\nAvance automático:",
                    estado["avance_automatico"]
                )

            elif key in (ord("q"), ord("Q")):
                break

            if not estado["pausado"]:
                # Saltar frames adicionales si SALTO_FRAMES > 1.
                for _ in range(SALTO_FRAMES - 1):
                    cap_l.grab()
                    cap_r.grab()

                frame_actual += SALTO_FRAMES

    finally:
        csv_file.close()

        cap_l.release()
        cap_r.release()

        if writer_video_l is not None:
            writer_video_l.release()

        if writer_video_r is not None:
            writer_video_r.release()

        if writer_depth is not None:
            writer_depth.release()

        cv2.destroyAllWindows()

    guardar_grafico(
        resultados_tiempo,
        resultados_altura,
        grafico_path
    )

    print("\n===================================")
    print("PROCESAMIENTO FINALIZADO")
    print("===================================")

    print("CSV:")
    print(csv_path)

    print("\nVideo cámara izquierda:")
    print(video_l_path)

    print("\nVideo cámara derecha:")
    print(video_r_path)

    print("\nVideo profundidad:")
    print(depth_path)

    print("\nGráfico:")
    print(grafico_path)


if __name__ == "__main__":
    main()