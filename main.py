import os
import time
import csv
from datetime import datetime
from collections import deque

import cv2
import numpy as np


cv2.startWindowThread()


# =========================================================
# DLL THORLABS
# =========================================================

DLL_PATH = (
    r"D:\MDT\Scientific Camera Interfaces"
    r"\SDK\Native Toolkit\dlls\Native_64_lib"
)

os.add_dll_directory(DLL_PATH)


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

CAMARA_L_SERIAL = "18842"
CAMARA_R_SERIAL = "36930"

EXPOSICION_US = 40

FPS_OBJETIVO = 30.0
FRAME_TIME = 1.0 / FPS_OBJETIVO

VIDEO_CODEC = "mp4v"
ANCHO_VIDEO_SALIDA = 960
ANCHO_VISTA_CAMARA = 600

# Orientación definitiva del montaje.
ROTACION_L = cv2.ROTATE_90_CLOCKWISE
ROTACION_R = cv2.ROTATE_90_COUNTERCLOCKWISE

MOSTRAR_OVERLAY = True
MOSTRAR_LINEAS_HORIZONTALES = False
SEPARACION_LINEAS_PX = 80

GUARDAR_CSV = True
TIMEOUT_FRAME_S = 1.0


# =========================================================
# ALINEACIÓN MANUAL SIN CALIBRACIÓN
# =========================================================

MIN_PARES_MANUALES = 8
MAX_PARES_MANUALES = 40

RANSAC_THRESHOLD_PX = 4.0
MIN_INLIERS = 6
PERMITIR_ESCALA = False


# =========================================================
# DISPARIDAD Y ALTURA POR REFERENCIA
# =========================================================

# Distancia conocida del plano inicial.
Z_REF_MM = 900.0

MIN_DISPARITY = -128
NUM_DISPARITIES = 256
BLOCK_SIZE = 7

DISPARIDAD_ABS_MIN = 1.0
DISPARIDAD_ABS_MAX = 250.0
MIN_PIXELES_VALIDOS = 80

PERCENTIL_SUPERFICIE = 85.0
MARGEN_LIMITE_DISPARIDAD = 1.0
VENTANA_MEDIANA = 5

# 1 = medir en cada frame.
# 2 = medir cada 2 frames, normalmente más fluido.
PROCESAR_DISPARIDAD_CADA_N_FRAMES = 2


# =========================================================
# ESTADO
# =========================================================

estado = {
    "activo": True,
    "grabando": False,
    "t_inicio_grabacion": None,
    "frame_global": 0,
    "frame_grabado": 0,
    "d_ref": None,
    "K_ref": None,
    "ultima_disparidad": None,
    "ultima_distancia_mm": None,
    "ultima_altura_mm": None,
    "ultimos_pixeles_validos": 0,
}

roi_data = {
    "seleccionado": False,
    "x": 0,
    "y": 0,
    "w": 0,
    "h": 0,
}

historial_tiempos = deque(maxlen=60)
historial_altura = deque(maxlen=VENTANA_MEDIANA)



def aplicar_rotacion(img, lado):
    """
    Aplica la rotación correspondiente a cada cámara.

    Cámara L:
        sin rotación.

    Cámara R:
        rotación de 180 grados.
    """

    if lado == "L":
        rotacion = ROTACION_L
    elif lado == "R":
        rotacion = ROTACION_R
    else:
        raise ValueError('lado debe ser "L" o "R".')

    if rotacion is None:
        return img.copy()

    return cv2.rotate(img, rotacion)




def put_text_outline(
    img,
    text,
    org,
    scale=0.65,
    color=(255, 255, 255),
    thickness=2,
):
    """
    Dibuja texto legible con contorno negro.
    """

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




def dibujar_lineas_horizontales(
    img,
    separacion=80,
):
    """
    Dibuja líneas horizontales solamente para visualización.
    """

    salida = img.copy()
    h, w = salida.shape[:2]

    for y in range(0, h, separacion):
        cv2.line(
            salida,
            (0, y),
            (w - 1, y),
            (0, 255, 0),
            1,
        )

    return salida




def calcular_tamano_con_ancho(
    img,
    ancho_salida,
):
    """
    Calcula un tamaño manteniendo la relación de aspecto.
    Fuerza dimensiones pares para VideoWriter.
    """

    h, w = img.shape[:2]

    escala = ancho_salida / float(w)
    alto_salida = int(round(h * escala))

    if ancho_salida % 2 != 0:
        ancho_salida += 1

    if alto_salida % 2 != 0:
        alto_salida += 1

    return int(ancho_salida), int(alto_salida)




def redimensionar_para_vista(
    img,
    ancho_salida,
):
    """
    Redimensiona una imagen para mostrarla en pantalla.
    """

    out_w, out_h = calcular_tamano_con_ancho(
        img,
        ancho_salida,
    )

    return cv2.resize(
        img,
        (out_w, out_h),
        interpolation=cv2.INTER_AREA,
    )




def convertir_a_8bits_bgr(
    image_buffer,
    ancho,
    alto,
):
    """
    Convierte el buffer monocromático de la cámara a BGR de 8 bits.

    La normalización se hace de forma independiente para cada frame,
    igual que en el código anterior.
    """

    img_16 = np.array(
        image_buffer,
        dtype=np.uint16,
    ).reshape(alto, ancho)

    minimo = int(img_16.min())
    maximo = int(img_16.max())

    if maximo > minimo:
        img_8 = cv2.normalize(
            img_16,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)
    else:
        img_8 = np.zeros(
            img_16.shape,
            dtype=np.uint8,
        )

    return cv2.cvtColor(
        img_8,
        cv2.COLOR_GRAY2BGR,
    )




def calcular_fps_real():
    """
    Calcula FPS real usando los últimos tiempos de captura.
    """

    ahora = time.perf_counter()
    historial_tiempos.append(ahora)

    if len(historial_tiempos) < 2:
        return 0.0

    intervalo = (
        historial_tiempos[-1]
        - historial_tiempos[0]
    )

    if intervalo <= 0:
        return 0.0

    return (
        len(historial_tiempos) - 1
    ) / intervalo




def inicializar_camaras():
    """
    Detecta y configura las dos cámaras por número de serie.

    Asignación fija:
        36933 -> cámara izquierda (L / cámara 1)
        36930 -> cámara derecha (R / cámara 2)

    No depende del orden en que Windows o el SDK detecten las cámaras.
    """

    from thorlabs_tsi_sdk.tl_camera import (
        TLCameraSDK,
        OPERATION_MODE,
    )

    sdk = TLCameraSDK()
    seriales_detectados = list(
        sdk.discover_available_cameras()
    )

    print("\n===================================")
    print("CÁMARAS DETECTADAS")
    print("===================================")
    print(seriales_detectados)

    faltantes = []

    if CAMARA_L_SERIAL not in seriales_detectados:
        faltantes.append(
            f"L={CAMARA_L_SERIAL}"
        )

    if CAMARA_R_SERIAL not in seriales_detectados:
        faltantes.append(
            f"R={CAMARA_R_SERIAL}"
        )

    if faltantes:
        sdk.dispose()

        raise RuntimeError(
            "No se encontraron las cámaras requeridas: "
            + ", ".join(faltantes)
            + f". Detectadas: {seriales_detectados}"
        )

    # Se abren explícitamente por número de serie.
    cam_l = sdk.open_camera(
        CAMARA_L_SERIAL
    )

    cam_r = sdk.open_camera(
        CAMARA_R_SERIAL
    )

    for cam in (cam_l, cam_r):
        cam.exposure_time_us = EXPOSICION_US

        cam.frames_per_trigger_zero_for_unlimited = 0

        cam.operation_mode = (
            OPERATION_MODE.SOFTWARE_TRIGGERED
        )

        cam.arm(2)

    print("\nAsignación fija:")
    print(
        "Cámara L / cámara 1:",
        CAMARA_L_SERIAL,
    )
    print(
        "Cámara R / cámara 2:",
        CAMARA_R_SERIAL,
    )

    print("\nConfiguración:")
    print("Exposición:", EXPOSICION_US, "us")
    print("FPS objetivo:", FPS_OBJETIVO)

    print(
        "Resolución L:",
        cam_l.image_width_pixels,
        "x",
        cam_l.image_height_pixels,
    )

    print(
        "Resolución R:",
        cam_r.image_width_pixels,
        "x",
        cam_r.image_height_pixels,
    )

    print("Rotación L:", ROTACION_L)
    print("Rotación R:", ROTACION_R)

    return sdk, cam_l, cam_r




def esperar_frame(cam):
    """
    Espera un frame y lo convierte a BGR.
    """

    limite = time.perf_counter() + TIMEOUT_FRAME_S

    while time.perf_counter() < limite:
        frame = cam.get_pending_frame_or_null()

        if frame is not None:
            return convertir_a_8bits_bgr(
                frame.image_buffer,
                cam.image_width_pixels,
                cam.image_height_pixels,
            )

        time.sleep(0.0005)

    return None




def limpiar_frames_pendientes(cam):
    """
    Elimina frames antiguos del buffer para reducir latencia.
    Conserva el frame más reciente disponible.
    """

    ultimo = None

    while True:
        frame = cam.get_pending_frame_or_null()

        if frame is None:
            break

        ultimo = frame

    if ultimo is None:
        return None

    return convertir_a_8bits_bgr(
        ultimo.image_buffer,
        cam.image_width_pixels,
        cam.image_height_pixels,
    )




class SelectorPuntos:
    def __init__(self, frame_l, frame_r):
        if frame_l.shape[:2] != frame_r.shape[:2]:
            raise ValueError("Los frames deben tener el mismo tamaño.")

        self.frame_l = frame_l
        self.frame_r = frame_r
        self.h, self.w = frame_l.shape[:2]

        self.puntos_l = []
        self.puntos_r = []
        self.punto_l_pendiente = None

        self.ventana = "Seleccion manual de puntos"
        self.cancelado = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < self.w:
                if self.punto_l_pendiente is None:
                    self.punto_l_pendiente = (float(x), float(y))
                    print(
                        f"Punto izquierdo ({x}, {y}). "
                        "Ahora selecciona el mismo punto en la imagen derecha."
                    )
                else:
                    print(
                        "Ya hay un punto izquierdo pendiente. "
                        "Selecciona ahora la derecha."
                    )
            else:
                if self.punto_l_pendiente is None:
                    print("Primero selecciona el punto en la imagen izquierda.")
                    return

                xr = x - self.w
                punto_r = (float(xr), float(y))

                self.puntos_l.append(self.punto_l_pendiente)
                self.puntos_r.append(punto_r)

                print(
                    f"Par {len(self.puntos_l)}: "
                    f"L={self.punto_l_pendiente}, R={punto_r}"
                )

                self.punto_l_pendiente = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.punto_l_pendiente is not None:
                self.punto_l_pendiente = None
                print("Punto pendiente cancelado.")
            elif self.puntos_l:
                self.puntos_l.pop()
                self.puntos_r.pop()
                print(f"Último par eliminado. Quedan {len(self.puntos_l)}.")

    def dibujar(self):
        izquierda = self.frame_l.copy()
        derecha = self.frame_r.copy()

        for i, (pl, pr) in enumerate(
            zip(self.puntos_l, self.puntos_r),
            start=1,
        ):
            xl, yl = map(int, pl)
            xr, yr = map(int, pr)

            cv2.circle(izquierda, (xl, yl), 6, (0, 255, 0), -1)
            cv2.circle(derecha, (xr, yr), 6, (0, 255, 0), -1)

            put_text_outline(
                izquierda,
                str(i),
                (xl + 8, yl - 8),
                0.55,
                (0, 255, 0),
                1,
            )

            put_text_outline(
                derecha,
                str(i),
                (xr + 8, yr - 8),
                0.55,
                (0, 255, 0),
                1,
            )

        if self.punto_l_pendiente is not None:
            x, y = map(int, self.punto_l_pendiente)
            cv2.circle(izquierda, (x, y), 7, (0, 165, 255), -1)

        put_text_outline(
            izquierda,
            f"IZQUIERDA - pares: {len(self.puntos_l)}",
            (20, 30),
            0.65,
            (255, 255, 0),
        )

        put_text_outline(
            derecha,
            "DERECHA",
            (20, 30),
            0.65,
            (255, 255, 0),
        )

        put_text_outline(
            izquierda,
            (
                "Click L -> mismo punto R | ENTER terminar | "
                "Boton derecho deshacer | C limpiar | Q cancelar"
            ),
            (20, self.h - 20),
            0.42,
            (255, 255, 255),
            1,
        )

        return np.hstack([izquierda, derecha])

    def ejecutar(self):
        cv2.namedWindow(self.ventana, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.ventana, self.mouse_callback)

        print("\n===================================")
        print("SELECCIÓN MANUAL")
        print("===================================")
        print("Selecciona primero en la izquierda y luego en la derecha.")
        print("Usa detalles físicos, no textos ni rectángulos añadidos.")
        print("Usa puntos repartidos por toda la zona física visible.")
        print("ENTER termina, botón derecho deshace, C limpia, Q cancela.")

        while True:
            cv2.imshow(self.ventana, self.dibujar())
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):
                if self.punto_l_pendiente is not None:
                    print("Completa o cancela el punto pendiente.")
                    continue

                if len(self.puntos_l) < MIN_PARES_MANUALES:
                    print(
                        f"Necesitas al menos {MIN_PARES_MANUALES} pares. "
                        f"Actualmente tienes {len(self.puntos_l)}."
                    )
                    continue

                break

            if key in (ord("c"), ord("C")):
                self.puntos_l.clear()
                self.puntos_r.clear()
                self.punto_l_pendiente = None
                print("Puntos eliminados.")

            if key in (ord("q"), ord("Q"), 27):
                self.cancelado = True
                break

            if len(self.puntos_l) >= MAX_PARES_MANUALES:
                print(f"Se alcanzó el máximo de {MAX_PARES_MANUALES} pares.")
                break

        cv2.destroyWindow(self.ventana)

        if self.cancelado:
            raise RuntimeError("Selección manual cancelada.")

        return (
            np.asarray(self.puntos_l, dtype=np.float64),
            np.asarray(self.puntos_r, dtype=np.float64),
        )




def estimar_transformacion_rigida(puntos_l, puntos_r):
    """
    Alinea SOLO el eje vertical de la cámara derecha.

    La disparidad horizontal cambia naturalmente con la profundidad, por lo
    que NO se intenta hacer coincidir X entre ambas cámaras. Intentar ajustar
    X e Y simultáneamente con una transformación rígida provoca pocos inliers.

    Se estima:
      - una rotación global pequeña de la cámara derecha;
      - una traslación vertical;
      - ninguna escala;
      - ninguna perspectiva;
      - ninguna traslación horizontal artificial.

    La forma de los objetos se conserva.
    """

    if len(puntos_l) != len(puntos_r):
        raise ValueError("Cantidad distinta de puntos L y R.")

    if len(puntos_l) < MIN_PARES_MANUALES:
        raise ValueError(
            f"Se requieren al menos {MIN_PARES_MANUALES} pares de puntos."
        )

    puntos_l = np.asarray(puntos_l, dtype=np.float64)
    puntos_r = np.asarray(puntos_r, dtype=np.float64)

    validos = (
        np.isfinite(puntos_l).all(axis=1)
        & np.isfinite(puntos_r).all(axis=1)
    )

    puntos_l = puntos_l[validos]
    puntos_r = puntos_r[validos]

    if len(puntos_l) < MIN_PARES_MANUALES:
        raise RuntimeError("Quedaron muy pocos puntos válidos.")

    # -----------------------------------------------------
    # Búsqueda robusta del ángulo que minimiza únicamente
    # el error vertical. Se ignora el error horizontal,
    # porque ese desplazamiento ES la disparidad estéreo.
    # -----------------------------------------------------

    h_estimado = max(
        float(np.max(puntos_l[:, 1])),
        float(np.max(puntos_r[:, 1])),
        1.0,
    )
    w_estimado = max(
        float(np.max(puntos_l[:, 0])),
        float(np.max(puntos_r[:, 0])),
        1.0,
    )

    centro = np.array(
        [w_estimado / 2.0, h_estimado / 2.0],
        dtype=np.float64,
    )

    mejor = None

    # Primera búsqueda amplia.
    for angulo in np.linspace(-12.0, 12.0, 2401):
        theta = np.radians(angulo)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        xr = puntos_r[:, 0] - centro[0]
        yr = puntos_r[:, 1] - centro[1]

        y_rot = (
            sin_t * xr
            + cos_t * yr
            + centro[1]
        )

        ty = float(
            np.median(
                puntos_l[:, 1] - y_rot
            )
        )

        residuos = (
            puntos_l[:, 1]
            - (y_rot + ty)
        )

        score = float(
            np.median(
                np.abs(residuos)
            )
        )

        if mejor is None or score < mejor["score"]:
            mejor = {
                "angulo": float(angulo),
                "ty": ty,
                "score": score,
                "residuos": residuos,
            }

    # Segunda búsqueda fina alrededor del mejor ángulo.
    angulo_centro = mejor["angulo"]

    for angulo in np.linspace(
        angulo_centro - 0.15,
        angulo_centro + 0.15,
        601,
    ):
        theta = np.radians(angulo)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        xr = puntos_r[:, 0] - centro[0]
        yr = puntos_r[:, 1] - centro[1]

        y_rot = (
            sin_t * xr
            + cos_t * yr
            + centro[1]
        )

        ty = float(
            np.median(
                puntos_l[:, 1] - y_rot
            )
        )

        residuos = (
            puntos_l[:, 1]
            - (y_rot + ty)
        )

        score = float(
            np.median(
                np.abs(residuos)
            )
        )

        if score < mejor["score"]:
            mejor = {
                "angulo": float(angulo),
                "ty": ty,
                "score": score,
                "residuos": residuos,
            }

    residuos = mejor["residuos"]

    # -----------------------------------------------------
    # Rechazo robusto de correspondencias verticales malas.
    # -----------------------------------------------------

    mediana_residuo = float(np.median(residuos))

    mad = float(
        np.median(
            np.abs(
                residuos - mediana_residuo
            )
        )
    )

    sigma_robusta = max(
        1.4826 * mad,
        0.5,
    )

    umbral = max(
        RANSAC_THRESHOLD_PX,
        3.0 * sigma_robusta,
    )

    mascara = (
        np.abs(
            residuos - mediana_residuo
        )
        <= umbral
    )

    if int(np.count_nonzero(mascara)) < MIN_INLIERS:
        # No abortar por la disparidad horizontal: solo revisamos Y.
        # Se conservan los puntos con menor error vertical.
        orden = np.argsort(
            np.abs(
                residuos - mediana_residuo
            )
        )

        mascara = np.zeros(
            len(residuos),
            dtype=bool,
        )

        mascara[
            orden[:min(MIN_INLIERS, len(orden))]
        ] = True

    puntos_l_in = puntos_l[mascara]
    puntos_r_in = puntos_r[mascara]

    # Recalcular ángulo y ty usando solo inliers verticales.
    mejor_final = None

    for angulo in np.linspace(
        mejor["angulo"] - 0.25,
        mejor["angulo"] + 0.25,
        1001,
    ):
        theta = np.radians(angulo)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        xr = puntos_r_in[:, 0] - centro[0]
        yr = puntos_r_in[:, 1] - centro[1]

        y_rot = (
            sin_t * xr
            + cos_t * yr
            + centro[1]
        )

        ty = float(
            np.median(
                puntos_l_in[:, 1] - y_rot
            )
        )

        residuos_in = (
            puntos_l_in[:, 1]
            - (y_rot + ty)
        )

        score = float(
            np.median(
                np.abs(residuos_in)
            )
        )

        if mejor_final is None or score < mejor_final["score"]:
            mejor_final = {
                "angulo": float(angulo),
                "ty": ty,
                "score": score,
                "residuos": residuos_in,
            }

    angulo = mejor_final["angulo"]
    ty = mejor_final["ty"]

    # getRotationMatrix2D usa ángulo positivo antihorario.
    M = cv2.getRotationMatrix2D(
        (float(centro[0]), float(centro[1])),
        angulo,
        1.0,
    ).astype(np.float64)

    # No añadir traslación horizontal: X debe conservar la disparidad.
    M[0, 2] += 0.0
    M[1, 2] += ty

    puntos_r_transformados = cv2.transform(
        puntos_r.reshape(-1, 1, 2).astype(np.float32),
        M.astype(np.float32),
    ).reshape(-1, 2)

    error_vertical_todos = np.abs(
        puntos_l[:, 1]
        - puntos_r_transformados[:, 1]
    )

    error_vertical_inliers = error_vertical_todos[mascara]

    print("\n===================================")
    print("ALINEACIÓN VERTICAL SIN DEFORMACIÓN")
    print("===================================")
    print(
        f"Pares verticalmente válidos: "
        f"{int(np.count_nonzero(mascara))}/{len(puntos_l)}"
    )
    print(
        f"Ángulo aplicado a cámara derecha: "
        f"{angulo:.4f} grados"
    )
    print("Escala aplicada: 1.000000")
    print("Traslación horizontal artificial: 0.000 px")
    print(
        f"Traslación vertical: "
        f"{ty:.3f} px"
    )
    print(
        f"Error vertical mediano (todos): "
        f"{np.median(error_vertical_todos):.3f} px"
    )
    print(
        f"Error vertical mediano (válidos): "
        f"{np.median(error_vertical_inliers):.3f} px"
    )
    print(
        f"Error vertical P95 (válidos): "
        f"{np.percentile(error_vertical_inliers, 95):.3f} px"
    )
    print(
        "NOTA: las diferencias horizontales no se consideran error, "
        "porque corresponden a la disparidad estéreo."
    )

    return (
        M,
        mascara,
        float(np.median(error_vertical_inliers)),
        float(np.percentile(error_vertical_inliers, 95)),
    )


def alinear_par(frame_l, frame_r, M):
    """
    La imagen izquierda queda completamente intacta.
    Solo se rota y traslada la imagen derecha.
    """
    h, w = frame_l.shape[:2]

    alineada_l = frame_l.copy()

    alineada_r = cv2.warpAffine(
        frame_r,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    mascara = np.full((h, w), 255, dtype=np.uint8)

    mascara_r = cv2.warpAffine(
        mascara,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return alineada_l, alineada_r, mascara_r


def calcular_recorte_comun(frame_l, frame_r, M):
    alineada_l, alineada_r, mascara_r = alinear_par(
        frame_l,
        frame_r,
        M,
    )

    kernel = np.ones((9, 9), dtype=np.uint8)
    mascara_segura = cv2.erode(
        mascara_r,
        kernel,
        iterations=1,
    )

    puntos = cv2.findNonZero(mascara_segura)

    if puntos is None:
        raise RuntimeError("No existe una zona común después de alinear.")

    x, y, w, h = cv2.boundingRect(puntos)

    if w < 100 or h < 100:
        raise RuntimeError(
            f"Zona común demasiado pequeña: {w}x{h}."
        )

    print(f"Recorte común: x={x}, y={y}, w={w}, h={h}")

    return (x, y, w, h)


def alinear_y_recortar(frame_l, frame_r, M, crop):
    alineada_l, alineada_r, _ = alinear_par(
        frame_l,
        frame_r,
        M,
    )

    x, y, w, h = crop

    return (
        alineada_l[y:y + h, x:x + w],
        alineada_r[y:y + h, x:x + w],
    )




def seleccionar_roi(img):
    roi = cv2.selectROI(
        "Seleccionar ROI de superficie/espuma",
        img,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow("Seleccionar ROI de superficie/espuma")

    x, y, w, h = map(int, roi)

    if w <= 0 or h <= 0:
        print("ROI no seleccionado.")
        return False

    roi_data.update({
        "seleccionado": True,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
    })

    print(f"ROI: x={x}, y={y}, w={w}, h={h}")
    return True




def crear_stereo_sgbm():
    if NUM_DISPARITIES % 16 != 0:
        raise ValueError("NUM_DISPARITIES debe ser múltiplo de 16.")

    if BLOCK_SIZE % 2 == 0:
        raise ValueError("BLOCK_SIZE debe ser impar.")

    return cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
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


def calcular_disparidad(stereo, img_l, img_r):
    """
    Calcula el mapa de disparidad y elimina correctamente el valor inválido
    que StereoSGBM usa para los píxeles sin correspondencia.

    Con MIN_DISPARITY = -128, StereoSGBM devuelve aproximadamente -129
    para un píxel inválido. El código anterior aplicaba abs(-129)=129 y lo
    interpretaba como una disparidad real; por eso la lectura quedaba fija
    en 129 px y la altura permanecía en 0 mm.
    """

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    gray_l = clahe.apply(gray_l)
    gray_r = clahe.apply(gray_r)

    disp = stereo.compute(
        gray_l,
        gray_r,
    ).astype(np.float32) / 16.0

    # Rango teórico entregado por StereoSGBM.
    limite_inferior = float(MIN_DISPARITY)
    limite_superior = float(
        MIN_DISPARITY + NUM_DISPARITIES
    )

    # Valor de "sin correspondencia" de StereoSGBM:
    # minDisparity - 1. Para -128, es -129.
    valor_invalido_sgbm = float(
        MIN_DISPARITY - 1
    )

    invalidos = (
        ~np.isfinite(disp)
        | (disp <= valor_invalido_sgbm + 0.25)
        | (disp < limite_inferior + MARGEN_LIMITE_DISPARIDAD)
        | (disp > limite_superior - MARGEN_LIMITE_DISPARIDAD)
        | (np.abs(disp) < DISPARIDAD_ABS_MIN)
        | (np.abs(disp) > DISPARIDAD_ABS_MAX)
    )

    disp[invalidos] = np.nan

    return disp


def visualizar_disparidad(disp):
    validos = np.isfinite(disp)

    if not np.any(validos):
        return np.zeros(
            (disp.shape[0], disp.shape[1], 3),
            dtype=np.uint8,
        )

    valores = np.abs(disp[validos])
    minimo = np.percentile(valores, 5)
    maximo = np.percentile(valores, 95)

    normalizada = np.zeros_like(
        disp,
        dtype=np.float32,
    )

    if maximo > minimo:
        abs_disp = np.abs(disp)
        abs_disp = np.clip(abs_disp, minimo, maximo)
        normalizada = (abs_disp - minimo) / (maximo - minimo)

    normalizada = np.nan_to_num(
        normalizada,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    img = (normalizada * 255).astype(np.uint8)

    return cv2.applyColorMap(
        img,
        cv2.COLORMAP_JET,
    )


def obtener_disparidad_roi(disp):
    """
    Obtiene una disparidad representativa de la superficie dentro del ROI.

    No se usa la mediana de todo el ROI porque las burbujas pueden ocupar
    menos del 50 % del área. Se usa un percentil alto de |disparidad| para
    seguir las partes de la superficie que están más cerca de las cámaras.
    """

    if not roi_data["seleccionado"]:
        return None, 0

    x = roi_data["x"]
    y = roi_data["y"]
    w = roi_data["w"]
    h = roi_data["h"]

    roi = disp[y:y + h, x:x + w]

    if roi.size == 0:
        return None, 0

    validos = np.isfinite(roi)
    cantidad = int(np.count_nonzero(validos))

    if cantidad < MIN_PIXELES_VALIDOS:
        return None, cantidad

    valores = np.abs(roi[validos]).astype(np.float32)

    # Eliminar valores extremos residuales mediante percentiles robustos.
    p_bajo = float(np.percentile(valores, 5))
    p_alto = float(np.percentile(valores, 99))

    valores = valores[
        (valores >= p_bajo)
        & (valores <= p_alto)
    ]

    if len(valores) < MIN_PIXELES_VALIDOS:
        return None, len(valores)

    d_superficie = float(
        np.percentile(
            valores,
            PERCENTIL_SUPERFICIE,
        )
    )

    return d_superficie, len(valores)





# =========================================================
# CAPTURA DE UN PAR
# =========================================================

def capturar_par(cam_l, cam_r):
    img_l_raw = limpiar_frames_pendientes(cam_l)
    img_r_raw = limpiar_frames_pendientes(cam_r)

    if img_l_raw is None:
        img_l_raw = esperar_frame(cam_l)

    if img_r_raw is None:
        img_r_raw = esperar_frame(cam_r)

    if img_l_raw is None or img_r_raw is None:
        return None, None

    return (
        aplicar_rotacion(img_l_raw, "L"),
        aplicar_rotacion(img_r_raw, "R"),
    )


# =========================================================
# ALTURA BASADA EN REFERENCIA
# =========================================================

def actualizar_medicion(d_superficie, pixeles_validos):
    estado["ultima_disparidad"] = d_superficie
    estado["ultimos_pixeles_validos"] = pixeles_validos

    if (
        d_superficie is None
        or estado["K_ref"] is None
        or abs(d_superficie) < 1e-9
    ):
        estado["ultima_distancia_mm"] = None
        estado["ultima_altura_mm"] = None
        return

    z_actual = float(
        estado["K_ref"] / d_superficie
    )

    altura = float(
        abs(Z_REF_MM - z_actual)
    )

    historial_altura.append(altura)

    estado["ultima_distancia_mm"] = z_actual
    estado["ultima_altura_mm"] = float(
        np.median(historial_altura)
    )


def fijar_referencia():
    d_actual = estado["ultima_disparidad"]

    if d_actual is None:
        print(
            "\nNo hay disparidad válida para fijar la referencia. "
            "Revisa el ROI y el mapa de disparidad."
        )
        return

    estado["d_ref"] = float(d_actual)
    estado["K_ref"] = float(
        Z_REF_MM * d_actual
    )

    historial_altura.clear()

    estado["ultima_distancia_mm"] = Z_REF_MM
    estado["ultima_altura_mm"] = 0.0

    print("\nReferencia fijada:")
    print(f"d_ref = {estado['d_ref']:.6f} px")
    print(f"Z_ref = {Z_REF_MM:.3f} mm")
    print(f"K_ref = {estado['K_ref']:.6f}")


# =========================================================
# ARCHIVOS DE SALIDA
# =========================================================

def crear_rutas(experimento):
    base_dir = os.getcwd()

    return {
        "video_l": os.path.join(
            base_dir,
            f"{experimento}_cam1.mp4",
        ),
        "video_r": os.path.join(
            base_dir,
            f"{experimento}_cam2.mp4",
        ),
        "video_disp": os.path.join(
            base_dir,
            f"{experimento}_disparidad.mp4",
        ),
        "csv": os.path.join(
            base_dir,
            f"{experimento}_altura_online.csv",
        ),
        "alineacion": os.path.join(
            base_dir,
            f"{experimento}_alineacion_online.npz",
        ),
    }


def abrir_writers(rutas, frame_l):
    out_w, out_h = calcular_tamano_con_ancho(
        frame_l,
        ANCHO_VIDEO_SALIDA,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *VIDEO_CODEC
    )

    writer_l = cv2.VideoWriter(
        rutas["video_l"],
        fourcc,
        FPS_OBJETIVO,
        (out_w, out_h),
    )

    writer_r = cv2.VideoWriter(
        rutas["video_r"],
        fourcc,
        FPS_OBJETIVO,
        (out_w, out_h),
    )

    writer_disp = cv2.VideoWriter(
        rutas["video_disp"],
        fourcc,
        FPS_OBJETIVO,
        (out_w, out_h),
    )

    if not writer_l.isOpened():
        raise RuntimeError("No se pudo crear el video L.")

    if not writer_r.isOpened():
        writer_l.release()
        raise RuntimeError("No se pudo crear el video R.")

    if not writer_disp.isOpened():
        writer_l.release()
        writer_r.release()
        raise RuntimeError(
            "No se pudo crear el video de disparidad."
        )

    print("\nVideos configurados:")
    print("Tamaño:", out_w, "x", out_h)
    print("FPS:", FPS_OBJETIVO)
    print("L:", rutas["video_l"])
    print("R:", rutas["video_r"])
    print("Disparidad:", rutas["video_disp"])

    return (
        writer_l,
        writer_r,
        writer_disp,
        out_w,
        out_h,
    )


# =========================================================
# CONFIGURACIÓN INICIAL
# =========================================================

def configurar_alineacion_y_roi(cam_l, cam_r):
    print("\nCapturando frame para configuración inicial...")

    frame_l, frame_r = capturar_par(
        cam_l,
        cam_r,
    )

    if frame_l is None or frame_r is None:
        raise RuntimeError(
            "No se pudo capturar el par inicial."
        )

    selector = SelectorPuntos(
        frame_l,
        frame_r,
    )

    puntos_l, puntos_r = selector.ejecutar()

    M, mascara, error_mediano, error_p95 = (
        estimar_transformacion_rigida(
            puntos_l,
            puntos_r,
        )
    )

    crop = calcular_recorte_comun(
        frame_l,
        frame_r,
        M,
    )

    alineada_l, alineada_r = alinear_y_recortar(
        frame_l,
        frame_r,
        M,
        crop,
    )

    vista_l = dibujar_lineas_horizontales(
        alineada_l,
        SEPARACION_LINEAS_PX,
    )

    vista_r = dibujar_lineas_horizontales(
        alineada_r,
        SEPARACION_LINEAS_PX,
    )

    vista_l = redimensionar_para_vista(
        vista_l,
        ANCHO_VISTA_CAMARA,
    )

    vista_r = redimensionar_para_vista(
        vista_r,
        ANCHO_VISTA_CAMARA,
    )

    if vista_l.shape[:2] != vista_r.shape[:2]:
        vista_r = cv2.resize(
            vista_r,
            (vista_l.shape[1], vista_l.shape[0]),
        )

    ventana = "Confirmar alineacion vertical"

    cv2.namedWindow(
        ventana,
        cv2.WINDOW_NORMAL,
    )

    cv2.imshow(
        ventana,
        np.hstack([vista_l, vista_r]),
    )

    print("\nA = aceptar")
    print("Q = rechazar")

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("a"), ord("A")):
            break

        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(ventana)
            raise RuntimeError(
                "Alineación rechazada."
            )

    cv2.destroyWindow(ventana)

    seleccionar_roi(alineada_l)

    return (
        M,
        crop,
        puntos_l,
        puntos_r,
        mascara,
        error_mediano,
        error_p95,
    )


# =========================================================
# VISUALIZACIÓN EN VIVO
# =========================================================

def crear_vista_online(
    alineada_l,
    alineada_r,
    fps_real,
):
    vista_l = alineada_l.copy()
    vista_r = alineada_r.copy()

    if roi_data["seleccionado"]:
        x = roi_data["x"]
        y = roi_data["y"]
        w = roi_data["w"]
        h = roi_data["h"]

        cv2.rectangle(
            vista_l,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        cv2.rectangle(
            vista_r,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

    if MOSTRAR_OVERLAY:
        put_text_outline(
            vista_l,
            f"FPS real: {fps_real:.1f}",
            (20, 35),
            0.6,
            (
                (0, 255, 0)
                if fps_real >= 27.0
                else (0, 165, 255)
            ),
        )

        d = estado["ultima_disparidad"]
        z = estado["ultima_distancia_mm"]
        altura = estado["ultima_altura_mm"]

        put_text_outline(
            vista_l,
            (
                f"d superficie: {d:.3f} px"
                if d is not None
                else "d superficie: ---"
            ),
            (20, 70),
            0.6,
        )

        put_text_outline(
            vista_l,
            (
                f"Z: {z:.2f} mm"
                if z is not None
                else "Z: ---"
            ),
            (20, 105),
            0.6,
        )

        put_text_outline(
            vista_l,
            (
                f"Altura: {altura:.2f} mm"
                if altura is not None
                else "Altura: ---"
            ),
            (20, 140),
            0.7,
            (
                (0, 255, 0)
                if altura is not None
                else (0, 165, 255)
            ),
        )

        put_text_outline(
            vista_l,
            "R referencia | G grabar | S detener | P ROI | Q salir",
            (20, 180),
            0.48,
            (220, 220, 220),
        )

        if estado["grabando"]:
            put_text_outline(
                vista_l,
                "REC",
                (20, 225),
                1.0,
                (0, 0, 255),
            )

    if MOSTRAR_LINEAS_HORIZONTALES:
        vista_l = dibujar_lineas_horizontales(
            vista_l,
            SEPARACION_LINEAS_PX,
        )

        vista_r = dibujar_lineas_horizontales(
            vista_r,
            SEPARACION_LINEAS_PX,
        )

    vista_l = redimensionar_para_vista(
        vista_l,
        ANCHO_VISTA_CAMARA,
    )

    vista_r = redimensionar_para_vista(
        vista_r,
        ANCHO_VISTA_CAMARA,
    )

    if vista_l.shape[:2] != vista_r.shape[:2]:
        vista_r = cv2.resize(
            vista_r,
            (vista_l.shape[1], vista_l.shape[0]),
        )

    return np.hstack([
        vista_l,
        vista_r,
    ])


# =========================================================
# BUCLE PRINCIPAL
# =========================================================

def ejecutar_online(
    sdk,
    cam_l,
    cam_r,
    rutas,
    M,
    crop,
):
    stereo = crear_stereo_sgbm()

    writer_l = None
    writer_r = None
    writer_disp = None

    csv_file = None
    csv_writer = None

    ultima_disp_color = None

    cv2.namedWindow(
        "Estereo online",
        cv2.WINDOW_NORMAL,
    )

    cv2.namedWindow(
        "Disparidad online",
        cv2.WINDOW_NORMAL,
    )

    try:
        if GUARDAR_CSV:
            csv_file = open(
                rutas["csv"],
                "w",
                newline="",
                encoding="utf-8",
            )

            csv_writer = csv.writer(
                csv_file
            )

            csv_writer.writerow([
                "frame_global",
                "frame_grabado",
                "timestamp",
                "tiempo_grabacion_s",
                "fps_real",
                "disparidad_superficie_px",
                "pixeles_validos",
                "distancia_mm",
                "altura_mm",
                "d_ref_px",
                "K_ref",
                "z_ref_mm",
            ])

        print("\n===================================")
        print("CONTROLES")
        print("===================================")
        print("R: fijar nivel inicial como referencia")
        print("G: iniciar grabación")
        print("S: detener grabación")
        print("P: seleccionar otro ROI")
        print("Q: salir")

        while estado["activo"]:
            inicio_ciclo = time.perf_counter()

            frame_l, frame_r = capturar_par(
                cam_l,
                cam_r,
            )

            if frame_l is None or frame_r is None:
                print(
                    "No se pudo obtener un par de frames."
                )
                continue

            alineada_l, alineada_r = alinear_y_recortar(
                frame_l,
                frame_r,
                M,
                crop,
            )

            fps_real = calcular_fps_real()

            if (
                estado["frame_global"]
                % PROCESAR_DISPARIDAD_CADA_N_FRAMES
                == 0
            ):
                disp = calcular_disparidad(
                    stereo,
                    alineada_l,
                    alineada_r,
                )

                d_superficie, pixeles_validos = (
                    obtener_disparidad_roi(
                        disp
                    )
                )

                actualizar_medicion(
                    d_superficie,
                    pixeles_validos,
                )

                ultima_disp_color = visualizar_disparidad(
                    disp
                )

            if ultima_disp_color is None:
                ultima_disp_color = np.zeros(
                    alineada_l.shape,
                    dtype=np.uint8,
                )

            if writer_l is None:
                (
                    writer_l,
                    writer_r,
                    writer_disp,
                    out_w,
                    out_h,
                ) = abrir_writers(
                    rutas,
                    frame_l,
                )

            if estado["grabando"]:
                # Guardar los videos orientados pero sin alineación,
                # sin textos ni rectángulos, para reutilizarlos offline.
                video_l = cv2.resize(
                    frame_l,
                    (out_w, out_h),
                    interpolation=cv2.INTER_AREA,
                )

                video_r = cv2.resize(
                    frame_r,
                    (out_w, out_h),
                    interpolation=cv2.INTER_AREA,
                )

                video_disp = cv2.resize(
                    ultima_disp_color,
                    (out_w, out_h),
                    interpolation=cv2.INTER_AREA,
                )

                writer_l.write(video_l)
                writer_r.write(video_r)
                writer_disp.write(video_disp)

                tiempo_grabacion = (
                    time.perf_counter()
                    - estado["t_inicio_grabacion"]
                )

                timestamp = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S.%f"
                )[:-3]

                if csv_writer is not None:
                    csv_writer.writerow([
                        estado["frame_global"],
                        estado["frame_grabado"],
                        timestamp,
                        tiempo_grabacion,
                        fps_real,
                        (
                            ""
                            if estado["ultima_disparidad"] is None
                            else estado["ultima_disparidad"]
                        ),
                        estado["ultimos_pixeles_validos"],
                        (
                            ""
                            if estado["ultima_distancia_mm"] is None
                            else estado["ultima_distancia_mm"]
                        ),
                        (
                            ""
                            if estado["ultima_altura_mm"] is None
                            else estado["ultima_altura_mm"]
                        ),
                        (
                            ""
                            if estado["d_ref"] is None
                            else estado["d_ref"]
                        ),
                        (
                            ""
                            if estado["K_ref"] is None
                            else estado["K_ref"]
                        ),
                        Z_REF_MM,
                    ])

                    csv_file.flush()

                estado["frame_grabado"] += 1

            combined = crear_vista_online(
                alineada_l,
                alineada_r,
                fps_real,
            )

            cv2.imshow(
                "Estereo online",
                combined,
            )

            cv2.imshow(
                "Disparidad online",
                redimensionar_para_vista(
                    ultima_disp_color,
                    720,
                ),
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("r"), ord("R")):
                fijar_referencia()

            elif key in (ord("g"), ord("G")):
                if not estado["grabando"]:
                    estado["grabando"] = True
                    estado["t_inicio_grabacion"] = (
                        time.perf_counter()
                    )
                    estado["frame_grabado"] = 0

                    print(
                        "\nGrabación y registro iniciados."
                    )

            elif key in (ord("s"), ord("S")):
                estado["grabando"] = False
                print("\nGrabación detenida.")

            elif key in (ord("p"), ord("P")):
                seleccionar_roi(alineada_l)

                estado["d_ref"] = None
                estado["K_ref"] = None
                estado["ultima_distancia_mm"] = None
                estado["ultima_altura_mm"] = None
                historial_altura.clear()

                print(
                    "\nROI actualizado. "
                    "Debes volver a presionar R."
                )

            elif key in (ord("q"), ord("Q"), 27):
                estado["activo"] = False
                break

            estado["frame_global"] += 1

            tiempo_usado = (
                time.perf_counter()
                - inicio_ciclo
            )

            if tiempo_usado < FRAME_TIME:
                time.sleep(
                    FRAME_TIME - tiempo_usado
                )

    finally:
        if csv_file is not None:
            csv_file.close()

        if writer_l is not None:
            writer_l.release()

        if writer_r is not None:
            writer_r.release()

        if writer_disp is not None:
            writer_disp.release()

        cv2.destroyAllWindows()

        try:
            cam_l.dispose()
        except Exception:
            pass

        try:
            cam_r.dispose()
        except Exception:
            pass

        try:
            sdk.dispose()
        except Exception:
            pass

        print("\nRecursos liberados correctamente.")


# =========================================================
# MAIN
# =========================================================

def main():
    experimento = input(
        "Nombre de la experiencia "
        "(sin espacios, acentos ni signos): "
    ).strip()

    experimento = experimento.replace(
        " ",
        "_",
    )

    if not experimento:
        experimento = "prueba_estereo_online"

    rutas = crear_rutas(experimento)

    sdk = None
    cam_l = None
    cam_r = None

    try:
        sdk, cam_l, cam_r = inicializar_camaras()

        # Igual que en tu código online definitivo:
        # un disparo inicial comienza la adquisición.
        cam_l.issue_software_trigger()
        cam_r.issue_software_trigger()

        (
            M,
            crop,
            puntos_l,
            puntos_r,
            mascara,
            error_mediano,
            error_p95,
        ) = configurar_alineacion_y_roi(
            cam_l,
            cam_r,
        )

        np.savez(
            rutas["alineacion"],
            M=M,
            crop=np.asarray(
                crop,
                dtype=np.int32,
            ),
            puntos_l=puntos_l,
            puntos_r=puntos_r,
            inliers=mascara.astype(np.uint8),
            error_vertical_mediano_px=error_mediano,
            error_vertical_p95_px=error_p95,
            rotacion_l=np.asarray(int(ROTACION_L)),
            rotacion_r=np.asarray(int(ROTACION_R)),
            z_ref_mm=np.asarray(
                Z_REF_MM,
                dtype=np.float64,
            ),
        )

        print("\nAlineación guardada:")
        print(rutas["alineacion"])

        ejecutar_online(
            sdk,
            cam_l,
            cam_r,
            rutas,
            M,
            crop,
        )

    except Exception:
        if cam_l is not None:
            try:
                cam_l.dispose()
            except Exception:
                pass

        if cam_r is not None:
            try:
                cam_r.dispose()
            except Exception:
                pass

        if sdk is not None:
            try:
                sdk.dispose()
            except Exception:
                pass

        cv2.destroyAllWindows()
        raise


if __name__ == "__main__":
    main()