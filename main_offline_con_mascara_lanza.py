import os
import time
import csv
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURACIÓN
# =========================================================

VIDEO_L = r"D:\MDT\Stereovision\II38P_cam1.mp4"
VIDEO_R = r"D:\MDT\Stereovision\II38P_cam2.mp4"

CARPETA_SALIDA = r"D:\MDT\Stereovision\resultados_II38P"
NOMBRE_EXPERIENCIA = "II38P"

# Frame usado para seleccionar puntos correspondientes.
FRAME_ALINEACION = 500

# Frame usado exclusivamente para seleccionar la máscara de la lanza.
# Debe ser un frame donde la lanza sea claramente visible.
FRAME_LANZA = 15000

# =========================================================
# ORIENTACIÓN DE LOS VIDEOS
# =========================================================
#
# El código ONLINE definitivo ya guarda los videos orientados así:
#   cámara L (36933): 90° clockwise
#   cámara R (36930): 90° counterclockwise
#
# Los videos que vas a procesar ahora fueron grabados con el código
# ONLINE que ya aplicaba las rotaciones antes de guardarlos. Por eso
# el offline NO debe volver a rotarlos.
VIDEOS_YA_ROTADOS_ONLINE = True
#
# Para videos antiguos/crudos que NO hayan sido rotados durante la captura,
# cambia esta variable a False. En ese caso se aplicarán las rotaciones
# reales del montaje:
#   L -> 90° clockwise
#   R -> 90° counterclockwise
#
if VIDEOS_YA_ROTADOS_ONLINE:
    ROTACION_L = None
    ROTACION_R = None
else:
    ROTACION_L = cv2.ROTATE_90_CLOCKWISE
    ROTACION_R = cv2.ROTATE_90_COUNTERCLOCKWISE

FRAME_INICIAL = 0

# =========================================================
# MODO DE REPRODUCCIÓN
# =========================================================
#
# True:
#   El procesamiento intenta durar lo mismo que el video.
#   Si el cálculo no alcanza 30 FPS, omite automáticamente frames.
#
# False:
#   Procesa todos los frames, aunque tarde más que la duración del video.
MODO_TIEMPO_REAL = True

# Salto mínimo manual. En modo tiempo real el programa puede saltar
# más frames automáticamente cuando el procesamiento va atrasado.
SALTO_FRAMES = 1

# Distancia conocida del plano de referencia a las cámaras.
Z_REF_MM = 900.0

# Altura física inicial de la espuma cuando se presiona R.
# 1.2 cm = 12 mm.
ALTURA_REFERENCIA_MM = 12.0

# Selección manual de puntos.
MIN_PARES_MANUALES = 8
MAX_PARES_MANUALES = 40

# Rechazo de puntos atípicos.
RANSAC_THRESHOLD_PX = 4.0
MIN_INLIERS = 6

# Si es False, la transformación conserva exactamente la escala.
# Esto evita que el crisol se estire o encoja.
PERMITIR_ESCALA = False

# Disparidad.
MIN_DISPARITY = -128
NUM_DISPARITIES = 256
BLOCK_SIZE = 7

DISPARIDAD_ABS_MIN = 1.0
DISPARIDAD_ABS_MAX = 500.0
MIN_PIXELES_VALIDOS = 30

# La espuma ocupa solo una parte del ROI. La mediana de todo el ROI
# puede quedarse inmóvil. Se usa un percentil alto de la disparidad
# absoluta para seguir la superficie más cercana a las cámaras.
PERCENTIL_SUPERFICIE = 75.0

# Margen para descartar valores pegados a los límites de StereoSGBM.
MARGEN_LIMITE_DISPARIDAD = 1.0

# =========================================================
# MEJORA PARA VIDEOS YA GRABADOS CON EXCESO DE LUZ
# =========================================================
#
# Estas opciones no recuperan píxeles completamente saturados,
# pero mejoran las zonas que todavía conservan información.
MEJORAR_VIDEO_GRABADO = True

# Gamma > 1 oscurece las zonas brillantes.
GAMMA_CORRECCION = 1.5

# CLAHE moderado para recuperar contraste local.
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID = (8, 8)

# Descartar píxeles casi negros o casi blancos.
INTENSIDAD_MIN_VALIDA = 2
INTENSIDAD_MAX_VALIDA = 253

# Se conserva por compatibilidad, pero el código ya no rechaza el ROI
# completo por porcentaje inválido; usa MIN_PIXELES_VALIDOS.
FRACCION_MAX_SATURADA_ROI = 0.98

# Visualización y guardado.
MOSTRAR_LINEAS_EPIPOLARES = True
SEPARACION_LINEAS = 50
GUARDAR_VIDEOS = False
VIDEO_CODEC = "mp4v"
VENTANA_MEDIANA = 5


# =========================================================
# ESTADO
# =========================================================

roi_data = {
    "seleccionado": False,
    "x": 0,
    "y": 0,
    "w": 0,
    "h": 0,
}

# Zona de exclusión para la lanza.
# Esta región puede estar dentro del ROI, pero no participa
# en el cálculo de disparidad ni altura.
lanza_data = {
    "seleccionado": False,
    "x": 0,
    "y": 0,
    "w": 0,
    "h": 0,
}

estado = {
    "pausado": True,
    "guardando": False,
    "d_ref": None,
    "K_ref": None,
    "pixeles_validos_roi": 0,
    "fraccion_valida_roi": 0.0,
    "motivo_disparidad_invalida": "",
}

historial_altura = []

# Últimas disparidades válidas. Al presionar R se usa la mediana,
# evitando depender de un único frame ruidoso.
historial_disparidad_valida = []
MAX_HISTORIAL_DISPARIDAD = 15
tiempos_resultados = []
alturas_resultados = []


# =========================================================
# UTILIDADES
# =========================================================

def aplicar_rotacion(img, lado):
    """
    Aplica la rotación correspondiente a cada cámara.

    lado:
        "L" para cámara izquierda
        "R" para cámara derecha
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


def abrir_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video:\n{path}")
    return cap


def leer_frame(cap, indice, lado):
    """
    Lee un frame particular y aplica la rotación de su cámara.
    """

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        indice,
    )

    ok, frame = cap.read()

    if not ok:
        return None

    return aplicar_rotacion(
        frame,
        lado,
    )


def put_text_outline(
    img,
    text,
    org,
    scale=0.6,
    color=(255, 255, 255),
    thickness=2,
):
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


def redimensionar_para_pantalla(img, max_w=720, max_h=720):
    h, w = img.shape[:2]
    escala = min(max_w / float(w), max_h / float(h), 1.0)

    nuevo_w = max(2, int(round(w * escala)))
    nuevo_h = max(2, int(round(h * escala)))

    return cv2.resize(
        img,
        (nuevo_w, nuevo_h),
        interpolation=cv2.INTER_AREA,
    )


def dibujar_lineas(img, separacion=50):
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


# =========================================================
# SELECCIÓN MANUAL DE PUNTOS
# =========================================================

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


# =========================================================
# ALINEACIÓN VERTICAL SIN DEFORMACIÓN PROYECTIVA
# =========================================================

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



# =========================================================
# RANGO DE DISPARIDAD AUTOMÁTICO
# =========================================================

def configurar_rango_disparidad_automatico(
    puntos_l,
    puntos_r,
    M,
):
    """
    Estima automáticamente el signo y rango de disparidad usando los
    puntos manuales ya seleccionados.

    Esto evita usar un rango fijo equivocado, como 0...703 px, cuando
    la disparidad real puede ser negativa.
    """

    global MIN_DISPARITY
    global NUM_DISPARITIES
    global DISPARIDAD_ABS_MAX

    puntos_l = np.asarray(
        puntos_l,
        dtype=np.float32,
    )

    puntos_r = np.asarray(
        puntos_r,
        dtype=np.float32,
    )

    puntos_r_alineados = cv2.transform(
        puntos_r.reshape(-1, 1, 2),
        M.astype(np.float32),
    ).reshape(-1, 2)

    disparidades = (
        puntos_l[:, 0]
        - puntos_r_alineados[:, 0]
    )

    disparidades = disparidades[
        np.isfinite(disparidades)
    ]

    if len(disparidades) < 4:
        raise RuntimeError(
            "No hay suficientes puntos para estimar "
            "el rango de disparidad."
        )

    mediana = float(
        np.median(disparidades)
    )

    mad = float(
        np.median(
            np.abs(
                disparidades - mediana
            )
        )
    )

    dispersion = max(
        1.4826 * mad,
        8.0,
    )

    # Margen amplio para cubrir variaciones de profundidad de la espuma.
    margen = max(
        96.0,
        6.0 * dispersion,
    )

    limite_inferior = mediana - margen
    limite_superior = mediana + margen

    min_disp = int(
        math.floor(
            limite_inferior / 16.0
        ) * 16
    )

    max_disp = int(
        math.ceil(
            limite_superior / 16.0
        ) * 16
    )

    num_disp = max_disp - min_disp

    num_disp = max(
        128,
        int(
            math.ceil(
                num_disp / 16.0
            ) * 16
        ),
    )

    # Evita un rango excesivo que vuelva el cálculo innecesariamente lento.
    num_disp = min(
        num_disp,
        512,
    )

    MIN_DISPARITY = min_disp
    NUM_DISPARITIES = num_disp

    DISPARIDAD_ABS_MAX = float(
        max(
            abs(MIN_DISPARITY),
            abs(
                MIN_DISPARITY
                + NUM_DISPARITIES
            ),
        )
        + 16.0
    )

    print("\n===================================")
    print("RANGO DE DISPARIDAD AUTOMÁTICO")
    print("===================================")
    print(
        "Disparidades manuales:",
        np.round(
            disparidades,
            2,
        ),
    )
    print(
        f"Mediana estimada: {mediana:.3f} px"
    )
    print(
        f"MIN_DISPARITY: {MIN_DISPARITY}"
    )
    print(
        f"NUM_DISPARITIES: {NUM_DISPARITIES}"
    )
    print(
        "Rango SGBM:",
        MIN_DISPARITY,
        "a",
        MIN_DISPARITY
        + NUM_DISPARITIES
        - 1,
        "px",
    )

    return (
        MIN_DISPARITY,
        NUM_DISPARITIES,
        disparidades,
    )


# =========================================================
# ROI
# =========================================================

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


def seleccionar_mascara_lanza(img):
    """
    Permite seleccionar una zona rectangular que contiene la lanza.

    Esa zona se excluirá completamente del cálculo de disparidad
    dentro del ROI.
    """

    mascara = cv2.selectROI(
        "Seleccionar zona de la lanza",
        img,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow("Seleccionar zona de la lanza")

    x, y, w, h = map(int, mascara)

    if w <= 0 or h <= 0:
        lanza_data.update({
            "seleccionado": False,
            "x": 0,
            "y": 0,
            "w": 0,
            "h": 0,
        })

        print(
            "No se seleccionó máscara de lanza. "
            "La lanza no será excluida."
        )
        return False

    lanza_data.update({
        "seleccionado": True,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
    })

    print(
        f"Máscara de lanza: x={x}, y={y}, w={w}, h={h}"
    )

    return True


# =========================================================
# PREPROCESAMIENTO DE INTENSIDAD
# =========================================================

def aplicar_gamma_gray(gray, gamma):
    """
    Corrige brillo en una imagen de 8 bits.

    gamma > 1 oscurece regiones brillantes.
    gamma < 1 aclara regiones oscuras.
    """

    if gamma <= 0:
        raise ValueError("gamma debe ser mayor que cero.")

    tabla = np.array(
        [
            ((i / 255.0) ** gamma) * 255.0
            for i in range(256)
        ],
        dtype=np.float32,
    )

    tabla = np.clip(
        tabla,
        0,
        255,
    ).astype(np.uint8)

    return cv2.LUT(
        gray,
        tabla,
    )


def mejorar_gray_para_estereo(gray):
    """
    Mejora contraste sin inventar textura.

    1. Comprime las altas luces con gamma.
    2. Aplica CLAHE moderado.
    """

    if not MEJORAR_VIDEO_GRABADO:
        return gray.copy()

    corregida = aplicar_gamma_gray(
        gray,
        GAMMA_CORRECCION,
    )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )

    return clahe.apply(
        corregida
    )


def calcular_mascara_intensidad_valida(
    gray_l_original,
    gray_r_original,
):
    """
    Conserva solo píxeles con intensidad útil en ambas cámaras.
    """

    return (
        (gray_l_original >= INTENSIDAD_MIN_VALIDA)
        & (gray_l_original <= INTENSIDAD_MAX_VALIDA)
        & (gray_r_original >= INTENSIDAD_MIN_VALIDA)
        & (gray_r_original <= INTENSIDAD_MAX_VALIDA)
    )


# =========================================================
# DISPARIDAD
# =========================================================

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

    El valor inválido de StereoSGBM es minDisparity - 1.

    En esta versión el rango se amplía a 0...703 px porque la lectura de
    127.000 px observada anteriormente correspondía al límite superior
    artificial del rango -128...127, no a la superficie real.
    """

    gray_l_original = cv2.cvtColor(
        img_l,
        cv2.COLOR_BGR2GRAY,
    )

    gray_r_original = cv2.cvtColor(
        img_r,
        cv2.COLOR_BGR2GRAY,
    )

    mascara_intensidad = calcular_mascara_intensidad_valida(
        gray_l_original,
        gray_r_original,
    )

    gray_l = mejorar_gray_para_estereo(
        gray_l_original
    )

    gray_r = mejorar_gray_para_estereo(
        gray_r_original
    )

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
        | (~mascara_intensidad)
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

    La lanza se excluye mediante su máscara. Los píxeles saturados, oscuros
    o sin correspondencia quedan como NaN. No se rechaza todo el ROI solo
    porque tenga muchas zonas inválidas: basta con que sobreviva una cantidad
    mínima de píxeles útiles.
    """

    estado["motivo_disparidad_invalida"] = ""
    estado["pixeles_validos_roi"] = 0
    estado["fraccion_valida_roi"] = 0.0

    if not roi_data["seleccionado"]:
        estado["motivo_disparidad_invalida"] = "ROI no seleccionado"
        return None, 0

    x = roi_data["x"]
    y = roi_data["y"]
    w = roi_data["w"]
    h = roi_data["h"]

    roi = disp[y:y + h, x:x + w].copy()

    if roi.size == 0:
        estado["motivo_disparidad_invalida"] = "ROI vacío"
        return None, 0

    # Excluir la lanza aunque esté dentro del ROI principal.
    if lanza_data["seleccionado"]:
        lx = lanza_data["x"] - x
        ly = lanza_data["y"] - y
        lw = lanza_data["w"]
        lh = lanza_data["h"]

        x1 = max(0, lx)
        y1 = max(0, ly)
        x2 = min(w, lx + lw)
        y2 = min(h, ly + lh)

        if x2 > x1 and y2 > y1:
            roi[y1:y2, x1:x2] = np.nan

    validos = np.isfinite(roi)
    cantidad = int(np.count_nonzero(validos))

    estado["pixeles_validos_roi"] = cantidad
    estado["fraccion_valida_roi"] = float(
        cantidad / float(roi.size)
    )

    if cantidad < MIN_PIXELES_VALIDOS:
        estado["motivo_disparidad_invalida"] = (
            f"solo {cantidad} píxeles válidos; "
            f"se requieren {MIN_PIXELES_VALIDOS}"
        )
        return None, cantidad

    valores = np.abs(
        roi[validos]
    ).astype(np.float32)

    # Eliminar solamente extremos residuales.
    if len(valores) >= 100:
        p_bajo = float(np.percentile(valores, 2))
        p_alto = float(np.percentile(valores, 99.5))

        valores = valores[
            (valores >= p_bajo)
            & (valores <= p_alto)
        ]

    if len(valores) < MIN_PIXELES_VALIDOS:
        estado["motivo_disparidad_invalida"] = (
            f"quedaron {len(valores)} píxeles tras filtrar"
        )
        return None, len(valores)

    d_superficie = float(
        np.percentile(
            valores,
            PERCENTIL_SUPERFICIE,
        )
    )

    if not np.isfinite(d_superficie):
        estado["motivo_disparidad_invalida"] = (
            "percentil de disparidad no finito"
        )
        return None, len(valores)

    return d_superficie, len(valores)


# =========================================================
# DISTANCIA Y ALTURA
# =========================================================

def disparidad_a_distancia(d):
    if d is None or estado["K_ref"] is None:
        return None

    if abs(d) < 1e-9:
        return None

    return float(estado["K_ref"] / d)


def calcular_altura(z_actual):
    """
    Calcula la altura absoluta de espuma.

    Al presionar R, la superficie observada corresponde a una altura
    física inicial de ALTURA_REFERENCIA_MM = 12 mm.

    Si la superficie se acerca a las cámaras, z_actual disminuye y la
    altura aumenta. Si se aleja, la altura disminuye.
    """

    if z_actual is None:
        return None

    cambio_altura_mm = float(
        Z_REF_MM - z_actual
    )

    altura_absoluta_mm = float(
        ALTURA_REFERENCIA_MM
        + cambio_altura_mm
    )

    # Evita alturas físicas negativas por ruido extremo.
    return max(
        0.0,
        altura_absoluta_mm,
    )


def suavizar_altura(valor):
    if valor is None:
        return None

    historial_altura.append(float(valor))

    if len(historial_altura) > VENTANA_MEDIANA:
        historial_altura.pop(0)

    return float(np.median(historial_altura))


# =========================================================
# GRÁFICO
# =========================================================

def guardar_grafico(ruta):
    if not alturas_resultados:
        print("No hay datos para crear el gráfico.")
        return

    fig, ax = plt.subplots()

    ax.plot(
        tiempos_resultados,
        alturas_resultados,
        label="Altura estéreo",
    )

    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Altura absoluta de espuma [mm]")
    ax.set_title("Altura absoluta de espuma por estereovisión")
    ax.grid(True)
    ax.legend()

    fig.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(
        CARPETA_SALIDA,
        exist_ok=True,
    )

    ruta_npz = os.path.join(
        CARPETA_SALIDA,
        f"{NOMBRE_EXPERIENCIA}_alineacion_rigida.npz",
    )

    ruta_csv = os.path.join(
        CARPETA_SALIDA,
        f"{NOMBRE_EXPERIENCIA}_altura.csv",
    )

    ruta_video_l = os.path.join(
        CARPETA_SALIDA,
        f"{NOMBRE_EXPERIENCIA}_left.mp4",
    )

    ruta_video_r = os.path.join(
        CARPETA_SALIDA,
        f"{NOMBRE_EXPERIENCIA}_right_alineado.mp4",
    )

    ruta_video_disp = os.path.join(
        CARPETA_SALIDA,
        f"{NOMBRE_EXPERIENCIA}_disparidad.mp4",
    )

    ruta_grafico = os.path.join(
        CARPETA_SALIDA,
        f"{NOMBRE_EXPERIENCIA}_grafico.png",
    )

    cap_l = abrir_video(VIDEO_L)
    cap_r = abrir_video(VIDEO_R)

    frames_l = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(frames_l, frames_r)

    fps_l = float(cap_l.get(cv2.CAP_PROP_FPS))
    fps_r = float(cap_r.get(cv2.CAP_PROP_FPS))

    fps = fps_l if fps_l > 0 else fps_r

    if fps <= 0:
        # Los videos del código online definitivo se guardan a 30 FPS.
        fps = 30.0

    print("\n===================================")
    print("VIDEOS")
    print("===================================")
    print("Video izquierdo:", VIDEO_L)
    print("Video derecho:", VIDEO_R)
    print("Frames izquierda:", frames_l)
    print("Frames derecha:", frames_r)
    print("FPS izquierda:", fps_l)
    print("FPS derecha:", fps_r)
    print("Videos ya rotados por el código online:", VIDEOS_YA_ROTADOS_ONLINE)
    print("Rotación adicional cámara izquierda:", ROTACION_L)
    print("Rotación adicional cámara derecha:", ROTACION_R)
    print("Frame para seleccionar la lanza:", FRAME_LANZA)
    print("Mejora para videos brillantes:", MEJORAR_VIDEO_GRABADO)
    print("Gamma:", GAMMA_CORRECCION)
    print(
        "Rango de intensidad válido:",
        INTENSIDAD_MIN_VALIDA,
        "-",
        INTENSIDAD_MAX_VALIDA,
    )
    print(
        "Rango de disparidad SGBM:",
        MIN_DISPARITY,
        "a",
        MIN_DISPARITY + NUM_DISPARITIES - 1,
        "px",
    )
    print(
        "Percentil de superficie:",
        PERCENTIL_SUPERFICIE,
    )

    frame_alin_l = leer_frame(
        cap_l,
        FRAME_ALINEACION,
        "L",
    )

    frame_alin_r = leer_frame(
        cap_r,
        FRAME_ALINEACION,
        "R",
    )

    if frame_alin_l is None or frame_alin_r is None:
        raise RuntimeError("No se pudo leer FRAME_ALINEACION.")

    selector = SelectorPuntos(
        frame_alin_l,
        frame_alin_r,
    )

    puntos_l, puntos_r = selector.ejecutar()

    M, mascara_inliers, error_mediano, error_p95 = (
        estimar_transformacion_rigida(
            puntos_l,
            puntos_r,
        )
    )

    configurar_rango_disparidad_automatico(
        puntos_l,
        puntos_r,
        M,
    )

    crop = calcular_recorte_comun(
        frame_alin_l,
        frame_alin_r,
        M,
    )

    prueba_l, prueba_r = alinear_y_recortar(
        frame_alin_l,
        frame_alin_r,
        M,
        crop,
    )

    vista_l = dibujar_lineas(
        prueba_l,
        SEPARACION_LINEAS,
    )

    vista_r = dibujar_lineas(
        prueba_r,
        SEPARACION_LINEAS,
    )

    vista_l = redimensionar_para_pantalla(vista_l)
    vista_r = redimensionar_para_pantalla(vista_r)

    if vista_l.shape[:2] != vista_r.shape[:2]:
        vista_r = cv2.resize(
            vista_r,
            (vista_l.shape[1], vista_l.shape[0]),
        )

    cv2.namedWindow(
        "Confirmar alineacion vertical",
        cv2.WINDOW_NORMAL,
    )

    cv2.imshow(
        "Confirmar alineacion vertical",
        np.hstack([vista_l, vista_r]),
    )

    print("\n===================================")
    print("CONFIRMACIÓN")
    print("===================================")
    print("La imagen izquierda debe quedar intacta.")
    print("La derecha solo debe rotar y desplazarse verticalmente.")
    print("A = aceptar")
    print("Q = rechazar")

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("a"), ord("A")):
            break

        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow("Confirmar alineacion vertical")
            raise RuntimeError(
                "Alineación vertical rechazada. "
                "Vuelve a seleccionar puntos o cambia FRAME_ALINEACION."
            )

    cv2.destroyWindow("Confirmar alineacion vertical")

    # =====================================================
    # SELECCIÓN DE LA LANZA EN UN FRAME DONDE ES VISIBLE
    # =====================================================

    if FRAME_LANZA < 0 or FRAME_LANZA >= total_frames:
        raise ValueError(
            f"FRAME_LANZA={FRAME_LANZA} está fuera del video. "
            f"El rango válido es 0 a {total_frames - 1}."
        )

    frame_lanza_l = leer_frame(
        cap_l,
        FRAME_LANZA,
        "L",
    )

    frame_lanza_r = leer_frame(
        cap_r,
        FRAME_LANZA,
        "R",
    )

    if frame_lanza_l is None or frame_lanza_r is None:
        raise RuntimeError(
            f"No se pudo leer FRAME_LANZA={FRAME_LANZA}."
        )

    lanza_l_alineada, lanza_r_alineada = alinear_y_recortar(
        frame_lanza_l,
        frame_lanza_r,
        M,
        crop,
    )

    print("\n===================================")
    print("SELECCIÓN DE LA LANZA")
    print("===================================")
    print(
        f"Se utilizará el frame {FRAME_LANZA}, "
        "donde la lanza debe ser visible."
    )
    print(
        "Selecciona un rectángulo que cubra completamente la lanza "
        "y deja un pequeño margen alrededor."
    )

    seleccionar_mascara_lanza(
        lanza_l_alineada
    )

    np.savez(
        ruta_npz,
        M=M,
        crop=np.asarray(crop, dtype=np.int32),
        puntos_l=puntos_l,
        puntos_r=puntos_r,
        inliers=mascara_inliers.astype(np.uint8),
        error_vertical_mediano_px=error_mediano,
        error_vertical_p95_px=error_p95,
        permitir_escala=np.asarray(PERMITIR_ESCALA),
        videos_ya_rotados_online=np.asarray(VIDEOS_YA_ROTADOS_ONLINE),
        rotacion_l=np.asarray(
            -1 if ROTACION_L is None else int(ROTACION_L)
        ),
        rotacion_r=np.asarray(
            -1 if ROTACION_R is None else int(ROTACION_R)
        ),
        z_ref_mm=np.asarray(
            Z_REF_MM,
            dtype=np.float64,
        ),
        altura_referencia_mm=np.asarray(
            ALTURA_REFERENCIA_MM,
            dtype=np.float64,
        ),
        lanza_seleccionada=np.asarray(
            lanza_data["seleccionado"]
        ),
        lanza_rect=np.asarray(
            [
                lanza_data["x"],
                lanza_data["y"],
                lanza_data["w"],
                lanza_data["h"],
            ],
            dtype=np.int32,
        ),
        frame_lanza=np.asarray(
            FRAME_LANZA,
            dtype=np.int32,
        ),
        min_disparity=np.asarray(
            MIN_DISPARITY,
            dtype=np.int32,
        ),
        num_disparities=np.asarray(
            NUM_DISPARITIES,
            dtype=np.int32,
        ),
    )

    print("\nAlineación guardada:")
    print(ruta_npz)

    cap_l.set(
        cv2.CAP_PROP_POS_FRAMES,
        FRAME_INICIAL,
    )

    cap_r.set(
        cv2.CAP_PROP_POS_FRAMES,
        FRAME_INICIAL,
    )

    stereo = crear_stereo_sgbm()

    _, _, ancho_salida, alto_salida = crop

    fourcc = cv2.VideoWriter_fourcc(
        *VIDEO_CODEC
    )

    fps_salida = max(
        0.1,
        fps / SALTO_FRAMES,
    )

    writer_l = writer_r = writer_disp = None

    if GUARDAR_VIDEOS:
        writer_l = cv2.VideoWriter(
            ruta_video_l,
            fourcc,
            fps_salida,
            (ancho_salida, alto_salida),
        )

        writer_r = cv2.VideoWriter(
            ruta_video_r,
            fourcc,
            fps_salida,
            (ancho_salida, alto_salida),
        )

        writer_disp = cv2.VideoWriter(
            ruta_video_disp,
            fourcc,
            fps_salida,
            (ancho_salida, alto_salida),
        )

        if not writer_l.isOpened():
            raise RuntimeError("No se pudo crear el video izquierdo.")

        if not writer_r.isOpened():
            raise RuntimeError("No se pudo crear el video derecho.")

        if not writer_disp.isOpened():
            raise RuntimeError("No se pudo crear el video de disparidad.")

    csv_file = open(
        ruta_csv,
        "w",
        newline="",
        encoding="utf-8",
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "frame",
        "tiempo_s",
        "disparidad_superficie_px",
        "pixeles_validos",
        "distancia_mm",
        "altura_mm",
        "d_ref_px",
        "K_ref",
        "altura_referencia_mm",
    ])

    frame_actual = FRAME_INICIAL
    frame_l = None
    frame_r = None

    # Referencias temporales para sincronizar el procesamiento con
    # la duración real del video.
    tiempo_base_reproduccion = None
    frame_base_reproduccion = FRAME_INICIAL

    cv2.namedWindow(
        "Estereo offline",
        cv2.WINDOW_NORMAL,
    )

    cv2.namedWindow(
        "Disparidad",
        cv2.WINDOW_NORMAL,
    )

    print("\n===================================")
    print("CONTROLES")
    print("===================================")
    print("ESPACIO: pausar/reanudar")
    print("R: fijar referencia inicial")
    print("G: comenzar guardado")
    print("S: detener guardado")
    print("P: seleccionar otro ROI")
    print("L: seleccionar/actualizar máscara de lanza")
    print("Q: salir")
    print("Modo tiempo real:", MODO_TIEMPO_REAL)
    if MODO_TIEMPO_REAL:
        print(
            "El programa omitirá frames automáticamente "
            "si el cálculo no alcanza 30 FPS."
        )

    try:
        while frame_actual < total_frames:
            if (
                not estado["pausado"]
                or frame_l is None
                or frame_r is None
            ):
                ok_l, nuevo_l = cap_l.read()
                ok_r, nuevo_r = cap_r.read()

                if not ok_l or not ok_r:
                    print("Fin de los videos.")
                    break

                frame_l = aplicar_rotacion(nuevo_l, "L")
                frame_r = aplicar_rotacion(nuevo_r, "R")

            alineada_l, alineada_r = alinear_y_recortar(
                frame_l,
                frame_r,
                M,
                crop,
            )

            if not roi_data["seleccionado"]:
                estado["pausado"] = True
                seleccionar_roi(alineada_l)

            disp = calcular_disparidad(
                stereo,
                alineada_l,
                alineada_r,
            )

            d_superficie, pixeles_validos = obtener_disparidad_roi(
                disp
            )

            if d_superficie is not None:
                historial_disparidad_valida.append(
                    float(d_superficie)
                )

                if (
                    len(historial_disparidad_valida)
                    > MAX_HISTORIAL_DISPARIDAD
                ):
                    historial_disparidad_valida.pop(0)

            z_actual = disparidad_a_distancia(
                d_superficie
            )

            altura = calcular_altura(
                z_actual
            )

            altura_suave = suavizar_altura(
                altura
            )

            display_l = alineada_l.copy()
            display_r = alineada_r.copy()

            x = roi_data["x"]
            y = roi_data["y"]
            rw = roi_data["w"]
            rh = roi_data["h"]

            cv2.rectangle(
                display_l,
                (x, y),
                (x + rw, y + rh),
                (0, 255, 0),
                2,
            )

            cv2.rectangle(
                display_r,
                (x, y),
                (x + rw, y + rh),
                (0, 255, 0),
                2,
            )

            if lanza_data["seleccionado"]:
                lx = lanza_data["x"]
                ly = lanza_data["y"]
                lw = lanza_data["w"]
                lh = lanza_data["h"]

                cv2.rectangle(
                    display_l,
                    (lx, ly),
                    (lx + lw, ly + lh),
                    (0, 0, 255),
                    2,
                )

                cv2.rectangle(
                    display_r,
                    (lx, ly),
                    (lx + lw, ly + lh),
                    (0, 0, 255),
                    2,
                )

                put_text_outline(
                    display_l,
                    "LANZA EXCLUIDA",
                    (lx, max(20, ly - 10)),
                    0.45,
                    (0, 0, 255),
                    1,
                )

            tiempo_s = frame_actual / fps

            put_text_outline(
                display_l,
                f"Frame: {frame_actual}/{total_frames}",
                (20, 35),
                0.6,
            )

            put_text_outline(
                display_l,
                f"Tiempo: {tiempo_s:.2f} s",
                (20, 70),
                0.6,
            )

            put_text_outline(
                display_l,
                (
                    f"d superficie: {d_superficie:.3f} px"
                    if d_superficie is not None
                    else "d superficie: ---"
                ),
                (20, 105),
                0.6,
            )

            put_text_outline(
                display_l,
                (
                    f"Z: {z_actual:.2f} mm"
                    if z_actual is not None
                    else "Z: ---"
                ),
                (20, 140),
                0.6,
            )

            put_text_outline(
                display_l,
                (
                    f"Altura: {altura_suave:.2f} mm"
                    if altura_suave is not None
                    else "Altura: ---"
                ),
                (20, 175),
                0.65,
                (
                    (0, 255, 0)
                    if altura_suave is not None
                    else (0, 165, 255)
                ),
            )

            if (
                d_superficie is not None
                and d_superficie
                >= MIN_DISPARITY + NUM_DISPARITIES
                - 2.0 * MARGEN_LIMITE_DISPARIDAD
            ):
                put_text_outline(
                    display_l,
                    "ADVERTENCIA: DISPARIDAD CERCA DEL LIMITE",
                    (20, 210),
                    0.55,
                    (0, 0, 255),
                )

            put_text_outline(
                display_l,
                (
                    f"Validos ROI: {estado['pixeles_validos_roi']} "
                    f"({100.0 * estado['fraccion_valida_roi']:.1f}%)"
                ),
                (20, 210),
                0.52,
                (
                    (0, 255, 0)
                    if estado["pixeles_validos_roi"] >= MIN_PIXELES_VALIDOS
                    else (0, 165, 255)
                ),
            )

            put_text_outline(
                display_l,
                "ALINEACION VERTICAL - SIN DEFORMACION",
                (20, 245),
                0.55,
                (255, 255, 0),
            )

            if estado["pausado"]:
                put_text_outline(
                    display_l,
                    "PAUSA",
                    (20, 285),
                    0.9,
                    (0, 165, 255),
                )

            if estado["guardando"]:
                put_text_outline(
                    display_l,
                    "GUARDANDO",
                    (20, 330),
                    0.9,
                    (0, 0, 255),
                )

            put_text_outline(
                display_l,
                (
                    "ESPACIO pausa | R referencia | "
                    "G/S guardar | P ROI | L lanza | Q salir"
                ),
                (20, alto_salida - 20),
                0.42,
            )

            if MOSTRAR_LINEAS_EPIPOLARES:
                display_l = dibujar_lineas(
                    display_l,
                    SEPARACION_LINEAS,
                )

                display_r = dibujar_lineas(
                    display_r,
                    SEPARACION_LINEAS,
                )

            vista_l = redimensionar_para_pantalla(
                display_l
            )

            vista_r = redimensionar_para_pantalla(
                display_r
            )

            if vista_l.shape[:2] != vista_r.shape[:2]:
                vista_r = cv2.resize(
                    vista_r,
                    (vista_l.shape[1], vista_l.shape[0]),
                )

            cv2.imshow(
                "Estereo offline",
                np.hstack([vista_l, vista_r]),
            )

            disp_color = visualizar_disparidad(
                disp
            )

            cv2.imshow(
                "Disparidad",
                redimensionar_para_pantalla(
                    disp_color
                ),
            )

            if estado["guardando"] and not estado["pausado"]:
                csv_writer.writerow([
                    frame_actual,
                    tiempo_s,
                    "" if d_superficie is None else d_superficie,
                    pixeles_validos,
                    "" if z_actual is None else z_actual,
                    "" if altura_suave is None else altura_suave,
                    "" if estado["d_ref"] is None else estado["d_ref"],
                    "" if estado["K_ref"] is None else estado["K_ref"],
                    ALTURA_REFERENCIA_MM,
                ])

                csv_file.flush()

                if altura_suave is not None:
                    tiempos_resultados.append(
                        tiempo_s
                    )

                    alturas_resultados.append(
                        altura_suave
                    )

                if GUARDAR_VIDEOS:
                    writer_l.write(display_l)
                    writer_r.write(display_r)
                    writer_disp.write(disp_color)

            if estado["pausado"]:
                delay = 0
            elif MODO_TIEMPO_REAL:
                delay = 1
            else:
                delay = max(
                    1,
                    int(
                        1000
                        * SALTO_FRAMES
                        / fps
                    ),
                )

            key = cv2.waitKey(delay) & 0xFF

            if key == ord(" "):
                estaba_pausado = estado["pausado"]
                estado["pausado"] = not estado["pausado"]

                if estaba_pausado and not estado["pausado"]:
                    tiempo_base_reproduccion = time.perf_counter()
                    frame_base_reproduccion = frame_actual

            elif key in (ord("r"), ord("R")):
                candidatos_ref = list(
                    historial_disparidad_valida
                )

                if (
                    not candidatos_ref
                    and d_superficie is not None
                ):
                    candidatos_ref = [
                        float(d_superficie)
                    ]

                if not candidatos_ref:
                    print("\nNo hay disparidades válidas recientes.")
                    print(
                        "Motivo actual:",
                        estado["motivo_disparidad_invalida"],
                    )
                    print(
                        "Píxeles válidos en ROI:",
                        estado["pixeles_validos_roi"],
                    )
                    print(
                        "Rango SGBM actual:",
                        MIN_DISPARITY,
                        "a",
                        MIN_DISPARITY
                        + NUM_DISPARITIES
                        - 1,
                        "px",
                    )
                    print(
                        "Deja reproducir algunos frames con el ROI "
                        "sobre la espuma y vuelve a presionar R."
                    )
                else:
                    d_referencia = float(
                        np.median(
                            np.asarray(
                                candidatos_ref,
                                dtype=np.float64,
                            )
                        )
                    )

                    estado["d_ref"] = d_referencia
                    estado["K_ref"] = (
                        Z_REF_MM
                        * d_referencia
                    )

                    historial_altura.clear()

                    print("\nReferencia fijada correctamente:")
                    print(
                        f"d_ref = {d_referencia:.6f} px"
                    )
                    print(
                        f"Muestras usadas = "
                        f"{len(candidatos_ref)}"
                    )
                    print(
                        f"Z_ref = {Z_REF_MM:.3f} mm"
                    )
                    print(
                        f"Altura inicial = "
                        f"{ALTURA_REFERENCIA_MM:.3f} mm "
                        f"({ALTURA_REFERENCIA_MM / 10.0:.2f} cm)"
                    )
                    print(
                        f"K_ref = "
                        f"{estado['K_ref']:.6f}"
                    )

            elif key in (ord("g"), ord("G")):
                estado["guardando"] = True
                print("Guardado iniciado.")

            elif key in (ord("s"), ord("S")):
                estado["guardando"] = False
                print("Guardado detenido.")

            elif key in (ord("p"), ord("P")):
                estado["pausado"] = True
                roi_data["seleccionado"] = False
                estado["d_ref"] = None
                estado["K_ref"] = None

                historial_altura.clear()
                historial_disparidad_valida.clear()

                seleccionar_roi(alineada_l)

            elif key in (ord("l"), ord("L")):
                estado["pausado"] = True
                seleccionar_mascara_lanza(alineada_l)

                estado["d_ref"] = None
                estado["K_ref"] = None
                historial_altura.clear()
                historial_disparidad_valida.clear()

                print(
                    "\nMáscara de lanza actualizada. "
                    "Debes volver a presionar R."
                )

            elif key in (ord("q"), ord("Q")):
                break

            if not estado["pausado"]:
                if MODO_TIEMPO_REAL:
                    if tiempo_base_reproduccion is None:
                        tiempo_base_reproduccion = time.perf_counter()
                        frame_base_reproduccion = frame_actual

                    tiempo_transcurrido = (
                        time.perf_counter()
                        - tiempo_base_reproduccion
                    )

                    frame_objetivo = (
                        frame_base_reproduccion
                        + int(tiempo_transcurrido * fps)
                    )

                    siguiente_frame = max(
                        frame_actual + SALTO_FRAMES,
                        frame_objetivo,
                    )

                    siguiente_frame = min(
                        siguiente_frame,
                        total_frames,
                    )

                    cap_l.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        siguiente_frame,
                    )

                    cap_r.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        siguiente_frame,
                    )

                    frame_actual = siguiente_frame

                else:
                    for _ in range(SALTO_FRAMES - 1):
                        cap_l.grab()
                        cap_r.grab()

                    frame_actual += SALTO_FRAMES

    finally:
        csv_file.close()
        cap_l.release()
        cap_r.release()

        if writer_l is not None:
            writer_l.release()

        if writer_r is not None:
            writer_r.release()

        if writer_disp is not None:
            writer_disp.release()

        cv2.destroyAllWindows()

    guardar_grafico(ruta_grafico)

    print("\n===================================")
    print("PROCESAMIENTO FINALIZADO")
    print("===================================")
    print("Alineación:", ruta_npz)
    print("CSV:", ruta_csv)
    print("Video izquierdo:", ruta_video_l)
    print("Video derecho alineado:", ruta_video_r)
    print("Video de disparidad:", ruta_video_disp)
    print("Gráfico:", ruta_grafico)
    print("Modo tiempo real usado:", MODO_TIEMPO_REAL)
    if MODO_TIEMPO_REAL:
        print(
            "Advertencia: para mantener la duración real, "
            "se pudieron omitir frames."
        )


if __name__ == "__main__":
    main()