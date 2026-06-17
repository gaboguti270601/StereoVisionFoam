import os
import cv2
import numpy as np
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading

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
# RUTAS
# =========================================================

CALIB_FILE = (
    r"D:\MDT\Stereovision"
    r"\calibration\stereo_calibration.npz"
)

# =========================================================
# PARÁMETROS GENERALES
# =========================================================

ANGULO_GRADOS = 3.47

# Ya comprobaste que la imagen cruda se ve bien con 40 us.
EXPOSICION_US = 40

FPS_OBJETIVO = 5
FRAME_TIME = 1.0 / FPS_OBJETIVO

VIDEO_CODEC = "mp4v"

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

# =========================================================
# RECTIFICACIÓN
# =========================================================

USAR_RECTIFICACION = True

# El programa probará automáticamente:
# - sin rotación
# - 90° antihorario
# - 90° horario
# - 180°
SELECCIONAR_ROTACION_AUTOMATICA = True

# Solo se usa si SELECCIONAR_ROTACION_AUTOMATICA = False.
ROTACION_MANUAL = cv2.ROTATE_90_COUNTERCLOCKWISE

# Mínimo porcentaje de píxeles no negros para considerar
# válida una imagen rectificada.
MINIMO_PIXELES_NO_NEGROS = 0.01

# Dibuja líneas horizontales para comprobar visualmente
# la alineación epipolar.
DIBUJAR_LINEAS_EPIPOLARES = True
SEPARACION_LINEAS_PX = 100

# =========================================================
# ROI
# =========================================================

USAR_ROI = True

roi_data = {
    "seleccionado": False,
    "x": 0,
    "y": 0,
    "w": 0,
    "h": 0
}

# =========================================================
# BUFFERS DEL GRÁFICO
# =========================================================

tiempos = deque(maxlen=300)
alturas = deque(maxlen=300)

lock = threading.Lock()

estado = {
    "grabando": False,
    "z_ref": None,
    "activo": True,
    "t_inicio": None,
    "rectificacion_ok": False,
    "rotacion_nombre": "sin determinar"
}

# =========================================================
# UTILIDADES
# =========================================================

def put_text_outline(img, text, org, scale, color):
    """
    Escribe texto con borde negro para mejorar visibilidad.
    """

    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        8,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        3,
        cv2.LINE_AA
    )


def dibujar_lineas_horizontales(img, separacion=100):
    """
    Dibuja líneas horizontales para revisar si puntos
    correspondientes están en la misma fila.
    """

    resultado = img.copy()
    h, w = resultado.shape[:2]

    for y in range(0, h, separacion):
        cv2.line(
            resultado,
            (0, y),
            (w - 1, y),
            (0, 255, 0),
            1
        )

    return resultado


def calcular_tamano_salida(img, ancho_salida=720):
    """
    Calcula un tamaño de video conservando la relación
    de aspecto del frame procesado.
    """

    h, w = img.shape[:2]

    escala = ancho_salida / float(w)
    alto_salida = int(round(h * escala))

    # VideoWriter funciona mejor con dimensiones pares.
    if alto_salida % 2 != 0:
        alto_salida += 1

    return ancho_salida, alto_salida


def imagen_valida(img):
    """
    Comprueba que una imagen rectificada no esté compuesta
    casi completamente por píxeles negros.
    """

    if img is None or img.size == 0:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    proporcion_no_negra = np.count_nonzero(gray > 2) / gray.size

    return (
        img.max() >= 5
        and np.mean(img) > 0.1
        and proporcion_no_negra >= MINIMO_PIXELES_NO_NEGROS
    )


def imprimir_estadisticas(nombre, img):
    """
    Imprime estadísticas básicas de una imagen.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    proporcion_no_negra = (
        np.count_nonzero(gray > 2) / gray.size
    )

    print(f"\n{nombre}")
    print("shape:", img.shape)
    print("min:", int(img.min()))
    print("max:", int(img.max()))
    print("mean:", float(np.mean(img)))
    print(
        "píxeles no negros:",
        f"{100.0 * proporcion_no_negra:.2f}%"
    )


# =========================================================
# ROI
# =========================================================

def seleccionar_roi_en_vivo(img):
    """
    Permite seleccionar el ROI sobre la cámara izquierda
    rectificada.
    """

    ventana = (
        "Seleccionar ROI - superficie del fundido/espuma"
    )

    roi = cv2.selectROI(
        ventana,
        img,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(ventana)

    x, y, w, h = roi

    if w == 0 or h == 0:
        print("\nROI no seleccionado.")
        return None

    print("\nROI seleccionado:")
    print(f"x={x}, y={y}, w={w}, h={h}")

    return int(x), int(y), int(w), int(h)


# =========================================================
# CALIBRACIÓN
# =========================================================

def evaluar_mapas(map1, map2, nombre):
    """
    Revisa qué porcentaje de coordenadas de los mapas cae
    dentro de la imagen de origen.
    """

    try:
        map_x, map_y = cv2.convertMaps(
            map1,
            map2,
            cv2.CV_32FC1
        )
    except cv2.error as error:
        print(f"\nNo fue posible evaluar mapas {nombre}:")
        print(error)
        return 0.0

    h, w = map_x.shape

    validos = (
        (map_x >= 0)
        & (map_x < w)
        & (map_y >= 0)
        & (map_y < h)
        & np.isfinite(map_x)
        & np.isfinite(map_y)
    )

    porcentaje = (
        100.0 * np.count_nonzero(validos) / validos.size
    )

    print(f"\nMapas {nombre}:")
    print("shape:", map_x.shape)
    print(
        "map_x min/max:",
        float(np.nanmin(map_x)),
        float(np.nanmax(map_x))
    )
    print(
        "map_y min/max:",
        float(np.nanmin(map_y)),
        float(np.nanmax(map_y))
    )
    print(
        "coordenadas dentro de la imagen:",
        f"{porcentaje:.2f}%"
    )

    if porcentaje < 10:
        print(
            "ADVERTENCIA: casi todas las coordenadas "
            "caen fuera de la imagen."
        )
    elif porcentaje < 50:
        print(
            "ADVERTENCIA: los mapas tienen cobertura baja."
        )
    else:
        print("Cobertura geométrica razonable.")

    return porcentaje


def cargar_calibracion():
    """
    Carga y evalúa el archivo NPZ.
    """

    if not os.path.isfile(CALIB_FILE):
        raise FileNotFoundError(
            f"No existe el archivo:\n{CALIB_FILE}"
        )

    data = np.load(CALIB_FILE)

    claves_necesarias = [
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
        clave
        for clave in claves_necesarias
        if clave not in data.files
    ]

    if faltantes:
        raise KeyError(
            "Faltan variables en la calibración: "
            + ", ".join(faltantes)
        )

    baseline = float(np.linalg.norm(data["T"]))

    print("\n===================================")
    print("CALIBRACIÓN CARGADA")
    print("===================================")

    print("Archivo:", CALIB_FILE)
    print("Variables:", data.files)

    print("\nK_l:")
    print(data["K_l"])

    print("\nD_l:")
    print(data["D_l"])

    print("\nK_r:")
    print(data["K_r"])

    print("\nD_r:")
    print(data["D_r"])

    print("\nT:")
    print(data["T"])

    print(f"\nBaseline del NPZ: {baseline:.4f} mm")
    print(
        f"Factor de corrección: {FACTOR_CORRECCION:.6f}"
    )

    print("\nmap_l1 shape:", data["map_l1"].shape)
    print("map_l2 shape:", data["map_l2"].shape)
    print("map_r1 shape:", data["map_r1"].shape)
    print("map_r2 shape:", data["map_r2"].shape)

    if "image_size" in data.files:
        print("image_size guardado:", data["image_size"])

    evaluar_mapas(
        data["map_l1"],
        data["map_l2"],
        "cámara izquierda"
    )

    evaluar_mapas(
        data["map_r1"],
        data["map_r2"],
        "cámara derecha"
    )

    return data


# =========================================================
# CÁMARAS
# =========================================================

def inicializar_camaras():
    """
    Detecta, abre y configura las dos cámaras.
    """

    from thorlabs_tsi_sdk.tl_camera import (
        TLCameraSDK,
        OPERATION_MODE
    )

    sdk = TLCameraSDK()
    camaras = sdk.discover_available_cameras()

    print("\n===================================")
    print("CÁMARAS DETECTADAS")
    print("===================================")
    print(camaras)

    if len(camaras) < 2:
        sdk.dispose()
        raise RuntimeError(
            "No se detectaron dos cámaras Thorlabs."
        )

    cam_l = sdk.open_camera(camaras[0])
    cam_r = sdk.open_camera(camaras[1])

    for cam in (cam_l, cam_r):
        cam.exposure_time_us = EXPOSICION_US
        cam.frames_per_trigger_zero_for_unlimited = 0
        cam.operation_mode = (
            OPERATION_MODE.SOFTWARE_TRIGGERED
        )
        cam.arm(2)

    print(
        f"\nExposición configurada: "
        f"{EXPOSICION_US} us"
    )

    print(
        "Resolución cámara L:",
        cam_l.image_width_pixels,
        "x",
        cam_l.image_height_pixels
    )

    print(
        "Resolución cámara R:",
        cam_r.image_width_pixels,
        "x",
        cam_r.image_height_pixels
    )

    return sdk, cam_l, cam_r


# =========================================================
# CAPTURA
# =========================================================

def capturar_frame(cam):
    """
    Captura un frame y lo convierte a 8 bits BGR.
    """

    frame = None
    tiempo_limite = time.time() + 2.0

    while frame is None and time.time() < tiempo_limite:
        frame = cam.get_pending_frame_or_null()

        if frame is None:
            time.sleep(0.001)

    if frame is None:
        return None

    w = cam.image_width_pixels
    h = cam.image_height_pixels

    img = np.array(
        frame.image_buffer,
        dtype=np.uint16
    ).reshape(h, w)

    # Conversión para visualización/procesamiento.
    # Se usa el mismo procedimiento en ambas cámaras.
    minimo = int(img.min())
    maximo = int(img.max())

    if maximo > minimo:
        img_8bit = cv2.normalize(
            img,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)
    else:
        img_8bit = np.zeros(
            img.shape,
            dtype=np.uint8
        )

    return cv2.cvtColor(
        img_8bit,
        cv2.COLOR_GRAY2BGR
    )


# =========================================================
# ROTACIÓN Y RECTIFICACIÓN
# =========================================================

def aplicar_rotacion(img, rotacion):
    """
    Aplica una rotación OpenCV o devuelve copia sin rotar.
    """

    if rotacion is None:
        return img.copy()

    return cv2.rotate(img, rotacion)


def seleccionar_mejor_rotacion(
    img_l,
    img_r,
    map_l1,
    map_l2,
    map_r1,
    map_r2
):
    """
    Prueba todas las orientaciones compatibles y elige la
    que produzca mayor proporción de imagen rectificada útil.
    """

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
        ("180_grados", cv2.ROTATE_180)
    ]

    mejor = None

    print("\n===================================")
    print("PRUEBA AUTOMÁTICA DE ROTACIÓN")
    print("===================================")

    for nombre, rotacion in candidatos:
        prueba_l = aplicar_rotacion(img_l, rotacion)
        prueba_r = aplicar_rotacion(img_r, rotacion)

        print(f"\nCandidato: {nombre}")
        print("Frame:", prueba_l.shape[:2])
        print("Mapa:", map_l1.shape[:2])

        if prueba_l.shape[:2] != map_l1.shape[:2]:
            print("No coincide con la resolución del mapa.")
            continue

        rect_l = cv2.remap(
            prueba_l,
            map_l1,
            map_l2,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        rect_r = cv2.remap(
            prueba_r,
            map_r1,
            map_r2,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        gray_l = cv2.cvtColor(
            rect_l,
            cv2.COLOR_BGR2GRAY
        )

        gray_r = cv2.cvtColor(
            rect_r,
            cv2.COLOR_BGR2GRAY
        )

        proporcion_l = (
            np.count_nonzero(gray_l > 2) / gray_l.size
        )

        proporcion_r = (
            np.count_nonzero(gray_r > 2) / gray_r.size
        )

        score = (
            proporcion_l
            + proporcion_r
            + np.mean(gray_l) / 255.0
            + np.mean(gray_r) / 255.0
        )

        print(
            "Píxeles útiles L:",
            f"{100.0 * proporcion_l:.2f}%"
        )

        print(
            "Píxeles útiles R:",
            f"{100.0 * proporcion_r:.2f}%"
        )

        print("Score:", float(score))

        if mejor is None or score > mejor["score"]:
            mejor = {
                "nombre": nombre,
                "rotacion": rotacion,
                "score": score,
                "rect_l": rect_l,
                "rect_r": rect_r
            }

    if mejor is None:
        raise RuntimeError(
            "Ninguna orientación coincide con "
            "la resolución de los mapas."
        )

    print("\nRotación seleccionada:")
    print(mejor["nombre"])
    print("Score:", mejor["score"])

    return mejor["rotacion"], mejor["nombre"]


def rectificar_par(
    img_l,
    img_r,
    rotacion,
    map_l1,
    map_l2,
    map_r1,
    map_r2
):
    """
    Rota y rectifica un par de imágenes.
    """

    img_l_rot = aplicar_rotacion(img_l, rotacion)
    img_r_rot = aplicar_rotacion(img_r, rotacion)

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

    ok = (
        imagen_valida(rect_l)
        and imagen_valida(rect_r)
    )

    return (
        img_l_rot,
        img_r_rot,
        rect_l,
        rect_r,
        ok
    )


# =========================================================
# DISPARIDAD
# =========================================================

def crear_matcher_estereo():
    """
    Crea el objeto StereoSGBM una sola vez.
    """

    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISPARIDADES,
        blockSize=BLOCK_SIZE,
        P1=8 * BLOCK_SIZE**2,
        P2=32 * BLOCK_SIZE**2,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )


def calcular_disparidad(stereo, img_l, img_r):
    """
    Calcula disparidad sobre imágenes rectificadas.
    """

    gray_l = cv2.cvtColor(
        img_l,
        cv2.COLOR_BGR2GRAY
    )

    gray_r = cv2.cvtColor(
        img_r,
        cv2.COLOR_BGR2GRAY
    )

    disp = stereo.compute(
        gray_l,
        gray_r
    ).astype(np.float32) / 16.0

    return disp


# =========================================================
# PROFUNDIDAD
# =========================================================

def disparidad_a_profundidad(disp, Q):
    """
    Reproyecta la disparidad a coordenadas 3D.
    """

    puntos_3d = cv2.reprojectImageTo3D(
        disp,
        Q
    )

    depth = puntos_3d[:, :, 2].astype(np.float32)

    invalidos = (
        ~np.isfinite(depth)
        | (disp <= 0)
        | (depth < 100)
        | (depth > 3000)
    )

    depth[invalidos] = np.nan

    return depth


# =========================================================
# DETECCIÓN DE SUPERFICIE
# =========================================================

def detectar_superficie(gray):
    """
    Segmentación inicial de la superficie mediante Otsu.
    """

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


# =========================================================
# ALTURA
# =========================================================

def obtener_profundidad_superficie(depth_roi, mask):
    """
    Obtiene la profundidad mediana de la superficie
    segmentada.
    """

    validos = (
        (mask > 0)
        & np.isfinite(depth_roi)
    )

    cantidad = int(np.count_nonzero(validos))

    if cantidad < 50:
        return None, cantidad

    z_superficie = float(
        np.nanmedian(depth_roi[validos])
    )

    return z_superficie, cantidad


def estimar_altura(z_superficie, z_ref):
    """
    Calcula el cambio de altura respecto a la referencia.
    """

    altura_mm = (
        abs(z_ref - z_superficie)
        * np.cos(np.radians(ANGULO_GRADOS))
        * FACTOR_CORRECCION
    )

    return float(altura_mm)


# =========================================================
# VISUALIZACIÓN DE PROFUNDIDAD
# =========================================================

def visualizar_depth(depth):
    """
    Convierte un mapa de profundidad en una imagen coloreada.
    """

    if depth is None:
        return np.zeros(
            (480, 640, 3),
            dtype=np.uint8
        )

    depth_vis = depth.copy()
    validos = np.isfinite(depth_vis)

    if not np.any(validos):
        resultado = np.zeros(
            depth_vis.shape,
            dtype=np.uint8
        )

        return cv2.applyColorMap(
            resultado,
            cv2.COLORMAP_JET
        )

    minimo = np.percentile(
        depth_vis[validos],
        5
    )

    maximo = np.percentile(
        depth_vis[validos],
        95
    )

    if maximo <= minimo:
        normalizada = np.zeros(
            depth_vis.shape,
            dtype=np.uint8
        )
    else:
        recortada = np.clip(
            depth_vis,
            minimo,
            maximo
        )

        recortada = (
            recortada - minimo
        ) / (
            maximo - minimo
        )

        recortada = np.nan_to_num(
            recortada,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        normalizada = (
            recortada * 255
        ).astype(np.uint8)

    return cv2.applyColorMap(
        normalizada,
        cv2.COLORMAP_JET
    )


# =========================================================
# GRÁFICO
# =========================================================

def hilo_grafico(grafico_path):
    """
    Muestra y guarda el gráfico de altura.
    """

    fig, ax = plt.subplots()

    linea, = ax.plot(
        [],
        [],
        label="Altura espuma"
    )

    punto, = ax.plot([], [], "ro")

    texto = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes
    )

    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Altura espuma [mm]")
    ax.set_title(
        "Altura de espuma en tiempo real"
    )
    ax.grid(True)
    ax.legend()

    def actualizar(_):
        if not estado["activo"]:
            plt.close(fig)
            return linea, punto, texto

        with lock:
            if len(alturas) == 0:
                return linea, punto, texto

            t = list(tiempos)
            h = list(alturas)

        linea.set_data(t, h)
        punto.set_data([t[-1]], [h[-1]])
        texto.set_text(f"{h[-1]:.2f} mm")

        ax.set_xlim(
            max(0, t[-1] - 60),
            t[-1] + 1
        )

        limite_superior = max(
            max(h) * 1.2,
            10
        )

        ax.set_ylim(0, limite_superior)

        return linea, punto, texto

    anim = animation.FuncAnimation(
        fig,
        actualizar,
        interval=500,
        cache_frame_data=False
    )

    plt.show()

    with lock:
        if len(alturas) == 0:
            print(
                "\nNo se guardó gráfico porque "
                "no existen datos de altura."
            )
            return

        t = list(tiempos)
        h = list(alturas)

    fig_save, ax_save = plt.subplots()

    ax_save.plot(
        t,
        h,
        label="Altura espuma"
    )

    ax_save.scatter(
        t[-1],
        h[-1]
    )

    ax_save.set_xlabel("Tiempo [s]")
    ax_save.set_ylabel("Altura espuma [mm]")
    ax_save.set_title(
        "Altura de espuma medida por estereovisión"
    )
    ax_save.grid(True)
    ax_save.legend()

    fig_save.savefig(
        grafico_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig_save)

    print("\nGráfico guardado en:")
    print(grafico_path)


# =========================================================
# HILO PRINCIPAL DE CÁMARAS
# =========================================================

def hilo_camaras(
    cal,
    sdk,
    cam_l,
    cam_r,
    csv_path,
    video_l_path,
    video_r_path
):
    global USAR_ROI

    Q = cal["Q"]

    map_l1 = cal["map_l1"]
    map_l2 = cal["map_l2"]
    map_r1 = cal["map_r1"]
    map_r2 = cal["map_r2"]

    stereo = crear_matcher_estereo()

    cv2.namedWindow(
        "Camaras Estereo",
        cv2.WINDOW_NORMAL
    )

    cv2.namedWindow(
        "Mapa de Profundidad",
        cv2.WINDOW_NORMAL
    )

    csv_file = None
    video_l = None
    video_r = None

    rotacion = None
    rotacion_seleccionada = False

    diagnostico_impreso = False

    try:
        csv_file = open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        )

        writer = csv.writer(csv_file)

        writer.writerow([
            "timestamp",
            "tiempo_s",
            "altura_mm",
            "altura_cm",
            "z_ref_mm",
            "z_superficie_mm",
            "rectificacion_ok",
            "rotacion"
        ])

        fourcc = cv2.VideoWriter_fourcc(
            *VIDEO_CODEC
        )

        frame_id = 0

        while estado["activo"]:
            tiempo_frame_inicio = time.time()

            cam_l.issue_software_trigger()
            cam_r.issue_software_trigger()

            img_l_raw = capturar_frame(cam_l)
            img_r_raw = capturar_frame(cam_r)

            if img_l_raw is None or img_r_raw is None:
                print(
                    "No fue posible obtener "
                    "un par de frames."
                )
                continue

            if frame_id == 0:
                imprimir_estadisticas(
                    "Frame RAW izquierdo",
                    img_l_raw
                )

                imprimir_estadisticas(
                    "Frame RAW derecho",
                    img_r_raw
                )

            # =============================================
            # DETERMINAR ROTACIÓN
            # =============================================

            if not rotacion_seleccionada:
                if SELECCIONAR_ROTACION_AUTOMATICA:
                    rotacion, nombre_rotacion = (
                        seleccionar_mejor_rotacion(
                            img_l_raw,
                            img_r_raw,
                            map_l1,
                            map_l2,
                            map_r1,
                            map_r2
                        )
                    )
                else:
                    rotacion = ROTACION_MANUAL
                    nombre_rotacion = "manual"

                estado["rotacion_nombre"] = (
                    nombre_rotacion
                )

                rotacion_seleccionada = True

            # =============================================
            # RECTIFICAR
            # =============================================

            if USAR_RECTIFICACION:
                (
                    img_l_rot,
                    img_r_rot,
                    rect_l,
                    rect_r,
                    rectificacion_ok
                ) = rectificar_par(
                    img_l_raw,
                    img_r_raw,
                    rotacion,
                    map_l1,
                    map_l2,
                    map_r1,
                    map_r2
                )
            else:
                img_l_rot = aplicar_rotacion(
                    img_l_raw,
                    rotacion
                )

                img_r_rot = aplicar_rotacion(
                    img_r_raw,
                    rotacion
                )

                rect_l = img_l_rot.copy()
                rect_r = img_r_rot.copy()

                rectificacion_ok = False

            estado["rectificacion_ok"] = (
                rectificacion_ok
            )

            if not diagnostico_impreso:
                print("\n===================================")
                print("DIAGNÓSTICO DEL PRIMER FRAME")
                print("===================================")

                print(
                    "Rotación:",
                    estado["rotacion_nombre"]
                )

                imprimir_estadisticas(
                    "Frame rotado izquierdo",
                    img_l_rot
                )

                imprimir_estadisticas(
                    "Frame rotado derecho",
                    img_r_rot
                )

                if rect_l is not None:
                    imprimir_estadisticas(
                        "Frame rectificado izquierdo",
                        rect_l
                    )

                if rect_r is not None:
                    imprimir_estadisticas(
                        "Frame rectificado derecho",
                        rect_r
                    )

                print(
                    "\nRectificación válida:",
                    rectificacion_ok
                )

                if not rectificacion_ok:
                    print(
                        "\nLa adquisición funciona, "
                        "pero los mapas de rectificación "
                        "no generan imágenes utilizables."
                    )

                    print(
                        "No se calculará profundidad ni "
                        "altura mientras esto ocurra."
                    )

                diagnostico_impreso = True

            # =============================================
            # IMÁGENES PARA VISUALIZAR
            # =============================================

            if (
                USAR_RECTIFICACION
                and rectificacion_ok
            ):
                procesada_l = rect_l
                procesada_r = rect_r
                modo_texto = "RECTIFICADA"
            else:
                # Se muestran imágenes crudas/rotadas para
                # no dejar la ventana completamente negra.
                procesada_l = img_l_rot
                procesada_r = img_r_rot
                modo_texto = (
                    "RAW - RECTIFICACION FALLIDA"
                    if USAR_RECTIFICACION
                    else "SIN RECTIFICAR"
                )

            # =============================================
            # INICIALIZAR VIDEO WRITERS
            # =============================================

            if video_l is None or video_r is None:
                out_w, out_h = calcular_tamano_salida(
                    procesada_l,
                    ancho_salida=720
                )

                video_l = cv2.VideoWriter(
                    video_l_path,
                    fourcc,
                    FPS_OBJETIVO,
                    (out_w, out_h)
                )

                video_r = cv2.VideoWriter(
                    video_r_path,
                    fourcc,
                    FPS_OBJETIVO,
                    (out_w, out_h)
                )

                if not video_l.isOpened():
                    raise RuntimeError(
                        "No se pudo abrir el video "
                        "de la cámara izquierda."
                    )

                if not video_r.isOpened():
                    raise RuntimeError(
                        "No se pudo abrir el video "
                        "de la cámara derecha."
                    )

                print(
                    "\nTamaño de videos:",
                    out_w,
                    "x",
                    out_h
                )

            # =============================================
            # ROI
            # =============================================

            if (
                USAR_ROI
                and not roi_data["seleccionado"]
                and rectificacion_ok
            ):
                roi = seleccionar_roi_en_vivo(
                    procesada_l
                )

                if roi is not None:
                    (
                        roi_data["x"],
                        roi_data["y"],
                        roi_data["w"],
                        roi_data["h"]
                    ) = roi

                    roi_data["seleccionado"] = True
                else:
                    USAR_ROI = False

            # =============================================
            # DISPARIDAD Y PROFUNDIDAD
            # =============================================

            disp = None
            depth = None

            if rectificacion_ok:
                disp = calcular_disparidad(
                    stereo,
                    procesada_l,
                    procesada_r
                )

                depth = disparidad_a_profundidad(
                    disp,
                    Q
                )

            display_l = procesada_l.copy()
            display_r = procesada_r.copy()

            roi_depth = None
            gray_roi = None

            # =============================================
            # APLICAR ROI
            # =============================================

            if (
                rectificacion_ok
                and depth is not None
                and USAR_ROI
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

            elif (
                rectificacion_ok
                and depth is not None
            ):
                roi_depth = depth

                gray_roi = cv2.cvtColor(
                    procesada_l,
                    cv2.COLOR_BGR2GRAY
                )

            # =============================================
            # ALTURA
            # =============================================

            texto_altura = "Altura: ---"
            z_superficie = None
            altura_mm = None
            cantidad_validos = 0

            if (
                rectificacion_ok
                and estado["z_ref"] is not None
                and gray_roi is not None
                and roi_depth is not None
            ):
                mask = detectar_superficie(gray_roi)

                (
                    z_superficie,
                    cantidad_validos
                ) = obtener_profundidad_superficie(
                    roi_depth,
                    mask
                )

                if z_superficie is not None:
                    altura_mm = estimar_altura(
                        z_superficie,
                        estado["z_ref"]
                    )

                    texto_altura = (
                        f"Altura: {altura_mm:.2f} mm"
                    )

            # =============================================
            # REGISTRO
            # =============================================

            if estado["grabando"]:
                timestamp = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S.%f"
                )[:-3]

                tiempo_s = (
                    time.time() - estado["t_inicio"]
                )

                if altura_mm is not None:
                    with lock:
                        tiempos.append(tiempo_s)
                        alturas.append(altura_mm)

                writer.writerow([
                    timestamp,
                    tiempo_s,
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
                    rectificacion_ok,
                    estado["rotacion_nombre"]
                ])

                csv_file.flush()

            # =============================================
            # OVERLAY
            # =============================================

            titulo_1 = f"Camara 1 - {modo_texto}"
            titulo_2 = f"Camara 2 - {modo_texto}"

            put_text_outline(
                display_l,
                titulo_1,
                (40, 60),
                0.9,
                (255, 255, 0)
            )

            put_text_outline(
                display_r,
                titulo_2,
                (40, 60),
                0.9,
                (255, 255, 0)
            )

            put_text_outline(
                display_l,
                f"Frame: {frame_id}",
                (40, 115),
                0.8,
                (255, 255, 255)
            )

            put_text_outline(
                display_l,
                texto_altura,
                (40, 170),
                0.9,
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
                (40, 225),
                0.75,
                (
                    (0, 255, 0)
                    if rectificacion_ok
                    else (0, 0, 255)
                )
            )

            put_text_outline(
                display_l,
                (
                    "Rotacion: "
                    + estado["rotacion_nombre"]
                ),
                (40, 275),
                0.7,
                (255, 255, 255)
            )

            if estado["z_ref"] is not None:
                put_text_outline(
                    display_l,
                    (
                        f"Z ref: "
                        f"{estado['z_ref']:.2f} mm"
                    ),
                    (40, 325),
                    0.7,
                    (220, 220, 220)
                )

            put_text_outline(
                display_l,
                (
                    "R: referencia | G: grabar | "
                    "S: detener | P: nuevo ROI | Q: salir"
                ),
                (40, 380),
                0.55,
                (220, 220, 220)
            )

            if estado["grabando"]:
                put_text_outline(
                    display_l,
                    "REC",
                    (40, 435),
                    1.1,
                    (0, 0, 255)
                )

            if DIBUJAR_LINEAS_EPIPOLARES:
                display_l = dibujar_lineas_horizontales(
                    display_l,
                    SEPARACION_LINEAS_PX
                )

                display_r = dibujar_lineas_horizontales(
                    display_r,
                    SEPARACION_LINEAS_PX
                )

            # =============================================
            # MOSTRAR
            # =============================================

            vista_w, vista_h = calcular_tamano_salida(
                display_l,
                ancho_salida=640
            )

            vista_l = cv2.resize(
                display_l,
                (vista_w, vista_h)
            )

            vista_r = cv2.resize(
                display_r,
                (vista_w, vista_h)
            )

            combined = np.hstack([
                vista_l,
                vista_r
            ])

            cv2.imshow(
                "Camaras Estereo",
                combined
            )

            if depth is not None:
                depth_color = visualizar_depth(depth)
            else:
                depth_color = np.zeros(
                    procesada_l.shape,
                    dtype=np.uint8
                )

                put_text_outline(
                    depth_color,
                    "Sin profundidad: rectificacion no valida",
                    (40, 80),
                    0.8,
                    (0, 0, 255)
                )

            cv2.imshow(
                "Mapa de Profundidad",
                depth_color
            )

            # =============================================
            # GUARDAR VIDEO
            # =============================================

            if estado["grabando"]:
                out_w = int(
                    video_l.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )

                out_h = int(
                    video_l.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )

                video_l.write(
                    cv2.resize(
                        display_l,
                        (out_w, out_h)
                    )
                )

                video_r.write(
                    cv2.resize(
                        display_r,
                        (out_w, out_h)
                    )
                )

            # =============================================
            # TECLAS
            # =============================================

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("r"), ord("R")):
                if (
                    not rectificacion_ok
                    or gray_roi is None
                    or roi_depth is None
                ):
                    print(
                        "\nNo se puede capturar referencia: "
                        "la rectificación/profundidad "
                        "no es válida."
                    )
                else:
                    mask = detectar_superficie(
                        gray_roi
                    )

                    (
                        nueva_ref,
                        cantidad_validos
                    ) = obtener_profundidad_superficie(
                        roi_depth,
                        mask
                    )

                    if nueva_ref is None:
                        print(
                            "\nNo se pudo capturar "
                            "la referencia."
                        )

                        print(
                            "Píxeles válidos:",
                            cantidad_validos
                        )
                    else:
                        estado["z_ref"] = nueva_ref

                        print(
                            "\nReferencia capturada:",
                            f"{nueva_ref:.3f} mm"
                        )

                        print(
                            "Píxeles válidos:",
                            cantidad_validos
                        )

            elif key in (ord("g"), ord("G")):
                estado["grabando"] = True
                estado["t_inicio"] = time.time()

                print("\nGrabación iniciada.")

            elif key in (ord("s"), ord("S")):
                estado["grabando"] = False

                print("\nGrabación detenida.")

            elif key in (ord("p"), ord("P")):
                if rectificacion_ok:
                    roi_data["seleccionado"] = False
                    estado["z_ref"] = None

                    print(
                        "\nSeleccione un nuevo ROI."
                    )
                else:
                    print(
                        "\nNo se puede seleccionar ROI "
                        "sin rectificación válida."
                    )

            elif key in (ord("q"), ord("Q")):
                estado["activo"] = False
                break

            frame_id += 1

            tiempo_transcurrido = (
                time.time() - tiempo_frame_inicio
            )

            if tiempo_transcurrido < FRAME_TIME:
                time.sleep(
                    FRAME_TIME - tiempo_transcurrido
                )

    except Exception as error:
        estado["activo"] = False

        print("\nERROR EN EL HILO DE CÁMARAS:")
        print(type(error).__name__)
        print(error)

    finally:
        if csv_file is not None:
            csv_file.close()

        if video_l is not None:
            video_l.release()

        if video_r is not None:
            video_r.release()

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

        estado["activo"] = False

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
        "_"
    )

    if not experimento:
        experimento = "prueba_estereo"

    base_dir = os.getcwd()

    csv_path = os.path.join(
        base_dir,
        f"{experimento}.csv"
    )

    video_l_path = os.path.join(
        base_dir,
        f"{experimento}_cam1.mp4"
    )

    video_r_path = os.path.join(
        base_dir,
        f"{experimento}_cam2.mp4"
    )

    grafico_path = os.path.join(
        base_dir,
        f"{experimento}_grafico_altura.png"
    )

    cal = cargar_calibracion()

    sdk, cam_l, cam_r = inicializar_camaras()

    hilo = threading.Thread(
        target=hilo_camaras,
        args=(
            cal,
            sdk,
            cam_l,
            cam_r,
            csv_path,
            video_l_path,
            video_r_path
        ),
        daemon=True
    )

    hilo.start()

    hilo_grafico(grafico_path)

    estado["activo"] = False

    hilo.join()


if __name__ == "__main__":
    main()