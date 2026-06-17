
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

# Número de serie fijo para cada lado.
# La cámara 36933 será siempre la izquierda (L / cámara 1).
# La cámara 36930 será siempre la derecha (R / cámara 2).
CAMARA_L_SERIAL = "36933"
CAMARA_R_SERIAL = "36930"

# Exposición usada actualmente.
EXPOSICION_US = 40

# Vista y videos a 30 FPS.
FPS_OBJETIVO = 30.0
FRAME_TIME = 1.0 / FPS_OBJETIVO

VIDEO_CODEC = "mp4v"

# Los videos se guardarán en 960 x 720 si la cámara entrega
# imágenes con relación 4:3, como 1440 x 1080.
ANCHO_VIDEO_SALIDA = 960

# Tamaño aproximado de cada cámara en la ventana.
ANCHO_VISTA_CAMARA = 640

# La cámara izquierda gira 1 vez las manecillas del reloj 90 grados
ROTACION_L = cv2.ROTATE_90_CLOCKWISE

# La cámara derecha gira 1 vez al contrario de las manecillas del reloj 90 grados
ROTACION_R = cv2.ROTATE_90_COUNTERCLOCKWISE

# Mostrar información solo en pantalla.
# Los videos se guardan LIMPIOS, sin textos ni líneas.
MOSTRAR_OVERLAY = True

# Mostrar líneas horizontales solo en pantalla.
MOSTRAR_LINEAS_HORIZONTALES = False
SEPARACION_LINEAS_PX = 80

# Guardar CSV con información temporal de cada par grabado.
GUARDAR_CSV = True

# Tiempo máximo para esperar un frame de cada cámara.
TIMEOUT_FRAME_S = 1.0


# =========================================================
# ESTADO
# =========================================================

estado = {
    "activo": True,
    "grabando": False,
    "t_inicio_grabacion": None,
    "frame_global": 0,
    "frame_grabado": 0,
}

# Para calcular FPS real de manera estable.
historial_tiempos = deque(maxlen=60)


# =========================================================
# UTILIDADES
# =========================================================

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


# =========================================================
# CÁMARAS THORLABS
# =========================================================

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


# =========================================================
# ARCHIVOS DE SALIDA
# =========================================================

def crear_rutas(experimento):
    """
    Crea las rutas de salida en la carpeta actual.
    """

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
        "csv": os.path.join(
            base_dir,
            f"{experimento}_captura.csv",
        ),
    }


def abrir_writers(
    rutas,
    frame_l,
):
    """
    Abre VideoWriter a 30 FPS.

    Los videos se guardan sin overlays para que luego puedan
    usarse en el procesamiento offline.
    """

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

    if not writer_l.isOpened():
        raise RuntimeError(
            "No se pudo abrir el video de la cámara izquierda."
        )

    if not writer_r.isOpened():
        writer_l.release()

        raise RuntimeError(
            "No se pudo abrir el video de la cámara derecha."
        )

    print("\nVideos configurados:")
    print("Tamaño:", out_w, "x", out_h)
    print("FPS guardado:", FPS_OBJETIVO)
    print("Cámara L:", rutas["video_l"])
    print("Cámara R:", rutas["video_r"])
    print(
        "Los videos se guardarán en mp4 sin overlays, para procesamiento offline."
    )

    return writer_l, writer_r, out_w, out_h


# =========================================================
# VISUALIZACIÓN
# =========================================================

def crear_vista(
    img_l,
    img_r,
    fps_real,
):
    """
    Genera la vista combinada para pantalla.
    Los overlays no se guardan en los videos.
    """

    vista_l = img_l.copy()
    vista_r = img_r.copy()

    if MOSTRAR_OVERLAY:
        put_text_outline(
            vista_l,
            "Camara 1 - RAW",
            (25, 40),
            0.75,
            (255, 255, 0),
        )

        put_text_outline(
            vista_r,
            "Camara 2 - RAW - rotada 180",
            (25, 40),
            0.75,
            (255, 255, 0),
        )

        put_text_outline(
            vista_l,
            f"FPS real: {fps_real:.1f}",
            (25, 80),
            0.65,
            (
                (0, 255, 0)
                if fps_real >= 27.0
                else (0, 165, 255)
            ),
        )

        put_text_outline(
            vista_l,
            f"Frame: {estado['frame_global']}",
            (25, 120),
            0.65,
            (255, 255, 255),
        )

        put_text_outline(
            vista_l,
            "G: grabar | S: detener | Q: salir",
            (25, 160),
            0.55,
            (230, 230, 230),
        )

        if estado["grabando"]:
            tiempo_grabacion = (
                time.perf_counter()
                - estado["t_inicio_grabacion"]
            )

            put_text_outline(
                vista_l,
                "REC",
                (25, 210),
                1.0,
                (0, 0, 255),
            )

            put_text_outline(
                vista_l,
                f"Grabacion: {tiempo_grabacion:.1f} s",
                (25, 250),
                0.65,
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
            (
                vista_l.shape[1],
                vista_l.shape[0],
            ),
            interpolation=cv2.INTER_AREA,
        )

    return np.hstack([
        vista_l,
        vista_r,
    ])


# =========================================================
# BUCLE PRINCIPAL
# =========================================================

def ejecutar_captura(
    sdk,
    cam_l,
    cam_r,
    rutas,
):
    """
    Captura, muestra y guarda ambos videos.

    No usa calibración.
    No rectifica.
    No calcula profundidad.
    """

    writer_l = None
    writer_r = None
    csv_file = None
    csv_writer = None

    out_w = None
    out_h = None

    cv2.namedWindow(
        "Camaras Estereo RAW",
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

            csv_writer = csv.writer(csv_file)

            csv_writer.writerow([
                "frame_global",
                "frame_grabado",
                "timestamp",
                "tiempo_grabacion_s",
                "fps_real",
                "fps_objetivo",
                "exposicion_us",
            ])

        # Un disparo inicial inicia la adquisición continua,
        # igual que en tu configuración anterior.
        cam_l.issue_software_trigger()
        cam_r.issue_software_trigger()

        print("\n===================================")
        print("CONTROLES")
        print("===================================")
        print("G: iniciar grabación")
        print("S: detener grabación")
        print("Q: salir")
        print(
            "\nLa pantalla intenta actualizarse a 30 FPS. "
            "El FPS real aparece en la ventana."
        )

        while estado["activo"]:
            inicio_ciclo = time.perf_counter()

            # Leer el frame más reciente disponible.
            img_l_raw = limpiar_frames_pendientes(cam_l)
            img_r_raw = limpiar_frames_pendientes(cam_r)

            # Si todavía no había frame pendiente, esperar uno.
            if img_l_raw is None:
                img_l_raw = esperar_frame(cam_l)

            if img_r_raw is None:
                img_r_raw = esperar_frame(cam_r)

            if img_l_raw is None or img_r_raw is None:
                print(
                    "No se pudo obtener un par de frames."
                )
                continue

            # Orientación real del montaje:
            # L sin rotar, R rotada 180°.
            img_l = aplicar_rotacion(
                img_l_raw,
                "L",
            )

            img_r = aplicar_rotacion(
                img_r_raw,
                "R",
            )

            fps_real = calcular_fps_real()

            if writer_l is None or writer_r is None:
                (
                    writer_l,
                    writer_r,
                    out_w,
                    out_h,
                ) = abrir_writers(
                    rutas,
                    img_l,
                )

            # Guardar imágenes limpias, antes de dibujar overlays.
            if estado["grabando"]:
                frame_l_guardado = cv2.resize(
                    img_l,
                    (out_w, out_h),
                    interpolation=cv2.INTER_AREA,
                )

                frame_r_guardado = cv2.resize(
                    img_r,
                    (out_w, out_h),
                    interpolation=cv2.INTER_AREA,
                )

                writer_l.write(
                    frame_l_guardado
                )

                writer_r.write(
                    frame_r_guardado
                )

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
                        FPS_OBJETIVO,
                        EXPOSICION_US,
                    ])

                    csv_file.flush()

                estado["frame_grabado"] += 1

            combined = crear_vista(
                img_l,
                img_r,
                fps_real,
            )

            cv2.imshow(
                "Camaras Estereo RAW",
                combined,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("g"), ord("G")):
                if not estado["grabando"]:
                    estado["grabando"] = True
                    estado["t_inicio_grabacion"] = (
                        time.perf_counter()
                    )
                    estado["frame_grabado"] = 0

                    print("\nGrabación iniciada a 30 FPS.")

            elif key in (ord("s"), ord("S")):
                if estado["grabando"]:
                    estado["grabando"] = False

                    print("\nGrabación detenida.")
                    print(
                        "Frames guardados:",
                        estado["frame_grabado"],
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
        experimento = "prueba_estereo"

    rutas = crear_rutas(
        experimento
    )

    sdk, cam_l, cam_r = inicializar_camaras()

    ejecutar_captura(
        sdk,
        cam_l,
        cam_r,
        rutas,
    )


if __name__ == "__main__":
    main()
