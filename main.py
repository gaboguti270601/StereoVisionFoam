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

os.add_dll_directory(
    r"D:\MDT\Scientific Camera Interfaces\SDK\Native Toolkit\dlls\Native_64_lib"
)

# =========================================================
# PARÁMETROS GENERALES
# =========================================================

CALIB_FILE = "stereo_calibration.npz"

ANGULO_GRADOS = 3.47
EXPOSICION_US = 28712

NUM_DISPARIDADES = 64
BLOCK_SIZE = 11

# Baseline físico / baseline MATLAB
# Baseline físico: 75 mm
# Baseline MATLAB: 125.2247934179211 mm
FACTOR_CORRECCION = 75.0 / 125.2247934179211

FPS_OBJETIVO = 5
FRAME_TIME = 1.0 / FPS_OBJETIVO

VIDEO_CODEC = "mp4v"

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
# BUFFERS PARA GRÁFICO
# =========================================================

tiempos = deque(maxlen=300)
alturas = deque(maxlen=300)

lock = threading.Lock()

estado = {
    "grabando": False,
    "z_ref": None,
    "activo": True,
    "t_inicio": None
}

# =========================================================
# TEXTO CON BORDE
# =========================================================

def put_text_outline(img, text, org, scale, color):
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        10,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        4,
        cv2.LINE_AA
    )


# =========================================================
# SELECCIÓN DE ROI
# =========================================================

def seleccionar_roi_en_vivo(rect_l):
    roi = cv2.selectROI(
        "Seleccionar ROI - superficie del fundido/espuma",
        rect_l,
        False
    )

    cv2.destroyWindow("Seleccionar ROI - superficie del fundido/espuma")

    x, y, w, h = roi

    if w == 0 or h == 0:
        print("ROI no seleccionado.")
        return None

    print("\nROI seleccionado:")
    print(f"x={x}, y={y}, w={w}, h={h}")

    return int(x), int(y), int(w), int(h)


# =========================================================
# CARGAR CALIBRACIÓN
# =========================================================

def cargar_calibracion():
    data = np.load(CALIB_FILE)

    print("\n=== CALIBRACIÓN CARGADA ===")
    print(f"Baseline MATLAB: {np.linalg.norm(data['T']):.2f} mm")
    print(f"Factor corrección: {FACTOR_CORRECCION:.4f}")

    print("map_l1 shape:", data["map_l1"].shape)
    print("map_l2 shape:", data["map_l2"].shape)
    print("map_r1 shape:", data["map_r1"].shape)
    print("map_r2 shape:", data["map_r2"].shape)

    if "image_size" in data:
        print("image_size guardado:", data["image_size"])

    return data


# =========================================================
# INICIALIZAR CÁMARAS
# =========================================================

def inicializar_camaras():
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE

    sdk = TLCameraSDK()
    cams = sdk.discover_available_cameras()

    print("\n=== CÁMARAS DETECTADAS ===")
    print(cams)

    if len(cams) < 2:
        raise RuntimeError("No se detectaron 2 cámaras.")

    cam_l = sdk.open_camera(cams[0])
    cam_r = sdk.open_camera(cams[1])

    for cam in [cam_l, cam_r]:
        cam.exposure_time_us = EXPOSICION_US
        cam.frames_per_trigger_zero_for_unlimited = 0
        cam.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGERED
        cam.arm(2)

    return sdk, cam_l, cam_r


# =========================================================
# CAPTURAR FRAME
# =========================================================

def capturar_frame(cam):
    frame = None
    timeout = time.time() + 2.0

    while frame is None and time.time() < timeout:
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

    img_8bit = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    return cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)


# =========================================================
# CALCULAR DISPARIDAD
# =========================================================

def calcular_disparidad(img_l, img_r):
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISPARIDADES,
        blockSize=BLOCK_SIZE,
        P1=8 * 3 * BLOCK_SIZE**2,
        P2=32 * 3 * BLOCK_SIZE**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    disp = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

    return disp


# =========================================================
# DISPARIDAD A PROFUNDIDAD
# =========================================================

def disparidad_a_profundidad(disp, Q):
    pts = cv2.reprojectImageTo3D(disp, Q)

    depth = pts[:, :, 2]

    depth[
        (disp <= 0) |
        (depth < 100) |
        (depth > 3000)
    ] = np.nan

    return depth


# =========================================================
# DETECCIÓN AUTOMÁTICA DE SUPERFICIE
# =========================================================

def detectar_superficie(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)

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
# ESTIMAR ALTURA
# =========================================================

def estimar_altura(depth_roi, mask, z_ref):
    valid = (mask > 0) & np.isfinite(depth_roi)

    if np.sum(valid) < 50:
        return None, None

    z_superficie = np.nanmedian(depth_roi[valid])

    altura_mm = (
        abs(z_ref - z_superficie)
        * np.cos(np.radians(ANGULO_GRADOS))
        * FACTOR_CORRECCION
    )

    return altura_mm, z_superficie


# =========================================================
# VISUALIZACIÓN MAPA DE PROFUNDIDAD
# =========================================================

def visualizar_depth(depth):
    depth_vis = depth.copy()
    mask = np.isfinite(depth_vis)

    if np.any(mask):
        min_val = np.percentile(depth_vis[mask], 5)
        max_val = np.percentile(depth_vis[mask], 95)

        if max_val > min_val:
            depth_vis = np.clip(depth_vis, min_val, max_val)
            depth_vis = (depth_vis - min_val) / (max_val - min_val)
        else:
            depth_vis = np.zeros_like(depth_vis)
    else:
        depth_vis = np.zeros_like(depth_vis)

    depth_vis = np.nan_to_num(depth_vis)
    depth_vis = (depth_vis * 255).astype(np.uint8)

    depth_color = cv2.applyColorMap(
        depth_vis,
        cv2.COLORMAP_JET
    )

    return depth_color


# =========================================================
# GRÁFICO EN TIEMPO REAL Y GUARDADO
# =========================================================

def hilo_grafico(grafico_path):
    fig, ax = plt.subplots()

    linea, = ax.plot([], [], label="Altura espuma")
    punto, = ax.plot([], [], "ro")
    texto = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Altura espuma [mm]")
    ax.set_title("Altura de espuma en tiempo real")
    ax.grid(True)
    ax.legend()

    def update(frame):
        with lock:
            if len(alturas) == 0:
                return linea,

            t = list(tiempos)
            h = list(alturas)

        linea.set_data(t, h)
        punto.set_data([t[-1]], [h[-1]])
        texto.set_text(f"{h[-1]:.2f} mm")

        ax.set_xlim(max(0, t[-1] - 60), t[-1] + 1)
        ax.set_ylim(0, max(h) * 1.2 if max(h) > 0 else 10)

        return linea,

    animation.FuncAnimation(
        fig,
        update,
        interval=500,
        cache_frame_data=False
    )

    plt.show()

    # Guardar gráfico al cerrar ventana
    with lock:
        if len(alturas) > 0:
            t = list(tiempos)
            h = list(alturas)

            fig_save, ax_save = plt.subplots()

            ax_save.plot(t, h, label="Altura espuma")
            ax_save.scatter(t[-1], h[-1])

            ax_save.set_xlabel("Tiempo [s]")
            ax_save.set_ylabel("Altura espuma [mm]")
            ax_save.set_title("Altura de espuma medida por estereovisión")
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

        else:
            print("\nNo se guardó gráfico porque no hay datos de altura.")


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

    cv2.namedWindow("Camaras Estereo", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mapa de Profundidad", cv2.WINDOW_NORMAL)

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)

    writer.writerow([
        "timestamp",
        "tiempo_s",
        "altura_mm",
        "altura_cm",
        "z_ref_mm",
        "z_superficie_mm"
    ])

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)

    out_w = 960
    out_h = 720

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

    frame_id = 0

    while estado["activo"]:

        t0 = time.time()

        cam_l.issue_software_trigger()
        cam_r.issue_software_trigger()

        img_l = capturar_frame(cam_l)
        img_r = capturar_frame(cam_r)

        if img_l is None or img_r is None:
            continue

        # =================================================
        # CHEQUEO DE RESOLUCIÓN
        # =================================================

        h_img, w_img = img_l.shape[:2]

        if map_l1.shape[:2] != (h_img, w_img):
            print("\nERROR DE RESOLUCIÓN")
            print("Frame shape:", img_l.shape)
            print("map_l1 shape:", map_l1.shape)
            print("El mapa de rectificación no coincide con el frame.")
            break

        # =================================================
        # RECTIFICACIÓN
        # =================================================

        rect_l = cv2.remap(
            img_l,
            map_l1,
            map_l2,
            cv2.INTER_LINEAR
        )

        rect_r = cv2.remap(
            img_r,
            map_r1,
            map_r2,
            cv2.INTER_LINEAR
        )

        if rect_l.max() == 0 or rect_r.max() == 0:
            print("\nADVERTENCIA: rectificación negra.")
            print("Rect L min/max:", rect_l.min(), rect_l.max())
            print("Rect R min/max:", rect_r.min(), rect_r.max())

        # =================================================
        # SELECCIONAR ROI UNA SOLA VEZ
        # =================================================

        if USAR_ROI and not roi_data["seleccionado"]:

            roi = seleccionar_roi_en_vivo(rect_l)

            if roi is not None:
                x_roi, y_roi, w_roi, h_roi = roi

                roi_data["x"] = x_roi
                roi_data["y"] = y_roi
                roi_data["w"] = w_roi
                roi_data["h"] = h_roi
                roi_data["seleccionado"] = True

            else:
                print("No se seleccionó ROI. Se usará imagen completa.")
                USAR_ROI = False

        # =================================================
        # DISPARIDAD Y PROFUNDIDAD
        # =================================================

        disp = calcular_disparidad(rect_l, rect_r)
        depth = disparidad_a_profundidad(disp, Q)

        display_l = rect_l.copy()
        display_r = rect_r.copy()

        # =================================================
        # APLICAR ROI
        # =================================================

        if USAR_ROI and roi_data["seleccionado"]:

            x = roi_data["x"]
            y = roi_data["y"]
            w = roi_data["w"]
            h = roi_data["h"]

            roi_l = rect_l[y:y+h, x:x+w]
            roi_depth = depth[y:y+h, x:x+w]

            gray = cv2.cvtColor(roi_l, cv2.COLOR_BGR2GRAY)

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
            roi_depth = depth
            gray = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)

        texto_altura = "Altura: ---"

        h_val = None
        z_superficie = None

        # =================================================
        # CÁLCULO DE ALTURA
        # =================================================

        if estado["z_ref"] is not None:
            mask = detectar_superficie(gray)

            h_val, z_superficie = estimar_altura(
                roi_depth,
                mask,
                estado["z_ref"]
            )

            if h_val is not None:
                texto_altura = f"Altura: {h_val:.2f} mm"

                if estado["grabando"]:
                    timestamp = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    tiempo_s = time.time() - estado["t_inicio"]

                    with lock:
                        tiempos.append(tiempo_s)
                        alturas.append(h_val)

                    writer.writerow([
                        timestamp,
                        tiempo_s,
                        h_val,
                        h_val / 10.0,
                        estado["z_ref"],
                        z_superficie
                    ])

        # =================================================
        # OVERLAY
        # =================================================

        put_text_outline(
            display_l,
            "Camara 1 rectificada",
            (50, 80),
            1.3,
            (255, 255, 0)
        )

        put_text_outline(
            display_r,
            "Camara 2 rectificada",
            (50, 80),
            1.3,
            (255, 255, 0)
        )

        put_text_outline(
            display_l,
            f"Frame: {frame_id}",
            (50, 150),
            1.0,
            (255, 255, 255)
        )

        put_text_outline(
            display_l,
            texto_altura,
            (50, 220),
            1.3,
            (0, 255, 0)
        )

        if estado["z_ref"] is not None:
            put_text_outline(
                display_l,
                f"Z ref: {estado['z_ref']:.2f} mm",
                (50, 290),
                0.9,
                (200, 200, 200)
            )

        put_text_outline(
            display_l,
            "R: Ref | G: Grabar | S: Stop | Q: Salir",
            (50, 360),
            0.8,
            (200, 200, 200)
        )

        if estado["grabando"]:
            put_text_outline(
                display_l,
                "REC",
                (50, 430),
                1.5,
                (0, 0, 255)
            )

        # =================================================
        # MOSTRAR VENTANAS
        # =================================================

        combined = np.hstack([
            cv2.resize(display_l, (out_w, out_h)),
            cv2.resize(display_r, (out_w, out_h))
        ])

        cv2.imshow("Camaras Estereo", combined)

        depth_color = visualizar_depth(depth)

        cv2.imshow("Mapa de Profundidad", depth_color)

        # =================================================
        # GUARDAR VIDEO SOLO SI ESTÁ GRABANDO
        # =================================================

        if estado["grabando"]:
            video_l.write(cv2.resize(display_l, (out_w, out_h)))
            video_r.write(cv2.resize(display_r, (out_w, out_h)))

        # =================================================
        # TECLAS
        # =================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            mask = detectar_superficie(gray)
            valid = (mask > 0) & np.isfinite(roi_depth)

            if np.sum(valid) > 50:
                estado["z_ref"] = np.nanmedian(roi_depth[valid])
                print(f"Referencia capturada: {estado['z_ref']:.2f} mm")
            else:
                print("No se pudo capturar referencia.")

        elif key == ord("g"):
            estado["grabando"] = True
            estado["t_inicio"] = time.time()
            print("Grabando...")

        elif key == ord("s"):
            estado["grabando"] = False
            print("Stop.")

        elif key == ord("q"):
            break

        frame_id += 1

        dt = time.time() - t0

        if dt < FRAME_TIME:
            time.sleep(FRAME_TIME - dt)

    csv_file.close()
    video_l.release()
    video_r.release()

    cv2.destroyAllWindows()

    cam_l.dispose()
    cam_r.dispose()
    sdk.dispose()


# =========================================================
# MAIN
# =========================================================

def main():

    experimento = input(
        "Nombre de la experiencia "
        "(sin espacios, acentos ni signos de puntuación): "
    ).strip().replace(" ", "_")

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

    t = threading.Thread(
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

    t.start()

    hilo_grafico(grafico_path)

    estado["activo"] = False

    t.join()


if __name__ == "__main__":
    main()