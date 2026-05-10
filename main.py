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

os.add_dll_directory(r"D:\MDT\Scientific Camera Interfaces\SDK\Native Toolkit\dlls\Native_64_lib")

# PARÁMETROS
CALIB_FILE = "stereo_calibration.npz"

ANGULO_GRADOS = 3.47
EXPOSICION_US = 28712

NUM_DISPARIDADES = 64
BLOCK_SIZE = 11

FACTOR_CORRECCION = 75.0 / 143.46

FPS_OBJETIVO = 5
FRAME_TIME = 1.0 / FPS_OBJETIVO

# BUFFERS
alturas = deque(maxlen=300)
lock = threading.Lock()

estado = {
    "grabando": False,
    "z_ref": None,
    "activo": True
}

# TEXTO
def put_text_outline(img, text, org, scale, color):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 10, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 4, cv2.LINE_AA)

# CALIBRACIÓN
def cargar_calibracion():
    data = np.load(CALIB_FILE)
    print(f"Calibración cargada. Baseline: {np.linalg.norm(data['T']):.2f} mm")
    return data

# CÁMARAS
def inicializar_camaras():
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE

    sdk = TLCameraSDK()
    cams = sdk.discover_available_cameras()
    print("Cámaras detectadas:", cams)

    cam_l = sdk.open_camera(cams[0])
    cam_r = sdk.open_camera(cams[1])

    for cam in [cam_l, cam_r]:
        cam.exposure_time_us = EXPOSICION_US
        cam.frames_per_trigger_zero_for_unlimited = 0
        cam.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGERED
        cam.arm(2)

    return sdk, cam_l, cam_r

# CAPTURA
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

    img = np.array(frame.image_buffer, dtype=np.uint16).reshape(h, w)

    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)

# DISPARIDAD
def calcular_disparidad(img_l, img_r):
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISPARIDADES,
        blockSize=BLOCK_SIZE,
        P1=8 * 3 * BLOCK_SIZE**2,
        P2=32 * 3 * BLOCK_SIZE**2,
    )

    return stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

# PROFUNDIDAD
def disparidad_a_profundidad(disp, Q):
    pts = cv2.reprojectImageTo3D(disp, Q)
    depth = pts[:, :, 2]
    depth[(disp <= 0) | (depth > 2000) | (depth < 100)] = np.nan
    return depth

# DETECCIÓN
def detectar_superficie(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

# ALTURA
def estimar_altura(depth, mask, z_ref):
    valid = (mask > 0) & ~np.isnan(depth)
    if np.sum(valid) < 50:
        return None
    z = np.nanmedian(depth[valid])
    return abs(z_ref - z) * np.cos(np.radians(ANGULO_GRADOS)) * FACTOR_CORRECCION

# GRÁFICO
def hilo_grafico():
    fig, ax = plt.subplots()

    linea, = ax.plot([], [])
    punto, = ax.plot([], [], 'ro')
    texto = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def update(frame):
        with lock:
            if len(alturas) == 0:
                return linea,

            h = list(alturas)
            t = list(range(len(h)))  # ✔ FIX importante

        linea.set_data(t, h)
        punto.set_data([t[-1]], [h[-1]])
        texto.set_text(f"{h[-1]:.2f} mm")

        ax.set_xlim(max(0, t[-1]-60), t[-1]+1)
        ax.set_ylim(0, max(h)*1.2 if max(h)>0 else 10)

        return linea,

    anim = animation.FuncAnimation(fig, update, interval=500, cache_frame_data=False)
    plt.show()

# HILO CÁMARAS
def hilo_camaras(cal, sdk, cam_l, cam_r, csv_path, video_l_path, video_r_path):

    Q = cal['Q']

    cv2.namedWindow("Camaras Estereo", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mapa de Profundidad", cv2.WINDOW_NORMAL)

    # CSV
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp", "altura_mm"])

    # Video writers
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    w, h = 960, 720

    video_l = cv2.VideoWriter(video_l_path, fourcc, FPS_OBJETIVO, (w, h))
    video_r = cv2.VideoWriter(video_r_path, fourcc, FPS_OBJETIVO, (w, h))

    frame_id = 0

    while estado["activo"]:

        t0 = time.time()

        cam_l.issue_software_trigger()
        cam_r.issue_software_trigger()

        img_l = capturar_frame(cam_l)
        img_r = capturar_frame(cam_r)

        if img_l is None or img_r is None:
            continue

        rect_l = img_l
        rect_r = img_r

        disp = calcular_disparidad(rect_l, rect_r)
        depth = disparidad_a_profundidad(disp, Q)

        gray = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)

        display_l = rect_l.copy()
        display_r = rect_r.copy()

        texto_altura = "Altura: ---"

        if estado["z_ref"] is not None:
            mask = detectar_superficie(gray)
            h_val = estimar_altura(depth, mask, estado["z_ref"])

            if h_val is not None:
                texto_altura = f"Altura: {h_val:.2f} mm"

                if estado["grabando"]:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    with lock:
                        alturas.append(h_val)

                    writer.writerow([timestamp, h_val])

        put_text_outline(display_l, "Camara 1", (50,80), 1.6, (255,255,0))
        put_text_outline(display_l, texto_altura, (50,240), 1.6, (0,255,0))

        combined = np.hstack([
            cv2.resize(display_l, (w,h)),
            cv2.resize(display_r, (w,h))
        ])

        cv2.imshow("Camaras Estereo", combined)

        video_l.write(cv2.resize(display_l, (w,h)))
        video_r.write(cv2.resize(display_r, (w,h)))

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            estado["z_ref"] = np.nanmedian(depth)
            print("Referencia capturada")

        elif key == ord('g'):
            estado["grabando"] = True
            print("Grabando")

        elif key == ord('s'):
            estado["grabando"] = False
            print("Stop")

        elif key == ord('q'):
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

# MAIN
def main():

    experimento = input("Nombre de la experiencia(NOTA: Verificar que este bien escrito y que no contenga espacios): ").strip().replace(" ", "_")

    base_dir = os.getcwd()

    csv_path = os.path.join(base_dir, f"{experimento}.csv")
    video_l_path = os.path.join(base_dir, f"{experimento}_cam1.avi")
    video_r_path = os.path.join(base_dir, f"{experimento}_cam2.avi")

    cal = cargar_calibracion()
    sdk, cam_l, cam_r = inicializar_camaras()

    t = threading.Thread(
        target=hilo_camaras,
        args=(cal, sdk, cam_l, cam_r, csv_path, video_l_path, video_r_path),
        daemon=True
    )
    t.start()

    hilo_grafico()

    estado["activo"] = False
    t.join()

if __name__ == "__main__":
    main()