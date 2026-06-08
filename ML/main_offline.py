import cv2
import numpy as np
import os
from glob import glob

# =========================
# CONFIG
# =========================

CALIB_FILE = "stereo_calibration.npz"

LEFT_DIR  = r"D:\MDT\Pruebas\I28P\1"
RIGHT_DIR = r"D:\MDT\Pruebas\I28P\2"

ANGULO_GRADOS = 3.47
FACTOR_CORRECCION = 75.0 / 143.46

NUM_DISPARIDADES = 64
BLOCK_SIZE = 11

MAX_FRAMES = 90000
SKIP_EVERY = 10

# =========================
# CALIBRACIÓN
# =========================

def cargar_calibracion():
    data = np.load(CALIB_FILE)
    baseline = np.linalg.norm(data["T"])
    print("Baseline:", baseline)
    return data

# =========================
# TIMESTAMP
# =========================

def extraer_segundos(nombre):
    ts = nombre.split("_", 1)[1].replace(".avi", "")
    date_part, time_part = ts.split("T")
    h, m, s = time_part.split("-")
    return int(h)*3600 + int(m)*60 + float(s)

# =========================
# MATCHING ROBUSTO
# =========================

def emparejar_videos(left_files, right_files, tol=3.0):

    pares = []
    usados = set()

    right_data = []
    for r in right_files:
        right_data.append((extraer_segundos(os.path.basename(r)), r))

    for l in left_files:

        t_l = extraer_segundos(os.path.basename(l))

        best = None
        best_diff = 1e9
        best_idx = -1

        for i, (t_r, r_path) in enumerate(right_data):

            if i in usados:
                continue

            diff = abs(t_l - t_r)

            if diff < best_diff and diff <= tol:
                best_diff = diff
                best = r_path
                best_idx = i

        if best is not None:
            pares.append((l, best))
            usados.add(best_idx)

    return pares

# =========================
# DISPARIDAD (OPTIMIZADO)
# =========================

def crear_stereo():

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISPARIDADES,
        blockSize=BLOCK_SIZE,
        P1=8 * 3 * BLOCK_SIZE**2,
        P2=32 * 3 * BLOCK_SIZE**2,
        uniquenessRatio=10,
        speckleWindowSize=80,
        speckleRange=32
    )

    return stereo

def calcular_disparidad(stereo, img_l, img_r):

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    disp = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

    if np.std(disp) < 0.5:
        return None

    return disp

# =========================
# PROFUNDIDAD
# =========================

def depth_from_disp(disp, Q):

    pts = cv2.reprojectImageTo3D(disp, Q)
    depth = pts[:, :, 2]

    depth[(disp <= 0) | (depth < 0) | (depth > 5000)] = np.nan

    return depth

# =========================
# SUPERFICIE
# =========================

def detectar_superficie(gray):

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask

# =========================
# ALTURA
# =========================

def estimar_altura(depth, mask, z_ref):

    valid = (mask > 0) & np.isfinite(depth)

    if np.count_nonzero(valid) < 50:
        return None

    z = np.nanmedian(depth[valid])

    if np.isnan(z):
        return None

    return abs(z_ref - z) * np.cos(np.radians(ANGULO_GRADOS)) * FACTOR_CORRECCION

# =========================
# MAIN
# =========================

def main():

    cal = cargar_calibracion()
    Q = cal["Q"]

    left_files  = sorted(glob(os.path.join(LEFT_DIR, "*.avi")))
    right_files = sorted(glob(os.path.join(RIGHT_DIR, "*.avi")))

    print("Videos L:", len(left_files), "R:", len(right_files))

    pares = emparejar_videos(left_files, right_files, tol=3.0)

    print("Pares sincronizados:", len(pares))

    stereo = crear_stereo()

    resultados = []

    for i, (l_path, r_path) in enumerate(pares):

        print("\nProcesando:", os.path.basename(l_path))

        cap_l = cv2.VideoCapture(l_path)
        cap_r = cv2.VideoCapture(r_path)

        frame_id = 0
        z_ref = None
        ref_ok = False

        while True:

            ret_l, img_l = cap_l.read()
            ret_r, img_r = cap_r.read()

            if not ret_l or not ret_r:
                break

            if frame_id > MAX_FRAMES:
                print("Límite de frames alcanzado")
                break

            if frame_id % SKIP_EVERY != 0:
                frame_id += 1
                continue

            disp = calcular_disparidad(stereo, img_l, img_r)

            if disp is None:
                frame_id += 1
                continue

            depth = depth_from_disp(disp, Q)

            gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            mask = detectar_superficie(gray)

            # Z0 automático
            if not ref_ok:
                valid = (mask > 0) & np.isfinite(depth)
                if np.count_nonzero(valid) > 100:
                    z_ref = np.nanmedian(depth[valid])
                    ref_ok = True
                    print("Z0:", z_ref)

            if ref_ok:
                h = estimar_altura(depth, mask, z_ref)
                if h is not None:
                    resultados.append((i, frame_id, h))

            if frame_id % 200 == 0:
                print(f"Video {i} Frame {frame_id}")

            frame_id += 1

        cap_l.release()
        cap_r.release()

    print("\n=== RESULTADOS ===")

    if len(resultados) > 0:
        alturas = [r[2] for r in resultados]

        print("Frames válidos:", len(resultados))
        print("Promedio:", np.mean(alturas))
        print("Std:", np.std(alturas))
    else:
        print("Sin datos válidos")

    np.save("resultados_altura.npy", np.array(resultados))
    print("Guardado: resultados_altura.npy")


if __name__ == "__main__":
    main()