import cv2
import numpy as np
import glob
import os

# ─── PARÁMETROS DEL TABLERO ───────────────────────────────────────────────────
BOARD_COLS    = 7      # esquinas internas horizontales
BOARD_ROWS    = 5      # esquinas internas verticales
SQUARE_SIZE   = 5.0    # tamaño del cuadrado en mm

# ─── RUTAS ────────────────────────────────────────────────────────────────────
LEFT_DIR  = "D:/MDT/Calibracion/Calibracion2/1"
RIGHT_DIR = "D:/MDT/Calibracion/Calibracion2/2"
OUTPUT    = "stereo_calibration1.npz"

# ─── PREPARAR PUNTOS 3D DEL TABLERO ───────────────────────────────────────────
objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
objp *= SQUARE_SIZE

obj_points = []
pts_left   = []
pts_right  = []

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Flags de detección más permisivos
detect_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_NORMALIZE_IMAGE +
                cv2.CALIB_CB_FILTER_QUADS)

# ─── FUNCIÓN DE PREPROCESAMIENTO ──────────────────────────────────────────────
def preprocesar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Ecualización CLAHE (más suave que equalizeHist global)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray

# ─── CARGAR Y PROCESAR PARES ──────────────────────────────────────────────────
left_images  = sorted(glob.glob(os.path.join(LEFT_DIR,  "*.jpg")))
right_images = sorted(glob.glob(os.path.join(RIGHT_DIR, "*.jpg")))

if len(left_images) != len(right_images):
    print(f"Cantidad de imágenes no coincide: {len(left_images)} izq, {len(right_images)} der")
    exit()

print(f"Pares encontrados: {len(left_images)}")

pares_validos = 0
image_size    = None

for i, (l_path, r_path) in enumerate(zip(left_images, right_images)):
    img_l = cv2.imread(l_path)
    img_r = cv2.imread(r_path)

    gray_l = preprocesar(img_l)
    gray_r = preprocesar(img_r)

    if image_size is None:
        image_size = gray_l.shape[::-1]

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (BOARD_COLS, BOARD_ROWS), detect_flags)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, (BOARD_COLS, BOARD_ROWS), detect_flags)

    if ret_l and ret_r:
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        obj_points.append(objp)
        pts_left.append(corners_l)
        pts_right.append(corners_r)
        pares_validos += 1
        print(f"Par {i+1:02d}: {os.path.basename(l_path)} — detectado")

        # Mostrar detección visualmente
        vis_l = cv2.drawChessboardCorners(img_l.copy(), (BOARD_COLS, BOARD_ROWS), corners_l, ret_l)
        vis_r = cv2.drawChessboardCorners(img_r.copy(), (BOARD_COLS, BOARD_ROWS), corners_r, ret_r)
        combined = np.hstack([
            cv2.resize(vis_l, (640, 480)),
            cv2.resize(vis_r, (640, 480))
        ])
        cv2.imshow(f"Par {i+1} - Presiona tecla para continuar", combined)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    else:
        print(f"Par {i+1:02d}: {os.path.basename(l_path)} — RECHAZADO "
              f"(izq={'OK' if ret_l else 'FALLO'}, der={'OK' if ret_r else 'FALLO'})")

print(f"\nPares válidos: {pares_validos} / {len(left_images)}")

if pares_validos < 10:
    print("Se recomiendan al menos 10 pares válidos.")
if pares_validos < 4:
    print("Insuficiente para calibrar.")
    exit()

# ─── CALIBRACIÓN INDIVIDUAL ───────────────────────────────────────────────────
print("\nCalibrando cámara izquierda...")
ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(obj_points, pts_left, image_size, None, None)
print(f"Error de reproyección: {ret_l:.4f} px")

print("Calibrando cámara derecha...")
ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(obj_points, pts_right, image_size, None, None)
print(f"Error de reproyección: {ret_r:.4f} px")

# ─── CALIBRACIÓN ESTÉREO ──────────────────────────────────────────────────────
print("\nCalibrando sistema estéreo...")

ret_stereo, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
    obj_points, pts_left, pts_right,
    K_l, D_l, K_r, D_r,
    image_size,
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC
)

print("T:", T)
print("Norma T (baseline):", np.linalg.norm(T))

print("R:", R)

print(f"Error de reproyección estéreo: {ret_stereo:.4f} px")
if ret_stereo < 1.0:
    print("Calibración buena")
elif ret_stereo < 2.0:
    print("Calibración aceptable")
else:
    print("Error alto, necesitas más pares o mejor tablero")

# ─── RECTIFICACIÓN ────────────────────────────────────────────────────────────
print("\nCalculando mapas de rectificación...")

R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv2.stereoRectify(
    K_l, D_l, K_r, D_r,
    image_size, R, T,
    alpha=0
)

map_l1, map_l2 = cv2.initUndistortRectifyMap(K_l, D_l, R_l, P_l, image_size, cv2.CV_16SC2)
map_r1, map_r2 = cv2.initUndistortRectifyMap(K_r, D_r, R_r, P_r, image_size, cv2.CV_16SC2)

# ─── GUARDAR PARÁMETROS ───────────────────────────────────────────────────────
np.savez(OUTPUT,
    K_l=K_l, D_l=D_l,
    K_r=K_r, D_r=D_r,
    R=R, T=T, E=E, F=F,
    R_l=R_l, R_r=R_r,
    P_l=P_l, P_r=P_r,
    Q=Q,
    map_l1=map_l1, map_l2=map_l2,
    map_r1=map_r1, map_r2=map_r2,
    image_size=np.array(image_size)
)

print(f"\nParámetros guardados en: {OUTPUT}")
print(f"  Baseline (distancia entre cámaras): {np.linalg.norm(T):.2f} mm")

# ─── VERIFICACIÓN VISUAL ──────────────────────────────────────────────────────
print("\nVerificación visual de rectificación (presiona tecla para cerrar)...")

img_l = cv2.imread(left_images[0])
img_r = cv2.imread(right_images[0])

rect_l = cv2.remap(img_l, map_l1, map_l2, cv2.INTER_LINEAR)
rect_r = cv2.remap(img_r, map_r1, map_r2, cv2.INTER_LINEAR)

combined = np.hstack([
    cv2.resize(rect_l, (640, 480)),
    cv2.resize(rect_r, (640, 480))
])
for y in range(0, combined.shape[0], 40):
    cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

cv2.imshow("Rectificacion - las lineas verdes deben alinear en ambas imagenes", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()