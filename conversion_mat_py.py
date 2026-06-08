import numpy as np
import scipy.io
import cv2

# =========================================================
# CONFIGURACIÓN
# =========================================================

MAT_FILE = r"D:\MDT\Stereovision\stereo_params_matlab.mat"
SALIDA = r"D:\MDT\Stereovision\stereo_calibration.npz"

# MATLAB calibró imágenes con ImageSize = [1440 1080]
# MATLAB: [height width]
# OpenCV usa image_size = (width, height)
# Por eso:
IMAGE_SIZE = (1080, 1440)

# =========================================================
# CARGAR PARÁMETROS DESDE MATLAB
# =========================================================

mat = scipy.io.loadmat(MAT_FILE)

K_l = mat["K_l"].astype(np.float64)
K_r = mat["K_r"].astype(np.float64)

D_l = mat["D_l"].flatten().astype(np.float64)
D_r = mat["D_r"].flatten().astype(np.float64)

R = mat["R"].astype(np.float64)
T = mat["T"].flatten().astype(np.float64)

print("\n===================================")
print("PARÁMETROS CARGADOS DESDE MATLAB")
print("===================================")

print("K_l:\n", K_l)
print("D_l:", D_l)

print("\nK_r:\n", K_r)
print("D_r:", D_r)

print("\nR shape:", R.shape)
print("T:", T)
print("Baseline MATLAB:", np.linalg.norm(T), "mm")

if "imageSize_l" in mat:
    print("\nImageSize MATLAB cámara 1:")
    print(mat["imageSize_l"])

if "imageSize_r" in mat:
    print("\nImageSize MATLAB cámara 2:")
    print(mat["imageSize_r"])

print("\nIMAGE_SIZE usado en OpenCV:", IMAGE_SIZE)

# =========================================================
# RECTIFICACIÓN ESTÉREO
# =========================================================

R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv2.stereoRectify(
    K_l,
    D_l,
    K_r,
    D_r,
    IMAGE_SIZE,
    R,
    T,
    alpha=0
)

# =========================================================
# MAPAS DE RECTIFICACIÓN
# =========================================================

map_l1, map_l2 = cv2.initUndistortRectifyMap(
    K_l,
    D_l,
    R_l,
    P_l,
    IMAGE_SIZE,
    cv2.CV_16SC2
)

map_r1, map_r2 = cv2.initUndistortRectifyMap(
    K_r,
    D_r,
    R_r,
    P_r,
    IMAGE_SIZE,
    cv2.CV_16SC2
)

# =========================================================
# GUARDAR CALIBRACIÓN
# =========================================================

np.savez(
    SALIDA,
    K_l=K_l,
    D_l=D_l,
    K_r=K_r,
    D_r=D_r,
    R=R,
    T=T,
    R_l=R_l,
    R_r=R_r,
    P_l=P_l,
    P_r=P_r,
    Q=Q,
    map_l1=map_l1,
    map_l2=map_l2,
    map_r1=map_r1,
    map_r2=map_r2,
    image_size=np.array(IMAGE_SIZE)
)

print("\n===================================")
print("CALIBRACIÓN CONVERTIDA")
print("===================================")

print("Guardado:", SALIDA)

baseline = np.linalg.norm(T)
factor = 75.0 / baseline

print("Baseline MATLAB:", baseline, "mm")
print("Factor corrección recomendado:", factor)

print("\nMapas generados:")
print("map_l1 shape:", map_l1.shape)
print("map_l2 shape:", map_l2.shape)
print("map_r1 shape:", map_r1.shape)
print("map_r2 shape:", map_r2.shape)

print("\nimage_size guardado:", IMAGE_SIZE)