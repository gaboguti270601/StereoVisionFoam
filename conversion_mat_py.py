import numpy as np
import scipy.io
import cv2

MAT_FILE = "stereo_params_matlab.mat"
SALIDA = "stereo_calibration.npz"

mat = scipy.io.loadmat(MAT_FILE)

K_l = mat["K_l"]
D_l = mat["D_l"].flatten()
K_r = mat["K_r"]
D_r = mat["D_r"].flatten()
R = mat["R"]
T = mat["T"].flatten()

# IMPORTANTE:
# OpenCV usa image_size = (width, height)
image_size = (1080, 1440)  # usa esto si tus frames son shape (1440,1080)

R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv2.stereoRectify(
    K_l, D_l,
    K_r, D_r,
    image_size,
    R, T,
    alpha=0
)

map_l1, map_l2 = cv2.initUndistortRectifyMap(
    K_l, D_l, R_l, P_l, image_size, cv2.CV_16SC2
)

map_r1, map_r2 = cv2.initUndistortRectifyMap(
    K_r, D_r, R_r, P_r, image_size, cv2.CV_16SC2
)

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
    image_size=np.array(image_size)
)

print("Guardado:", SALIDA)
print("Baseline MATLAB:", np.linalg.norm(T), "mm")
print("Factor corrección recomendado:", 75.0 / np.linalg.norm(T))