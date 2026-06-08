import cv2
import numpy as np
import os

# =========================================================
# CONFIGURACIÓN
# =========================================================

VIDEO_L = r"D:\MDT\Stereovision\II35P_cam1.mp4"
VIDEO_R = r"D:\MDT\Stereovision\II35P_cam2.mp4"

CALIB_FILE = r"D:\MDT\Stereovision\stereo_calibration.npz"

FRAME_TEST = 100

# Probar rotaciones
ROTACIONES = {
    "sin_rotar": None,
    "clockwise": cv2.ROTATE_90_CLOCKWISE,
    "counterclockwise": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180
}

# =========================================================
# FUNCIONES
# =========================================================

def leer_frame(video_path, frame_no):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nVideo: {os.path.basename(video_path)}")
    print("Total frames:", total)
    print("FPS:", fps)

    frame_no = min(frame_no, total - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    ret, frame = cap.read()

    cap.release()

    if not ret:
        raise RuntimeError("No se pudo leer frame")

    return frame


def stats(nombre, img):
    print(f"\n{nombre}")
    print("shape:", img.shape)
    print("min:", img.min())
    print("max:", img.max())
    print("mean:", np.mean(img))


def redimensionar(img, ancho_max=700, alto_max=700):
    h, w = img.shape[:2]
    escala = min(ancho_max / w, alto_max / h)

    if escala < 1:
        return cv2.resize(img, (int(w * escala), int(h * escala)))

    return img


# =========================================================
# MAIN
# =========================================================

cal = np.load(CALIB_FILE)

map_l1 = cal["map_l1"]
map_l2 = cal["map_l2"]
map_r1 = cal["map_r1"]
map_r2 = cal["map_r2"]

print("\n===================================")
print("CALIBRACIÓN")
print("===================================")
print("map_l1 shape:", map_l1.shape)
print("map_l2 shape:", map_l2.shape)
print("map_r1 shape:", map_r1.shape)
print("map_r2 shape:", map_r2.shape)

if "image_size" in cal:
    print("image_size:", cal["image_size"])

frame_l = leer_frame(VIDEO_L, FRAME_TEST)
frame_r = leer_frame(VIDEO_R, FRAME_TEST)

stats("Frame L crudo", frame_l)
stats("Frame R crudo", frame_r)

mejor = None

for nombre, rot in ROTACIONES.items():

    print("\n===================================")
    print("PROBANDO:", nombre)
    print("===================================")

    if rot is None:
        img_l = frame_l.copy()
        img_r = frame_r.copy()
    else:
        img_l = cv2.rotate(frame_l, rot)
        img_r = cv2.rotate(frame_r, rot)

    stats("L después rotación", img_l)
    stats("R después rotación", img_r)

    if map_l1.shape[:2] != img_l.shape[:2]:
        print("No coincide resolución con mapas.")
        print("map_l1:", map_l1.shape[:2])
        print("img_l:", img_l.shape[:2])
        continue

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

    stats("L rectificada", rect_l)
    stats("R rectificada", rect_r)

    score = np.mean(rect_l) + np.mean(rect_r) + rect_l.max() + rect_r.max()

    print("Score:", score)

    if mejor is None or score > mejor["score"]:
        mejor = {
            "nombre": nombre,
            "score": score,
            "rect_l": rect_l,
            "rect_r": rect_r,
            "img_l": img_l,
            "img_r": img_r
        }

    cv2.imshow(f"L cruda/rotada - {nombre}", redimensionar(img_l))
    cv2.imshow(f"R cruda/rotada - {nombre}", redimensionar(img_r))
    cv2.imshow(f"L rectificada - {nombre}", redimensionar(rect_l))
    cv2.imshow(f"R rectificada - {nombre}", redimensionar(rect_r))

    print("Presiona una tecla para probar la siguiente rotación...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\n===================================")
print("MEJOR OPCIÓN")
print("===================================")

if mejor is not None:
    print("Mejor rotación:", mejor["nombre"])
    print("Score:", mejor["score"])

    cv2.imshow("Mejor L cruda/rotada", redimensionar(mejor["img_l"]))
    cv2.imshow("Mejor R cruda/rotada", redimensionar(mejor["img_r"]))
    cv2.imshow("Mejor L rectificada", redimensionar(mejor["rect_l"]))
    cv2.imshow("Mejor R rectificada", redimensionar(mejor["rect_r"]))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Ninguna rotación coincidió con los mapas.")