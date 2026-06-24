import cv2
import tkinter as tk
import numpy as np


def get_scale(width, height, window_fraction=0.45):
    root = tk.Tk()
    root.withdraw()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    root.destroy()

    scale_width = (screen_width * window_fraction) / width
    scale_height = (screen_height * 0.85) / height

    return min(scale_width, scale_height, 1.0)


def automatic_canny_from_gradient(
    gray_img,
    low_percentile=70,
    high_percentile=90,
    aperture_size=3,
    use_l2_gradient=True
):
    # Gradiente horizontal
    sobel_x = cv2.Sobel(
        gray_img,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    # Gradiente vertical
    sobel_y = cv2.Sobel(
        gray_img,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    # Magnitud del gradiente
    magnitude = cv2.magnitude(
        sobel_x,
        sobel_y
    )

    # Ignorar gradientes nulos
    valid_gradients = magnitude[magnitude > 0]

    if valid_gradients.size == 0:
        empty_edges = np.zeros_like(gray_img)
        return empty_edges, 0, 1

    # Calcular umbrales mediante percentiles
    threshold_low = int(
        np.percentile(
            valid_gradients,
            low_percentile
        )
    )

    threshold_high = int(
        np.percentile(
            valid_gradients,
            high_percentile
        )
    )

    # Limitar umbrales al rango válido de Canny
    threshold_low = max(
        1,
        min(threshold_low, 254)
    )

    threshold_high = max(
        threshold_low + 1,
        min(threshold_high, 255)
    )

    edges = cv2.Canny(
        gray_img,
        threshold1=threshold_low,
        threshold2=threshold_high,
        apertureSize=aperture_size,
        L2gradient=use_l2_gradient
    )

    return edges, threshold_low, threshold_high


def reproduce_canny_video(
    video_path,
    start_frame=0,

    clahe_clip_limit=1.2,
    clahe_grid_size=(8, 8),
    clahe_brightness_alpha=0.55,
    clahe_brightness_beta=0,

    bilateral_diameter=9,
    bilateral_sigma_color=45,
    bilateral_sigma_space=45,

    low_percentile=70,
    high_percentile=90,
    canny_aperture_size=3,

    use_morph_close=False,
    morph_kernel_size=3
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: no se pudo abrir el video.")
        return

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    if start_frame < 0 or start_frame >= total_frames:
        print(
            "Error: el frame inicial está fuera "
            "del rango del video."
        )
        cap.release()
        return

    # ---------------------------------------------------------
    # Validar parámetros
    # ---------------------------------------------------------

    if canny_aperture_size not in [3, 5, 7]:
        canny_aperture_size = 3

    if low_percentile < 0:
        low_percentile = 0

    if high_percentile > 100:
        high_percentile = 100

    if low_percentile >= high_percentile:
        cap.release()
        return

    if bilateral_diameter < 1:
        bilateral_diameter = 1

    if morph_kernel_size < 1:
        morph_kernel_size = 1

    if morph_kernel_size % 2 == 0:
        morph_kernel_size += 1

    # ---------------------------------------------------------
    # Crear objetos reutilizables
    # ---------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_grid_size
    )

    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (morph_kernel_size, morph_kernel_size)
    )

    # ---------------------------------------------------------
    # Leer primer frame para obtener dimensiones
    # ---------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    ret, first_frame = cap.read()

    if not ret:
        print("Error: no se pudo leer el frame inicial.")
        cap.release()
        return

    height, width = first_frame.shape[:2]

    scale = get_scale(
        width,
        height
    )

    display_width = int(width * scale)
    display_height = int(height * scale)

    # Volver al frame inicial
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    delay = max(
        1,
        int(round(1000 / fps))
    )

    paused = False

    print("--------------------------------------")
    print("Información del video")
    print("--------------------------------------")
    print(f"Resolución: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total de frames: {total_frames}")
    print(f"Frame inicial: {start_frame}")

    print("\nParámetros CLAHE:")
    print(f"Clip limit: {clahe_clip_limit}")
    print(f"Grid size: {clahe_grid_size}")
    print(f"Alpha visual CLAHE: {clahe_brightness_alpha}")

    print("\nFiltro bilateral:")
    print(f"Diámetro: {bilateral_diameter}")
    print(f"Sigma color: {bilateral_sigma_color}")
    print(f"Sigma espacio: {bilateral_sigma_space}")

    print("\nCanny automático:")
    print(f"Percentil inferior: {low_percentile}")
    print(f"Percentil superior: {high_percentile}")
    print(f"Aperture size: {canny_aperture_size}")

    print("\nMorfología:")
    print(f"Cierre activado: {use_morph_close}")
    print(f"Kernel morfológico: {morph_kernel_size}")

    print("\nControles:")
    print("Espacio: pausar o reanudar")
    print("R: reiniciar")
    print("Q o ESC: salir")

    # ---------------------------------------------------------
    # Bucle de reproducción
    # ---------------------------------------------------------

    while True:

        if not paused:

            ret, frame = cap.read()

            if not ret:
                print("Fin del video.")
                break

            current_frame = int(
                cap.get(cv2.CAP_PROP_POS_FRAMES)
            ) - 1

            # -------------------------------------------------
            # 1. Escala de grises
            # -------------------------------------------------

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # -------------------------------------------------
            # 2. CLAHE sobre imagen gris original
            # -------------------------------------------------

            gray_clahe = clahe.apply(gray)

            # -------------------------------------------------
            # 3. Oscurecer CLAHE solo para mostrar
            # -------------------------------------------------

            gray_clahe_display = cv2.convertScaleAbs(
                gray_clahe,
                alpha=clahe_brightness_alpha,
                beta=clahe_brightness_beta
            )

            # -------------------------------------------------
            # 4. Filtro bilateral para reducir ruido
            # -------------------------------------------------

            gray_filtered = cv2.bilateralFilter(
                gray_clahe,
                d=bilateral_diameter,
                sigmaColor=bilateral_sigma_color,
                sigmaSpace=bilateral_sigma_space
            )

            # -------------------------------------------------
            # 5. Canny automático basado en gradiente
            # -------------------------------------------------

            edges, auto_low, auto_high = (
                automatic_canny_from_gradient(
                    gray_filtered,
                    low_percentile=low_percentile,
                    high_percentile=high_percentile,
                    aperture_size=canny_aperture_size,
                    use_l2_gradient=True
                )
            )

            # -------------------------------------------------
            # 6. Cierre morfológico opcional
            # -------------------------------------------------

            if use_morph_close:
                edges = cv2.morphologyEx(
                    edges,
                    cv2.MORPH_CLOSE,
                    morph_kernel
                )

            # -------------------------------------------------
            # 7. Redimensionar para mostrar
            # -------------------------------------------------

            frame_small = cv2.resize(
                frame,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA
            )

            clahe_small = cv2.resize(
                gray_clahe_display,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA
            )

            edges_small = cv2.resize(
                edges,
                (display_width, display_height),
                interpolation=cv2.INTER_NEAREST
            )

            # -------------------------------------------------
            # 8. Agregar textos
            # -------------------------------------------------

            cv2.putText(
                frame_small,
                f"Frame: {current_frame}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                clahe_small,
                f"Frame: {current_frame}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                255,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                edges_small,
                f"Frame: {current_frame}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                255,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                edges_small,
                f"Canny: {auto_low}-{auto_high}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                255,
                2,
                cv2.LINE_AA
            )

            # -------------------------------------------------
            # 9. Mostrar ventanas
            # -------------------------------------------------

            cv2.imshow(
                "Video original",
                frame_small
            )

            cv2.imshow(
                (
                    "Video CLAHE oscurecido - "
                    f"alpha={clahe_brightness_alpha}"
                ),
                clahe_small
            )

            cv2.imshow(
                "Video Canny automatico por gradiente",
                edges_small
            )

        key = cv2.waitKey(
            delay if not paused else 30
        ) & 0xFF

        # Salir
        if key == ord("q") or key == 27:
            break

        # Pausar o reanudar
        elif key == ord(" "):

            paused = not paused

            if paused:
                print("Video pausado.")
            else:
                print("Video reanudado.")

        # Reiniciar
        elif key == ord("r"):

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                start_frame
            )

            paused = False

            print(
                f"Video reiniciado desde "
                f"el frame {start_frame}."
            )

    cap.release()
    cv2.destroyAllWindows()


# =============================================================
# CONFIGURACIÓN
# =============================================================

VIDEO_PATH = r"D:\MDT\Stereovision\II38P_cam1.mp4"

START_FRAME = 19000

# -------------------------------------------------------------
# CLAHE
# -------------------------------------------------------------

CLAHE_CLIP_LIMIT = 1.2
CLAHE_GRID_SIZE = (8, 8)

# Solo modifica cómo se muestra la ventana CLAHE.
CLAHE_BRIGHTNESS_ALPHA = 0.45
CLAHE_BRIGHTNESS_BETA = 0

# -------------------------------------------------------------
# Filtro bilateral
# -------------------------------------------------------------

BILATERAL_DIAMETER = 9
BILATERAL_SIGMA_COLOR = 45
BILATERAL_SIGMA_SPACE = 45

# -------------------------------------------------------------
# Canny automático por gradiente
# -------------------------------------------------------------

LOW_PERCENTILE = 70
HIGH_PERCENTILE = 90

CANNY_APERTURE_SIZE = 3

# -------------------------------------------------------------
# Cierre morfológico
# -------------------------------------------------------------

USE_MORPH_CLOSE = False
MORPH_KERNEL_SIZE = 3


# =============================================================
# EJECUCIÓN
# =============================================================

reproduce_canny_video(
    video_path=VIDEO_PATH,
    start_frame=START_FRAME,

    clahe_clip_limit=CLAHE_CLIP_LIMIT,
    clahe_grid_size=CLAHE_GRID_SIZE,
    clahe_brightness_alpha=CLAHE_BRIGHTNESS_ALPHA,
    clahe_brightness_beta=CLAHE_BRIGHTNESS_BETA,

    bilateral_diameter=BILATERAL_DIAMETER,
    bilateral_sigma_color=BILATERAL_SIGMA_COLOR,
    bilateral_sigma_space=BILATERAL_SIGMA_SPACE,

    low_percentile=LOW_PERCENTILE,
    high_percentile=HIGH_PERCENTILE,
    canny_aperture_size=CANNY_APERTURE_SIZE,

    use_morph_close=USE_MORPH_CLOSE,
    morph_kernel_size=MORPH_KERNEL_SIZE
)