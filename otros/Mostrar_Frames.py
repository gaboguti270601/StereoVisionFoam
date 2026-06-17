import cv2
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np

def get_scale(width, height):
    """Calcula la escala para ajustar la imagen a la pantalla."""
    root = tk.Tk()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.destroy()
    escala = min(sw / width, sh / height)
    return escala if escala < 1 else 1

def sobel_edge_detection(gray_img, ksize=3, threshold=None):
    """
    Detecta bordes usando el operador Sobel.
    Retorna la imagen de magnitud del gradiente (uint8, rango 0-255).
    Si se da un threshold, aplica umbral binario.
    """
    # Calcular gradientes en X e Y
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Magnitud del gradiente
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    # Normalizar a 0-255 y convertir a uint8
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if threshold is not None:
        # Aplicar umbral: píxeles mayores a threshold se convierten en 255, otros en 0
        _, magnitude = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
    
    return magnitude

def test(ruta_video, frame_no, sobel_ksize=3, sobel_thresh=None):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return

    # Saltar al frame deseado
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    if not ret:
        print(f"No se pudo leer el frame {frame_no}")
        cap.release()
        return

    # Redimensionar para mostrar
    h, w = frame.shape[:2]
    escala = get_scale(w, h)
    frame_small = cv2.resize(frame, (int(w * escala), int(h * escala)))
    cv2.imshow(f'Frame original {frame_no}', frame_small)

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar detección de bordes con Sobel
    edges = sobel_edge_detection(gray, ksize=sobel_ksize, threshold=sobel_thresh)
    
    # Redimensionar el resultado para mostrar
    edges_small = cv2.resize(edges, (int(w * escala), int(h * escala)))
    cv2.imshow(f'Bordes Sobel {frame_no}', edges_small)
    
    # Calcular histogramas
    hist_original = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_edges = cv2.calcHist([edges], [0], None, [256], [0, 256])
    
    # Mostrar histogramas comparativos
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Histograma - Imagen original (gris)")
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.plot(hist_original, color='black')
    plt.xlim([0, 256])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.title("Histograma - Bordes Sobel")
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.plot(hist_edges, color='black')
    plt.xlim([0, 256])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    print("Presiona cualquier tecla sobre las ventanas de OpenCV para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

# Ruta del video
VIDEO_PATH = r'D:\MDT\Pruebas\II28P\1\1_2026-04-28T13-20-01.875.avi'

# Parámetros ajustables:
#   sobel_ksize  : tamaño del kernel (3, 5, 7... normalmente 3)
#   sobel_thresh : si se proporciona, umbraliza para obtener bordes binarios.
#                  Ejemplo: 50. Déjalo como None para ver la magnitud continua.
test(VIDEO_PATH, 10000, sobel_ksize=3, sobel_thresh=None)