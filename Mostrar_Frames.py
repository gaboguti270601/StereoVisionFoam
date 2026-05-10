import cv2
import tkinter as tk

def get_scale(width, height):
    root = tk.Tk()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.destroy()
    escala = min(sw / width, sh / height)
    return escala if escala < 1 else 1

def test(ruta_video, frame_no):
    cap = cv2.VideoCapture(ruta_video)

    # Saltar al frame deseado
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()

    h, w = frame.shape[:2]
    escala = get_scale(w, h)
    frame_small = cv2.resize(frame, (int(w * escala), int(h * escala)))

    cv2.imshow(f'Frame {frame_no}', frame_small)  # Mostrar el redimensionado
    print(f"Mostrando frame {frame_no}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

VIDEO_PATH = r'D:\MDT\Pruebas\II28P\1\1_2026-04-28T13-20-01.875.avi'
test(VIDEO_PATH, 71087)