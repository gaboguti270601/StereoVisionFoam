import cv2

cap = cv2.VideoCapture("D:/MDT/Pruebas/II29P/1/1_2026-04-28T17-54-23.672.avi")

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Total frames:", frames)

cap.release()