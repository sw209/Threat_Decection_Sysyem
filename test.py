import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        print(i)

    cap.release()