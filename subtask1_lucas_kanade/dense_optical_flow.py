import cv2
import numpy as np

video_path = "../data/input_video.mp4"

cap = cv2.VideoCapture(video_path)

ret, frame1 = cap.read()
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:

    ret, frame2 = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    hsv = np.zeros_like(frame2)
    hsv[...,1] = 255

    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    overlay = cv2.addWeighted(frame2, 0.7, flow_rgb, 0.3, 0)
    cv2.imshow("Dense Optical Flow", overlay)

    if cv2.waitKey(30) & 0xff == 27:
        break

    prev_gray = gray

cap.release()
cv2.destroyAllWindows()