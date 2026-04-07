import cv2
import numpy as np

def detect_features(gray_frame):

    points = cv2.goodFeaturesToTrack(
        gray_frame,
        mask = None,
        maxCorners = 400,
        qualityLevel = 0.3,
        minDistance = 7,
        blockSize = 7
    )

    return points

def compute_optical_flow(prev_gray, curr_gray, prev_points):

    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_points,
        None,
        winSize = (15,15),
        maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]

    return good_old, good_new


def draw_flow(frame, mask, old_points, new_points):

    for i, (new, old) in enumerate(zip(new_points, old_points)):

        # subsample vectors to avoid clutter
        if i % 3 != 0:
            continue

        a, b = new.ravel()
        c, d = old.ravel()

        dx = a - c
        dy = b - d

        magnitude = np.sqrt(dx**2 + dy**2)

        # normalize magnitude for coloring
        mag_norm = min((magnitude / 10)**0.5, 1.0)

        # color mapping
        # slow → yellow (0,255,255)
        # fast → green (0,255,0)
        color = (
            0,
            int(255),
            int(255 * (1 - mag_norm))
        )

        scale = 5
        end_x = int(a + dx * scale)
        end_y = int(b + dy * scale)

        mask = cv2.arrowedLine(
            mask,
            (int(a), int(b)),
            (end_x, end_y),
            color,
            2,
            tipLength=0.3
        )

        frame = cv2.circle(
            frame,
            (int(a), int(b)),
            3,
            (0,0,255),
            -1
        )

    output = cv2.add(frame, mask)

    return output, mask

video_path = "input_video.mp4"

cap = cv2.VideoCapture(video_path)

ret, first_frame = cap.read()
if not ret:
    print("Error reading video")
    exit()

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# detect initial feature points
points = detect_features(prev_gray)

# create mask for motion trails
mask = np.zeros_like(first_frame)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    old_points, new_points = compute_optical_flow(
        prev_gray,
        gray,
        points
    )

    # remove noisy vectors
    filtered_old = []
    filtered_new = []

    for new, old in zip(new_points, old_points):

        dx = new[0] - old[0]
        dy = new[1] - old[1]

        motion = np.sqrt(dx**2 + dy**2)

        # keep only meaningful motion
        if 2 < motion < 50:
            filtered_old.append(old)
            filtered_new.append(new)

    if len(filtered_old) == 0:
        points = detect_features(gray)
        prev_gray = gray.copy()
        continue

    filtered_old = np.array(filtered_old)
    filtered_new = np.array(filtered_new)

    # fade previous trails slightly
    mask = cv2.addWeighted(mask, 0.9, np.zeros_like(mask), 0.1, 0)

    output, mask = draw_flow(
        frame,
        mask,
        filtered_old,
        filtered_new
    )

    cv2.imshow("Lucas-Kanade Optical Flow", output)

    if cv2.waitKey(30) & 0xff == 27:
        break

    prev_gray = gray.copy()
    points = filtered_new.reshape(-1,1,2)

    # re-detect features if too few remain
    if len(points) < 50:
        points = detect_features(gray)

cap.release()
cv2.destroyAllWindows()