import sys
import cv2
import numpy as np
import pybullet as p
import time
from simulation_setup import setup_simulation

# constants
WIDTH, HEIGHT = 320, 240
MAX_VELOCITY = 15.0
STEER_GAIN = 0.003
MIN_FEATURES = 20

LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

FEATURE_PARAMS = dict(maxCorners=150, qualityLevel=0.2, minDistance=5, blockSize=7)


def get_frame(car_id):
    car_pos, car_orn = p.getBasePositionAndOrientation(car_id)
    rot_mat = p.getMatrixFromQuaternion(car_orn)

    forward = [rot_mat[0], rot_mat[3], rot_mat[6]]
    up = [0, 0, 1]

    cam_pos = [
        car_pos[0] + forward[0]*0.4,
        car_pos[1] + forward[1]*0.4,
        car_pos[2] + 0.3
    ]

    target = [
        cam_pos[0] + forward[0],
        cam_pos[1] + forward[1],
        cam_pos[2]
    ]

    view_mat = p.computeViewMatrix(cam_pos, target, up)
    proj_mat = p.computeProjectionMatrixFOV(60, WIDTH/HEIGHT, 0.1, 100.0)

    _, _, rgb, _, _ = p.getCameraImage(
        WIDTH, HEIGHT, view_mat, proj_mat, renderer=p.ER_TINY_RENDERER
    )

    frame = np.reshape(rgb, (HEIGHT, WIDTH, 4))[:, :, :3]
    frame = frame.astype(np.uint8)
    frame = np.ascontiguousarray(frame)

    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

try:
    car_id, steer_joints, motor_joints = setup_simulation(gui=True)
except Exception as e:
    print(f"Setup failed: {e}")
    sys.exit()

prev_frame = get_frame(car_id)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)

print("Navigation started. Press 'q' to exit.")

try:
    while True:

        p.stepSimulation()

        frame = get_frame(car_id)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        steer_val = 0.0

        # re-detect if tracking is lost or too few features
        if p0 is None or len(p0) < MIN_FEATURES:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **FEATURE_PARAMS)
            prev_gray = frame_gray.copy()
            continue

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, frame_gray, p0, None, **LK_PARAMS
        )

        if p1 is None:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **FEATURE_PARAMS)
            prev_gray = frame_gray.copy()
            continue

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) == 0:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **FEATURE_PARAMS)
            prev_gray = frame_gray.copy()
            continue

        # ==============================
        # FLOW-BASED STEERING
        # ==============================
        for (new, old) in zip(good_new, good_old):

            x, y = new.ravel()
            dx = x - old.ravel()[0]

            # weight by closeness (bottom = more important)
            weight = (y / HEIGHT) ** 2

            if x < WIDTH / 2:
                steer_val += abs(dx) * weight
            else:
                steer_val -= abs(dx) * weight

            # draw flow
            cv2.line(frame,
                    (int(old.ravel()[0]), int(old.ravel()[1])),
                    (int(x), int(y)),
                    (0, 255, 0), 1)

        # ==============================
        # UPDATE TRACKING
        # ==============================
        if len(good_new) < MIN_FEATURES:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **FEATURE_PARAMS)
        else:
            p0 = good_new.reshape(-1, 1, 2)

        # ==============================
        # CONTROL
        # ==============================
        target_steer = np.clip(steer_val * STEER_GAIN, -0.6, 0.6)

        for sj in steer_joints:
            p.setJointMotorControl2(
                car_id, sj,
                p.POSITION_CONTROL,
                targetPosition=target_steer
            )

        for mj in motor_joints:
            p.setJointMotorControl2(
                car_id, mj,
                p.VELOCITY_CONTROL,
                targetVelocity=MAX_VELOCITY,
                force=500
            )

        # ==============================
        # DISPLAY
        # ==============================
        cv2.putText(frame, f"Steer: {target_steer:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Optical Flow Navigation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = frame_gray.copy()

        time.sleep(1./60.)

except Exception as e:
    print(f"Runtime error: {e}")

p.disconnect()
cv2.destroyAllWindows()