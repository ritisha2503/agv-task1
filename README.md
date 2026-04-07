# AGV Task 1 - Optical Flow

This repository contains the implementation of optical flow techniques as part of the AGV Software Task.

The goal of this project is to estimate motion between consecutive frames of a video using computer vision techniques.

---

## Overview

Optical flow is a method used to estimate the motion of objects between frames in a video sequence. This project focuses on:

- Lucas-Kanade Optical Flow
- (Bonus) Dense Optical Flow 
- Navigation

The output is visualized as motion overlays on the original video.

---

## Features

1. Lucas-Kanade Optical Flow
- Tracks distinct feature points (corners) instead of all pixels
- Uses pyramidal Lucas–Kanade for robust small-to-moderate motion
- Produces clean motion vectors (dx, dy) for each tracked point
- Includes filtering & subsampling to reduce noise and clutter
- Supports temporal visualization (motion trails) using a mask

2. (Bonus) Dense Optical Flow
- Reads input video and processes frame-by-frame
- Computes dense optical flow using Farneback algorithm
- Visualizes motion using HSV color encoding
- Overlays motion on original frames for better understanding

3. Navigation
- Uses sparse optical flow (Lucas–Kanade) for real-time tracking
- Converts motion into steering decisions based on left/right flow
- Applies distance-aware weighting (closer points influence more)
- Includes robustness checks (re-detect features when tracking fails)
- Integrates with PyBullet physics engine for closed-loop control

---

## Concepts Used

- Optical Flow
- Computer Vision
- Frame Differencing
- HSV Color Space Visualization
- Vector Magnitude and Direction

---

## Project Structure
agv-task1/
│
├── subtask1_lucas_kanade/
│ ├── main.py
│ ├── dense_optical_flow.py
│ └── input_video.mp4
│
├── subtask2_navigation/
│ ├── main.py
│ └── simulation_setup.py
│
├── README.md
└── requirements.txt

---

## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/ritisha2503/agv-task1.git
cd agv-task1

2. Install dependencies:
pip install -r requirements.txt

3. For subtask 1, enter the folder of subtask 1:
cd subtask1_lucas_kanade

4. Run the script:
python main.py

5. Also, check the bonus task:
python dense_optical_flow.py

6. Exit the folder for subtask 1:
cd ..

7. For subtask 2, enter the folder of subtask 2:
cd subtask2_navigation

8. Run the scriptL
python main.py