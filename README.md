# AGV Task 1 - Optical Flow

This repository contains the implementation of optical flow techniques as part of the AGV Software Task.

The goal of this project is to estimate motion between consecutive frames of a video using computer vision techniques.

---

## Overview

Optical flow is a method used to estimate the motion of objects between frames in a video sequence. This project focuses on:

- Dense Optical Flow (Farneback Method)
- (Optional) Lucas-Kanade Optical Flow

The output is visualized as motion overlays on the original video.

---

## Features

- Reads input video and processes frame-by-frame
- Computes dense optical flow using Farneback algorithm
- Visualizes motion using HSV color encoding
- Overlays motion on original frames for better understanding

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
│ ├── dense_optical_flow.py
│ └── (other scripts)
│
├── input_video.mp4
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

3. Run the script:
python dense_optical_flow.py