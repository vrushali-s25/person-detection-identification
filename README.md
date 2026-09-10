# Person Detection and Identification System

## Project Overview

This project is a Computer Vision based Person Detection and Identification System.

The system uses YOLO for person detection and InsightFace for face detection and identification.

It can process images, videos, and webcam input and can generate detection results in CSV format.

## Technologies Used

- Python
- OpenCV
- YOLO
- Ultralytics
- InsightFace
- NumPy
- Pandas
- Streamlit

## Features

- Person detection using YOLO
- Face detection using InsightFace
- Face identification of registered/authorized sample people
- Known and Unknown classification
- People counting
- Image testing
- Video testing
- Webcam testing
- CSV result generation

## Project Workflow

```text
Input Image / Video / Webcam
            |
            v
      YOLO Person Detection
            |
            v
       Face Detection
            |
            v
      Face Identification
            |
            v
       Known / Unknown
            |
            v
       Results + CSV