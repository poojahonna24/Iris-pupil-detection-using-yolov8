# Iris and Pupil Detection using YOLOv8

## Overview
This project focuses on detecting and segmenting the iris and pupil using YOLOv8 and OpenCV with Haar Cascade classifiers. The workflow includes:
- Real-time face and eye detection
- Image enhancement using CLAHE
- Specular reflection removal
- YOLOv8-based iris and pupil detection
- Bounding box generation
- Performance evaluation using precision, recall, F1-score, and mAP

## Repository Structure
```
📂 Project Root
 ├── 📂 weights             # Trained YOLOv8 model (yolov8s.pt)
 ├── 📂 enhanced_eyes       # Final output images with bounding boxes
 ├── 📂 graphical_results   # Graphs, performance metrics, and classification reports
 ├── 📂 predict             # YOLOv8 inference results
 ├── iris_pupil.py          # Python script for real-time detection and inference
 ├── enhancedeye_code.py    # Image enhancement and preprocessing
 ├── README.md              # Project documentation
 ├── Classification_Report.pdf  # YOLOv8 performance metrics
 ├── Table(iris_pupil).pdf  # Iris-to-pupil ratio detection results
```

## Installation & Setup
### Step 1: Clone the YOLOv8 Repository
```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install ultralytics
```

### Step 2: Download the Dataset
Download the Iris-Pupil dataset from Roboflow and place it inside the `ultralytics` folder.

### Step 3: Train the YOLOv8 Model
```bash
python ultralytics/yolo/v8/detect/train.py --img 640 --batch 4 --epochs 100 --data irispupille/data.yaml --weights yolov8s.pt --name iris_pupil_detection
```
The trained model weights (`best.pt`) will be stored in `runs/detect/iris_pupil_detection/`.

### Step 4: Run the Detection Script
```bash
python iris_pupil.py
```
The script will:
- Detect and select the eyes.
- Enhance images using CLAHE.
- Detect the iris and pupil with YOLOv8.

## Results & Evaluation
- Processed images with bounding boxes are stored in `enhanced_eyes/`.
- Performance metrics and graphs are available in `graphical_results/`.
- Classification report and detection results are documented in:
  - **📄 Classification_Report.pdf** (YOLOv8 performance metrics)
  - **📄 Table(iris_pupil).pdf** (Iris-to-pupil ratio detection results)

## Suggested Approach & Methodology
### 1. Face & Eye Detection
- Used Haar Cascade Classifier to detect face and eyes.
- Allowed manual selection of the eye region.

### 2. Image Enhancement
- Applied **CLAHE (Contrast Limited Adaptive Histogram Equalization)**.
- Removed specular reflections for enhanced clarity.

### 3. YOLOv8 Model for Iris-Pupil Detection
- Trained on **310 training images, 90 validation images, 45 test images**.
- Achieved **mAP@0.5: 0.9912** and **F1-score: 0.9721**.

### 4. Evaluation Metrics
| Metric      | Value  |
|------------|--------|
| Precision  | 99.12% |
| Recall     | 95.73% |
| mAP@0.5    | 99.12% |

## Conclusion
The project successfully detects the iris and pupil with high accuracy. The trained YOLOv8 model generalizes well across different eye conditions, making it suitable for applications such as biometric authentication, medical imaging, and gaze tracking.
