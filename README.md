# Iris pupil detection using yolov8
 
Iris and Pupil Detection using YOLOv8 and OpenCV

This project focuses on detecting and segmenting the iris and pupil using YOLOv8 and OpenCV with Haar Cascade classifiers. It includes real-time eye detection, enhancement (CLAHE), and specular reflection removal.

1. Project Overview

Detect face and eyes using Haar Cascade Classifier.

Manually select the eye region.

Apply CLAHE enhancement and specular reflection removal.

Detect iris and pupil using a fine-tuned YOLOv8 model.

Generate bounding boxes for the iris and pupil.

Evaluate results using precision, recall, F1-score, and mAP.

2. Repository Structure

📂 Project Root
 ├── 📂 weights             # Trained YOLOv8 model (yolov8s.pt)
 ├── 📂 enhanced_eyes       # Final output images with bounding boxes
 ├── 📂 graphical_results   # Graphs, performance metrics, and classification reports
 ├── 📂 predict             # YOLOv8 inference results
 ├── iris_pupil.py          # Python script for real-time eye detection and YOLOv8 inference
 ├── enhancedeye_code.py    # Image enhancement and preprocessing
 ├── README.md              # Project documentation
 ├── Classification_Report.pdf  # YOLOv8 performance metrics
 ├── Table(iris_pupil).pdf  # Iris-to-pupil ratio detection results

3. Installation & Setup

Step 1: Clone the YOLOv8 Repository

git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install ultralytics

Step 2: Download the Dataset

Download the Iris-Pupil dataset from Roboflow and place it inside the ultralytics folder.

Step 3: Train the YOLOv8 Model

python ultralytics/yolo/v8/detect/train.py --img 640 --batch 4 --epochs 100 --data irispupille/data.yaml --weights yolov8s.pt --name iris_pupil_detection

The trained model weights (best.pt) will be stored in runs/detect/iris_pupil_detection/.

Step 4: Run the Detection Script

python iris_pupil.py

The script will:

Detect and select the eyes.

Enhance images using CLAHE.

Detect the iris and pupil with YOLOv8.

4. Results & Evaluation

The processed images with bounding boxes are stored in enhanced_eyes/.

The performance metrics and graphs are available in graphical_results/.

Access the classification report and detection results:

📄 Classification Report (YOLOv8)

📄 Iris-Pupil Detection Table

5. Suggested Approach & Methodology

Face & Eye Detection

Used Haar Cascade Classifier to detect face and eyes.

Allowed manual selection of the eye pair.

Image Enhancement

Applied CLAHE (Contrast Limited Adaptive Histogram Equalization).

Removed specular reflections to enhance clarity.

YOLOv8 Model for Iris-Pupil Detection

Trained YOLOv8 on 310 training images, 90 validation images, 45 test images.

Achieved mAP@0.5: 0.9912 and F1-score: 0.9721.

Evaluation Metrics

Precision: 99.12%

Recall: 95.73%

mAP@0.5: 99.12%

6. Conclusion

The project successfully detects the iris and pupil with high accuracy. The trained YOLOv8 model generalizes well across various eye conditions. The approach can be extended for biometric authentication, medical imaging, and gaze tracking applications.

