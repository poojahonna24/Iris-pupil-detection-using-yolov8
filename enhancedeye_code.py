import ultralytics
import cv2
import os

# Load the YOLOv8 model
model_path = "D:\\irispupilfinal\\runs\\detect\\custom_yolov8\\weights\\best.pt"
model = ultralytics.YOLO(model_path)

# Set confidence threshold to 0.5
model.overrides['conf'] = 0.5  # Confidence threshold

# Input and output directories
enhanced_images_dir ="D:\\irispupilfinal\\enhanced_eyes"  # Directory containing generic images
output_dir = "D:\\irispupilfinal\\fullyfinal"  # Output directory for processed images
os.makedirs(output_dir, exist_ok=True)

# Class indices for iris and pupil
IRIS_CLASS = 0  # Class ID for iris
PUPIL_CLASS = 1  # Class ID for pupil

# Function to process an image and calculate the IP ratio
def process_image(image_path):
    try:
        results = model(image_path)  # YOLOv8 model inference
        img = cv2.imread(image_path)

        if img is None:
            print(f"Failed to read image: {image_path}")
            return

        # Initialize variables to store the highest confidence boxes for each class
        best_iris = None
        best_pupil = None

        # Process detection results
        for box in results[0].boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            if cls == IRIS_CLASS:
                if best_iris is None or conf > best_iris[1]:
                    best_iris = ((x_min, y_min, x_max, y_max), conf)
            elif cls == PUPIL_CLASS:
                if best_pupil is None or conf > best_pupil[1]:
                    best_pupil = ((x_min, y_min, x_max, y_max), conf)

        # Draw the best bounding boxes on the image
        iris_width = None
        pupil_width = None

        if best_iris:
            (x_min, y_min, x_max, y_max), conf = best_iris
            iris_width = x_max - x_min
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue for iris
            label = f"Iris {conf:.2f}"
            cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if best_pupil:
            (x_min, y_min, x_max, y_max), conf = best_pupil
            pupil_width = x_max - x_min
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green for pupil
            label = f"Pupil {conf:.2f}"
            cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Compute and log the IP ratio
        filename = os.path.basename(image_path)
        if iris_width and pupil_width:
            ratio = iris_width / pupil_width  # Iris-to-pupil ratio
            print(f"{filename}: IP Ratio = {ratio:.2f}")
        else:
            print(f"{filename}: IP Ratio = Not Detected")

        # Save the processed image with bounding boxes
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Process all valid images in the directory
for filename in os.listdir(enhanced_images_dir):
    filepath = os.path.join(enhanced_images_dir, filename)

    # Ensure the file is a valid image format
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        process_image(filepath)
    else:
        print(f"Skipped non-image file: {filename}")
