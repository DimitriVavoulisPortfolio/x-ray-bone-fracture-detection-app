import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

def draw_boxes(image, boxes, class_names):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.xyxy[0].tolist() + [box.conf.item(), box.cls.item()]
        label = f"{class_names[int(cls)]}: {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1-10), label, fill="red")
    return image

def process_image(image_path, model, class_names):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get predictions
    results = model(image)
    
    # Draw boxes on image
    pil_image = Image.fromarray(image)
    result_image = draw_boxes(pil_image, results[0].boxes, class_names)
    
    # Save result
    output_path = os.path.join(os.path.dirname(__file__), f"output_{os.path.basename(image_path)}")
    result_image.save(output_path)
    print(f"Processed {image_path} -> {output_path}")

def process_images(input_path, weights_path, class_names):
    model = YOLO(weights_path)
    
    if os.path.isfile(input_path):
        process_image(input_path, model, class_names)
    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_path, filename)
                process_image(image_path, model, class_names)
    else:
        print(f"Invalid input path: {input_path}")

if __name__ == "__main__":
    input_path = input("Enter the path to the image or directory containing images: ").strip()
    weights_path = "C:/Github-uploading/x-ray-bone-fracture-detection-app/yolov8l_quick_run/weights/best.pt"  # Update this with your actual model path
    class_names = ["elbow positive", "fingers positive", "forearm fracture", "humerus fracture", "humerus", "shoulder fracture", "wrist positive"]
    
    process_images(input_path, weights_path, class_names)
    print("Processing complete!")