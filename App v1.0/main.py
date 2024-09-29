import eel
import os
import tempfile
import sys
import logging
from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO
import base64
from io import BytesIO

# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        logging.info("Application starting")
        
        # Get the absolute path of the script
        script_dir = Path(__file__).resolve().parent
        logging.info(f"Script directory: {script_dir}")

        # Set up web directory
        web_dir = script_dir / "web"
        logging.info(f"Web directory: {web_dir}")

        if not web_dir.is_dir():
            raise FileNotFoundError(f"Web directory not found at {web_dir}")

        # Initialize eel with your web files folder
        eel.init(str(web_dir))

        # Load the YOLOv8 model
        model_path = script_dir / "weights" / "best.pt"
        logging.info(f"Looking for model at: {model_path}")

        # Check if the model file exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Import YOLO here to catch any import errors
        model = YOLO(str(model_path))
        logging.info("Model loaded successfully")

        @eel.expose
        def process_image(image_data_base64):
            logging.info("Processing image")
            try:
                # Decode the base64-encoded image data
                image_data = base64.b64decode(image_data_base64.split(',')[0])

                # Open and process the image
                with Image.open(BytesIO(image_data)) as image:
                    logging.info(f"Image opened successfully: {image.format}, {image.size}, {image.mode}")
                    
                    results = model(image)
                    logging.info("Model prediction completed")

                    # Process the results
                    processed_image = image.copy()
                    draw = ImageDraw.Draw(processed_image)
                    
                    detections = []
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            conf = box.conf[0]
                            cls = int(box.cls[0])
                            
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                            draw.text((x1, y1-10), f"Class {cls}: {conf:.2f}", fill="red")
                            logging.info(f"Drew bounding box: {x1}, {y1}, {x2}, {y2}, Class {cls}, Confidence {conf:.2f}")
                            
                            detections.append({
                                'class': cls,
                                'confidence': float(conf),
                            })

                    # Encode the processed image as base64
                    buffered = BytesIO()
                    processed_image.save(buffered, format="PNG")
                    processed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                return {
                    'processedImage': processed_image_base64,
                    'detections': detections,
                }
            except Exception as e:
                logging.exception(f"Error processing image: {str(e)}")
                print(f"Error processing image: {str(e)}")
                return None

        @eel.expose
        def get_class_names():
            class_names = ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']
            logging.info(f"Returning class names: {class_names}")
            return class_names
        
        # Start the app
        eel.start('index.html', mode='chrome', size=(800, 600))
        
    except Exception as e:
        logging.exception(f"An error occurred during startup: {str(e)}")
        print(f"An error occurred during startup: {str(e)}")
    finally:
        logging.info("Application exiting")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()