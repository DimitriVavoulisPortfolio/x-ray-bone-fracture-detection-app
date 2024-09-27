import os
import yaml
import logging
import time
import torch
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import shutil
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, cohen_kappa_score
from scipy.interpolate import interp1d

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def verify_directory(path, dir_type):
    if not os.path.isdir(path):
        raise NotADirectoryError(f"The {dir_type} path is not a valid directory: {path}")
    if not os.path.exists(os.path.join(path, 'images')) or not os.path.exists(os.path.join(path, 'labels')):
        raise FileNotFoundError(f"The {dir_type} directory must contain 'images' and 'labels' subdirectories.")

def split_valid_dataset(valid_path, split_ratio=0.5):
    images_path = os.path.join(valid_path, 'images')
    labels_path = os.path.join(valid_path, 'labels')
    
    image_files = os.listdir(images_path)
    
    val_files, test_files = train_test_split(image_files, test_size=split_ratio, random_state=42)
    
    test_path = os.path.join(os.path.dirname(valid_path), 'test')
    os.makedirs(os.path.join(test_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_path, 'labels'), exist_ok=True)
    
    for file in test_files:
        shutil.move(os.path.join(images_path, file), os.path.join(test_path, 'images', file))
        label_file = os.path.splitext(file)[0] + '.txt'
        if os.path.exists(os.path.join(labels_path, label_file)):
            shutil.move(os.path.join(labels_path, label_file), os.path.join(test_path, 'labels', label_file))
    
    return test_path

def create_data_yaml(train_path, val_path, test_path):
    with open(os.path.join(os.path.dirname(train_path), 'data.yaml'), 'r') as f:
        data = yaml.safe_load(f)
    
    data['train'] = train_path
    data['val'] = val_path
    data['test'] = test_path
    
    yaml_path = 'bone_fracture_detection_data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    return yaml_path

def calculate_iou(box1, box2):
    # Calculate IoU between two bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def calculate_map(true_boxes, pred_boxes, iou_threshold=0.5):
    # Calculate mAP
    aps = []
    for c in range(len(true_boxes)):
        true_class = true_boxes[c]
        pred_class = pred_boxes[c]
        
        if len(true_class) == 0:
            continue
        
        detected = [False] * len(true_class)
        confidences = [box[4] for box in pred_class]
        sort_index = np.argsort(confidences)[::-1]
        
        tp = np.zeros(len(pred_class))
        fp = np.zeros(len(pred_class))
        
        for i, pred_idx in enumerate(sort_index):
            pred_box = pred_class[pred_idx]
            max_iou = 0
            max_idx = -1
            
            for j, true_box in enumerate(true_class):
                iou = calculate_iou(pred_box[:4], true_box[:4])
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            
            if max_iou >= iou_threshold:
                if not detected[max_idx]:
                    tp[i] = 1
                    detected[max_idx] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(true_class)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        ap = np.trapz(precisions, recalls)
        aps.append(ap)
    
    return np.mean(aps)

def calculate_froc(true_boxes, pred_boxes, num_images, max_fps=8):
    # Calculate FROC curve
    fps_per_image = []
    sensitivities = []
    
    for threshold in np.linspace(0, 1, 100):
        fp = 0
        tp = 0
        fn = 0
        
        for i in range(len(true_boxes)):
            true_class = true_boxes[i]
            pred_class = pred_boxes[i]
            
            detected = [False] * len(true_class)
            
            for pred_box in pred_class:
                if pred_box[4] < threshold:
                    continue
                
                max_iou = 0
                max_idx = -1
                
                for j, true_box in enumerate(true_class):
                    iou = calculate_iou(pred_box[:4], true_box[:4])
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = j
                
                if max_iou >= 0.5:
                    if not detected[max_idx]:
                        tp += 1
                        detected[max_idx] = True
                    else:
                        fp += 1
                else:
                    fp += 1
            
            fn += sum(1 for d in detected if not d)
        
        fps_per_image.append(fp / num_images)
        sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    
    # Interpolate FROC curve
    interp_fps = np.linspace(0, max_fps, 100)
    interp_sens = interp1d(fps_per_image, sensitivities, kind='linear', fill_value="extrapolate")(interp_fps)
    
    return interp_fps, interp_sens

def calculate_metrics(true_boxes, pred_boxes, num_classes, num_images):
    iou_thresholds = [0.5, 0.75]
    confidences = []
    tp_confidence = []
    fp_confidence = []
    fn_confidence = []
    class_metrics = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(num_classes)}
    total_detections = 0
    total_detection_time = 0
    
    for i in range(len(true_boxes)):
        true_class = true_boxes[i]
        pred_class = pred_boxes[i]
        
        start_time = time.time()
        
        detected = [False] * len(true_class)
        
        for pred_box in pred_class:
            confidences.append(pred_box[4])
            max_iou = 0
            max_idx = -1
            pred_class_id = int(pred_box[5])
            
            for j, true_box in enumerate(true_class):
                iou = calculate_iou(pred_box[:4], true_box[:4])
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            
            if max_iou >= 0.5:
                if not detected[max_idx]:
                    class_metrics[pred_class_id]['tp'] += 1
                    tp_confidence.append(pred_box[4])
                    detected[max_idx] = True
                else:
                    class_metrics[pred_class_id]['fp'] += 1
                    fp_confidence.append(pred_box[4])
            else:
                class_metrics[pred_class_id]['fp'] += 1
                fp_confidence.append(pred_box[4])
        
        for j, d in enumerate(detected):
            if not d:
                true_class_id = int(true_class[j][5])
                class_metrics[true_class_id]['fn'] += 1
                fn_confidence.append(0)  # Assign 0 confidence to false negatives
        
        total_detections += len(pred_class)
        total_detection_time += time.time() - start_time
    
    # Calculate overall metrics
    total_tp = sum(cm['tp'] for cm in class_metrics.values())
    total_fp = sum(cm['fp'] for cm in class_metrics.values())
    total_fn = sum(cm['fn'] for cm in class_metrics.values())
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f2 = 5 * (precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
    
    # Calculate mAP and AR for different IoU thresholds
    mAP = {iou: calculate_map(true_boxes, pred_boxes, iou) for iou in iou_thresholds}
    AR = {iou: sum(calculate_map(true_boxes, pred_boxes, iou) for _ in range(10)) / 10 for iou in iou_thresholds}
    
    # Calculate FROC curve
    froc_fps, froc_sens = calculate_froc(true_boxes, pred_boxes, num_images)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for class_id, cm in class_metrics.items():
        tp, fp, fn = cm['tp'], cm['fp'], cm['fn']
        per_class_metrics[class_id] = {
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1': 2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))) if ((tp / (tp + fp)) + (tp / (tp + fn))) > 0 else 0
        }
    
    return {
        'accuracy': (total_tp + total_fn) / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'f2_score': f2,
        'false_positive_rate': total_fp / (total_fp + total_fn) if (total_fp + total_fn) > 0 else 0,
        'false_negative_rate': total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
        'mAP': mAP,
        'AR': AR,
        'average_iou': sum(calculate_iou(pred[:4], true[:4]) for pred, true in zip(pred_boxes, true_boxes) if calculate_iou(pred[:4], true[:4]) > 0.5) / total_tp if total_tp > 0 else 0,
        'average_confidence': {
            'overall': np.mean(confidences) if confidences else 0,
            'true_positives': np.mean(tp_confidence) if tp_confidence else 0,
            'false_positives': np.mean(fp_confidence) if fp_confidence else 0,
            'false_negatives': np.mean(fn_confidence) if fn_confidence else 0
        },
        'per_class_metrics': per_class_metrics,
        'average_detection_time': total_detection_time / total_detections if total_detections > 0 else 0,
        'froc': {'fps': froc_fps.tolist(), 'sensitivity': froc_sens.tolist()},
        'confusion_matrix': confusion_matrix([b[5] for boxes in true_boxes for b in boxes], [b[5] for boxes in pred_boxes for b in boxes]).tolist(),
        'cohen_kappa': cohen_kappa_score([b[5] for boxes in true_boxes for b in boxes], [b[5] for boxes in pred_boxes for b in boxes])
    }

def train_and_test():
    logging.info("Starting training and testing process for bone fracture detection")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        logging.warning("CUDA is not available. Using CPU. This may significantly slow down training.")
        device = 'cpu'

    try:
        # Get dataset paths
        dataset_path = input("Enter the directory path for the dataset: ").strip()
        train_path = os.path.join(dataset_path, 'train')
        valid_path = os.path.join(dataset_path, 'valid')
        
        # Verify directories
        verify_directory(train_path, "train dataset")
        verify_directory(valid_path, "validation dataset")
        
        # Split validation dataset
        test_path = split_valid_dataset(valid_path)
        logging.info(f"Split validation dataset. New test dataset path: {test_path}")

        # Create data.yaml file
        data_yaml = create_data_yaml(train_path, valid_path, test_path)
        logging.info(f"Created data YAML file: {data_yaml}")

        # Initialize YOLOv8x model
        model = YOLO('yolov8m.pt')
        logging.info("YOLOv8m model initialized")

        # Train the model with optimized parameters
        results = model.train(
            data=data_yaml,
            epochs=150,  # Increased epochs, early stopping will prevent overfitting
            imgsz=640,  # Increased image size for better detail in medical images
            batch=32,  # Adjust based on GPU memory. Reduce if OOM errors occur
            patience=20,  # Increased patience for early stopping
            device=device,
            workers=4,
            project='bone_fracture_detection',
            name='yolov8m_optimized_run',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',  # AdamW often performs better than Adam
            lr0=0.002,  # Initial learning rate
            lrf=0.01,  # Final learning rate fraction
            momentum=0.937,  # SGD momentum/Adam beta1
            weight_decay=0.0005,  # Optimizer weight decay
            warmup_epochs=3,  # Warmup epochs for learning rate
            warmup_momentum=0.8,  # Warmup initial momentum
            warmup_bias_lr=0.1,  # Warmup initial bias lr
            box=7.5,  # Box loss gain
            cls=0.5,  # Cls loss gain
            dfl=1.5,  # DFL loss gain
            close_mosaic=10,  # Disable mosaic augmentation for final 10 epochs
            amp=True,  # Automatic Mixed Precision
            fraction=1.0,  # Dataset fraction to train on
            profile=False,  # Profile ONNX/TensorRT speed
            save=True,
            save_period=-1,  # Save checkpoint every x epochs (-1 for last epoch only)
            seed=0,  # Global training seed
            # Augmentation parameters
            hsv_h=0.015,  # Hue augmentation
            hsv_s=0.2,  # Saturation augmentation
            hsv_v=0.1,  # Value augmentation
            degrees=0.0,  # Rotate (+/- deg)
            translate=0.1,  # Translate (+/- fraction)
            scale=0.3,  # Scale (+/- gain)
            shear=0.0,  # Shear (+/- deg)
            perspective=0.0,  # Perspective (+/- fraction), 0.0 for medical images
            flipud=0.0,  # Vertical flip probability, 0.0 for medical images
            fliplr=0.5,  # Horizontal flip probability
            mosaic=0.0,  # Mosaic probability, 0.0 for medical images
            mixup=0.0,  # Mixup probability, 0.0 for medical images
            copy_paste=0.0  # Copy-paste probability, 0.0 for medical images
        )
        logging.info("Training complete!")

        # Save the final model
        model.save('bone_fracture_detection_yolov8x_final.pt')
        logging.info("Final model saved as 'bone_fracture_detection_yolov8x_final.pt'")

        # Validate the model
        val_results = model.val(data=data_yaml, device=device)
        logging.info(f"Validation results: {val_results}")

        # Test the model on the new test set
        logging.info("Starting testing on the new test set")
        test_results = model.predict(source=test_path, save=True, save_txt=True, device=device)
        
        # Prepare data for metric calculation
        true_boxes = []
        pred_boxes = []
        for result in test_results:
            true_boxes.append(result.boxes.cpu().numpy())
            pred_boxes.append(result.boxes.cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(true_boxes, pred_boxes, model.model.nc, len(test_results))
        
        # Log results
        logging.info("Test results on the new test set:")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Precision: {metrics['precision']:.4f}")
        logging.info(f"Recall: {metrics['recall']:.4f}")
        logging.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logging.info(f"F2 Score: {metrics['f2_score']:.4f}")
        logging.info(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
        logging.info(f"False Negative Rate: {metrics['false_negative_rate']:.4f}")
        logging.info(f"mAP@0.5: {metrics['mAP'][0.5]:.4f}")
        logging.info(f"mAP@0.75: {metrics['mAP'][0.75]:.4f}")
        logging.info(f"AR@0.5: {metrics['AR'][0.5]:.4f}")
        logging.info(f"AR@0.75: {metrics['AR'][0.75]:.4f}")
        logging.info(f"Average IoU: {metrics['average_iou']:.4f}")
        logging.info(f"Average Detection Time: {metrics['average_detection_time']:.4f} seconds")
        logging.info(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        
        logging.info("Average Confidence Scores:")
        for k, v in metrics['average_confidence'].items():
            logging.info(f"  {k.capitalize()}: {v:.4f}")
        
        logging.info("Per-class Metrics:")
        for class_id, class_metrics in metrics['per_class_metrics'].items():
            logging.info(f"  Class {class_id}:")
            for metric, value in class_metrics.items():
                logging.info(f"    {metric.capitalize()}: {value:.4f}")
        
        logging.info("Confusion Matrix:")
        for row in metrics['confusion_matrix']:
            logging.info(f"  {row}")
        
        # Save FROC curve data
        froc_data = {
            'fps': metrics['froc']['fps'],
            'sensitivity': metrics['froc']['sensitivity']
        }
        with open('froc_curve_data.yaml', 'w') as f:
            yaml.dump(froc_data, f)
        logging.info("FROC curve data saved to 'froc_curve_data.yaml'")

    except Exception as e:
        logging.error(f"An error occurred during training or testing: {str(e)}")
        raise

if __name__ == '__main__':
    train_and_test()
