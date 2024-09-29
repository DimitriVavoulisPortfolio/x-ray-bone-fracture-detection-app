# X-Ray Bone Fracture Detection App

## Project Overview

This project is an advanced machine learning application designed to detect bone fractures in X-ray images with milisecond response time. It utilizes state-of-the-art computer vision techniques and deep learning models to analyze X-ray images and identify potential fractures, serving as a diagnostic aid for medical professionals.

### Key Features

- YOLOv8l model for bone fracture detection
- User-friendly interface for uploading and analyzing X-ray images
- Real-time fracture detection and visualization 
- Comprehensive performance metrics adhering to medical imaging standards
- Designed as a SaaS prototype for potential clinical integration

## Dataset

The model was trained on the [Bone Fracture Detection Computer Vision Dataset](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project) from Kaggle. This dataset contains a diverse collection of X-ray images with labeled bone fractures, allowing for robust model training and evaluation.

## Project Structure

1. **Data Processing**: Scripts for medical image preprocessing and augmentation
2. **Model Training**: Implementation of the deep learning model and training pipeline
3. **Model Evaluation**: Scripts for rigorous testing and performance evaluation
4. **Web Application**: User interface for uploading X-rays and viewing results
5. **API**: Backend API for handling image processing and model inference
6. **Documentation**: Comprehensive documentation on model architecture, training process, and usage guidelines

## Model Performance

The YOLOv8 model achieved the following performance metrics on the test set:

- mAP50: 0.37
- mAP50-95: 0.18
- Precision: 0.46
- Recall: 0.26

These metrics suggest that the model is capable of detecting bone fractures with reasonable accuracy, but there is still room for improvement. The project team is continuously working on optimizing the model's performance and exploring strategies to enhance the detection capabilities.

## Quick Start Guide

1. Clone the repository:

   ```
   git clone https://github.com/DimitriVavoulisPortfolio/x-ray-bone-fracture-detection-app
   ```

2. Install dependencies:
   ```
   pip install tensorflow torch torchvision pandas numpy scikit-learn matplotlib seaborn pillow opencv-python-headless flask gunicorn pytest pylint black isort
   ```
3. To test the app:
   start the launcher on the App v1.0 folder

DISCLAIMER: This is a prototype, it's not a replacement to medical professionals and future iterations while better in quality are still not a replacement, consult a medical professional on case of injury

## Usage

1. Upload an X-ray image through the secure web interface.
2. The application processes the image and displays results with fracture probability scores.
3. Review the detection results, including highlighted regions of interest.
4. (Optional) Provide feedback to improve model performance.

## Clinical Validation

This tool is intended to assist, not replace, professional medical judgment. All results should be interpreted by qualified medical professionals in the context of clinical findings and patient history.

## Data Privacy and Security

This application adheres to strict medical data privacy standards. All uploaded images and results are encrypted and securely handled in compliance with healthcare data protection regulations.

## Future Work

- Enhanced YOLOv8l model for better bone fracture diagnosis
- Enhance the user interface for better visualization of results
- Integrate with medical record systems for seamless workflow integration
- Expand the dataset to improve model generalization across diverse populations

## Contributing

We welcome contributions to improve the project. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure all contributions adhere to medical software development standards and practices.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kaggle and the dataset contributors for providing the bone fracture X-ray dataset
- The open-source community for the tools and libraries used in this project
- Medical advisors and clinicians for their invaluable input and validation

## Contact

For any questions, feedback, or collaboration opportunities, please open an issue in this repository or contact [Dimitri Vavoulis] at [dimitrivavoulis3@gmail.com].

## Disclaimer

This software is not certified for clinical use and is currently in the research and development phase. It should not be used for making clinical decisions without proper validation and regulatory approval.


