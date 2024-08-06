# Alzheimer's Imaging CNN

## Description
This project implements a Convolutional Neural Network (CNN) to classify MRI images for Alzheimer's disease detection. The CNN architecture consists of 12 layers, aiming to effectively analyze and classify medical imaging data. The project includes preprocessing of image data, model training, and evaluation, along with visualization of training results and performance metrics.

## Features
- **Image Preprocessing**: Validates and preprocesses MRI images to ensure uniformity.
- **CNN Architecture**: Implements a 12-layer Convolutional Neural Network with dropout for regularization.
- **Model Training**: Trains the model using categorized MRI images for Alzheimer's detection.
- **Performance Evaluation**: Includes accuracy tracking, confusion matrix generation, and visualization of training and validation metrics.
- **Interactive Visualization**: Displays the confusion matrix and training/validation performance graphs.

## Prerequisites
- Python 3.x
- pip (Python package manager)
- TensorFlow
- OpenCV
- Matplotlib
- NumPy
- Seaborn

## Installation
1. Clone the repository:
2. Navigate to the project directory
3. Install the required dependencies: `pip install -r requirements.txt`

   ## Usage

1. **Prepare Your Data**: Ensure the MRI image dataset is organized in the following structure:
    ```
    Alzheimer's Dataset/
    ├── train/
    ├── test/
    └── val/
    ```
    Each subdirectory (`train`, `test`, `val`) should contain images categorized by class (e.g., `Mild_Demented`, `Moderate_Demented`, etc.).

2. **Run the Model**:
    Execute the script to start training the CNN:
   `image_classification.ipynb`

3. **Monitor Training**:
    - **Training and Validation Accuracy**: Track the model's performance over epochs.
    - **Confusion Matrix**: View the confusion matrix to understand classification performance.

4. **Visualize Results**:
    - The script generates plots showing training and validation accuracy/loss.
    - Confusion matrix visualization is provided to analyze model predictions.

## Files Needed
Ensure the following files are in your project directory:
- `image_classification.ipynb`: Main script for running the CNN training and evaluation
- `requirements.txt`: List of Python dependencies
- Your dataset directory (`Alzheimer's Dataset`) with subdirectories for training, testing, and validation images

## Technologies Used
- Python
- TensorFlow: For deep learning and CNN implementation
- OpenCV: For image processing
- Matplotlib: For plotting training and evaluation metrics
- NumPy: For numerical operations
- Seaborn: For enhanced confusion matrix visualization

## Notes
- Ensure that the MRI images are properly organized and preprocessed before running the model.
- Training time may vary depending on hardware specifications and dataset size.

## Troubleshooting
If you encounter any issues:
- Verify that all image files are correctly placed and formatted.
- Check for any missing dependencies and install them as needed.
- Review the console output for specific error messages and adjust the code accordingly.

## Acknowledgements
- Medical imaging datasets and resources used in this project; https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images
- TensorFlow and other libraries for their robust deep learning tools.

