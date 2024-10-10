# Plant Disease Detection Using Image Processing and Machine Learning
This MATLAB project implements a plant disease detection system using image processing and machine learning techniques. It leverages a dataset of plant leaf images to classify various plant diseases using Support Vector Machines (SVM) and K-Nearest Neighbors (KNN) classifiers.

###Project Overview  
The project involves several key steps, including:

Image preprocessing (resizing, grayscale conversion, edge detection, and segmentation).
Feature extraction based on color properties of the images.
Training multi-class classifiers (SVM and KNN) to detect plant diseases.
Model evaluation through confusion matrices and accuracy metrics.
Code Structure
Image Datastore: The code uses an image datastore to manage the dataset of plant leaf images. Images are loaded and resized for consistency.
Feature Extraction: Custom feature extraction functions extract color features (red, green, blue channel averages) from each image.
Classification: Two machine learning classifiers, SVM and KNN, are trained using extracted features to perform multi-class classification of plant diseases.
Segmentation and Edge Detection: The images are segmented using thresholding, and edges are detected using the Canny edge detection algorithm.
Visualization: The code visualizes a few images, their segmented versions, and edges, along with confusion matrices for both classifiers.
Key Steps
Image Loading and Preprocessing:

Images are loaded from a dataset of plant leaves and resized to a consistent dimension of 256x256 pixels.
The first 50 images are selected for training and testing.
Dataset Splitting:

The dataset is split into training (70%) and testing (30%) subsets using a random shuffle.
Feature Extraction:

Color features (average RGB values) are extracted from each image and used as input for the classifiers.
Training Models:

SVM (Support Vector Machine): A multi-class SVM model is trained using extracted features.
KNN (K-Nearest Neighbors): A KNN model is trained using the same feature set for comparison.
Model Evaluation:

Both models are evaluated on the test set. Accuracy is calculated, and confusion matrices are visualized to show classification performance.
Results
The models classify plant diseases based on color features, and their performance is evaluated with accuracy metrics and confusion matrices.

Future Enhancements
This project can be further improved by:

Including more advanced feature extraction techniques such as texture or shape analysis.
Increasing the dataset size for better generalization of the models.
Experimenting with deep learning models (e.g., Convolutional Neural Networks) for improved accuracy.
