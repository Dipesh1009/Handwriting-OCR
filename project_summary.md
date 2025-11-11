# Project Summary: Handwriting Recognition with CRNN

This document provides a detailed summary of the handwriting recognition project. The project uses a Convolutional Recurrent Neural Network (CRNN) to recognize handwritten text from images.

## 1. Project Overview

The project is a handwriting recognition system that can predict text from an image of a handwritten word or sentence. It consists of the following key components:

-   **A deep learning model:** A CRNN model built with TensorFlow and Keras.
-   **A training pipeline:** A script to train the model on the IAM Handwriting Database.
-   **A web application:** A user-friendly web interface built with Streamlit that allows users to upload an image and get a prediction.
-   **A REST API:** A FastAPI application that exposes the model as a REST service.

The trained model is converted to the ONNX (Open Neural Network Exchange) format for efficient inference.

## 2. File Descriptions

Here is a breakdown of the most important files in the project:

-   **`app.py`**: This file contains the FastAPI application that serves the model as a REST API. It defines a `/predict/` endpoint that accepts an image file and returns the predicted text.

-   **`streamlit_app.py`**: This is the main file for the Streamlit web application. It provides a user-friendly interface for the handwriting recognition model. Users can upload an image, and the app will display the prediction.

-   **`train.py`**: This is the main script for training the model. It performs the following steps:
    1.  Loads the IAM Handwriting Database.
    2.  Preprocesses the data, including resizing images and encoding labels.
    3.  Splits the dataset into training and validation sets.
    4.  Builds and compiles the CRNN model.
    5.  Trains the model using the `fit` method.
    6.  Saves the trained model and converts it to ONNX format.

-   **`model.py`**: This file defines the architecture of the CRNN model using Keras. The model consists of:
    -   Convolutional layers for feature extraction from the input image.
    -   Recurrent layers (Bidirectional LSTMs) to process the sequence of features extracted by the convolutional layers.
    -   A dense layer with a softmax activation function to output the probability distribution over the vocabulary.

-   **`inferenceModel.py`**: This file defines the `ImageToWordModel` class, which is used for making predictions with the trained ONNX model. It handles the image preprocessing and the decoding of the model's output.

-   **`configs.py`**: This file defines the `ModelConfigs` class, which stores the configuration for the model training, including hyperparameters like learning rate, batch size, and image dimensions.

-   **`mltu/` directory**: This directory contains a custom Python library that provides a framework for the entire project. It includes modules for:
    -   **`configs.py`**: A base class for model configurations.
    -   **`dataProvider.py`**: A class for creating data providers that efficiently load and preprocess data for training.
    -   **`tensorflow/losses.py`**: Contains the `CTCloss` function.
    -   **`tensorflow/callbacks.py`**: Includes custom Keras callbacks like `Model2onnx` for converting the model to ONNX format.

## 3. Core Concepts and Methods

### 3.1. CRNN (Convolutional Recurrent Neural Network)

The project uses a CRNN model, which is a popular architecture for image-based sequence recognition tasks like handwriting recognition. It combines the strengths of two types of neural networks:

-   **Convolutional Neural Networks (CNNs):** The convolutional layers are used to extract visual features from the input image.
-   **Recurrent Neural Networks (RNNs):** The recurrent layers (in this case, Bidirectional LSTMs) are used to process the sequence of features extracted by the CNN and capture the contextual dependencies in the text.

### 3.2. CTC (Connectionist Temporal Classification) Loss

The model is trained with the CTC loss function. This is a special type of loss function that is designed for sequence-to-sequence problems where the alignment between the input and output sequences is unknown. In handwriting recognition, the CTC loss allows the model to learn to predict a sequence of characters without needing to know the exact position of each character in the input image.

### 3.3. ONNX (Open Neural Network Exchange)

After training, the Keras model is converted to the ONNX format. ONNX is an open standard for representing machine learning models. Using ONNX has several advantages:

-   **Interoperability:** ONNX models can be run on a variety of platforms and in different programming languages.
-   **Performance:** ONNX runtimes are often highly optimized for inference, which can lead to faster prediction times.

## 4. Dataset

The model is trained on the **IAM Handwriting Database**, which is a large dataset of handwritten English text. The `train.py` script processes the `sentences.txt` file from the dataset to create a list of image paths and their corresponding labels. The dataset is then split into training and validation sets.

## 5. How to Run the Project

### 5.1. Training the Model

To train the model, you can run the `train.py` script:

```bash
python train.py
```

You can modify the `TEST_SUBSET` variable in `train.py` to train on a smaller subset of the data for faster development and testing.

### 5.2. Running the Web Application

To start the Streamlit web application, run the following command:

```bash
streamlit run streamlit_app.py
```

This will open a new tab in your browser with the web application. You can then upload an image of handwritten text to get a prediction.

### 5.3. Running the REST API

To start the FastAPI server, run the following command:

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can send a POST request with an image file to the `/predict/` endpoint to get a prediction.
