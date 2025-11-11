# Project Summary: Handwriting Recognition with CRNN

This document provides a detailed and elaborate summary of the handwriting recognition project. The project uses a Convolutional Recurrent Neural Network (CRNN) to recognize handwritten text from images.

## 1. Project Overview

The project is a comprehensive handwriting recognition system that can predict text from an image of a handwritten word or sentence. It is a complete end-to-end solution, from training a deep learning model to deploying it as a web application and a REST API.

The core of the project is a CRNN model built with TensorFlow and Keras. This model is specifically designed for image-based sequence recognition tasks. The project also includes a training pipeline to train the model on the IAM Handwriting Database, a well-known dataset in the field of handwriting recognition.

For deployment, the project offers two options:
-   A user-friendly web application built with Streamlit, which allows users to upload an image and get a prediction in real-time.
-   A REST API built with FastAPI, which exposes the model as a service that can be integrated into other applications.

A key feature of the project is the use of the ONNX (Open Neural Network Exchange) format for the trained model. This ensures that the model is portable and can be deployed on a wide range of platforms with high performance.

The project is well-structured and makes use of a custom library `mltu` which encapsulates much of the boilerplate code for data handling, training, and inference, promoting code reuse and maintainability.

## 2. File Descriptions

Here is a more detailed breakdown of the most important files in the project:

-   **`app.py`**: This file contains the FastAPI application that serves the model as a REST API.
    -   It initializes a FastAPI app and loads the trained `ImageToWordModel` from the `inferenceModel.py` file.
    -   It defines a `/predict/` endpoint that accepts an image file as an `UploadFile`.
    -   Inside the `predict` function, it reads the image file, decodes it into an OpenCV image, and then passes it to the `model.predict()` method to get the predicted text.
    -   The predicted text is then returned as a JSON response.
    -   It also includes a root endpoint `/` that returns a welcome message.

-   **`streamlit_app.py`**: This is the main file for the Streamlit web application.
    -   It uses Streamlit's features to create an interactive and visually appealing user interface.
    -   The `load_model` function is decorated with `@st.cache_resource` to ensure that the model is loaded only once, which improves the performance of the app.
    -   The UI includes a title, a description of the project, and instructions on how to use the app.
    -   The `st.file_uploader` widget allows users to upload an image.
    -   When an image is uploaded, it is converted to an OpenCV image and passed to the `model.predict()` method.
    -   The predicted text is then displayed to the user in a styled "prediction box". The app also includes some fun elements like balloons on a successful prediction.

-   **`train.py`**: This is the main script for training the model.
    -   **Data Loading and Preprocessing:** It reads the `sentences.txt` file from the IAM dataset, which contains the labels for the handwritten text images. It then creates a list of image paths and their corresponding labels. It also determines the vocabulary and the maximum text length from the dataset.
    -   **Configuration:** It creates an instance of the `ModelConfigs` class and saves the vocabulary and maximum text length to a `configs.yaml` file.
    -   **Data Provider:** It uses the `DataProvider` class from the `mltu` library to create a data pipeline for training and validation. This data provider handles batching, shuffling, and preprocessing of the data. The preprocessing steps include reading the images, resizing them, and encoding the labels into a numerical format.
    -   **Model Creation:** It calls the `train_model` function from `model.py` to create the CRNN model.
    -   **Model Compilation:** It compiles the model with the Adam optimizer, the `CTCloss` function, and the `CERMetric` and `WERMetric` for monitoring the performance.
    -   **Callbacks:** It defines several Keras callbacks to be used during training:
        -   `EarlyStopping`: To stop the training if the validation performance does not improve.
        -   `ModelCheckpoint`: To save the best model based on the validation CER.
        -   `TrainLogger`: A custom callback to log the training progress to a file.
        -   `TensorBoard`: To visualize the training process in TensorBoard.
        -   `ReduceLROnPlateau`: To reduce the learning rate if the validation performance plateaus.
        -   `Model2onnx`: A custom callback to convert the saved model to ONNX format after training.
    -   **Training:** It trains the model using the `model.fit()` method, passing the training and validation data providers and the callbacks.
    -   **Saving Datasets:** After training, it saves the training and validation datasets as CSV files for future reference.

-   **`model.py`**: This file defines the architecture of the CRNN model using the Keras functional API. The `train_model` function is the core of this file, and it constructs the model layer by layer.

    ```python
    from keras import layers
    from keras.models import Model

    from mltu.tensorflow.model_utils import residual_block


    def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
        
        inputs = layers.Input(shape=input_dim, name="input")
    ```
    - **`inputs = layers.Input(shape=input_dim, name="input")`**: This line defines the input layer of the model. It specifies the shape of the input data, which is an image with dimensions `input_dim` (height, width, channels).

    ```python
        # normalize images here instead in preprocessing step
        input = layers.Lambda(lambda x: x / 255)(inputs)
    ```
    - **`input = layers.Lambda(lambda x: x / 255)(inputs)`**: This is a normalization layer. It takes the input image and divides each pixel value by 255. This scales the pixel values from the range [0, 255] to [0, 1], which is a common practice in deep learning to help the model converge faster.

    ```python
        x1 = residual_block(input, 32, activation=activation, skip_conv=True, strides=1, dropout=dropout)

        x2 = residual_block(x1, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
        x3 = residual_block(x2, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

        x4 = residual_block(x3, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
        x5 = residual_block(x4, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

        x6 = residual_block(x5, 128, activation=activation, skip_conv=True, strides=2, dropout=dropout)
        x7 = residual_block(x6, 128, activation=activation, skip_conv=True, strides=1, dropout=dropout)

        x8 = residual_block(x7, 128, activation=activation, skip_conv=True, strides=2, dropout=dropout)
        x9 = residual_block(x8, 128, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    ```
    - **`residual_block(...)`**: These lines define the convolutional part of the model. The model uses a series of residual blocks to extract features from the input image.
        - **`residual_block`**: This is a custom function (likely defined in `mltu.tensorflow.model_utils`) that creates a block of convolutional layers with a skip connection. Skip connections help to prevent the vanishing gradient problem in deep networks.
        - **`32`, `64`, `128`**: These numbers represent the number of filters in the convolutional layers. As we go deeper into the network, the number of filters increases, which allows the model to learn more complex features.
        - **`strides=2`**: A stride of 2 in the convolutional layers is used for downsampling the feature maps. This reduces the spatial dimensions of the feature maps, which helps to reduce the computational cost and also increases the receptive field of the subsequent layers.

    ```python
        squeezed = layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)
    ```
    - **`squeezed = layers.Reshape(...)`**: This layer reshapes the output of the convolutional layers to prepare it for the recurrent layers. The 2D feature maps are "squeezed" into a sequence of feature vectors.

    ```python
        blstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(squeezed)
        blstm = layers.Dropout(dropout)(blstm)

        blstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(blstm)
        blstm = layers.Dropout(dropout)(blstm)
    ```
    - **`blstm = layers.Bidirectional(layers.LSTM(...))`**: These lines define the recurrent part of the model.
        - **`layers.LSTM(256, return_sequences=True)`**: This is a Long Short-Term Memory (LSTM) layer. LSTMs are a type of RNN that are very effective at learning long-range dependencies in sequential data. `return_sequences=True` is important because we want the LSTM to output a sequence of vectors (one for each time step) that can be passed to the next layer.
        - **`layers.Bidirectional(...)`**: This wrapper makes the LSTM layer bidirectional. A bidirectional LSTM processes the sequence in both forward and backward directions, which allows it to learn from both past and future context. This is very useful for handwriting recognition, as the context of a character can depend on the characters that come before and after it.
        - **`layers.Dropout(dropout)`**: Dropout is a regularization technique that helps to prevent overfitting. It randomly sets a fraction of the input units to 0 at each update during training time.

    ```python
        output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm)
    ```
    - **`output = layers.Dense(...)`**: This is the final output layer of the model.
        - **`output_dim + 1`**: The number of units in this layer is the size of the vocabulary plus one. The extra unit is for the "blank" label required by the CTC loss function.
        - **`activation="softmax"`**: The softmax activation function is used to output a probability distribution over the vocabulary for each time step.

    ```python
        model = Model(inputs=inputs, outputs=output)
        return model
    ```
    - **`model = Model(inputs=inputs, outputs=output)`**: This line creates the Keras `Model` object by specifying the input and output layers.
    - **`return model`**: The function returns the compiled Keras model.

-   **`inferenceModel.py`**: This file defines the `ImageToWordModel` class, which is used for making predictions with the trained ONNX model.

    ```python
    import cv2
    import typing
    import numpy as np

    from mltu.inferenceModel import OnnxInferenceModel
    from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
    from mltu.transformers import ImageResizer

    class ImageToWordModel(OnnxInferenceModel):
    ```
    - **`class ImageToWordModel(OnnxInferenceModel):`**: This defines the `ImageToWordModel` class, which inherits from `OnnxInferenceModel` from the `mltu` library. This base class likely handles the loading of the ONNX model and setting up the inference session.

    ```python
        def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.char_list = char_list
    ```
    - **`def __init__(...)`**: This is the constructor for the class.
        - **`super().__init__(*args, **kwargs)`**: This calls the constructor of the parent class (`OnnxInferenceModel`) to initialize the ONNX inference session.
        - **`self.char_list = char_list`**: This stores the character list (vocabulary) that is needed for decoding the model's output.

    ```python
        def predict(self, image: np.ndarray):
    ```
    - **`def predict(self, image: np.ndarray):`**: This is the main method for making predictions. It takes an image (as a NumPy array) as input.

    ```python
            image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shapes[0][1:3][::-1])
    ```
    - **`image = ImageResizer.resize_maintaining_aspect_ratio(...)`**: This line preprocesses the input image. It resizes the image to the size expected by the model while maintaining the aspect ratio. This is important to avoid distorting the characters in the image.

    ```python
            image_pred = np.expand_dims(image, axis=0).astype(np.float32)
    ```
    - **`image_pred = np.expand_dims(image, axis=0).astype(np.float32)`**: This line prepares the image for inference.
        - **`np.expand_dims(image, axis=0)`**: The model expects a batch of images as input, so this adds an extra dimension to the image array to create a batch of size 1.
        - **`.astype(np.float32)`**: This converts the image data type to `float32`, which is the data type expected by the model.

    ```python
            preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
    ```
    - **`preds = self.model.run(...)`**: This is where the actual inference happens.
        - **`self.model.run(...)`**: This method of the ONNX runtime session executes the model.
        - **`self.output_names`**: This specifies the names of the output nodes of the model that we want to get.
        - **`{self.input_names[0]: image_pred}`**: This is a dictionary that maps the input names of the model to the input data.
        - **`[0]`**: The `run` method returns a list of outputs, so we take the first element, which is the output of our model.

    ```python
            text = ctc_decoder(preds, self.char_list)[0]
    ```
    - **`text = ctc_decoder(preds, self.char_list)[0]`**: This line decodes the output of the model to get the final predicted text.
        - **`ctc_decoder`**: This function (from `mltu.utils.text_utils`) implements the CTC decoding algorithm. It takes the probability distribution from the model and the character list as input and returns the most likely text sequence.
        - **`[0]`**: The `ctc_decoder` can decode a batch of predictions, so we take the first element to get the text for our single image.

    ```python
            return text
    ```
    - **`return text`**: The method returns the predicted text.

    ```python
    if __name__ == "__main__":
    ```
    - **`if __name__ == "__main__":`**: This block of code is executed only when the script is run directly (not when it is imported as a module). It is used for testing the `ImageToWordModel` class.
        - It loads the model configuration, creates an instance of the `ImageToWordModel`, and then iterates over the validation set.
        - For each image in the validation set, it makes a prediction and then calculates the Character Error Rate (CER) and Word Error Rate (WER) to evaluate the model's performance.

-   **`configs.py`**: This file defines the `ModelConfigs` class, which inherits from `BaseModelConfigs` in the `mltu` library.
    -   It defines all the hyperparameters and configurations for the model and training process, such as the image dimensions (`height`, `width`), `batch_size`, `learning_rate`, `train_epochs`, etc.
    -   It also defines the path where the trained model and other artifacts will be saved.

-   **`mltu/` directory**: This directory contains a custom Python library that provides a framework for the entire project.
    -   **`configs.py`**: The `BaseModelConfigs` class provides a simple way to manage and save model configurations as YAML files.
    -   **`dataProvider.py`**: The `DataProvider` class is a powerful and flexible data loader that can handle various types of datasets. It supports data preprocessing, augmentation, batching, shuffling, and caching.
    -   **`tensorflow/losses.py`**: The `CTCloss` class is a Keras-compatible implementation of the CTC loss function.
    -   **`tensorflow/callbacks.py`**: This file contains several useful custom Keras callbacks:
        -   `Model2onnx`: Converts the trained Keras model to ONNX format.
        -   `TrainLogger`: Logs the training progress to a file.
        -   `WarmupCosineDecay`: A learning rate scheduler that implements a warmup phase followed by a cosine decay.

## 3. Core Concepts and Methods

### 3.1. CRNN (Convolutional Recurrent Neural Network)

The CRNN architecture is a powerful combination of CNNs and RNNs for sequence recognition tasks.

-   **Convolutional Layers (The Feature Extractor):** The initial layers of the CRNN are convolutional. Their job is to scan the input image and extract a sequence of feature vectors. Each feature vector corresponds to a "receptive field" or a vertical slice of the image. This sequence of feature vectors is then passed to the recurrent layers.

-   **Recurrent Layers (The Sequence Labeler):** The recurrent layers are typically LSTMs or GRUs. In this project, Bidirectional LSTMs are used. These layers are designed to handle sequential data. They process the sequence of feature vectors from the convolutional layers and learn the contextual dependencies between them. The bidirectional nature of the LSTMs allows them to learn from both the left-to-right and right-to-left context, which is crucial for recognizing text.

### 3.2. CTC (Connectionist Temporal Classification) Loss

The CTC loss function is a key component of the project. It is used to train the CRNN model for sequence labeling tasks where the alignment between the input and the output is not known.

-   **The Alignment Problem:** In handwriting recognition, it is difficult to know which part of the image corresponds to which character in the label. The CTC loss function solves this problem by allowing the model to output a probability distribution over the vocabulary for each time step (i.e., for each feature vector from the convolutional layers).

-   **The "Blank" Label:** The CTC loss introduces a special "blank" label, which the model can output when it is not confident about any character. This allows the model to have multiple valid alignments for the same label. For example, for the word "cat", the model could output "c-a-t", "cc-aa-t", or "c-aat", where "-" represents the blank label. All of these would be collapsed to "cat" by the CTC decoding process.

### 3.3. ONNX (Open Neural Network Exchange)

ONNX is an open format for representing machine learning models. By converting the trained Keras model to ONNX, the project gains several benefits:

-   **Portability:** ONNX models can be run on a wide range of platforms, from cloud servers to edge devices.
-   **Performance:** The ONNX runtime is highly optimized for inference and can provide significant speedups compared to running the model in its original framework.
-   **Interoperability:** ONNX allows developers to use their preferred framework for training and then deploy the model in a different framework or language.

## 4. Dataset

The model is trained on the **IAM Handwriting Database**, which is one of the most widely used datasets for handwriting recognition research.

-   **Content:** The dataset contains a large number of images of handwritten English text, along with the corresponding ground truth transcriptions. The images are of varying quality and style, which makes the dataset challenging and suitable for training robust models.
-   **Data Parsing:** The `train.py` script parses the `sentences.txt` file from the dataset. This file contains information about each sentence in the dataset, including the path to the image file and the transcription. The script extracts this information and creates a list of (image_path, label) pairs.

## 5. How to Run the Project

### 5.1. Installation

Before running the project, you need to install the required dependencies. You can do this by running:

```bash
pip install -r requirements.txt
```

You also need to download the IAM Handwriting Database and extract it to the `dataset/archive` directory.

### 5.2. Training the Model

To train the model, you can run the `train.py` script:

```bash
python train.py
```

You can adjust the hyperparameters in `configs.py` and `train.py` to experiment with different model configurations. For example, you can change the `TEST_SUBSET` variable in `train.py` to train on a smaller subset of the data for faster development.

### 5.3. Running the Web Application

To start the Streamlit web application, run the following command:

```bash
streamlit run streamlit_app.py
```

This will open a new tab in your browser with the web application. You can then upload an image of handwritten text to get a prediction.

### 5.4. Running the REST API

To start the FastAPI server, run the following command:

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can use a tool like `curl` or Postman to send a POST request with an image file to the `/predict/` endpoint to get a prediction.
