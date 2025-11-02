import streamlit as st
from PIL import Image
import numpy as np
import cv2
from mltu.configs import BaseModelConfigs
from inferenceModel import ImageToWordModel

st.title("Handwriting Recognition")

# Load the model
@st.cache_resource
def load_model():
    configs = BaseModelConfigs.load("Models/04_sentence_recognition/202510280421/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # Convert bytes data to numpy array
    nparr = np.frombuffer(bytes_data, np.uint8)
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Predicting...")

    try:
        prediction_text = model.predict(image)
        st.write(f"**Prediction:** {prediction_text}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")