
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from mltu.configs import BaseModelConfigs
from inferenceModel import ImageToWordModel
import base64

# Page Config
st.set_page_config(
    page_title="The Magical Quill",
    page_icon="✍️",
    layout="wide"
)

# Inject CSS for gradient background and uploader highlight
st.markdown("""
<style>
@keyframes mild-glow {
    0% { text-shadow: 0 0 3px rgba(255, 215, 0, 0.1); } /* 10% opacity */
    50% { text-shadow: 0 0 6px rgba(255, 215, 0, 0.2); } /* 20% opacity */
    100% { text-shadow: 0 0 3px rgba(255, 215, 0, 0.1); }
}

h1, h2, h3 {
    transition: text-shadow 0.4s ease-in-out; /* Smooth transition for hover */
}

h1:hover, h2:hover, h3:hover {
    text-shadow: 0 0 5px #FFD700; /* Milder golden glow on hover */
}

h4 {
    animation: mild-glow 5s ease-in-out infinite; /* Very slow animation */
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #0f0c29, #302b63, #24243e);
}

[data-testid="stFileUploader"] {
    border: 1px solid #FFD700; /* Golden border */
    border-radius: 5px;
    padding: 10px;
    transition: box-shadow 0.8s ease-in-out; /* Slow glow animation */
}

[data-testid="stFileUploader"]:hover {
    box-shadow: 0 0 15px #FFD700; /* Golden glow on hover */
}

.prediction-box {
    border: 1px solid #FFD700;
    border-radius: 5px;
    padding: 10px;
    transition: box-shadow 0.8s ease-in-out;
}

.prediction-box:hover {
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.3); /* 30% opacity golden glow */
}
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    configs = BaseModelConfigs.load("Models/04_sentence_recognition/202510280421/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    return model

model = load_model()

# App Title
st.title("✍️ The Magical Quill")
st.markdown("---")

# About Section
st.header("The Legend of the Magical Quill")
st.markdown("""
In a library lost to time, there was a magical quill that could bring any handwritten word to life. Scribes would spend years learning its secrets, watching as their ink-on-paper creations danced and swirled into reality.

This app is our attempt to recreate a piece of that magic. We've trained a digital scribe—a powerful deep learning model—to read the words you provide. While it can't make them dance (yet!), it can decipher the most intricate of handwriting, turning your images into digital text in the blink of an eye.
""")
st.markdown("---")


# How to Use Section
st.header("Harness the Magic")
st.markdown("""
1.  **Find a suitable scroll:** For the best results, use a clear, well-lit image of a single word or a short line of text. The model loves high-contrast images (e.g., dark ink on a light background).
    *   **Ideal Dimensions:** Aim for images with a width-to-height ratio of around 10:1 (e.g., 1000x100 pixels). The model works best with text that is relatively horizontal.
    *   **Text Length:** The scribe is most proficient with single words or short phrases, ideally up to 20-30 characters. Longer sentences might be split or less accurately recognized.
2.  **Present it to the quill:** Upload your image using the uploader below.
3.  **Witness the magic:** The digital scribe will analyze the image and reveal the hidden text.
""")
st.markdown("---")

# Handwriting Recognition Section
st.header("✨ Unleash the Digital Scribe! ✨")
st.markdown("#### Got a handwritten note you can't decipher? A secret message? Or just curious? Upload it here and let the magic begin!")
uploaded_file = st.file_uploader("Choose an image to bring to life...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # Convert bytes data to numpy array
    nparr = np.frombuffer(bytes_data, np.uint8)
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not decode the image. Please ensure it's a valid image file.")
    else:
        left, center, right = st.columns([1, 3, 1])
        with center:
            st.markdown("<div style='text-align: center;'><h3>Message to be deciphered:</h3></div>", unsafe_allow_html=True)
            
            # Encode image to base64
            _, buffer = cv2.imencode('.png', image)
            img_str = base64.b64encode(buffer).decode()
            st.markdown(f"""
<div style='display: flex; justify-content: center; flex-direction: column; align-items: center;'>
    <img src='data:image/png;base64,{img_str}' width='600' />
    <p style='font-size: 14px; color: #AAAAAA; margin-top: 5px;'>Your Scroll</p>
</div>
""", unsafe_allow_html=True)
            
            st.markdown("<div style='text-align: center;'><h3 style='font-weight: bold; font-size: 28px;'>The Scribe's Prediction:</h3></div>", unsafe_allow_html=True)
            with st.spinner("The quill is writing..."):
                try:
                    prediction_text = model.predict(image)
                    st.markdown(f"<div class='prediction-box' style='text-align: center;'><p style='font-size: 24px; font-weight: bold; color: #FFD700;'>{prediction_text}</p></div>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True) # Add spacing
                    st.balloons()
                    st.markdown("<div style='text-align: center;'><h4>The ancient script has been deciphered! The quill returns to its slumber, awaiting the next enigma. Feel free to present another scroll to awaken its magic.</h4></div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"The magic fizzled... Error: {e}")





# Content of requirements.txt
# keras
# numpy
# onnx
# onnxruntime
# pandas
# Pillow
# PyYAML
# qqdm
# scipy
# tensorflow
# tf2onnx
# tqdm
# ultralytics
# fastapi
# uvicorn
# python-multipart
