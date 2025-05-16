
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("models/chest_xray_cnn.keras")
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Pneumonia Identification System
         """
         )

file = st.file_uploader("Please upload a chest scan file", type=["jpg","jpeg", "png"])

def preprocess_image(uploaded_file):
    img_height, img_width = 180, 180
    img = image.load_img(uploaded_file, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

if file is None:
    st.text("Please upload an image file")
else:
    class_names = ['NORMAL', 'PNEUMONIA']
    
    # Show uploaded image
    uploaded_img = Image.open(file).convert("RGB")
    st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)

    img_arr = preprocess_image(file)

    # Predict
    prediction = model.predict(img_arr)[0][0]  # Get scalar
    confidence = prediction if prediction > 0.5 else 1 - prediction
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    
    st.write(f"### Prediction: `{label}`")
    st.write(f"**Confidence:** `{confidence:.4f}`")