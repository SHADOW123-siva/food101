import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os

# Custom loss function to handle deserialization
from tensorflow.keras.losses import SparseCategoricalCrossentropy

class CustomSparseCategoricalCrossentropy(SparseCategoricalCrossentropy):
    def __init__(self, reduction='auto', name='sparse_categorical_crossentropy', from_logits=False, ignore_class=None):
        super().__init__(reduction=reduction, name=name, from_logits=from_logits, ignore_class=ignore_class)

# Register the custom loss function
tf.keras.utils.get_custom_objects().update({
    'SparseCategoricalCrossentropy': CustomSparseCategoricalCrossentropy
})

# Set working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'Food_classifier.h5')

# Load the pre-trained model
model = tf.keras.models.load_model(model_path, custom_objects={'SparseCategoricalCrossentropy': CustomSparseCategoricalCrossentropy})

# Define class labels for Fashion MNIST dataset
class_names = ['burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature', 'dal_makhani', 'dhokla', 'fried_rice', 'idli', 'jalebi','kadhai_paneer','kathi_roll','kulfi','masala_dosa','momos','paani_puri','pakode','pav_bhaji','pizza','samosa']

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = cv2.imread(image)
    img = cv2.resize(img,(28, 28))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

# Streamlit App
st.title('Food Item Classifier')

uploaded_image = st.file_uploader("Upload a Fashion MNIST image (Don't upload high quality images)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = cv2.imread(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image
            img_array = preprocess_image(uploaded_image)

            # Make a prediction using the pre-trained model
            result = model.predict(img_array)
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            st.success(f'Prediction: {prediction}')
