import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
st.set_page_config(page_title="Medicinal Plant Identifier", page_icon="🌿", layout="centered")

# --- Load model and plant info ---
MODEL_PATH = r'C:\Users\AHMAD ALI\Desktop\medicinal-plant-identifier\models\medicinal_plant_model.h5'
INFO_PATH = r'C:\Users\AHMAD ALI\Desktop\medicinal-plant-identifier\plant_info.json'

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Load plant info
@st.cache_data
def load_plant_info():
    with open(INFO_PATH, 'r') as f:
        return json.load(f)

plant_info = load_plant_info()

# Class mapping
class_indices = {
    0: 'Aloevera', 1: 'Amla', 2: 'Amruta_Balli', 3: 'Arali', 4: 'Ashoka', 5: 'Ashwagandha',
    6: 'Avacado', 7: 'Bamboo', 8: 'Basale', 9: 'Betel', 10: 'Betel_Nut', 11: 'Brahmi',
    12: 'Castor', 13: 'Curry_Leaf', 14: 'Doddapatre', 15: 'Ekka', 16: 'Ganike', 17: 'Gauva',
    18: 'Geranium', 19: 'Henna', 20: 'Hibiscus', 21: 'Honge', 22: 'Insulin', 23: 'Jasmine',
    24: 'Lemon', 25: 'Lemon_grass', 26: 'Mango', 27: 'Mint', 28: 'Nagadali', 29: 'Neem',
    30: 'Nithyapushpa', 31: 'Nooni', 32: 'Pappaya', 33: 'Pepper', 34: 'Pomegranate',
    35: 'Raktachandini', 36: 'Rose', 37: 'Sapota', 38: 'Tulasi', 39: 'Wood_sorel'
}

st.title("🌿 Medicinal Plant Identifier")
st.markdown("Upload an image of a plant leaf to identify the plant and explore its **medicinal uses**.")

# Sidebar for file upload
with st.sidebar:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

# Process if image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)

    with st.spinner("🔍 Classifying..."):
        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions)) * 100
        plant_name = class_indices.get(predicted_class_index, "Unknown")

    # Prediction Output
    st.markdown(
        f"""
        <div style='background-color: black; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3>🌱 Predicted Plant: <span style='color: #2e7d32;'>{plant_name}</span></h3>
            <p style='font-size: 18px;'>Confidence: <strong>{confidence:.2f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display plant info
    if plant_name in plant_info:
        st.markdown("### 🌿 Medicinal Benefits")
        for benefit in plant_info[plant_name].get("Benefits", []):
            st.markdown(f"- ✅ {benefit}")

        st.markdown("### 🧪 Applications")
        for app in plant_info[plant_name].get("Applications", []):
            st.markdown(f"- 🧬 {app}")
    else:
        st.warning("⚠️ No additional information found for this plant.")

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ for Herbal Awareness</center>", unsafe_allow_html=True)
