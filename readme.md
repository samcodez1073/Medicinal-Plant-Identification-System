# 🌿 Medicinal Plant Identifier

This project is a deep learning-powered **web application** that classifies medicinal plants based on leaf images and provides information about their medicinal benefits and applications.

Built using **TensorFlow**, **Keras**, and **Streamlit**, it combines a MobileNetV2-based model with a clean user interface for educational and herbal research purposes.

---

## 📸 Features

- ✅ Upload a plant leaf image and get predictions.
- 🌱 Displays predicted plant name with confidence score.
- 💊 Provides medicinal benefits and applications from a JSON knowledge base.
- ⚙️ Includes custom training script with fine-tuning using MobileNetV2.

---

## 💠 Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- Pillow (PIL)
- NumPy
- MobileNetV2 (Transfer Learning)

---

## 📁 Project Structure

```
medicinal-plant-identifier/
├── app.py                         # Streamlit app (main interface)
├── plant_info.json                # JSON file with plant details
├── data/
│   └── Medicinal plant dataset/   # Image folders for training
├── Model/
│   ├── model.py                   # Model training and fine-tuning script
│   └── medicinal_plant_model.h5   # Trained Keras model
```

---

## 🚀 How to Run the App

### 1. Install Dependencies

```bash
pip install streamlit tensorflow pillow numpy
```

### 2. Run the Web App

```bash
streamlit run app.py
```

### 3. Upload an Image

Upload a plant leaf image (JPG, JPEG, or PNG) and get instant predictions with confidence and medicinal info.

---

## 🧠 Model Training

### Dataset

Use a directory structure like:

```
data/
└── Medicinal plant dataset/
    ├── Aloevera/
    ├── Neem/
    └── ... (40 plant folders)
```

### Training Steps

- Uses **MobileNetV2** as base model.
- Image augmentation for robust learning.
- Initial training with frozen base, followed by fine-tuning.
- Model saved as `Model/medicinal_plant_model.h5`.

Run `Model/model.py` to retrain the model with updated data.

---

## 📖 Example Plant Info (JSON format)

```json
{
  "Neem": {
    "Benefits": ["Anti-bacterial", "Boosts immunity"],
    "Applications": ["Used in skin creams", "Dental care products"]
  },
  "Tulasi": {
    "Benefits": ["Reduces stress", "Fights infections"],
    "Applications": ["Herbal tea", "Ayurvedic medicine"]
  }
}
```

---

## 🧪 Future Enhancements

- 📲 Deploy to cloud (Streamlit Cloud, Hugging Face Spaces, etc.)
- 📱 Mobile-friendly version
- 🌐 Multilingual support
- 📦 Build a REST API

---

## 👌 Acknowledgements

- MobileNetV2 - TensorFlow Applications

---

## 🧑‍💻 Author

**Sayyed Ahmed Ali**\
Feel free to connect and suggest improvements!

