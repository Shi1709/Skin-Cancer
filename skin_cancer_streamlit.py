import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Skin Cancer Detection (CNN)", page_icon="🩺", layout="wide")

MODEL_PATH = "final_model.keras"
TARGET_SIZE = (224, 224)

# ------------------------------
# APP FUNCTIONS
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load Keras model and align class names with output."""
    model = tf.keras.models.load_model(model_path, compile=False)

    # Original 7 ISIC class names
    class_names = [
        "Actinic Keratoses (AK)",
        "Basal Cell Carcinoma (BCC)",
        "Benign Keratosis (BKL)",
        "Dermatofibroma (DF)",
        "Melanoma (MEL)",
        "Nevus (NV)",
        "Vascular Lesion (VASC)"
    ]

    # Adjust class_names to match model output
    num_classes = model.output_shape[-1]
    if len(class_names) > num_classes:
        class_names = class_names[:num_classes]
    elif len(class_names) < num_classes:
        class_names += [f"Class {i}" for i in range(len(class_names), num_classes)]

    return model, class_names

def preprocess_image(img: Image.Image, target_size=(224, 224)):
    """Convert uploaded image to model input."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def get_top_k(probs: np.ndarray, class_names: list, k=3):
    """Return top-k class names and probabilities."""
    top_idx = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in top_idx]

# ------------------------------
# APP LAYOUT
# ------------------------------
st.title("🩺 Skin Cancer Detection – CNN")
st.caption("Upload a dermoscopic image to get a model prediction. This is a clinical aid, not a diagnosis.")

# Load model
model = None
CLASS_NAMES = None
if os.path.exists(MODEL_PATH):
    with st.spinner("Loading model..."):
        model, CLASS_NAMES = load_model(MODEL_PATH)
        st.success(f"Model loaded: `{MODEL_PATH}`")
else:
    st.warning(f"Model file `{MODEL_PATH}` not found. Place your trained Keras model in the app directory.")

# Image uploader
uploaded = st.file_uploader("Upload a dermoscopic image (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    orig_img = Image.open(uploaded)
    st.image(orig_img, caption="Uploaded Image", use_container_width=True)

    arr = preprocess_image(orig_img, TARGET_SIZE)
    if model is not None and CLASS_NAMES is not None:
        with st.spinner("Inferencing..."):
            logits = model.predict(arr, verbose=0)[0]
            probs = tf.nn.softmax(logits).numpy()

        # Safe prediction mapping
        pred_idx = int(np.argmax(probs))
        pred_idx = min(pred_idx, len(CLASS_NAMES)-1)
        pred_label = CLASS_NAMES[pred_idx]
        pred_conf = float(probs[pred_idx])

        st.subheader("Prediction Result")
        st.metric("Predicted Class", pred_label, f"{pred_conf*100:.2f}%")

        st.write("Top Probabilities:")
        for lbl, p in get_top_k(probs, CLASS_NAMES, k=3):
            st.write(f"{lbl}: {p*100:.2f}%")
