from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess


BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "seg"
MODEL_DIR = BASE_DIR / "Model"
MODELS = [
    {
        "name": "EfficientNetB0",
        "path": MODEL_DIR / "scene_efficientnetb0.h5",
        "preprocess": eff_preprocess,
    },
    {
        "name": "MobileNetV2",
        "path": MODEL_DIR / "scene_mobilenetv2.h5",
        "preprocess": mob_preprocess,
    },
]

IMG_SIZE = (224, 224)


def load_class_names(train_dir: Path) -> list[str]:
    if not train_dir.exists():
        return []
    return sorted([p.name for p in train_dir.iterdir() if p.is_dir()])


@st.cache_resource
def load_model(model_path: Path, model_mtime: float):
    return tf.keras.models.load_model(model_path)


def predict_image(model, image: Image.Image, preprocess_fn) -> tuple[int, float, np.ndarray]:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_fn(img_array)

    preds = model.predict(img_array, verbose=0)[0]
    best_idx = int(np.argmax(preds))
    best_score = float(preds[best_idx])
    return best_idx, best_score, preds


def format_top_k(preds: np.ndarray, class_names: list[str], k: int = 3) -> list[tuple[str, float]]:
    top_idx = np.argsort(preds)[::-1][:k]
    results = []
    for idx in top_idx:
        if class_names and idx < len(class_names):
            label = class_names[idx]
        else:
            label = f"class_{idx}"
        results.append((label, float(preds[idx])))
    return results


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp_vals = np.exp(logits)
    return exp_vals / np.sum(exp_vals)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    log_probs = np.log(clipped)
    scaled = log_probs / temperature
    return softmax(scaled)


def entropy_score(probs: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def entropy_contribution(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return -clipped * np.log(clipped)




st.set_page_config(page_title="Scene Classification Demo", layout="centered")

st.title("Scene Classification Demo")
st.write("Upload an image to get predicted scene labels from both models.")

class_names = load_class_names(TRAIN_DIR)
if not class_names:
    st.warning("No class folders found in seg/. Predictions will show class index.")

ood_method = st.selectbox(
    "OOD method",
    ["confidence", "entropy"],
    index=0,
)
temperature = st.slider("Temperature (for scaling)", 0.5, 5.0, 1.0, 0.1)

if class_names:
    entropy_max = float(np.log(len(class_names)))
else:
    entropy_max = 2.5

if ood_method == "confidence":
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.6, 0.01)
elif ood_method == "entropy":
    threshold = st.slider("Entropy threshold", 0.0, entropy_max, 0.9, 0.01)

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "gif"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Input image", use_container_width=True)

    available = [m for m in MODELS if m["path"].exists()]
    missing = [m for m in MODELS if not m["path"].exists()]

    for model_info in missing:
        st.error(f"Model not found: {model_info['path']}")

    if not available:
        st.stop()

    for model_info in available:
        st.subheader(model_info["name"])
        model_path = model_info["path"]
        model = load_model(model_path, model_path.stat().st_mtime)
        best_idx, best_score, preds = predict_image(
            model,
            image,
            model_info["preprocess"],
        )

        preds = apply_temperature(preds, temperature)
        best_idx = int(np.argmax(preds))
        best_score = float(preds[best_idx])

        if class_names and best_idx < len(class_names):
            best_label = class_names[best_idx]
        else:
            best_label = f"class_{best_idx}"

        entropy_val = entropy_score(preds)

        if ood_method == "confidence":
            is_unknown = best_score < threshold
            score_label = f"confidence {best_score:.3f}"
        elif ood_method == "entropy":
            is_unknown = entropy_val > threshold
            score_label = f"entropy {entropy_val:.3f}"

        if is_unknown:
            st.warning(f"Low confidence: {score_label}. Label = UNKNOWN")
        else:
            st.success(f"Prediction: {best_label} ({score_label})")

        if ood_method == "entropy":
            per_class_scores = entropy_contribution(preds)
            score_column = "entropy_contribution"
            table_title = "All class entropy contributions:"
        else:
            per_class_scores = preds
            score_column = "confidence"
            table_title = "All class confidences:"

        if class_names:
            all_rows = [(name, float(score)) for name, score in zip(class_names, per_class_scores)]
        else:
            all_rows = [(f"class_{idx}", float(score)) for idx, score in enumerate(per_class_scores)]

        all_df = pd.DataFrame(all_rows, columns=["label", score_column]).sort_values(
            score_column,
            ascending=False,
        )
        st.write(table_title)
        st.dataframe(all_df, hide_index=True, use_container_width=True)

