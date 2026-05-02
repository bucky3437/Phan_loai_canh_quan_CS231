from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from pathlib import Path

# ====== Cấu hình ======
BASE_DIR = Path(r"c:\Users\Admin\OneDrive\Documents\File_save_bai_tap_random\Train")
SEG_DIR = BASE_DIR / "seg"
OUTPUT_DIR = BASE_DIR / "images"

# Tạo folder output nếu chưa có
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Models to generate
MODELS = [
    ("scene_mobilenetv2.h5", "MobileNetV2", "examples_6classes_mobilenetv2.png"),
    ("scene_efficientnetb0.h5", "EfficientNetB0", "examples_6classes_efficientnetb0.png")
]

class_names = sorted([d for d in os.listdir(SEG_DIR) if os.path.isdir(SEG_DIR / d)])
print(f"Classes: {class_names}\n")

# ====== Lấy 1 ảnh mẫu từ mỗi lớp ======
sample_images = []
true_labels = []

for class_name in class_names:
    class_dir = SEG_DIR / class_name
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if images:
        img_path = class_dir / images[0]
        img = Image.open(img_path).convert('RGB')
        sample_images.append(img)
        true_labels.append(class_name)
        print(f"Loaded: {class_name} - {img_path.name}")
    else:
        print(f"No images found in {class_dir}")

print(f"\nLoaded {len(sample_images)} sample images")

for model_file, model_name, output_file in MODELS:
    MODEL_PATH = BASE_DIR / model_file
    OUTPUT_PATH = OUTPUT_DIR / output_file

    if "mobilenet" in model_file.lower():
        preprocess_func = tf.keras.applications.mobilenet_v2.preprocess_input
    else:
        preprocess_func = tf.keras.applications.efficientnet.preprocess_input

    print(f"\nLoading model: {model_name}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # ====== Dự đoán cho mỗi ảnh ======
    print("Making predictions...")
    predictions = []
    for i, img in enumerate(sample_images):
        # Resize và tiền xử lý
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = preprocess_func(img_array)
        img_batch = np.expand_dims(img_array, axis=0)

        # Dự đoán
        pred = model.predict(img_batch, verbose=0)
        pred_class_idx = np.argmax(pred)
        pred_class = class_names[pred_class_idx]
        pred_conf = pred[0, pred_class_idx]

        predictions.append((pred_class, pred_conf))
        print(f"{true_labels[i]:12} -> {pred_class:12} (conf: {pred_conf:.4f})")

    # ====== Tạo grid 2x3 với 6 ảnh ======
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=100)
    axes = axes.flatten()

    for idx, (ax, img, true_label, (pred_label, pred_conf)) in enumerate(
        zip(axes, sample_images, true_labels, predictions)
    ):
        # Hiển thị ảnh
        ax.imshow(img)
        ax.axis('off')

        # Tạo text nhãn
        text = f"True: {true_label}\nPred: {pred_label}\nConf: {pred_conf:.3f}"

        # Màu sắc: xanh nếu đúng, đỏ nếu sai
        color = 'green' if true_label == pred_label else 'red'

        # Thêm text box dưới ảnh
        ax.text(0.5, -0.15, text,
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=11, weight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                color=color)

    plt.suptitle(f'Scene Classification Examples - {model_name}',
                 fontsize=16, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ====== Lưu ảnh ======
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved to: {OUTPUT_PATH}")
    plt.close()

print("\n✓ Hoàn tất! Đã tạo ảnh cho cả 2 model.")
