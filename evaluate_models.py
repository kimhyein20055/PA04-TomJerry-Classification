
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. Project settings
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
TEST_DIR = BASE_DIR / "dataset" / "test"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model folder names
MODELS = [
    "model1_tom200_jerry200",
    "model2_tom800_jerry800",
    "model3_tom200_jerry800",
    "model4_tom800_jerry200",
    "model5_tom500_jerry500"
]

CLASSES = ["tom", "jerry"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

IMAGE_SIZE = (224, 224)

# ==========================================
# 2. Keras compatibility
# ==========================================

class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, groups=None, **kwargs):
        super().__init__(*args, **kwargs)


# ==========================================
# 3. Image preprocessing
# ==========================================

def preprocess_image(image_path):

    with Image.open(image_path) as image:
        image = image.convert("RGB")

        image = ImageOps.fit(
            image,
            IMAGE_SIZE,
            method=Image.Resampling.LANCZOS
        )

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

    # Normalize image values to [-1, 1]
    image_array = (image_array / 127.5) - 1.0

    return image_array


# ==========================================
# 4. Load fixed test dataset
# ==========================================

test_images = []
true_labels = []
filenames = []

for class_name in CLASSES:

    folder = TEST_DIR / class_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Test folder not found: {folder}"
        )

    images = sorted([
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    print(f"{class_name}: {len(images)} test images")

    if len(images) != 200:
        raise ValueError(
            f"{class_name} must have exactly 200 test images."
        )

    for image_path in images:

        test_images.append(
            preprocess_image(image_path)
        )

        true_labels.append(class_name)
        filenames.append(image_path.name)


X_test = np.stack(test_images)
y_true = np.array(true_labels)

print("\nTotal test images:", len(X_test))

# ==========================================
# 5. Evaluate five models
# ==========================================

summary = []

for model_name in MODELS:

    print(f"\n=== Evaluating {model_name} ===")

    folder = MODEL_DIR / model_name

    model_path = folder / "keras_model.h5"
    labels_path = folder / "labels.txt"

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    # Read class labels
    with open(
        labels_path,
        "r",
        encoding="utf-8-sig"
    ) as f:

        class_names = [
            line.strip().split(" ", 1)[-1].strip()
            for line in f
            if line.strip()
        ]

    print("Model classes:", class_names)

    if len(class_names) != 2 or set(class_names) != set(CLASSES):
        raise ValueError(
            f"Unexpected labels in {model_name}: {class_names}"
        )

    # Load trained Keras model
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "DepthwiseConv2D": CompatibleDepthwiseConv2D
        }
    )

    # Predict test images
    predictions = model.predict(
        X_test,
        batch_size=32,
        verbose=0
    )

    predicted_indices = np.argmax(
        predictions,
        axis=1
    )

    y_pred = np.array([
        class_names[index]
        for index in predicted_indices
    ])

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASSES,
            zero_division=0
        )
    )

    macro_f1 = np.mean(f1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    # Save class-level metrics
    class_results = pd.DataFrame({
        "Class": CLASSES,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Support": support
    })

    class_results.to_csv(
        RESULTS_DIR / f"{model_name}_class_metrics.csv",
        index=False
    )

    # Save individual predictions
    prediction_results = pd.DataFrame({
        "Filename": filenames,
        "True_Label": y_true,
        "Predicted_Label": y_pred,
        "Correct": y_true == y_pred
    })

    prediction_results.to_csv(
        RESULTS_DIR / f"{model_name}_predictions.csv",
        index=False
    )

    # Create confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASSES
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    ).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title(
        f"{model_name} Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / f"{model_name}_confusion_matrix.png",
        dpi=200
    )

    plt.close()

    # Collect model results
    summary.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro_F1": macro_f1,
        "Tom_Recall": recall[0],
        "Jerry_Recall": recall[1]
    })

    # Clear memory
    del model
    tf.keras.backend.clear_session()


# ==========================================
# 6. Save comparison results
# ==========================================

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)

print("\n=== Final Model Comparison ===")

print(
    summary_df.to_string(index=False)
)

print("\nEvaluation completed!")

print(
    "Results saved to:",
    RESULTS_DIR
)
