import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import joblib
import pandas as pd

# Load Phase 1 image model
print("🔄 Loading image classification model...")
image_model = load_model("skin_cancer_model.h5")

# Load Phase 2 clinical model and encoders
clinical_model = joblib.load("skin_cancer_model.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Binary labels for image model
image_classes = ["benign", "malignant"]

# Clinical input columns
clinical_cols = [
    "demographic_age_at_index",
    "demographic_gender",
    "demographic_ethnicity",
    "demographic_race",
    "diagnoses_tumor_grade",
    "diagnoses_prior_malignancy"
]


# --- Phase 1: Image classification ---
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    prediction = image_model.predict(img_tensor)[0][0]

    if prediction >= 0.5:
        label = "malignant"
        confidence = prediction * 100
    else:
        label = "benign"
        confidence = (1 - prediction) * 100

    return label, confidence

# --- Phase 2: Clinical classification ---
def run_clinical_model():
    print("\n🧠 Phase 2: Clinical Diagnosis Prediction")
    print("📥 Enter input in this format (comma-separated):")
    print("age,gender,ethnicity,race,tumor_grade,prior_malignancy")
    print("Example: 41.0,Female,Unknown,Black or African American,G2,No")

    raw_input = input("\nPaste your input here: ").strip()
    values = [v.strip() for v in raw_input.split(",")]

    if len(values) != 6:
        print("❌ Invalid input. Please enter exactly 6 comma-separated values.")
        return

    df = pd.DataFrame([values], columns=clinical_cols)

    # Encode all but age
    for col in clinical_cols:
        if col == "demographic_age_at_index":
            df[col] = df[col].astype(float)
        elif col == "diagnoses_prior_malignancy":
            df[col] = df[col].str.lower().map({'yes': 1, 'no': 0}).fillna(-1)
        else:
            df[col] = df[col].astype(str).str.strip().replace('', 'unknown')
            le = feature_encoders[col]
            df[col] = le.transform(df[col])


    prediction = clinical_model.predict(df)
    cancer_label = target_encoder.inverse_transform(prediction)
    
    print(f"\n🩺 Clinical Prediction: {cancer_label[0]}")

# === App Entry ===
print("\n👋 Welcome to the Dual AI Skin Cancer Classifier")

img_path = input("\n🖼 Please enter the path to your skin image: ").strip()

if not os.path.exists(img_path):
    print("❌ Image not found. Please check the path.")
    exit()

label, confidence = classify_image(img_path)
print(f"\n✅ Phase 1 Prediction: {label.capitalize()} ({confidence:.2f}% confidence)")

if label == "malignant":
    run_phase2 = input("\n⚠️ The lesion appears malignant. Run Phase 2 clinical analysis? (y/n): ").strip()
    if run_phase2.lower() == "y":
        run_clinical_model()
else:
    print("\n👍 The lesion appears benign. No further analysis required.")
