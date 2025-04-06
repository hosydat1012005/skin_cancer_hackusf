import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the merged dataset
df = pd.read_csv("data/clinical_input_examples/merged_skin_cancer_data.csv")

# Step 2: Drop 'diagnoses_tumor_grade' since it's nearly all null

# Step 3: Handle missing values in 'diagnoses_prior_malignancy'
# Map Yes → 1, No → 0, NaN → -1 to indicate unknown
df['diagnoses_prior_malignancy'] = df['diagnoses_prior_malignancy'].map({'Yes': 1, 'No': 0})
df['diagnoses_prior_malignancy'] = df['diagnoses_prior_malignancy'].fillna(-1)

# Step 4: Encode other categorical features
label_encoders = {}  # To store encoders if needed later

for col in ['demographic_gender','demographic_ethnicity','demographic_race','diagnoses_tumor_grade']:
     # Convert to string, clean whitespace, replace empty strings or NaNs with 'unknown'
    df[col] = df[col].astype(str).str.strip().replace('', 'unknown')
    df[col] = df[col].replace('nan', 'unknown')  # Handles actual NaN converted to 'nan'

    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# Step 5: Encode target label (type of skin cancer)
target_encoder = LabelEncoder()
df['type_of_skin_cancer'] = target_encoder.fit_transform(df['type_of_skin_cancer'])

# Step 6: Split features and target
X = df.drop('type_of_skin_cancer', axis=1)
y = df['type_of_skin_cancer']

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 8: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate the model
y_pred = model.predict(X_test)

print("✅ Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

print("\n🧾 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Step 10: Visualize feature importance
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 4))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()

import joblib

# Save the trained model
joblib.dump(model, 'skin_cancer_model.pkl')

# Save the label encoder for decoding predictions
joblib.dump(label_encoders, 'feature_encoders.pkl')        # for clinical inputs
joblib.dump(target_encoder, 'target_encoder.pkl')    # for output class

print("✅ Model and label encoder saved successfully.")

