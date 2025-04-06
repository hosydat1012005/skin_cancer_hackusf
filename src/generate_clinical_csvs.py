import med_minds
import pandas as pd
import os

# Create output directory if it doesn't exist
output_dir = "data/clinical_input_examples"
os.makedirs(output_dir, exist_ok=True)

# Diagnosis mapping: file-friendly name → full diagnosis name
cancer_map = {
    "Squamous_cell_carcinoma_NOS": "Squamous cell carcinoma, NOS",
    "Malignant_melanoma_NOS": "Malignant melanoma, NOS",
    "Acral_lentiginous_melanoma": "Acral lentiginous melanoma, malignant",
    "Verrucous_carcinoma_NOS": "Verrucous carcinoma, NOS",
    "Warty_carcinoma": "Warty carcinoma"
}

# SQL query template
query_template = """
SELECT 
    demographic_age_at_index,
    demographic_gender,
    demographic_ethnicity,
    demographic_race,
    diagnoses_tumor_grade,
    diagnoses_prior_malignancy,
    diagnoses_primary_diagnosis
FROM clinical
WHERE diagnoses_primary_diagnosis = '{diagnosis}'
"""

for filename, diagnosis in cancer_map.items():
    query = query_template.format(diagnosis=diagnosis)
    df = med_minds.query(query)

    # Drop critical fields only
    df_clean = df.dropna(subset=[
        "demographic_age_at_index",
        "demographic_gender"
    ])

    if not df_clean.empty:
        df_clean = df_clean.drop(columns=["diagnoses_primary_diagnosis"])
        path = os.path.join(output_dir, f"{filename}.csv")
        df_clean.to_csv(path, index=False)
        print(f"Saved: {path}")
    else:
        print(f"No usable data for '{diagnosis}'")
