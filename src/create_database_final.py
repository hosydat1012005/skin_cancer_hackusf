import os
import pandas as pd

# Folder containing CSV files
data_folder = 'data/clinical_input_examples/'

# Cancer type mapping from filename to label
filename_to_label = {
    'Acral_lentiginous_melanoma.csv': 'acral',
    'Malignant_melanoma_NOS.csv': 'malignant_melanoma',
    'Squamous_cell_carcinoma_NOS.csv': 'sqamous',
    'Verrucuous_carcinoma_NOS.csv': 'verrucuous',
    'Warty_carcinoma.csv': 'warty'
}

# List to collect DataFrames
dataframes = []

# Iterate through the filenames and process each file
for filename, label in filename_to_label.items():
    file_path = os.path.join(data_folder, filename)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['type_of_skin_cancer'] = label
        dataframes.append(df)
    else:
        print(f"File not found: {file_path}")

# Concatenate all DataFrames into one
merged_df = pd.concat(dataframes, ignore_index=True)

# Shuffle the dataset (optional but recommended)
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Preview the result
print("Merged Data Preview:")
print(merged_df.head())

# Save to new CSV file
output_path = os.path.join(data_folder, 'merged_skin_cancer_data.csv')
merged_df.to_csv(output_path, index=False)
print(f"\nMerged CSV saved to: {output_path}")
