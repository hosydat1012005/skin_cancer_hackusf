# DUAL-PHASE AI SKIN CANCER CLASSIFIER

This project implements a two-phase AI system for skin cancer diagnosis, combining image classification (using lesion photos) and clinical data analysis (via the NIH MINDS oncology database). It is built to assist healthcare professionals in early detection and cancer subtype classification.

## OVERVIEW

- Phase 1: Image-Based Classification
A Convolutional Neural Network (CNN) trained on a Kaggle dataset classifies skin lesion images as benign or malignant.

- Phase 2: Clinical Subtype Prediction
If Phase 1 predicts "malignant", the user may enter relevant clinical features. A Random Forest model then predicts the likely cancer subtype based on real-world patient records from the MINDS database.

## HOW TO RUN THE APP

 1. Install Requirements
pip install -r requirements.txt
 2. Run the Classifier App
python app.py
Upload an image (Phase 1)
If malignant, enter 6 comma-separated clinical features:
age,gender,ethnicity,race,tumor_grade,prior_malignancy
Example: 55.0,Female,Unknown,White,G1,No

- Phase 2: Clinical Data Source (MINDS)

This project uses real clinical data from the NIH MINDS (Multimodal INtegrated Data System), a powerful oncology database.

⚙️ Setup MINDS Locally (If Needed)
Install Docker
https://www.docker.com/products/docker-desktop/
Run PostgreSQL with MINDS schema:
docker run -d --name minds \
  -e POSTGRES_PASSWORD=my-secret-pw \
  -e POSTGRES_DB=minds \
  -p 5432:5432 postgres
Install the Python client:
pip install med-minds
Create a .env file:
HOST=127.0.0.1
PORT=5432
DB_USER=postgres
PASSWORD=my-secret-pw
DATABASE=minds
Download and populate data:
import med_minds
med_minds.update()

##  Large Dataset Note

⚠️ The full clinical dataset clinical.csv (~146MB) is NOT uploaded to GitHub due to GitHub's 100MB limit.

You can generate it manually by:
Running the below command to generate CSV from MINDS
python src/generate_clinical_csvs.py

📊 Example Clinical Inputs

Here are 5 supported cancer subtypes your model can classify:

Squamous cell carcinoma, NOS
Malignant melanoma, NOS
Acral lentiginous melanoma, malignant
Verrucous carcinoma, NOS
Warty carcinoma
Demo CSVs for each type are available in:

data/clinical_input_examples/

 Citation

Please cite the MINDS database when using this project in research:

@Article{s24051634,
    AUTHOR = {Tripathi, Aakash and Waqas, Asim and Venkatesan, Kavya and Yilmaz, Yasin and Rasool, Ghulam},
    TITLE = {Building Flexible, Scalable, and Machine Learning-Ready Multimodal Oncology Datasets},
    JOURNAL = {Sensors},
    VOLUME = {24},
    YEAR = {2024},
    NUMBER = {5},
    ARTICLE-NUMBER = {1634},
    URL = {https://www.mdpi.com/1424-8220/24/5/1634},
    ISSN = {1424-8220},
    DOI = {10.3390/s24051634}
}

## Project Member: Dat Ho, Trung Lam

