import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage

# importation des données
data_pdf = pd.read_csv("data/paysim_dataset.csv")

# split des données
features_pdf, target_pdf = data_pdf.drop("isFraud", axis=1), data_pdf["isFraud"]

X_train_pdf, X_test_pdf, y_train_pdf, y_test_pdf = train_test_split(
    features_pdf, target_pdf, test_size=0.15,
    shuffle=True, stratify=target_pdf, random_state=42)


# connection au client gcs
client = storage.Client()

# importer le train dataset
train_pdf = pd.concat([X_train_pdf, y_train_pdf], axis=1)
path = "gs://data-epip/train/paysim.parquet"
train_pdf.to_parquet(path, engine='pyarrow')

# importter le test dataset
test_pdf = pd.concat([X_test_pdf, y_test_pdf], axis=1)
path = "gs://data-epip/test/paysim.parquet"
test_pdf.to_parquet(path, engine='pyarrow')
