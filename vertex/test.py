import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    average_precision_score
)

path = "gs://data-epip/train/paysim.parquet"
test_pdf = pd.read_parquet(path, engine='pyarrow')


X_test, y_test = test_pdf.drop("isFraud", axis=1), test_pdf["isFraud"]
X_test.drop(["nameOrig", "nameDest", "step", "isFlaggedFraud"], axis=1, inplace=True)

model = joblib.load("model.joblib")
probas =  model.predict_proba(X_test)[:, 1]

precision, recall, threshold = precision_recall_curve(y_test, probas)
print(f"precision : {np.mean(precision)}")
print(f"auprc : {auc(recall, precision)}")
print(f"average_precision : {average_precision_score(y_test, probas)}")