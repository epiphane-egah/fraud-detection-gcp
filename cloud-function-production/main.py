from google.cloud import aiplatform
from utils import generate_transactions
from datetime import datetime
import random
import time

BUCKET_NAME = "predictions-fraud"
aiplatform.init(project="VOTRE_PROJECT_ID", location="europe-west1")
# endpoint = aiplatform.Endpoint("VOTRE_ENDPOINT_ID")

n_echantillon = 100
n = 1000
for _ in range(n):
    time.sleep(1)
    # génère une transaction
    df = generate_transactions(
        n=n_echantillon,
        fraud_rate=0.02
        )
    
    # vertex ai 
    try:
        df = df.drop(["isFlaggedFraud", "isFraud"], axis=1)
    except KeyError as error:
        print(error)
    instances = df.values
    # prediction = endpoint.predict(instances=instances)
    # df["predictions"] = prediction.predictions
    df["predictions"] = random.choices(population=[0,1],
                                       weights=[0.9, 0.1],
                                       k=(n_echantillon))
    df["temps"] = datetime.now()
    
    # sérialisation (dans un bucket)
    id = datetime.now().strftime("%y-%m-%d:%H-%M")
    df.to_parquet(f"gs://{BUCKET_NAME}/transactions_simulees_{id}.parquet",
                  engine="pyarrow")