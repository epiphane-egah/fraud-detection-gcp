#!/usr/bin/envbash
set -e

PROJECT_ID="fraude-detection-gcp"
REGION="europe-west1"
BUCKET="data-epip"      
REPO="ml"               
PIPELINE_NAME="ml-pipeline"
SERVICE_EMAIL="${PIPELINE_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
USER_EMAIL="$(gcloud config get-value account)"

# Création du projet
gcloud projects describe fraude-detection-gcp 1>/dev/null 2>&1 || \
    gcloud projects create fraude-detection-gcp

# Définir comme projet actuelle
gcloud config set project "$PROJECT_ID"

# Activer les APIs nécessaires
gcloud services enable storage.googleapis.com \
  aiplatform.googleapis.com \
  cloudfunctions.googleapis.com \
  cloudscheduler.googleapis.com \
  bigquery.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com

gcloud storage buckets describe gs://data-epip 1>/dev/null 2>&1 || \
  gcloud storage buckets create gs://data-epip \
  --location='europe-west1'


gcloud artifacts repositories describe ml --location='europe-west1' 1>/dev/null 2>&1 \
  gcloud artifacts repositories create ml \
  --repository-format=docker \
  --location=europe-west1

gcloud builds submit \
  --tag europe-west1-docker.pkg.dev/$PROJECT_ID/ml/fraud-train:latest .


# 1. Création du compte de service
gcloud iam service-accounts describe "$PIPELINE_NAME" 1>/dev/null 2>&1 || \
  gcloud iam service-accounts create "$PIPELINE_NAME" \
  --display-name="Vertex AI pipeline"

# 2. Rôles au niveau du projet
for ROLE in roles/aiplatform.user roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SERVICE_EMAIL}" \
    --role="$ROLE" \
    --condition=None
done

# 3. Accès au bucket (uniquement celui-ci)
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SERVICE_EMAIL}" \
  --role="roles/storage.objectAdmin"

# 4. Lecture de l'image Docker (uniquement ce dépôt)
gcloud artifacts repositories add-iam-policy-binding "$REPO" \
  --location="$REGION" \
  --member="serviceAccount:${SERVICE_EMAIL}" \
  --role="roles/artifactregistry.reader"

# 5. Droit d'agir "en tant que" ce compte de service
#    a) le compte lui-même (les jobs lancés par le pipeline s'exécutent sous son identité)
gcloud iam service-accounts add-iam-policy-binding "$SERVICE_EMAIL" \
  --member="serviceAccount:${SERVICE_EMAIL}" \
  --role="roles/iam.serviceAccountUser"

gcloud iam service-accounts add-iam-policy-binding "$SERVICE_EMAIL" \
  --member="user:${USER_EMAIL}" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SERVICE_EMAIL}" \
  --role="roles/storage.objectViewer"
  
gcloud storage buckets add-iam-policy-binding gs://data-epip \
  --member="serviceAccount:${SERVICE_EMAIL}" \
  --role="roles/storage.objectCreator"



