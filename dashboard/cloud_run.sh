# autoriser l'utilisation de l'api artifact registry
gcloud services enable artifactregistry.googleapis.com

# créer un repo
gcloud artifacts repositories describe fraude-detection-dash --location=europe-west1 &>/dev/null || \
gcloud artifacts repositories create fraude-detection-dash \
  --repository-format=docker \
  --location=europe-west1

# 3. Configurer l'authentification Docker pour pouvoir push
gcloud auth configure-docker europe-west1-docker.pkg.dev


TAG=$(date +%Y-%m-%d_%Hh%M)
docker buildx build --platform linux/amd64 \
    -t predictions-img:$TAG .
docker tag predictions-img:$TAG europe-west1-docker.pkg.dev/fraude-detection-gcp/fraude-detection-dash/predictions-img:$TAG
docker push europe-west1-docker.pkg.dev/fraude-detection-gcp/fraude-detection-dash/predictions-img:$TAG

gcloud run deploy fraud-monitoring-dash \
  --image=europe-west1-docker.pkg.dev/fraude-detection-gcp/fraude-detection-dash/predictions-img:$TAG \
  --platform=managed \
  --region=europe-west1 \
  --port=8050 \
  --allow-unauthenticated  
  #--service-account=container-runtime@jan26-cde-crypto.iam.gserviceaccount.com  \