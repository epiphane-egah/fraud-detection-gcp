# Projet de détection de fraude — GCP

## 0. Préparation de l'environnement GCP

- Création et configuration d'un projet Google Cloud Platform (GCP).
- Activation des API nécessaires, notamment Vertex AI et Cloud Storage.
- Création d'un bucket Google Cloud Storage (GCS) pour stocker les données, les modèles et les prédictions.
- Configuration de l'authentification et des permissions IAM.

**Infrastructure GCP :** Cloud Storage, IAM.



## 1. Exploration et prétraitement des données

- Exploration et analyse des données.
- Détection et traitement des valeurs aberrantes (outliers).
- Imputation des valeurs manquantes avec MICE.
- Analyse et traitement de l'asymétrie des données.
- Suréchantillonnage de la classe minoritaire avec SMOTE.



## 2. Préparation des données

- Division du dataset en deux ensembles :
  - 80 % pour l'entraînement.
  - 20 % pour le test.
- Préparation des données pour l'entraînement du modèle.

**Infrastructure GCP :** Cloud Storage.



## 3. Entraînement et évaluation du modèle

- Développement d'un modèle de classification binaire basé sur XGBoost.
- Recherche d'hyperparamètres avec Randomized Search.
- Évaluation des performances du modèle et analyse du surapprentissage (overfitting).
- Utilisation de métriques adaptées aux données déséquilibrées :
  - Average Precision.
  - AUPRC (Area Under the Precision-Recall Curve).

**Infrastructure GCP :** Vertex AI, Kubeflow.



## 4. Orchestration du pipeline ML

- Mise en place d'un pipeline de machine learning avec Kubeflow.
- Orchestration des différentes étapes du workflow.
- Exécution du pipeline sur Vertex AI à l'aide de l'API AI Platform.

**Infrastructure GCP :** Vertex AI, Kubeflow.



## 5. Enregistrement et déploiement du modèle

- Enregistrement du modèle entraîné dans le Vertex AI Model Registry.
- Déploiement du modèle sur un endpoint Vertex AI.
- Exposition d'un endpoint permettant d'envoyer des données et de récupérer des prédictions en production.

**Infrastructure GCP :** Vertex AI Model Registry, Vertex AI Endpoints.



## 6. Stockage des données et des prédictions

- Stockage des données utilisées par le système dans Google Cloud Storage.
- Sauvegarde des prédictions générées en production dans un bucket GCS.

**Infrastructure GCP :** Cloud Storage.



## 7. Monitoring et dashboard

- Développement d'un dashboard avec Dash.
- Hébergement du dashboard sur Google Cloud.
- Suivi de l'état de santé du système de détection de fraude et des prédictions réalisées en production.



**Technologies :** Dash, Google Cloud.


