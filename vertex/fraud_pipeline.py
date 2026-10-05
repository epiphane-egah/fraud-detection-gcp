from kfp import dsl, compiler
from kfp.dsl import Dataset, Input, Output, Model, Metrics
from google.cloud import aiplatform

PROJECT_ID = "fraude-detection-gcp"
REGION = "europe-west1"
BUCKET = "gs://data-epip"
PIPELINE_ROOT = f"{BUCKET}/pipeline_root"

# Image construite à partir du Dockerfile (contient utils.py)
IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/ml/fraud-train:latest"


# --- 1. Préparation des données ---
@dsl.component(base_image=IMAGE)
def prepare_data(
    train_uri: str,
    test_uri: str,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
):
    import pandas as pd

    drop_cols = ["nameOrig", "nameDest", "step", "isFlaggedFraud"]
    for uri, out in [(train_uri, train_data), (test_uri, test_data)]:
        df = pd.read_parquet(uri, engine="pyarrow").drop(columns=drop_cols)
        df.to_parquet(out.path, index=False)


# --- 2. Tuning + entraînement ---
@dsl.component(base_image=IMAGE)
def train_model(
    train_data: Input[Dataset],
    target: str,
    n_iter: int,
    cv_folds: int,
    n_jobs: int,
    model: Output[Model],
    metrics: Output[Metrics],
):
    import os
    import json
    import joblib
    import pandas as pd
    from scipy.stats import randint, loguniform, uniform
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from xgboost import XGBClassifier
    import utils

    skew_data = FunctionTransformer(utils.skew_data)
    clean_data = FunctionTransformer(utils.preprocess)

    df = pd.read_parquet(train_data.path)
    X_train, y_train = df.drop(columns=[target]), df[target]

    num_col_name = X_train.select_dtypes(include="number").columns.tolist()
    cat_col_name = X_train.select_dtypes(exclude="number").columns.tolist()

    num_pipeline = Pipeline([("skew_data", skew_data), ("scaler", MinMaxScaler())])
    cat_pipeline = Pipeline([("encoder", OneHotEncoder())])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_col_name),
            ("cat", cat_pipeline, cat_col_name),
        ]
    )

    # early_stopping_rounds et use_label_encoder retirés :
    # early stopping exige un eval_set (absent dans un RandomizedSearchCV)
    # et use_label_encoder n'existe plus dans XGBoost 2.x.
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
    )

    pipeline_xgb = ImbPipeline(
        steps=[
            ("clean_data", clean_data),
            ("preprocess", preprocess),
            ("smote", SMOTE(random_state=42)),
            ("classifier", classifier),
        ]
    )

    param_distributions = {
        "classifier__n_estimators": randint(200, 1200),
        "classifier__learning_rate": loguniform(0.01, 0.3),
        "classifier__max_depth": randint(3, 11),
        "classifier__min_child_weight": randint(1, 15),
        "classifier__subsample": uniform(0.6, 0.4),
        "classifier__colsample_bytree": uniform(0.6, 0.4),
        "classifier__reg_alpha": loguniform(1e-5, 10),
        "classifier__reg_lambda": loguniform(1e-3, 100),
        "classifier__gamma": loguniform(1e-5, 10),
        "classifier__max_bin": randint(64, 512),
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline_xgb,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=cv_folds),
        verbose=2,
        random_state=42,
        n_jobs=n_jobs,
    )
    random_search.fit(X_train, y_train)

    # Sauvegarde : dossier contenant model.joblib
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(random_search.best_estimator_, os.path.join(model.path, "model.joblib"))

    best_params = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in random_search.best_params_.items()
    }
    model.metadata["framework"] = "xgboost-imblearn-sklearn"
    model.metadata["best_params"] = json.dumps(best_params)
    metrics.log_metric("cv_best_average_precision", float(random_search.best_score_))


# --- 3. Évaluation sur le jeu de test ---
@dsl.component(base_image=IMAGE)
def evaluate_model(
    test_data: Input[Dataset],
    model: Input[Model],
    target: str,
    metrics: Output[Metrics],
) -> float:
    import os
    import joblib
    import pandas as pd
    from sklearn.metrics import (
        classification_report,
        average_precision_score,
        precision_score,
        recall_score,
        f1_score,
    )

    df = pd.read_parquet(test_data.path)
    X_test, y_test = df.drop(columns=[target]), df[target]

    clf = joblib.load(os.path.join(model.path, "model.joblib"))
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))

    auprc = float(average_precision_score(y_test, y_proba))
    metrics.log_metric("test_average_precision", auprc)
    metrics.log_metric("test_precision_fraud", float(precision_score(y_test, y_pred)))
    metrics.log_metric("test_recall_fraud", float(recall_score(y_test, y_pred)))
    metrics.log_metric("test_f1_fraud", float(f1_score(y_test, y_pred)))
    return auprc


# --- 4. Enregistrement dans le Model Registry ---
@dsl.component(base_image=IMAGE)
def register_model(
    model: Input[Model],
    project: str,
    region: str,
    display_name: str,
    serving_image: str,
):
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model.uri,
        serving_container_image_uri=serving_image,
    )


# --- Pipeline ---
@dsl.pipeline(name="fraud-training-pipeline", pipeline_root=PIPELINE_ROOT)
def fraud_pipeline(
    train_uri: str = f"{BUCKET}/train/paysim.parquet",
    test_uri: str = f"{BUCKET}/test/paysim.parquet",
    target: str = "isFraud",
    n_iter: int = 30,
    cv_folds: int = 10,
    n_jobs: int = -1,
    min_auprc: float = 0.90,
    serving_image: str = IMAGE,
):
    data_task = prepare_data(train_uri=train_uri, test_uri=test_uri)

    train_task = train_model(
        train_data=data_task.outputs["train_data"],
        target=target,
        n_iter=n_iter,
        cv_folds=cv_folds,
        n_jobs=n_jobs,
    )
    train_task.set_cpu_limit("8").set_memory_limit("32G")

    eval_task = evaluate_model(
        test_data=data_task.outputs["test_data"],
        model=train_task.outputs["model"],
        target=target,
    )

    with dsl.If(eval_task.outputs["Output"]  >= min_auprc, name="check-threshold"):
        register_model(
            model=train_task.outputs["model"],
            project=PROJECT_ID,
            region=REGION,
            display_name="fraud-xgboost",
            serving_image=serving_image,
        )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=fraud_pipeline,
        package_path="fraud_pipeline.json",
    )

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

    job = aiplatform.PipelineJob(
        display_name="fraud-training-run",
        template_path="fraud_pipeline.json",
        parameter_values={"n_iter": 1, "cv_folds": 3},
        enable_caching=True,
    )
    job.run(service_account=f"ml-pipeline@{PROJECT_ID}.iam.gserviceaccount.com")
