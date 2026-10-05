import argparse
import json
import os

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder
from xgboost import XGBClassifier

import utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", required=True)  # gs://.../train_data
    p.add_argument("--model-dir", required=True)   # gs://bucket/models/<run>
    p.add_argument("--target", default="isFraud")
    p.add_argument("--n-iter", type=int, default=30)
    p.add_argument("--cv-folds", type=int, default=10)
    p.add_argument("--n-jobs", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.train_data, engine="pyarrow")  # gcsfs gère gs://
    X_train, y_train = df.drop(columns=[args.target]), df[args.target]

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("skew_data", FunctionTransformer(utils.skew_data)),
                        ("scaler", MinMaxScaler()),
                    ]
                ),
                num_cols,
            ),
            ("cat", Pipeline([("encoder", OneHotEncoder())]), cat_cols),
        ]
    )

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
    )

    pipeline_xgb = ImbPipeline(
        steps=[
            ("clean_data", FunctionTransformer(utils.preprocess)),
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
        n_iter=args.n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=args.cv_folds),
        verbose=2,
        random_state=42,
        n_jobs=args.n_jobs,
    )
    random_search.fit(X_train, y_train)

    # Vertex monte les buckets GCS sous /gcs/ : on écrit comme sur un disque local.
    local_dir = args.model_dir.replace("gs://", "/gcs/", 1)
    os.makedirs(local_dir, exist_ok=True)
    joblib.dump(random_search.best_estimator_, os.path.join(local_dir, "model.joblib"))

    best_params = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in random_search.best_params_.items()
    }
    summary = {
        "cv_best_average_precision": float(random_search.best_score_),
        "best_params": best_params,
    }
    with open(os.path.join(local_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()