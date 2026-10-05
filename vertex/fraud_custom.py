from datetime import datetime
from kfp import dsl, compiler
from kfp.dsl import Dataset, Input, Model
from google.cloud import aiplatform
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from fraud_pipeline import (
    PROJECT_ID, REGION, BUCKET, PIPELINE_ROOT, IMAGE,
    prepare_data, evaluate_model, register_model,
)

SERVICE_ACCOUNT = f"ml-pipeline@{PROJECT_ID}.iam.gserviceaccount.com"


@dsl.component(base_image="python:3.11-slim")
def get_uri(data: Input[Dataset]) -> str:
    """Convertit un artefact en URI gs:// utilisable comme argument du job."""
    return data.uri


@dsl.pipeline(name="fraud-training-custom-job", pipeline_root=PIPELINE_ROOT)
def fraud_pipeline_cj(
    model_dir: str,  # ex. gs://data-epip/models/20261004-1530 (un dossier par run)
    train_uri: str = f"{BUCKET}/train/paysim.parquet",
    test_uri: str = f"{BUCKET}/test/paysim.parquet",
    target: str = "isFraud",
    n_iter: int = 30,
    cv_folds: int = 10,
    n_jobs: int = -1,
    min_auprc: float = 0.90,
    serving_image: str = IMAGE,  # à remplacer par une vraie image de serving
):
    data_task = prepare_data(train_uri=train_uri, test_uri=test_uri)
    train_uri_task = get_uri(data=data_task.outputs["train_data"])
    train_job = CustomTrainingJobOp(
        project=PROJECT_ID,
        location=REGION,
        display_name="fraud-xgb-train",
        service_account=SERVICE_ACCOUNT,
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": "n2-highmem-8"},  # 8 vCPU, 64 Go
                "replica_count": 1,
                "container_spec": {
                    "image_uri": IMAGE,
                    "command": ["python", "/app/train.py"],
                    "args": [
                        "--train-data", train_uri_task.output,
                        "--model-dir", model_dir,
                        "--target", target,
                        "--n-iter", n_iter,
                        "--cv-folds", cv_folds,
                        "--n-jobs", n_jobs,
                    ],
                },
            }
        ],
    )

    model_importer = dsl.importer(
        artifact_uri=model_dir,
        artifact_class=Model,
        reimport=False,
    ).after(train_job)

    eval_task = evaluate_model(
        test_data=data_task.outputs["test_data"],
        model=model_importer.outputs["artifact"],
        target=target,
    )

    with dsl.If(eval_task.outputs["Output"] >= min_auprc, name="check-threshold"):
        register_model(
            model=model_importer.outputs["artifact"],
            project=PROJECT_ID,
            region=REGION,
            display_name="fraud-xgboost",
            serving_image=serving_image,
        )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=fraud_pipeline_cj,
        package_path="fraud_pipeline_cj.json",
    )

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    job = aiplatform.PipelineJob(
        display_name="fraud-training-cj-run",
        template_path="fraud_pipeline_cj.json",
        parameter_values={
            "model_dir": f"{BUCKET}/models/{run_id}",
            "n_iter": 1,
            "cv_folds": 3,
        },
        enable_caching=True,
    )
    job.run(service_account=SERVICE_ACCOUNT)