from google.cloud import aiplatform


def upload_model_sample(
    project: str,
    display_name: str,
    image_uri: str,
    artifact_uri: str,
    metadata_schema_uri: str|None=None,
    location: str = "europe-west1",
    api_endpoint: str = "europe-west1-aiplatform.googleapis.com",
    timeout: int = 1800,
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.ModelServiceClient(client_options=client_options)
    model = {
        "display_name": display_name,
        "metadata_schema_uri": metadata_schema_uri,
        # The artifact_uri should be the path to a GCS directory containing
        # saved model artifacts.  The bucket must be accessible for the
        # project's AI Platform service account and in the same region as
        # the api endpoint.
        "artifact_uri": artifact_uri,
        "container_spec": {
            "image_uri": image_uri,
            "command": [],
            "args": [],
            "env": [],
            "ports": [],
            "predict_route": "",
            "health_route": "",
        },
    }
    parent = f"projects/{project}/locations/{location}"
    response = client.upload_model(parent=parent, model=model)
    print("Long running operation:", response.operation.name)
    upload_model_response = response.result(timeout=timeout)
    print("upload_model_response:", upload_model_response)


if __name__=="__main__":
    upload_model_sample(
    project="fraude-detection-gcp",

    display_name="fraud-classification",

    image_uri="europe-west1-docker.pkg.dev/fraude-detection-gcp/ml/fraud-train@sha256:fe9d04de1eaaa51677b1049e04a92877489360fa7021373d19afbcaa5ef01375",

    artifact_uri="gs://data-epip/models/20261107-222857",

    location="europe-west1",

    api_endpoint="europe-west1-aiplatform.googleapis.com",

    timeout=1800
)