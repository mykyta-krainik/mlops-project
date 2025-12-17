import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import tempfile

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from src.data.preprocessing import TextPreprocessor


sys.path.insert(0, "/opt/airflow")
sys.path.insert(0, "/opt/airflow/src")

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def get_minio_client():
    from minio import Minio

    return Minio(
        endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )


def check_new_data(**context) -> str:
    data_dir = Path("/opt/airflow/data")
    batch_dir = data_dir / "batches"

    if not batch_dir.exists():
        print(f"Batch directory not found: {batch_dir}")

        train_file = data_dir / "train.csv"
        if train_file.exists():
            context["ti"].xcom_push(key="files_to_process", value=[str(train_file)])
            context["ti"].xcom_push(key="is_batch", value=False)
            return "validate_data"
        else:
            print("No data files found")
            return "no_new_data"

    batch_files = sorted(batch_dir.glob("batch_*.csv"))

    if not batch_files:
        print("No batch files found")
        return "no_new_data"

    client = get_minio_client()
    bucket = "raw-data"

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    uploaded = set()
    try:
        objects = client.list_objects(bucket, prefix="batches/", recursive=True)
        uploaded = {obj.object_name.split("/")[-1] for obj in objects}
    except Exception as e:
        print(f"Error listing objects: {e}")

    new_files = [f for f in batch_files if f.name not in uploaded]

    if not new_files:
        print("All batch files already uploaded")
        return "no_new_data"

    print(f"Found {len(new_files)} new files to process")
    context["ti"].xcom_push(key="files_to_process", value=[str(f) for f in new_files])
    context["ti"].xcom_push(key="is_batch", value=True)

    return "validate_data"


def validate_data(**context):
    files = context["ti"].xcom_pull(key="files_to_process")

    required_columns = {
        "id",
        "comment_text",
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    }

    valid_files = []

    for file_path in files:
        try:
            df = pd.read_csv(file_path)

            missing = required_columns - set(df.columns)
            if missing or len(df) == 0:
                continue

            for col in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue

            valid_files.append(file_path)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    if not valid_files:
        raise ValueError("No valid files to upload")

    context["ti"].xcom_push(key="valid_files", value=valid_files)
    print(f"Validated {len(valid_files)} files")


def upload_to_minio(**context):
    valid_files = context["ti"].xcom_pull(key="valid_files")
    is_batch = context["ti"].xcom_pull(key="is_batch")

    print("is_batch", is_batch)

    client = get_minio_client()
    bucket = "raw-data"

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    uploaded_count = 0

    for file_path in valid_files:
        file_path = Path(file_path)

        if is_batch:
            object_name = f"batches/{file_path.name}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            object_name = f"train_{timestamp}.csv"

        try:
            client.fput_object(
                bucket_name=bucket,
                object_name=object_name,
                file_path=str(file_path),
                content_type="text/csv",
            )
            uploaded_count += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"Uploaded {uploaded_count}/{len(valid_files)} files")

    if uploaded_count == 0:
        raise ValueError("No files were uploaded successfully")

    context["ti"].xcom_push(key="uploaded_count", value=uploaded_count)


def trigger_preprocessing(**context):
    client = get_minio_client()
    raw_bucket = "raw-data"
    processed_bucket = "processed-data"

    if not client.bucket_exists(processed_bucket):
        client.make_bucket(processed_bucket)

    processed_files = set()
    try:
        objects = client.list_objects(processed_bucket, recursive=True)
        processed_files = {obj.object_name for obj in objects}
    except Exception as e:
        print(f"Error listing processed objects: {e}")

    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_html=True,
        remove_special_chars=True,
        remove_extra_whitespace=True,
    )

    raw_objects = list(client.list_objects(raw_bucket, recursive=True))
    processed_count = 0

    for obj in raw_objects:
        if obj.object_name in processed_files:
            print(f"Skipping already processed: {obj.object_name}")
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "raw.csv"
            processed_path = Path(tmpdir) / "processed.csv"

            client.fget_object(raw_bucket, obj.object_name, str(local_path))
            print(f"Downloaded: {obj.object_name}")

            df = pd.read_csv(local_path)
            df = preprocessor.preprocess_dataframe(df, text_column="comment_text")
            df.to_csv(processed_path, index=False)
            print(f"Preprocessed: {len(df)} rows")

            client.fput_object(
                bucket_name=processed_bucket,
                object_name=obj.object_name,
                file_path=str(processed_path),
                content_type="text/csv",
            )
            processed_count += 1

    print(f"Preprocessing complete. Processed {processed_count} files.")
    context["ti"].xcom_push(key="processed_count", value=processed_count)


with DAG(
    dag_id="data_ingestion",
    default_args=default_args,
    description="Ingest toxic comment data into Minio",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "data-ingestion"],
) as dag:
    start = EmptyOperator(task_id="start")
    check_data = BranchPythonOperator(
        task_id="check_new_data",
        python_callable=check_new_data,
    )
    no_data = EmptyOperator(task_id="no_new_data")
    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )
    upload = PythonOperator(
        task_id="upload_to_minio",
        python_callable=upload_to_minio,
    )

    preprocess = PythonOperator(
        task_id="trigger_preprocessing",
        python_callable=trigger_preprocessing,
    )

    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    start >> check_data
    check_data >> [validate, no_data]
    validate >> upload >> preprocess >> end
    no_data >> end
