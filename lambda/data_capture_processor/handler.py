import io
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import boto3
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODELS_BUCKET = os.environ.get("MODELS_BUCKET", "")
EVIDENTLY_BUCKET = os.environ.get("EVIDENTLY_BUCKET", "")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1000"))

s3_client = boto3.client("s3")


def parse_sagemaker_capture_data(content: str) -> List[Dict[str, Any]]:
    records = []
    
    for line in content.strip().split('\n'):
        if not line.strip():
            continue
            
        try:
            capture_record = json.loads(line)
            
            capture_data = capture_record.get("captureData", {})
            endpoint_input = capture_data.get("endpointInput", {})
            endpoint_output = capture_data.get("endpointOutput", {})
            event_metadata = capture_record.get("eventMetadata", {})
            
            input_data = endpoint_input.get("data", "")
            if input_data:
                try:
                    input_json = json.loads(input_data)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse input data: {input_data[:100]}")
                    continue
            else:
                continue
            
            output_data = endpoint_output.get("data", "")
            predictions = {}
            if output_data:
                try:
                    output_json = json.loads(output_data)
                    predictions = output_json.get("predictions", {})
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse output data: {output_data[:100]}")
            
            comment = input_json.get("comment", "")
            if not comment:
                comment = input_json.get("text", "")
            
            if not comment:
                logger.warning("No comment text found in input")
                continue
            
            record = {
                "timestamp": event_metadata.get("inferenceTime", datetime.utcnow().isoformat()),
                "request_id": event_metadata.get("inferenceId", ""),
                "text": comment,
                "text_length": len(comment),
                "word_count": len(comment.split()),
            }
            
            if predictions:
                for label, score in predictions.items():
                    record[f"pred_{label}"] = float(score)
                
                toxic_scores = [predictions.get(label, 0) for label in 
                               ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
                record["is_toxic"] = any(score > 0.5 for score in toxic_scores)
            
            records.append(record)
            
        except Exception as e:
            logger.error(f"Error parsing capture record: {e}")
            continue
    
    return records


def extract_text_features(text: str) -> Dict[str, float]:
    if not text:
        return {
            "special_char_ratio": 0.0,
            "url_count": 0,
            "avg_word_length": 0.0,
            "unique_word_ratio": 0.0,
        }
    
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    special_char_ratio = special_chars / len(text) if len(text) > 0 else 0.0
    
    url_count = text.lower().count("http://") + text.lower().count("https://")
    
    words = text.split()
    if words:
        avg_word_length = sum(len(word) for word in words) / len(words)
        unique_word_ratio = len(set(words)) / len(words)
    else:
        avg_word_length = 0.0
        unique_word_ratio = 0.0
    
    return {
        "special_char_ratio": special_char_ratio,
        "url_count": url_count,
        "avg_word_length": avg_word_length,
        "unique_word_ratio": unique_word_ratio,
    }


def process_capture_file(bucket: str, key: str) -> pd.DataFrame:
    logger.info(f"Processing capture file: s3://{bucket}/{key}")
    
    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")
    
    records = parse_sagemaker_capture_data(content)
    
    if not records:
        logger.warning(f"No records parsed from {key}")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    text_features = df["text"].apply(extract_text_features)
    features_df = pd.DataFrame(text_features.tolist())
    df = pd.concat([df, features_df], axis=1)
    
    logger.info(f"Parsed {len(df)} records from {key}")
    return df


def save_to_evidently_bucket(df: pd.DataFrame, timestamp: str) -> str:
    if df.empty:
        logger.warning("Empty DataFrame, skipping save")
        return ""
    
    key = f"predictions/{timestamp}.csv"
    
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    s3_client.put_object(
        Bucket=EVIDENTLY_BUCKET,
        Key=key,
        Body=buffer.getvalue(),
        ContentType="text/csv",
        Metadata={
            "record_count": str(len(df)),
            "processed_at": datetime.utcnow().isoformat(),
        }
    )
    
    s3_uri = f"s3://{EVIDENTLY_BUCKET}/{key}"
    logger.info(f"Saved {len(df)} records to {s3_uri}")
    return s3_uri


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    logger.info(f"Event: {json.dumps(event)}")
    
    processed_files = []
    total_records = 0
    
    try:
        for record in event.get("Records", []):
            s3_info = record.get("s3", {})
            bucket = s3_info.get("bucket", {}).get("name", "")
            key = s3_info.get("object", {}).get("key", "")
            
            if not bucket or not key:
                logger.warning("Missing bucket or key in S3 event")
                continue
            
            if not key.startswith("data-capture/"):
                logger.info(f"Skipping non-data-capture file: {key}")
                continue
            
            df = process_capture_file(bucket, key)
            
            if not df.empty:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                output_uri = save_to_evidently_bucket(df, timestamp)
                
                processed_files.append({
                    "input": f"s3://{bucket}/{key}",
                    "output": output_uri,
                    "records": len(df),
                })
                total_records += len(df)
        
        response = {
            "statusCode": 200,
            "body": {
                "message": "Data capture processing complete",
                "processed_files": len(processed_files),
                "total_records": total_records,
                "files": processed_files,
            }
        }
        
        logger.info(f"Response: {json.dumps(response)}")
        return response
    
    except Exception as e:
        logger.error(f"Error processing data capture: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": {"error": str(e)},
        }


if __name__ == "__main__":
    test_event = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "test-bucket"},
                    "object": {"key": "data-capture/test.jsonl"},
                }
            }
        ]
    }
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))
