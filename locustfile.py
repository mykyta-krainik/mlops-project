import json
import os
import random
import time

import boto3
from locust import User, between, events, task

ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "toxic-comment-staging")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
TARGET_VARIANT = os.environ.get("TARGET_VARIANT")

SAMPLE_COMMENTS = [
    "This is a great article, thank you for sharing!",
    "I completely disagree with this opinion.",
    "You are absolutely wrong and should be ashamed.",
    "Great work, keep it up!",
    "This is the stupidest thing I have ever read.",
    "I respectfully disagree with your analysis.",
    "Could you provide more sources for this claim?",
    "Thank you for the detailed explanation.",
    "This content is misleading and harmful.",
    "Interesting perspective, I had not considered that angle.",
]


class SageMakerUser(User):
    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        self._runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

    def _invoke(self, payload: dict, name: str) -> None:
        body = json.dumps(payload)
        start = time.perf_counter()
        exception = None
        response_length = 0

        try:
            kwargs = dict(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Accept="application/json",
                Body=body,
            )
            if TARGET_VARIANT:
                kwargs["TargetVariant"] = TARGET_VARIANT
            response = self._runtime.invoke_endpoint(**kwargs)
            result = response["Body"].read()
            response_length = len(result)
        except Exception as e:
            exception = e

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        events.request.fire(
            request_type="SAGEMAKER",
            name=name,
            response_time=elapsed_ms,
            response_length=response_length,
            exception=exception,
            context={},
        )

    @task(8)
    def predict_single(self) -> None:
        comment = random.choice(SAMPLE_COMMENTS)
        self._invoke({"comment": comment}, name="predict_single")

    @task(2)
    def predict_batch(self) -> None:
        comments = random.sample(SAMPLE_COMMENTS, k=min(10, len(SAMPLE_COMMENTS)))
        self._invoke({"comments": comments}, name="predict_batch")
