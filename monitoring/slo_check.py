"""
SLO check — queries CloudWatch Metrics for the production endpoint.

SLOs:
  P95 ModelLatency  < 500ms (CloudWatch reports in microseconds)
  Error rate        < 1%    (4XX + 5XX / Invocations)

Exit code:
  0 — all SLOs met
  1 — one or more SLOs breached

Usage:
  python monitoring/slo_check.py [--lookback-hours 24] [--endpoint mlops-toxic-prod]
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config

# SLO thresholds
LATENCY_SLO_US = 500_000   # 500ms in microseconds (CloudWatch unit for ModelLatency)
ERROR_RATE_SLO = 1.0        # percent


def get_metric_statistic(
    cw,
    metric_name: str,
    namespace: str,
    dimensions: list[dict],
    stat: str,
    period: int,
    start_time: datetime,
    end_time: datetime,
) -> float | None:
    response = cw.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=dimensions,
        StartTime=start_time,
        EndTime=end_time,
        Period=period,
        Statistics=[stat] if stat not in ("p95", "p99") else [],
        ExtendedStatistics=[stat] if stat in ("p95", "p99") else [],
    )
    datapoints = response.get("Datapoints", [])
    if not datapoints:
        return None
    # Return the most recent value
    datapoints.sort(key=lambda d: d["Timestamp"])
    dp = datapoints[-1]
    return dp.get(stat) or dp.get("ExtendedStatistics", {}).get(stat)


def check_latency(cw, endpoint_name: str, variant_name: str, start: datetime, end: datetime) -> dict:
    dims = [
        {"Name": "EndpointName", "Value": endpoint_name},
        {"Name": "VariantName", "Value": variant_name},
    ]
    p95 = get_metric_statistic(
        cw, "ModelLatency", "AWS/SageMaker", dims, "p95", 3600, start, end
    )
    result = {
        "metric": "ModelLatency_p95",
        "value_us": p95,
        "value_ms": round(p95 / 1000, 1) if p95 is not None else None,
        "threshold_ms": LATENCY_SLO_US / 1000,
        "passed": p95 is None or p95 < LATENCY_SLO_US,
        "note": "no data" if p95 is None else "",
    }
    return result


def check_error_rate(cw, endpoint_name: str, variant_name: str, start: datetime, end: datetime) -> dict:
    dims = [
        {"Name": "EndpointName", "Value": endpoint_name},
        {"Name": "VariantName", "Value": variant_name},
    ]
    period = int((end - start).total_seconds())

    def _sum(metric_name: str) -> float:
        response = cw.get_metric_statistics(
            Namespace="AWS/SageMaker",
            MetricName=metric_name,
            Dimensions=dims,
            StartTime=start,
            EndTime=end,
            Period=period,
            Statistics=["Sum"],
        )
        return sum(dp["Sum"] for dp in response.get("Datapoints", []))

    invocations = _sum("Invocations")
    errors_4xx = _sum("Invocation4XXErrors")
    errors_5xx = _sum("Invocation5XXErrors")
    total_errors = errors_4xx + errors_5xx

    if invocations == 0:
        error_rate = 0.0
        note = "no invocations"
    else:
        error_rate = 100.0 * total_errors / invocations
        note = ""

    return {
        "metric": "ErrorRate",
        "invocations": invocations,
        "errors_4xx": errors_4xx,
        "errors_5xx": errors_5xx,
        "error_rate_pct": round(error_rate, 3),
        "threshold_pct": ERROR_RATE_SLO,
        "passed": error_rate < ERROR_RATE_SLO,
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, default=config.sagemaker.prod_endpoint)
    parser.add_argument("--variant", type=str, default="blue")
    parser.add_argument("--lookback-hours", type=int, default=24)
    args = parser.parse_args()

    cw = boto3.client("cloudwatch", region_name=config.aws.region)

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=args.lookback_hours)

    print(f"SLO check: {args.endpoint} / {args.variant}")
    print(f"Window: {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} UTC\n")

    latency_result = check_latency(cw, args.endpoint, args.variant, start, end)
    error_result = check_error_rate(cw, args.endpoint, args.variant, start, end)

    results = {
        "endpoint": args.endpoint,
        "variant": args.variant,
        "window_hours": args.lookback_hours,
        "timestamp": end.isoformat(),
        "checks": [latency_result, error_result],
        "all_passed": latency_result["passed"] and error_result["passed"],
    }

    print(json.dumps(results, indent=2))

    if not results["all_passed"]:
        print("\nSLO BREACH DETECTED")
        sys.exit(1)

    print("\nAll SLOs passed.")


if __name__ == "__main__":
    main()
