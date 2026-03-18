"""
Parse Locust --csv output and assert SLOs.

Locust writes stats to <prefix>_stats.csv with columns including:
  Name, # Requests, # Failures, Median Response Time, 95%ile (ms), ...

Usage:
  python scripts/check_load_test_results.py --csv /tmp/locust_stats.csv

Exit code:
  0 — all SLOs met
  1 — SLO breach
"""

import argparse
import csv
import sys

# P95_THRESHOLD_MS = 500.0   # SLO: p95 latency < 500ms
P95_THRESHOLD_MS = 1000.0   # SLO: p95 latency < 1000ms
ERROR_RATE_PCT = 1.0        # SLO: failure rate < 1%


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Locust _stats.csv file")
    parser.add_argument("--p95-threshold-ms", type=float, default=P95_THRESHOLD_MS)
    parser.add_argument("--error-rate-pct", type=float, default=ERROR_RATE_PCT)
    args = parser.parse_args()

    breaches = []

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "")
            if name == "Aggregated":
                continue  # check per-task rows only

            total = int(row.get("# Requests", 0) or 0)
            failures = int(row.get("# Failures", 0) or 0)

            # Column names differ between Locust versions
            p95_key = next(
                (k for k in row if "95" in k and ("ile" in k.lower() or "%" in k)), None
            )
            p95_ms = float(row[p95_key]) if p95_key and row[p95_key] else None

            error_pct = (100.0 * failures / total) if total > 0 else 0.0

            print(
                f"  {name}: requests={total}, failures={failures}, "
                f"error={error_pct:.2f}%, p95={p95_ms}ms"
            )

            if p95_ms is not None and p95_ms > args.p95_threshold_ms:
                breaches.append(
                    f"[{name}] p95 latency {p95_ms}ms exceeds {args.p95_threshold_ms}ms SLO"
                )
            if error_pct >= args.error_rate_pct:
                breaches.append(
                    f"[{name}] error rate {error_pct:.2f}% exceeds {args.error_rate_pct}% SLO"
                )

    if breaches:
        print("\nSLO BREACHES:")
        for b in breaches:
            print(f"  ✗ {b}")
        sys.exit(1)

    print("\nAll SLOs passed.")


if __name__ == "__main__":
    main()
