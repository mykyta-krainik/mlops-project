import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))


def wait_for_service(url: str, timeout: int = 60, interval: int = 2) -> bool:
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(interval)

    return False


def demo_api_predictions(api_url: str):
    test_comments = [
        "This is a great article, thank you for sharing!",
        "You are an idiot and should be banned",
        "I disagree with your point but respect your opinion",
        "I will find you and make you pay for this",
        "This is absolutely disgusting behavior",
        "Bastard",
    ]

    print("Single predictions:\n")

    for comment in test_comments:
        try:
            response = requests.post(
                f"{api_url}/predict",
                json={"comment": comment},
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                print(f"Comment: '{comment[:50]}...'")
                print(f"  Action: {result['moderation_action']}")
                print(f"  Is Toxic: {result['is_toxic']}")
                print(f"  Top scores:")

                # Sort predictions by score
                sorted_preds = sorted(
                    result["predictions"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                for label, score in sorted_preds[:3]:
                    print(f"    - {label}: {score:.3f}")
                print()
            else:
                print(f"Error: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

    print("\nBatch predictions:\n")

    try:
        response = requests.post(
            f"{api_url}/predict/batch",
            json={"comments": test_comments},
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Processed {result['total']} comments")

            actions = {}
            for r in result["results"]:
                action = r["moderation_action"]
                actions[action] = actions.get(action, 0) + 1

            print("Action summary:")
            for action, count in actions.items():
                print(f"  - {action}: {count}")

        else:
            print(f"Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


def demo_health_check(api_url: str):
    try:
        response = requests.get(f"{api_url}/health", timeout=5)

        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result['status']}")
            print(f"Model Loaded: {result['model_loaded']}")
            print(f"Version: {result['version']}")
        else:
            print(f"Health check failed: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")


def demo_model_info(api_url: str):
    try:
        response = requests.get(f"{api_url}/model/info", timeout=5)

        if response.status_code == 200:
            result = response.json()
            print(f"Version: {result['version']}")
            print(f"Model Type: {result['model_type']}")
            print(f"Loaded: {result['loaded']}")
            print(f"Source: {result.get('source', 'N/A')}")
            print(f"Target Labels: {', '.join(result['target_labels'])}")
        else:
            print(f"Failed to get model info: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-url",
        default="http://localhost:5000",
        help="URL of the Flask API",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for API to be available",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout for waiting (seconds)",
    )
    args = parser.parse_args()

    print(f"API URL: {args.api_url}")

    if args.wait:
        print(f"\nWaiting for API to be available (timeout: {args.timeout}s)...")
        if not wait_for_service(f"{args.api_url}/health", timeout=args.timeout):
            print("ERROR: API not available")
            sys.exit(1)
        print("API is ready!\n")

    demo_health_check(args.api_url)
    demo_model_info(args.api_url)
    demo_api_predictions(args.api_url)

    print("For more information, see the README.md file.")
    print("\nUseful URLs:")
    print(f"  - API: {args.api_url}")
    print("  - Minio Console: http://localhost:9001")
    print("  - Airflow: http://localhost:8080")


if __name__ == "__main__":
    main()
