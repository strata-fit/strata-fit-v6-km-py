from __future__ import annotations

import argparse
import base64
import json
import os
import time

from getpass import getpass
from typing import Any

from vantage6.client import Client


TERMINAL_STATUSES = {
    "completed",
    "crashed",
    "failed",
    "cancelled",
    "non-existing Docker image",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the standalone STRATA-FIT KM algorithm to a Vantage6 deployment."
    )
    parser.add_argument("--host", default=os.getenv("V6_SERVER_HOST"))
    parser.add_argument("--port", type=int, default=int(os.getenv("V6_SERVER_PORT", "443")))
    parser.add_argument("--api-path", default=os.getenv("V6_API_PATH", "/api"))
    parser.add_argument("--username", default=os.getenv("V6_USERNAME"))
    parser.add_argument("--password", default=os.getenv("V6_PASSWORD"))
    parser.add_argument("--mfa-code", default=os.getenv("V6_MFA_CODE"))
    parser.add_argument("--organization-key", default=os.getenv("V6_ORGANIZATION_KEY"))
    parser.add_argument("--collaboration-id", type=int, default=_env_int("V6_COLLABORATION_ID"))
    parser.add_argument("--master-org-id", type=int, default=_env_int("V6_MASTER_ORG_ID"))
    parser.add_argument("--organization-ids", default=os.getenv("V6_ORGANIZATION_IDS"))
    parser.add_argument("--dataset-label", default=os.getenv("V6_DATASET_LABEL"))
    parser.add_argument("--image", default=os.getenv("V6_ALGO_IMAGE"))
    parser.add_argument("--timeout-s", type=int, default=int(os.getenv("V6_TASK_TIMEOUT_S", "900")))
    return parser.parse_args()


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw else None


def decode_result(value: Any) -> Any:
    if value is None or isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {"raw": str(value)}
    try:
        return json.loads(base64.b64decode(value).decode("utf-8"))
    except Exception:
        pass
    try:
        return json.loads(value)
    except Exception:
        return {"raw": value}


def build_client(args: argparse.Namespace) -> Client:
    if not args.host or not args.username:
        raise ValueError("V6_SERVER_HOST and V6_USERNAME are required")

    password = args.password or getpass("Password: ")
    client = Client(args.host, args.port, args.api_path)
    client.authenticate(args.username, password, mfa_code=args.mfa_code or None)
    client.setup_encryption(args.organization_key or None)
    return client


def resolve_org_ids(args: argparse.Namespace) -> list[int]:
    if not args.organization_ids:
        raise ValueError("V6_ORGANIZATION_IDS is required")
    return [int(part.strip()) for part in args.organization_ids.split(",") if part.strip()]


def wait_for_terminal(client: Client, task_id: int, timeout_s: int) -> str:
    deadline = time.time() + timeout_s
    status = None
    while time.time() < deadline:
        current = client.task.get(task_id).get("status")
        if current != status:
            print(f"task {task_id} status: {current}")
            status = current
        if current in TERMINAL_STATUSES:
            return str(current)
        time.sleep(2)
    raise TimeoutError(f"Task {task_id} did not finish before timeout")


def main() -> None:
    args = parse_args()
    client = build_client(args)

    if args.collaboration_id is None or args.master_org_id is None:
        raise ValueError("V6_COLLABORATION_ID and V6_MASTER_ORG_ID are required")
    if not args.dataset_label or not args.image:
        raise ValueError("V6_DATASET_LABEL and V6_ALGO_IMAGE are required")

    organization_ids = resolve_org_ids(args)
    task_input = {
        "method": "kaplan_meier_central",
        "kwargs": {
            "organizations_to_include": organization_ids,
        },
    }

    task = client.task.create(
        collaboration=args.collaboration_id,
        organizations=[args.master_org_id],
        name="strata-fit-km",
        image=args.image,
        description="Standalone STRATA-FIT Kaplan-Meier",
        databases=[{"label": args.dataset_label}],
        input_=task_input,
    )

    print(f"submitted task {task['id']}")
    status = wait_for_terminal(client, int(task["id"]), args.timeout_s)
    print(f"final status: {status}")

    result_rows = client.result.from_task(task_id=int(task["id"])).get("data", [])
    if not result_rows:
        raise RuntimeError("Task returned no result rows")

    decoded = decode_result(result_rows[0].get("result"))
    print(json.dumps(decoded, indent=2))


if __name__ == "__main__":
    main()
