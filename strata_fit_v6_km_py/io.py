from __future__ import annotations

import json

from pathlib import Path
from typing import Any


def serialize_payload(payload: Any) -> bytes:
    return json.dumps(payload).encode("utf-8")


def write_output(output_path: str | Path | None, payload: Any) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialize_payload(payload))
