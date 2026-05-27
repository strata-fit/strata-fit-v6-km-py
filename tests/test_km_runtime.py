from __future__ import annotations

import json

from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from strata_fit_v6_km_py.central import kaplan_meier_central, run_local_kaplan_meier
from strata_fit_v6_km_py.partial import (
    compute_d2t_prevalence_by_year_frame,
    compute_km_event_table_frame,
    compute_unique_event_times_frame,
    get_unique_event_times,
)
from strata_fit_v6_km_py.runtime import RunContext
from strata_fit_v6_km_py.types import DEFAULT_CUMULATIVE_INCIDENCE_COLUMN


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pat_ID": ["A1", "A1", "A1", "A1", "B1", "B1", "C1", "C1"],
            "Visit_months_from_diagnosis": [0, 6, 12, 18, 0, 12, 0, 6],
            "Year_diagnosis": [2010, 2010, 2010, 2010, 2012, 2012, 2014, 2014],
            "Age_diagnosis": [45, 45, 45, 45, 60, 60, 52, 52],
            "Sex": [1, 1, 1, 1, 0, 0, 1, 1],
            "RF_positivity": [1, 1, 1, 1, 0, 0, 1, 1],
            "anti_CCP": [1, 1, 1, 1, 0, 0, 1, 1],
            "DAS28": [5.0, 5.1, 5.2, 5.3, 2.1, 2.0, 4.1, 4.0],
            "Pat_global": [70, 72, 74, 76, 20, 18, 60, 62],
            "Ph_global": [68, 69, 71, 72, 20, 18, 61, 63],
            "CRP": [2.0, 2.1, 2.2, 2.3, 0.2, 0.2, 1.2, 1.1],
            "csDMARD1": [1, 1, 1, 1, 1, 1, 1, 1],
            "csDMARD2": [None, None, None, None, None, None, None, None],
            "csDMARD3": [None, None, None, None, None, None, None, None],
            "bDMARD": [1, 1, 2, 2, 1, 1, 1, 1],
            "tsDMARD": [None, None, None, None, None, None, None, None],
            "GC": [1, 1, 1, 1, 0, 0, 1, 1],
        }
    )


def test_partial_helpers_and_local_runner() -> None:
    dataset = _dataset()
    unique_event_times = compute_unique_event_times_frame(dataset)
    assert unique_event_times

    event_table_json = compute_km_event_table_frame(dataset, unique_event_times)
    event_table = pd.read_json(StringIO(event_table_json))
    assert {"removed", "observed", "interval", "censored", "at_risk"}.issubset(
        event_table.columns
    )

    prevalence_json = compute_d2t_prevalence_by_year_frame(dataset)
    prevalence_df = pd.read_json(StringIO(prevalence_json))
    assert {"Year_visit", "total_patients", "d2t_positive"}.issubset(prevalence_df.columns)

    result = run_local_kaplan_meier([dataset, dataset.copy(), dataset.copy()])
    km_df = pd.read_json(StringIO(result["km_result"]))
    assert DEFAULT_CUMULATIVE_INCIDENCE_COLUMN in km_df.columns
    assert result["metadata"]["included_organizations"] == [0, 1, 2]


def test_run_context_partial_writes_output(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    output_path = tmp_path / "out.json"
    dataset = _dataset()
    dataset.to_csv(dataset_path, index=False)

    context = RunContext(
        source=tmp_path / "run_context.json",
        payload={
            "entrypoint": {"name": "get_unique_event_times"},
            "arguments": {"named": {}},
            "inputs": [{"uri": str(dataset_path)}],
            "outputs": [{"uri": str(output_path)}],
        },
    )

    result = get_unique_event_times(run_context=context)
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == result


class _StubTaskClient:
    def __init__(self, parent: "_StubClient") -> None:
        self.parent = parent
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        *,
        input_: dict[str, Any],
        organizations: list[int] | None = None,
        name: str = "subtask",
        description: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "input_": input_,
                "organizations": organizations or [],
                "name": name,
                "description": description,
            }
        )
        task_id = len(self.calls)
        return {"id": task_id}


class _StubOrganizationClient:
    def list(self) -> list[dict[str, Any]]:
        return [{"id": 10}, {"id": 11}, {"id": 12}]


class _StubClient:
    def __init__(self, unique_event_times: list[float], event_table_json: str, prevalence_json: str) -> None:
        self.task = _StubTaskClient(self)
        self.organization = _StubOrganizationClient()
        self._results = {
            1: [unique_event_times, unique_event_times, unique_event_times],
            2: [event_table_json, event_table_json, event_table_json],
            3: [prevalence_json, prevalence_json, prevalence_json],
        }

    def wait_for_results(self, task_id: int, interval: float = 1.0) -> list[Any]:
        return self._results[task_id]


def test_central_orchestration_with_stub_client(tmp_path: Path) -> None:
    dataset = _dataset()
    unique_event_times = compute_unique_event_times_frame(dataset)
    event_table_json = compute_km_event_table_frame(dataset, unique_event_times)
    prevalence_json = compute_d2t_prevalence_by_year_frame(dataset)
    output_path = tmp_path / "central.json"

    client = _StubClient(unique_event_times, event_table_json, prevalence_json)
    result = kaplan_meier_central(
        organizations_to_include=[10, 11, 12],
        output_path=output_path,
        client=client,
    )

    assert [call["input_"]["method"] for call in client.task.calls] == [
        "get_unique_event_times",
        "get_km_event_table",
        "get_d2t_prevalence_by_year",
    ]
    assert {"series", "prevalence_series", "metadata", "km_result", "d2t_prevalence"} == set(
        result.keys()
    )
    assert output_path.exists()
