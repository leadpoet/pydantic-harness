"""One-shot file adapter for the Leadpoet Arena."""

from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable


INPUT_SCHEMA_VERSION = "leadpoet.lab_arena.icp_input.v1"
OUTPUT_SCHEMA_VERSION = "leadpoet.lab_arena.output.v1"
DEFAULT_INPUT_PATH = "/input/icp.json"
DEFAULT_OUTPUT_PATH = "/output/companies.json"


def _load_input(path: Path) -> tuple[dict[str, Any], str, int]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("Arena input is unavailable or invalid") from exc
    if not isinstance(document, dict):
        raise ValueError("Arena input must be a JSON object")
    allowed = {
        "schema_version",
        "icp",
        "evaluation_date",
        "company_limit",
        "provider_operations",
    }
    if set(document) - allowed:
        raise ValueError("Arena input has unknown fields")
    if document.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("Arena input schema is unsupported")
    icp = document.get("icp")
    if not isinstance(icp, dict):
        raise ValueError("Arena input must contain one ICP object")
    raw_date = str(document.get("evaluation_date") or "")
    try:
        evaluation_date = date.fromisoformat(raw_date).isoformat()
    except ValueError as exc:
        raise ValueError("Arena evaluation_date must use YYYY-MM-DD") from exc
    if evaluation_date != raw_date:
        raise ValueError("Arena evaluation_date must use YYYY-MM-DD")
    limit = document.get("company_limit", 5)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5:
        raise ValueError("Arena company_limit must be from 1 through 5")
    operations = document.get("provider_operations")
    if operations is not None and (
        not isinstance(operations, list)
        or not all(isinstance(item, str) for item in operations)
    ):
        raise ValueError("Arena provider_operations must be a string list")
    return dict(icp), evaluation_date, limit


def run_once(
    *,
    input_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    runner: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
) -> None:
    """Read one Arena ICP, call ``run_icp``, and write one result document."""

    icp, evaluation_date, company_limit = _load_input(Path(input_path))
    if runner is None:
        from harness import run_icp

        runner = run_icp

    previous = {
        name: os.environ.get(name)
        for name in ("BAKEOFF_EVALUATION_DATE", "BAKEOFF_MAX_COMPANIES")
    }
    os.environ["BAKEOFF_EVALUATION_DATE"] = evaluation_date
    os.environ["BAKEOFF_MAX_COMPANIES"] = str(company_limit)
    try:
        companies = runner(icp)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    from experiments.harness_bakeoff.models import validate_companies

    validated = validate_companies(companies, max_companies=company_limit)
    destination = Path(output_path)
    temporary = destination.with_name(destination.name + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(
            {"schema_version": OUTPUT_SCHEMA_VERSION, "companies": validated},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def main() -> int:
    try:
        run_once(
            input_path=os.environ.get("LAB_ARENA_INPUT_PATH", DEFAULT_INPUT_PATH),
            output_path=os.environ.get("LAB_ARENA_OUTPUT_PATH", DEFAULT_OUTPUT_PATH),
        )
    except Exception as exc:
        print(f"Arena run failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
