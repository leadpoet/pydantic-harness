"""Stable public entrypoint for the PydanticAI lead-sourcing harness."""

from experiments.harness_bakeoff.adapters.pydantic_ai import (
    get_last_usage,
    run_icp as _run_icp,
)


def run_icp(icp: dict) -> list[dict]:
    """Return up to five best-fit companies for one ICP."""

    return _run_icp(icp)


__all__ = ["get_last_usage", "run_icp"]
