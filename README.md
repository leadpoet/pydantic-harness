# Leadpoet PydanticAI Harness

An open-source PydanticAI agent for live B2B company sourcing.

The harness takes one ICP and returns up to five companies that match it and
have a verified, recent intent signal. Each result includes a clear sales-facing
reason and public evidence URLs.

This repository does not change Leadpoet production or the current daily
rebenchmark.

## Stable contract

The public entrypoint is:

```python
from harness import run_icp

companies = run_icp(icp)
```

Its signature is:

```python
def run_icp(icp: dict) -> list[dict]:
    """Return up to five best-fit companies, ranked best first."""
```

The normal output shape is:

```json
{
  "company_name": "Example",
  "company_website": "https://example.com/",
  "company_linkedin": "https://www.linkedin.com/company/example/",
  "industry": "Software",
  "employee_count": "51-200",
  "country": "United States",
  "state": "California",
  "fit_summary": "Why the company fits the ICP.",
  "fit_evidence_urls": ["https://example.com/about"],
  "intent_signals": [{
    "matched_icp_signal": 0,
    "description": "The required recent event.",
    "date": "2026-08-20",
    "why_now": "Why a sales representative should contact the company now.",
    "url": "https://example.com/news/event",
    "snippet": "Source text that supports the claim."
  }]
}
```

The host supplies these tools as plain JSON: `search_companies`,
`get_company_profile`, `get_company_events`, `search_web`, `fetch_page`, and
`submit_companies`.

## Install

Use Python 3.11 or newer and Node.js 20 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt \
  -r experiments/harness_bakeoff/adapters/requirements-pydantic-ai.txt
npm ci --prefix experiments/harness_bakeoff/deepline
export BAKEOFF_DEEPLINE_BIN="$PWD/experiments/harness_bakeoff/deepline/node_modules/.bin/deepline"
```

Set `OPENROUTER_API_KEY`, `DEEPLINE_API_KEY`, and `SCRAPINGDOG_API_KEY` in the
process environment. `EXA_API_KEY` is optional. Do not commit keys or private
ICP data.

## Run live sourcing

Store real ICPs in an uncommitted JSON file outside this repository. The file
can contain a JSON array or an object with an `icps` array.

```bash
python -m experiments.harness_bakeoff.runner preflight
python -m experiments.harness_bakeoff.runner all \
  --icp-file /absolute/path/to/icps.json \
  --evaluation-date YYYY-MM-DD
```

The smoke phase runs one live one-company attempt. The scored phase runs each
selected ICP twice. Each attempt uses a fresh process and the same provider,
token, time, and cost limits. Results must be written outside the repository.

## Provisional result

In the earlier internal live test, 6 of 16 returned companies met the full
sales-ready standard. This result is provisional because some test arms had
integration failures. No private ICP or result data is included here.

## License

MIT
