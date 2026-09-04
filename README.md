# Leadpoet PydanticAI Harness

An open-source PydanticAI agent for live B2B company sourcing.

The harness takes one ICP and returns up to five companies that match it and
have a verified, recent intent signal. Each result includes a clear sales-facing
reason and public evidence URLs.

This repository contains no Research Lab deployment or persistence code. A
host can call its stable function or run it through the Arena file adapter.

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
  "company_stage": "Series A",
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
  }],
  "required_attribute": {
    "text": "The required company characteristic.",
    "passed": true,
    "evidence_url": "https://example.com/about",
    "evidence_quote": "Source text that proves the characteristic.",
    "explanation": "Why the evidence satisfies the requirement."
  }
}
```

The harness can use these tools: `search_companies`, `get_company_profile`,
`get_company_events`, `search_web`, `fetch_page`, and `submit_companies`.

## Miner competition contract

Miners can fork this repository and change the model, harness, prompts,
routing, tools, and dependencies. The subnet requires only this final callable:

```python
def run_icp(icp: dict) -> list[dict]:
    """Return up to five companies, ranked best first."""
```

Each returned company must use the output fields shown in the stable contract
above. A submission must also contain an executable `/agent/run`. The included
[`agent/run`](agent/run) is the reference adapter. It reads
`/input/icp.json`, calls `run_icp`, and writes `/output/companies.json`.

The Arena input file has this shape:

```json
{
  "schema_version": "leadpoet.lab_arena.icp_input.v1",
  "icp": {"icp_id": "daily-icp-id"},
  "evaluation_date": "2026-09-04",
  "company_limit": 5,
  "provider_operations": ["openrouter.chat", "exa.search"]
}
```

The adapter writes this shape:

```json
{
  "schema_version": "leadpoet.lab_arena.output.v1",
  "companies": []
}
```

The host sets `LAB_ARENA_WORKER_SOCKET`. Provider calls contain only an
operation name and ordinary JSON parameters. The host adds its approved
provider credentials and enforces the same time, call, and cost limits for the
baseline and miner submissions. A submission does not receive provider keys.

Build the reference Linux AMD64 image with:

```bash
docker build --platform linux/amd64 -t your-registry/pydantic-harness:latest .
```

The competition does not require a Git commit, source digest, receipt,
manifest, replay proof, or GitHub attestation. The Arena can pin ordinary OCI
image bytes for the duration of one round so a run does not change midway.

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

For standalone testing only, set `OPENROUTER_API_KEY`, `DEEPLINE_API_KEY`, and
`SCRAPINGDOG_API_KEY` in the process environment. `EXA_API_KEY` is optional.
The Arena adapter does not read these keys. Do not commit keys or private ICP
data.

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

## One-shot host adapter

The stateless host adapter prints one `PYDANTIC_HARNESS_RESULT_JSON=` line for
reliable parsing. Run its live preflight once per daily batch. Pass the selected
model and its public pricing to each run; `run` does not repeat the paid model
probe.

```bash
python production_runner.py preflight
python production_runner.py run \
  --model openai/example \
  --model-pricing-json '{"prompt":"0.000001","completion":"0.000002"}' \
  --evaluation-date YYYY-MM-DD < /absolute/path/to/one-icp.json
```

Each `run` call reads one raw ICP JSON object and starts a fresh `run_icp`
worker process. The adapter does not deploy code or persist state.

## Provisional result

In the earlier internal live test, 6 of 16 returned companies met the full
sales-ready standard. This result is provisional because some test arms had
integration failures. No private ICP or result data is included here.

## License

MIT
