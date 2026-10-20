# CashBridge — SME Finance Copilot

International Track · Topic E · *AI as a Financial Participant*

CashBridge helps a small-business owner **see a cash shortfall before it happens**,
understand why, try non-loan fixes first, match bank payments to open invoices, and,
only if it still helps and is affordable, take a short-term loan with full cost
disclosure. Money only moves in a bank **sandbox**, and only after the owner approves
the exact terms.

**Code computes every number. The language model only writes words** (it fills
`{{placeholders}}`; any digit it types is rejected).

## What it does

| Capability | How |
|---|---|
| Cash-flow forecast & alerts | 2,000-path Monte Carlo (weekly block bootstrap, per-customer payment delays, backtest-calibrated band). Plain status: *Safe / Watch / Act now* with the probability of falling below the safety level. |
| Financing-option matching | Non-loan options first (chase late invoices, defer bills except payroll/tax, …), kept only if they materially reduce risk. Then three sandbox products checked against 9 rules incl. 80% repayment probability, debt-service limit and the legal cost cap. |
| Reconciliation | Proposes bank-credit ↔ invoice matches; owner confirms; forecast updates. |
| Business analysis | Week-over-week delta, upcoming bills, what-if simulation, grounded Q&A. |
| Safe execution | T0–T3 permission tiers; T3 = owner-only HMAC token bound to the terms, single use, 5 min, idempotency key, re-check, per-action and daily caps. |
| Compliance | Separate consents (PIPL / UU PDP), external-AI opt-in with pseudonymisation, AI-content labels, cost disclosure with 24% cap (CN) / OJK cap (ID), human review, erasure. |

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/seed_data.py                          # 3 demo businesses + consent
python scripts/run_web.py                            # http://127.0.0.1:8000
```

Log in with **demo / cashbridge-demo**. The demo clock is pinned to 2026-11-06
(`CASHBRIDGE_AS_OF`), so the three stories are reproducible on any day:

| Business | p(breach 14d) | Decision |
|---|---|---|
| Warung Sembako Bu Sari | 0% | No loan needed |
| Dapur Nusantara Catering | 95% | Loan blocked (repayment capacity); chasing invoices cuts risk to ~46% |
| Kirana Online Store | 72% | Loan available: IDR 4.5M, 18.9–19.9% a year all-in, repaid on time in 100% of simulations; chasing invoices alone cuts risk to ~1% |

With `MOCK_LLM=1` (default) answers come from the deterministic template engine. To use
a real model, set `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `MOCK_LLM=0`, and turn on
"Send pseudonymised figures to an external AI provider" for the business.

## Verify

```bash
python -m pytest -q                    # 100 tests
python scripts/run_redteam.py          # 30 attacks -> reports/redteam_report.md
python scripts/evaluate_forecast.py    # A/B/C evaluation -> reports/eval_report.md
python scripts/demo_run.py             # end-to-end sandbox run -> evidence/
```

## Deploy

```bash
export CASHBRIDGE_SECRET=$(python -c "import secrets;print(secrets.token_urlsafe(48))")
export DEMO_PASSWORD='choose-a-long-password'
docker compose up --build              # non-root, APP_ENV=production, SQLite on a volume
```

Serve behind HTTPS. Production refuses to start without the two secrets, disables
`/api/docs`, and sends HSTS. CI: `.github/workflows/ci.yml`.

## Submission checklist

| Item | Where |
|---|---|
| Technical documentation (architecture, algorithms, security design) | `docs/TECHNICAL_DOCUMENTATION.md` |
| Security self-assessment (tiers, known risks) | `docs/SECURITY_SELF_ASSESSMENT.md`, `reports/redteam_report.md` |
| Regulatory mapping (China, Indonesia) | `docs/COMPLIANCE.md` |
| Source code, README, deployment, tests | this repo, `Dockerfile`, `docker-compose.yml`, `tests/` |
| Execution evidence (sandbox logs) | `evidence/` (operation log, audit chain + verification, ledger) |
| Evaluation | `reports/eval_report.md`, `reports/eval_results.json` |

## Repository layout

```
app/          FastAPI app: guard/auth, pages, API, agent orchestrator, tools, policy,
              approval tokens, audit chain, templates, static (vendored Chart.js, fonts)
cashbridge/   engine (no LLM): forecast, gap, recommend, sizing, products, reconcile,
              ingest, seed, config
scripts/      seed_data, run_web, ask, evaluate_forecast, run_redteam, demo_run,
              fetch_datasets
tests/        100 tests
data/external Online Retail II daily aggregate (CC BY 4.0) for evaluation
docs/ reports/ evidence/
```

## Licences and data

Online Retail II (Chen 2012, UCI, CC BY 4.0, doi:10.24432/C5CG6D) — derived daily
aggregate only. Chart.js 4.4.1 (MIT), Roboto / Roboto Mono (Apache-2.0), Plus Jakarta
Sans (OFL-1.1); licence files are next to the assets. Demo businesses are synthetic.

> Forecasts are estimates, not financial advice. Every disbursement and repayment here
> is a sandbox simulation; no real money moves.
