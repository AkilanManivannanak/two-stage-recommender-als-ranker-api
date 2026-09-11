# CineWave — Salesforce integration

Apex + Lightning Web Component layer that consumes the CineWave recommender
through its GraphQL API. Salesforce is the front office; CineWave is the ML
backend. Nothing about the model runs in Apex.

```
Agentforce topic ─┐
Flow             ─┼─► CineWaveRecommenderAction  ─┐
LWC on a record  ─┴─► CineWaveController         ─┴─► CineWaveGraphQLClient
                                                        │  Named Credential
                                                        ▼
                                              POST /graphql  (FastAPI)
                                                        │
                                              ALS retrieval → LambdaRank rerank
```

## What's here

| File | Role |
|---|---|
| `CineWaveGraphQLClient.cls` | Single callout path. Named Credential, typed errors, handles GraphQL's 200-with-errors case. |
| `CineWaveRecommenderAction.cls` | `@InvocableMethod` for Agentforce / Flow / Prompt Builder. Bounded batch, degrades to `success=false` instead of throwing at a user. |
| `CineWaveController.cls` | `@AuraEnabled` controller. Read is `cacheable`, feedback write deliberately is not. |
| `CineWaveRecommendation.cls` | DTO. Keeps `alsScore` and `rankerScore` as separate fields. |
| `CineWaveCalloutMock.cls` | Nine response modes, including every failure branch. |
| `*Test.cls` | 30 test methods covering happy paths and all five failure modes. |
| `lwc/cineWaveRecommendations` | Lightning card. Shows both stage scores and labels metrics "offline". |

## Why GraphQL and not REST

The web client wants one fat `/recommend` response. An Agentforce action wants
three titles. A Lightning card wants posters plus both stage scores. Over REST
that is either several bespoke endpoints or heavy over-fetching across a metered
callout with a governor-limited timeout. One typed schema serves all three, each
asking for exactly what it renders — and Apex is statically typed, so a schema it
can introspect beats a JSON blob it must parse defensively.

## Deploy to a Developer Edition org

One command, once the prerequisites are in place:

```bash
./scripts/deploy.sh https://your-tunnel-url.ngrok-free.app
```

It checks the API answers a GraphQL query, validates the source offline, writes
the URL into the Named Credential, authorises the org, deploys, assigns the
permission set, and runs the Apex tests.

### Prerequisites

**1. An org** — free Developer Edition, no expiry:
<https://developer.salesforce.com/signup>

**2. The CLI**

```bash
npm install --global @salesforce/cli
sf --version
```

**3. A publicly reachable API.** Salesforce callouts cannot reach `localhost`.

```bash
# terminal 1 — the recommender
cd backend && PYTHONPATH=src python3 -m uvicorn recsys.serving.app:app --port 8000

# terminal 2 — a public tunnel
ngrok http 8000        # or: cloudflared tunnel --url http://localhost:8000
```

Pass the `https://` origin ngrok prints as the argument to `deploy.sh`.

### Validating without an org

`sf project deploy start` needs an authenticated org, so these checks run
offline first and catch most of what a failed deploy would have told you:

```bash
python3 scripts/validate_source.py
```

It verifies every class has a meta XML, API versions match, braces balance, all
endpoints resolve through a Named Credential rather than a hardcoded host, LWC
Apex imports point at real `@AuraEnabled` methods, the permission set names only
classes that exist, the invocable action is shaped for Flow, and every test of a
callout-making class sets a mock.

### Doing it by hand

```bash
sf org login web --alias cinewave-dev --set-default
# edit force-app/main/default/namedCredentials/CineWave_API.namedCredential-meta.xml
sf project deploy start --source-dir force-app
sf org assign permset --name CineWave_User
sf apex run test --code-coverage --result-format human --wait 10
sf apex run --file scripts/smoke.apex
```

## Add it to Agentforce

Setup → Agentforce Studio → your agent → **Topics** → New Action → **Apex** →
*Get CineWave Recommendations*. Map `userId` from the conversation context. The
action returns `summary` as a ready-to-speak sentence and `recommendations` as
structured rows.

## Add the LWC to a page

Setup → Lightning App Builder → edit a record or app page → drag **CineWave
Recommendations** on → set *CineWave user id* (1–6040 in the MovieLens
catalogue).

## Contract testing

The Apex builds its GraphQL documents as string literals, which no Salesforce
tool validates against the Python schema. `backend/tests/test_apex_contract.py`
parses those literals straight out of the `.cls` source and executes them against
the live schema, so a field rename on either side fails in CI rather than in a
customer's org:

```bash
cd backend && PYTHONPATH=src python3 -m pytest tests/test_apex_contract.py -q
```
