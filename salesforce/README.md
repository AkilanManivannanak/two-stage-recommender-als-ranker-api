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
| `*Test.cls` | 24 test methods covering happy paths and all five failure modes. |
| `lwc/cineWaveRecommendations` | Lightning card. Shows both stage scores and labels metrics "offline". |

## Why GraphQL and not REST

The web client wants one fat `/recommend` response. An Agentforce action wants
three titles. A Lightning card wants posters plus both stage scores. Over REST
that is either several bespoke endpoints or heavy over-fetching across a metered
callout with a governor-limited timeout. One typed schema serves all three, each
asking for exactly what it renders — and Apex is statically typed, so a schema it
can introspect beats a JSON blob it must parse defensively.

## Deploy to a Developer Edition org

**1. Get an org** — sign up free at
<https://developer.salesforce.com/signup> (Developer Edition, no expiry).

**2. Install the CLI**

```bash
npm install --global @salesforce/cli
sf --version
```

**3. Authorise the org**

```bash
cd salesforce
sf org login web --alias cinewave-dev --set-default
```

**4. Point the Named Credential at a reachable CineWave**

Apex callouts must reach a public HTTPS host — `localhost` will not work from
Salesforce. Expose the local API first:

```bash
# terminal 1 — the recommender
cd backend && PYTHONPATH=src python3 -m uvicorn recsys.serving.app:app --port 8000

# terminal 2 — a public tunnel
ngrok http 8000        # or: cloudflared tunnel --url http://localhost:8000
```

Put the resulting `https://...` origin into
`force-app/main/default/namedCredentials/CineWave_API.namedCredential-meta.xml`
(the `Url` parameter). It lives in metadata, never in Apex, so the same classes
promote from scratch org to sandbox to production without an edit.

**5. Deploy and assign**

```bash
sf project deploy start --source-dir force-app
sf org assign permset --name CineWave_User
```

**6. Run the Apex tests**

```bash
sf apex run test --code-coverage --result-format human --wait 10
```

**7. Try it end to end**

```bash
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
