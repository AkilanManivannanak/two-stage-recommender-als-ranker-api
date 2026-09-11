#!/usr/bin/env bash
# deploy.sh — take CineWave's Salesforce layer from source to a live org.
#
#   ./scripts/deploy.sh <https-url-of-a-running-cinewave-api> [org-alias]
#
# Salesforce callouts cannot reach localhost, so the first argument must be a
# public HTTPS origin. Start a tunnel in another terminal first:
#
#   cd backend && PYTHONPATH=src python3 -m uvicorn recsys.serving.app:app --port 8000
#   ngrok http 8000                 # or: cloudflared tunnel --url http://localhost:8000
#
set -euo pipefail
cd "$(dirname "$0")/.."

URL="${1:-}"
ALIAS="${2:-cinewave-dev}"
NC="force-app/main/default/namedCredentials/CineWave_API.namedCredential-meta.xml"

if [[ -z "$URL" ]]; then
  echo "usage: ./scripts/deploy.sh <https://your-tunnel-url> [org-alias]" >&2
  exit 2
fi
[[ "$URL" == https://* ]] || { echo "error: the URL must be https:// — Salesforce refuses http callouts" >&2; exit 2; }
URL="${URL%/}"

command -v sf >/dev/null || { echo "error: Salesforce CLI not found. npm install --global @salesforce/cli" >&2; exit 2; }

echo "==> 0/6  checking the API is reachable at $URL"
if ! curl -sf -m 20 "$URL/graphql" -H 'Content-Type: application/json' \
     -d '{"query":"query{modelMetrics{status ndcgAt10}}"}' | tee /dev/stderr | grep -q '"status"'; then
  echo "error: $URL/graphql did not answer a GraphQL query. Is the tunnel up and the API running?" >&2
  exit 1
fi

echo "==> 1/6  validating source offline"
python3 scripts/validate_source.py

echo "==> 2/6  pointing the Named Credential at $URL"
python3 - "$NC" "$URL" <<'PY'
import re, sys
path, url = sys.argv[1], sys.argv[2]
s = open(path).read()
s = re.sub(r"(<parameterName>Url</parameterName>\s*<parameterType>Url</parameterType>\s*<parameterValue>)[^<]*(</parameterValue>)",
           rf"\g<1>{url}\g<2>", s, flags=re.S)
open(path, "w").write(s)
print(f"    set to {url}")
PY

echo "==> 3/6  authorising the org (a browser window will open if needed)"
sf org display --target-org "$ALIAS" >/dev/null 2>&1 || sf org login web --alias "$ALIAS" --set-default

echo "==> 4/6  deploying"
sf project deploy start --source-dir force-app --target-org "$ALIAS" --wait 20

echo "==> 5/6  assigning the permission set"
sf org assign permset --name CineWave_User --target-org "$ALIAS" || true

echo "==> 6/6  running Apex tests"
sf apex run test --target-org "$ALIAS" --code-coverage --result-format human \
                 --wait 20 --test-level RunLocalTests

echo
echo "Deployed. End-to-end check against the live API:"
echo "  sf apex run --file scripts/smoke.apex --target-org $ALIAS"
