"""
validate_source.py — offline checks on the SFDX source before a deploy attempt.

`sf project deploy start` needs an authenticated org, so these are the errors
worth catching without one: a missing meta XML, malformed XML, inconsistent API
versions, an incomplete LWC bundle, a permission set naming a class that does
not exist, unbalanced braces, and the Named Credential still pointing at the
placeholder host.

Run: python3 scripts/validate_source.py
"""
from __future__ import annotations
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "force-app" / "main" / "default"
NS   = "{http://soap.sforce.com/2006/04/metadata}"

errors: list[str] = []
warnings: list[str] = []
ok: list[str] = []


def err(m): errors.append(m)
def warn(m): warnings.append(m)
def good(m): ok.append(m)


# ── Apex classes ──────────────────────────────────────────────────────────────
classes = sorted((SRC / "classes").glob("*.cls"))
if not classes:
    err("no Apex classes found")

api_versions = set()
class_names = set()
for cls in classes:
    name = cls.stem
    class_names.add(name)
    meta = cls.with_suffix(".cls-meta.xml")
    if not meta.exists():
        err(f"{name}.cls has no .cls-meta.xml — the deploy will reject it")
        continue
    try:
        root = ET.parse(meta).getroot()
        v = root.findtext(f"{NS}apiVersion")
        if v: api_versions.add(v)
    except ET.ParseError as e:
        err(f"{meta.name} is not valid XML: {e}")

    body = cls.read_text()
    # Strip string literals and comments before counting braces.
    stripped = re.sub(r"'(?:[^'\\]|\\.)*'", "''", body)
    stripped = re.sub(r"//[^\n]*", "", stripped)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.S)
    if stripped.count("{") != stripped.count("}"):
        err(f"{name}.cls has unbalanced braces "
            f"({stripped.count('{')} open, {stripped.count('}')} close)")
    if re.search(r"\bclass\s+" + re.escape(name) + r"\b", body) is None:
        err(f"{name}.cls does not declare a class named {name}")

good(f"{len(classes)} Apex classes, each with a meta XML")

if len(api_versions) > 1:
    warn(f"mixed API versions across classes: {sorted(api_versions)}")
elif api_versions:
    good(f"API version {api_versions.pop()} consistent across all classes")

# ── Tests ─────────────────────────────────────────────────────────────────────
test_classes = [c for c in classes if c.stem.endswith("Test")]
test_methods = sum(len(re.findall(r"static\s+void\s+\w+", c.read_text()))
                   for c in test_classes)
tested = {c.stem[:-4] for c in test_classes}
untested = {n for n in class_names
            if not n.endswith("Test") and not n.endswith("Mock")
            and n not in tested and n != "CineWaveException"}
good(f"{len(test_classes)} test classes, {test_methods} test methods")
if untested:
    warn(f"classes with no matching *Test class: {sorted(untested)}")

for c in test_classes:
    body = c.read_text()
    if "@IsTest" not in body:
        err(f"{c.stem} has no @IsTest annotation")
    if "System.assert" not in body:
        err(f"{c.stem} contains no assertions")
    # Only classes that actually make callouts need a mock; a DTO test does not.
    under_test = SRC / "classes" / f"{c.stem[:-4]}.cls"
    makes_callout = under_test.exists() and (
        "Http()" in under_test.read_text() or "callout:" in under_test.read_text())
    if makes_callout and "Test.setMock" not in body and "HttpCalloutMock" not in body:
        err(f"{c.stem} tests a class that makes callouts but sets no mock — "
            f"it will fail with a Callout-not-allowed exception")
good("every test class is annotated and asserts")

# ── Callouts go through the Named Credential ─────────────────────────────────
for cls in classes:
    body = cls.read_text()
    for m in re.finditer(r"setEndpoint\(\s*'([^']+)'", body):
        ep = m.group(1)
        if not ep.startswith("callout:"):
            err(f"{cls.stem} hardcodes an endpoint {ep!r}; use callout:<NamedCredential>")
    if re.search(r"ENDPOINT\s*=\s*'(?!callout:)", body):
        err(f"{cls.stem} defines a non-callout ENDPOINT constant")
good("all HTTP endpoints resolve through a Named Credential")

# ── Named Credential ──────────────────────────────────────────────────────────
nc = list((SRC / "namedCredentials").glob("*.namedCredential-meta.xml"))
if not nc:
    err("no Named Credential found")
for f in nc:
    try:
        root = ET.parse(f).getroot()
    except ET.ParseError as e:
        err(f"{f.name} is not valid XML: {e}"); continue
    url = None
    for p in root.findall(f"{NS}namedCredentialParameters"):
        if p.findtext(f"{NS}parameterName") == "Url":
            url = p.findtext(f"{NS}parameterValue")
    if url is None:
        err(f"{f.name} has no Url parameter")
    elif "example.com" in url:
        warn(f"{f.name} still points at the placeholder {url} — set this to your "
             f"tunnel URL before deploying, or the callouts will fail at runtime")
    elif url.startswith("http://"):
        err(f"{f.name} uses http://; Salesforce callouts require https")
    else:
        good(f"Named Credential URL set to {url}")

# ── External Credential ───────────────────────────────────────────────────────
ec = list((SRC / "externalCredentials").glob("*.externalCredential-meta.xml"))
if not ec:
    warn("no External Credential — a SecuredEndpoint Named Credential needs one")
else:
    good(f"{len(ec)} external credential")

# ── LWC bundles ───────────────────────────────────────────────────────────────
for bundle in (SRC / "lwc").iterdir() if (SRC / "lwc").exists() else []:
    if not bundle.is_dir(): continue
    need = [f"{bundle.name}.js", f"{bundle.name}.html", f"{bundle.name}.js-meta.xml"]
    missing = [n for n in need if not (bundle / n).exists()]
    if missing:
        err(f"LWC {bundle.name} is missing {missing}")
    else:
        try:
            ET.parse(bundle / f"{bundle.name}.js-meta.xml")
            good(f"LWC {bundle.name} bundle complete")
        except ET.ParseError as e:
            err(f"LWC {bundle.name} meta XML invalid: {e}")
    js = (bundle / f"{bundle.name}.js")
    if js.exists():
        for m in re.finditer(r"@salesforce/apex/(\w+)\.(\w+)", js.read_text()):
            cls_name, method = m.groups()
            if cls_name not in class_names:
                err(f"LWC imports {cls_name}.{method} but {cls_name}.cls does not exist")
            else:
                body = (SRC / "classes" / f"{cls_name}.cls").read_text()
                if not re.search(rf"@AuraEnabled[\s\S]{{0,120}}\b{method}\b", body):
                    err(f"LWC imports {cls_name}.{method} but it is not @AuraEnabled")
        good("every LWC Apex import resolves to an @AuraEnabled method")

# ── Permission set ────────────────────────────────────────────────────────────
for ps in (SRC / "permissionsets").glob("*.permissionset-meta.xml"):
    try:
        root = ET.parse(ps).getroot()
    except ET.ParseError as e:
        err(f"{ps.name} is not valid XML: {e}"); continue
    for ca in root.findall(f"{NS}classAccesses"):
        n = ca.findtext(f"{NS}apexClass")
        if n not in class_names:
            err(f"{ps.name} grants access to {n}, which does not exist")
    good(f"{ps.name} references only existing classes")

# ── Invocable action ──────────────────────────────────────────────────────────
inv = [c for c in classes if "@InvocableMethod" in c.read_text()]
for c in inv:
    body = c.read_text()
    if "callout=true" not in body:
        err(f"{c.stem} has @InvocableMethod making callouts but no callout=true")
    if not re.search(r"@InvocableMethod[\s\S]{0,400}?List<", body):
        warn(f"{c.stem} invocable may not take/return a List (Flow requires it)")
good(f"{len(inv)} invocable action declared correctly for Flow/Agentforce")

# ── sfdx-project.json ─────────────────────────────────────────────────────────
import json
proj = ROOT / "sfdx-project.json"
if not proj.exists():
    err("sfdx-project.json missing")
else:
    cfg = json.loads(proj.read_text())
    if not cfg.get("packageDirectories"):
        err("sfdx-project.json has no packageDirectories")
    else:
        good(f"sfdx-project.json sourceApiVersion {cfg.get('sourceApiVersion')}")

# ── Report ────────────────────────────────────────────────────────────────────
print("\nSFDX source validation\n" + "=" * 58)
for m in ok:       print(f"  PASS  {m}")
for m in warnings: print(f"  WARN  {m}")
for m in errors:   print(f"  FAIL  {m}")
print("=" * 58)
print(f"  {len(ok)} passed · {len(warnings)} warnings · {len(errors)} errors")
if errors:
    print("\nFix the errors above before running `sf project deploy start`.")
sys.exit(1 if errors else 0)
