"""
train_session_gru.py — train the session GRU on real MovieLens sessions.

Task     Next-genre prediction. Given the first n-1 events of a session, predict
         the primary genre of event n. This label exists in MovieLens, unlike
         "session intent", which does not and previously had to be fabricated.

Data     ML-1M interactions, sessionised on a 30-minute inactivity gap.

Split    BY USER. A user's sessions land entirely in train or entirely in
         validation, so the model cannot memorise a user and be graded on that
         same user's other sessions.

Baselines reported alongside the model, because an accuracy number without a
baseline says nothing:
    majority   always predict the most common next genre
    persist    predict the previous event's genre (sessions are genre-sticky,
               so this is the bar that actually has to be cleared)

Writes artifacts/bundle/session_gru.pkl
"""
from __future__ import annotations
import json, os, pickle, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recsys.serving.session_gru import (          # noqa: E402
    TrainableGRU, LinearHead, sequence_loss_and_grads, softmax)

DATA   = Path(os.environ.get("DATA_DIR",      ROOT / "data" / "processed"))
BUNDLE = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts")) / "bundle"
GAP_S      = int(os.environ.get("SESSION_GAP_S", str(30 * 60)))
MIN_EVENTS = 3
MAX_EVENTS = 20
HIDDEN_DIM = int(os.environ.get("GRU_HIDDEN", "32"))
EPOCHS     = int(os.environ.get("GRU_EPOCHS", "8"))
LR         = float(os.environ.get("GRU_LR", "0.05"))
CLIP       = 5.0
SEED       = int(os.environ.get("SEED", "42"))
BUNDLE.mkdir(parents=True, exist_ok=True)

t0 = time.time()
train_df  = pd.read_parquet(DATA / "train.parquet")
items     = pd.read_parquet(DATA / "items.parquet").set_index("item_id")
item_feat = pd.read_parquet(DATA / "features" / "item_features.parquet").set_index("item_id")

GENRES = sorted(items["primary_genre"].dropna().unique().tolist())
G_IX   = {g: i for i, g in enumerate(GENRES)}
N_CLASSES = len(GENRES)
N_EXTRA   = 6
INPUT_DIM = N_CLASSES + N_EXTRA
print(f"[gru] {N_CLASSES} genres · input_dim={INPUT_DIM} · hidden={HIDDEN_DIM}")

genre_of = items["primary_genre"].to_dict()
year_of  = items["year"].to_dict()
pop_of   = item_feat["item_cnt_total"].to_dict()
avg_of   = item_feat["item_avg_rating"].to_dict()
max_pop  = float(max(pop_of.values())) if pop_of else 1.0


def event_features(iid, rating, dt_s, position, prev_genre_ix):
    """One event -> INPUT_DIM floats. Everything here is known at request time."""
    v = np.zeros(INPUT_DIM, dtype=np.float64)
    g = genre_of.get(iid)
    if g in G_IX:
        v[G_IX[g]] = 1.0
    base = N_CLASSES
    v[base + 0] = float(rating) / 5.0
    v[base + 1] = 1.0 if rating >= 4 else 0.0
    v[base + 2] = np.log1p(float(pop_of.get(iid, 0.0))) / np.log1p(max_pop)
    v[base + 3] = float(avg_of.get(iid, 0.0)) / 5.0
    v[base + 4] = float(np.clip(np.log1p(max(dt_s, 0.0)) / np.log1p(GAP_S), 0.0, 1.0))
    v[base + 5] = float(min(position, MAX_EVENTS)) / MAX_EVENTS
    return v


# ── Sessionise ────────────────────────────────────────────────────────────────
print("[gru] sessionising real ML-1M interactions ...")
df = train_df.sort_values(["user_id", "timestamp"], kind="mergesort")
sessions_by_user: dict[int, list] = {}
cur_uid, cur, last_ts = None, [], None

for uid, iid, rating, ts in zip(df.user_id.to_numpy(), df.item_id.to_numpy(),
                                df.rating.to_numpy(), df.timestamp.to_numpy()):
    if uid != cur_uid or (last_ts is not None and ts - last_ts > GAP_S):
        if cur_uid is not None and len(cur) >= MIN_EVENTS:
            sessions_by_user.setdefault(int(cur_uid), []).append(cur)
        cur = []
        cur_uid = uid
    cur.append((int(iid), float(rating), int(ts)))
    last_ts = ts
if cur_uid is not None and len(cur) >= MIN_EVENTS:
    sessions_by_user.setdefault(int(cur_uid), []).append(cur)

n_sessions = sum(len(v) for v in sessions_by_user.values())
lengths = [len(s) for v in sessions_by_user.values() for s in v]
print(f"[gru] {n_sessions:,} sessions from {len(sessions_by_user):,} users "
      f"· median length {int(np.median(lengths))} · gap={GAP_S//60}min")


def to_example(sess):
    """(features for events 0..n-2, label = genre of event n-1)."""
    sess = sess[-(MAX_EVENTS + 1):]
    target_genre = genre_of.get(sess[-1][0])
    if target_genre not in G_IX:
        return None
    xs, prev_g = [], -1
    for pos, (iid, rating, ts) in enumerate(sess[:-1]):
        dt = ts - sess[pos - 1][2] if pos > 0 else 0.0
        xs.append(event_features(iid, rating, dt, pos, prev_g))
        prev_g = G_IX.get(genre_of.get(iid), -1)
    if not xs:
        return None
    prev_ix = G_IX.get(genre_of.get(sess[-2][0]), -1)
    return xs, G_IX[target_genre], prev_ix


# ── Split by user ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
users = np.array(sorted(sessions_by_user)); rng.shuffle(users)
cut = int(len(users) * 0.8)
train_users, val_users = set(users[:cut].tolist()), set(users[cut:].tolist())

def build(user_set):
    out = []
    for u in user_set:
        for s in sessions_by_user[u]:
            ex = to_example(s)
            if ex: out.append(ex)
    return out

train_ex, val_ex = build(train_users), build(val_users)
print(f"[gru] train {len(train_ex):,} sequences ({len(train_users):,} users) · "
      f"val {len(val_ex):,} sequences ({len(val_users):,} users)")
assert not (train_users & val_users), "user leakage between splits"

# ── Baselines ─────────────────────────────────────────────────────────────────
maj = Counter(lbl for _, lbl, _ in train_ex).most_common(1)[0][0]
maj_acc     = np.mean([lbl == maj for _, lbl, _ in val_ex])
persist_acc = np.mean([lbl == prev for _, lbl, prev in val_ex])
print(f"[gru] baselines on val — majority({GENRES[maj]})={maj_acc:.4f} "
      f"persist-previous-genre={persist_acc:.4f}")

# ── Train ─────────────────────────────────────────────────────────────────────
gru  = TrainableGRU(INPUT_DIM, HIDDEN_DIM, seed=SEED)
head = LinearHead(HIDDEN_DIM, N_CLASSES, seed=SEED + 1)

def evaluate(examples):
    correct = loss = 0.0
    for xs, lbl, _ in examples:
        h, _ = gru.forward(xs)
        p = softmax(head.logits(h))
        loss += -np.log(max(p[lbl], 1e-12))
        correct += int(np.argmax(p) == lbl)
    n = max(len(examples), 1)
    return correct / n, loss / n

history = []
best = {"val_acc": -1.0, "epoch": 0, "gru": None, "head": None}
order = np.arange(len(train_ex))
for epoch in range(EPOCHS):
    rng.shuffle(order)
    run_loss = run_correct = 0.0
    for count, i in enumerate(order, 1):
        xs, lbl, _ = train_ex[i]
        loss, correct, g, dW, db = sequence_loss_and_grads(gru, head, xs, lbl)
        run_loss += loss; run_correct += correct
        for name in TrainableGRU.PARAMS:
            grad = np.clip(g[name], -CLIP, CLIP)
            setattr(gru, name, gru.get(name) - LR * grad)
        head.W -= LR * np.clip(dW, -CLIP, CLIP)
        head.b -= LR * np.clip(db, -CLIP, CLIP)
    va, vl = evaluate(val_ex)
    history.append({"epoch": epoch + 1,
                    "train_acc": round(run_correct / len(train_ex), 4),
                    "train_loss": round(run_loss / len(train_ex), 4),
                    "val_acc": round(float(va), 4), "val_loss": round(float(vl), 4)})
    marker = ""
    if va > best["val_acc"]:
        # Keep the best epoch by held-out accuracy. Reporting the LAST epoch
        # would report whatever the final step happened to land on, which here
        # is a regression from epoch 2.
        best = {"val_acc": float(va), "epoch": epoch + 1,
                "gru": gru.state_dict(), "head": head.state_dict()}
        marker = "  <- best"
    print(f"  epoch {epoch+1}/{EPOCHS}  train_acc={history[-1]['train_acc']:.4f} "
          f"val_acc={va:.4f}  val_loss={vl:.4f}{marker}")

# Restore the best checkpoint before the final evaluation and before saving.
gru  = TrainableGRU.from_state(best["gru"])
head = LinearHead.from_state(best["head"])
val_acc, val_loss = evaluate(val_ex)
print(f"[gru] restored epoch {best['epoch']} (best held-out accuracy)")
result = {
    "task":             "next-genre prediction from a real ML-1M session prefix",
    "n_classes":        N_CLASSES,
    "genres":           GENRES,
    "input_dim":        INPUT_DIM,
    "hidden_dim":       HIDDEN_DIM,
    "epochs":           EPOCHS,
    "best_epoch":       best["epoch"],
    "session_gap_s":    GAP_S,
    "n_sessions":       n_sessions,
    "n_train_sequences": len(train_ex),
    "n_val_sequences":  len(val_ex),
    "split":            "by user, 80/20 — no user appears in both",
    "val_accuracy":     round(float(val_acc), 4),
    "val_loss":         round(float(val_loss), 4),
    "baseline_majority":       round(float(maj_acc), 4),
    "baseline_persist_genre":  round(float(persist_acc), 4),
    "beats_majority":  bool(val_acc > maj_acc),
    "beats_persist":   bool(val_acc > persist_acc),
    "history":          history,
    "trained_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "train_seconds":    round(time.time() - t0, 1),
}
with open(BUNDLE / "session_gru.pkl", "wb") as f:
    pickle.dump({"gru": gru.state_dict(), "head": head.state_dict(),
                 "genres": GENRES, "metrics": result}, f, protocol=4)
(BUNDLE / "session_gru_metrics.json").write_text(json.dumps(result, indent=2))

print(f"\n[gru] held-out accuracy {val_acc:.4f}  "
      f"(majority {maj_acc:.4f} · persist {persist_acc:.4f})")
print(f"[gru] wrote session_gru.pkl in {result['train_seconds']}s")
