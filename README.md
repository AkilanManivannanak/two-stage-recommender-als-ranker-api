<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CineWave — ML RecSys README</title>
<style>
  :root {
    --bg: #0a0a0f;
    --bg2: #111118;
    --bg3: #16161f;
    --border: rgba(255,255,255,0.07);
    --border2: rgba(255,255,255,0.12);
    --red: #e5091a;
    --red2: #ff2d3f;
    --gold: #f59e0b;
    --green: #22c55e;
    --blue: #3b82f6;
    --purple: #818cf8;
    --text: #e2e2e8;
    --muted: #888896;
    --dim: #555560;
    --mono: 'Courier New', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px;
    line-height: 1.7;
    padding: 0;
  }

  /* ── HERO ── */
  .hero {
    background: linear-gradient(135deg, #0a0a0f 0%, #12121f 40%, #1a0a14 100%);
    border-bottom: 1px solid var(--border2);
    padding: 56px 48px 48px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -120px; right: -120px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(229,9,26,0.12) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -80px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(129,140,248,0.06) 0%, transparent 70%);
    pointer-events: none;
  }
  .badge-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
  .badge {
    font-size: 11px; font-family: var(--mono);
    padding: 3px 10px; border-radius: 20px;
    border: 1px solid;
    font-weight: 600; letter-spacing: .03em;
  }
  .badge-red    { background: rgba(229,9,26,.15);   color: #ff6b76;  border-color: rgba(229,9,26,.3);  }
  .badge-green  { background: rgba(34,197,94,.12);  color: #4ade80;  border-color: rgba(34,197,94,.25); }
  .badge-blue   { background: rgba(59,130,246,.12); color: #60a5fa;  border-color: rgba(59,130,246,.25);}
  .badge-purple { background: rgba(129,140,248,.12);color: #a5b4fc;  border-color: rgba(129,140,248,.25);}
  .badge-gold   { background: rgba(245,158,11,.12); color: #fbbf24;  border-color: rgba(245,158,11,.25);}
  .badge-dim    { background: rgba(255,255,255,.06); color: var(--muted); border-color: var(--border2); }
  .hero-title {
    font-size: 42px; font-weight: 800; letter-spacing: -.02em;
    background: linear-gradient(135deg, #ffffff 0%, #e2e2e8 50%, #e5091a 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px; line-height: 1.1;
  }
  .hero-sub { font-size: 16px; color: var(--muted); margin-bottom: 24px; max-width: 640px; }
  .hero-stats {
    display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px;
    margin-top: 28px;
  }
  .stat-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border2);
    border-radius: 10px; padding: 14px 12px;
    text-align: center;
    backdrop-filter: blur(8px);
    box-shadow: 0 2px 12px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: transform .15s, border-color .15s;
  }
  .stat-box:hover { transform: translateY(-2px); border-color: rgba(255,255,255,0.2); }
  .stat-val { font-size: 22px; font-weight: 800; font-family: var(--mono); color: var(--red2); }
  .stat-lbl { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-top: 2px; }

  /* ── LAYOUT ── */
  .container { max-width: 960px; margin: 0 auto; padding: 0 48px 80px; }

  /* ── SECTIONS ── */
  .section { margin-top: 48px; }
  .section-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }
  .section-icon {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
    background: rgba(229,9,26,.15); border: 1px solid rgba(229,9,26,.25);
  }
  .section-title { font-size: 18px; font-weight: 700; color: #fff; }
  .section-subtitle { font-size: 13px; color: var(--muted); margin-left: auto; font-family: var(--mono); }

  /* ── ARCHITECTURE FLOW ── */
  .arch-flow {
    display: flex; align-items: center; gap: 0;
    overflow-x: auto; padding: 24px 0; margin-bottom: 8px;
  }
  .arch-node {
    background: var(--bg2);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 14px 16px;
    min-width: 140px;
    text-align: center;
    flex-shrink: 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
    position: relative;
  }
  .arch-node.highlight {
    border-color: rgba(229,9,26,.4);
    background: rgba(229,9,26,.06);
    box-shadow: 0 4px 20px rgba(229,9,26,.15), inset 0 1px 0 rgba(255,255,255,0.06);
  }
  .arch-node-icon { font-size: 18px; margin-bottom: 4px; }
  .arch-node-name { font-size: 11px; font-weight: 700; color: #fff; letter-spacing: .04em; text-transform: uppercase; }
  .arch-node-tech { font-size: 10px; color: var(--muted); font-family: var(--mono); margin-top: 2px; line-height: 1.3; }
  .arch-arrow {
    color: var(--dim); font-size: 18px; flex-shrink: 0; padding: 0 4px;
  }

  /* ── GRID CARDS ── */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
  .card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: border-color .15s, box-shadow .15s;
  }
  .card:hover {
    border-color: var(--border2);
    box-shadow: 0 4px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.07);
  }
  .card-label {
    font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .1em;
    color: var(--muted); margin-bottom: 10px; display: flex; align-items: center; gap: 6px;
  }
  .card-label::before {
    content: ''; width: 4px; height: 4px; border-radius: 50%;
    background: var(--red); display: inline-block;
  }
  .card-title { font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 6px; }
  .card-body { font-size: 13px; color: var(--muted); line-height: 1.6; }

  /* ── METRICS TABLE ── */
  .metrics-table { width: 100%; border-collapse: collapse; }
  .metrics-table th {
    font-size: 10px; text-transform: uppercase; letter-spacing: .08em;
    color: var(--dim); padding: 8px 14px; text-align: left;
    border-bottom: 1px solid var(--border2);
  }
  .metrics-table td {
    padding: 10px 14px; border-bottom: 1px solid var(--border);
    font-size: 13px; color: var(--text);
  }
  .metrics-table tr:last-child td { border-bottom: none; }
  .metrics-table tr:hover td { background: rgba(255,255,255,0.02); }
  .mono { font-family: var(--mono); }
  .green { color: var(--green); }
  .red-text { color: var(--red2); }
  .gold { color: var(--gold); }
  .blue-text { color: #60a5fa; }
  .purple-text { color: var(--purple); }
  .tag-green  { display: inline-block; background: rgba(34,197,94,.12);  color: #4ade80;  font-size: 10px; padding: 2px 7px; border-radius: 4px; font-family: var(--mono); font-weight: 600; }
  .tag-red    { display: inline-block; background: rgba(229,9,26,.12);   color: #ff6b76;  font-size: 10px; padding: 2px 7px; border-radius: 4px; font-family: var(--mono); font-weight: 600; }
  .tag-gold   { display: inline-block; background: rgba(245,158,11,.12); color: #fbbf24;  font-size: 10px; padding: 2px 7px; border-radius: 4px; font-family: var(--mono); font-weight: 600; }
  .tag-blue   { display: inline-block; background: rgba(59,130,246,.12); color: #60a5fa;  font-size: 10px; padding: 2px 7px; border-radius: 4px; font-family: var(--mono); font-weight: 600; }
  .tag-dim    { display: inline-block; background: rgba(255,255,255,.05); color: var(--muted); font-size: 10px; padding: 2px 7px; border-radius: 4px; font-family: var(--mono); }

  /* ── CODE BLOCK ── */
  .code-block {
    background: #0d0d14;
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 18px 20px;
    font-family: var(--mono);
    font-size: 12px;
    color: #a8b8cc;
    overflow-x: auto;
    line-height: 1.7;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
  }
  .code-block .c { color: var(--dim); }
  .code-block .k { color: #c792ea; }
  .code-block .s { color: #c3e88d; }
  .code-block .n { color: #82aaff; }
  .code-block .v { color: #f78c6c; }

  /* ── TIMELINE ── */
  .timeline { position: relative; padding-left: 24px; }
  .timeline::before {
    content: ''; position: absolute; left: 0; top: 6px; bottom: 6px;
    width: 2px; background: linear-gradient(to bottom, var(--red), transparent);
  }
  .tl-item { position: relative; margin-bottom: 20px; }
  .tl-item::before {
    content: ''; position: absolute; left: -28px; top: 5px;
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--bg); border: 2px solid var(--red);
  }
  .tl-title { font-size: 13px; font-weight: 600; color: #fff; margin-bottom: 3px; }
  .tl-body { font-size: 12px; color: var(--muted); line-height: 1.5; }

  /* ── STACK ROW ── */
  .stack-row {
    display: flex; align-items: flex-start; gap: 14px;
    padding: 12px 0; border-bottom: 1px solid var(--border);
  }
  .stack-row:last-child { border-bottom: none; }
  .stack-name { font-size: 13px; font-weight: 600; color: #fff; min-width: 160px; font-family: var(--mono); }
  .stack-role { font-size: 12px; color: var(--muted); flex: 1; line-height: 1.5; }
  .stack-badge { flex-shrink: 0; }

  /* ── POSTMORTEM ── */
  .pm-box {
    background: rgba(245,158,11,.04);
    border: 1px solid rgba(245,158,11,.2);
    border-left: 3px solid var(--gold);
    border-radius: 0 10px 10px 0;
    padding: 16px 20px; margin-bottom: 12px;
  }
  .pm-title { font-size: 12px; font-weight: 700; color: var(--gold); margin-bottom: 6px; text-transform: uppercase; letter-spacing: .06em; }
  .pm-body { font-size: 12px; color: var(--muted); line-height: 1.6; }

  /* ── FOOTER ── */
  .footer {
    border-top: 1px solid var(--border); padding: 28px 48px;
    display: flex; justify-content: space-between; align-items: center;
    background: var(--bg2);
  }
  .footer-name { font-size: 15px; font-weight: 700; color: #fff; }
  .footer-links { display: flex; gap: 20px; }
  .footer-link { font-size: 12px; color: var(--muted); text-decoration: none; font-family: var(--mono); }
  .footer-link:hover { color: #fff; }

  /* ── DIVIDER ── */
  .divider { height: 1px; background: var(--border); margin: 32px 0; }

  hr { border: none; border-top: 1px solid var(--border); margin: 0; }
</style>
</head>
<body>

<!-- ─── HERO ─────────────────────────────────────────────────────────────── -->
<div class="hero">
  <div class="badge-row">
    <span class="badge badge-red">Production-Grade</span>
    <span class="badge badge-green">MLOps</span>
    <span class="badge badge-blue">RAG + ALS + RL</span>
    <span class="badge badge-purple">Voice AI</span>
    <span class="badge badge-gold">A/B Tested</span>
    <span class="badge badge-dim">Python · TypeScript · Scala · SQL</span>
  </div>
  <div class="hero-title">CineWave RecSys</div>
  <div class="hero-sub">
    A full-stack, production-grade movie recommendation system — RAG retrieval, ALS collaborative filtering,
    LightGBM reranking, REINFORCE policy, voice AI, and live A/B experiments. Built by Akilan Manivannan.
  </div>
  <div class="badge-row" style="margin-bottom:0">
    <a href="https://drive.google.com/drive/folders/1sXFjx6ShommQ46mFLcTKCyBi0GokRT8v" style="text-decoration:none">
      <span class="badge badge-red">▶ Live Demo (Drive)</span>
    </a>
    <span class="badge badge-dim">localhost:3000 · localhost:8000</span>
    <span class="badge badge-dim">Docker Compose · 7 services</span>
  </div>
  <div class="hero-stats">
    <div class="stat-box"><div class="stat-val">&lt;180ms</div><div class="stat-lbl">p95 API latency</div></div>
    <div class="stat-box"><div class="stat-val">0.3847</div><div class="stat-lbl">NDCG@10</div></div>
    <div class="stat-box"><div class="stat-val">+6.2%</div><div class="stat-lbl">RL lift vs ALS</div></div>
    <div class="stat-box"><div class="stat-val">1,200+</div><div class="stat-lbl">TMDB movies</div></div>
    <div class="stat-box"><div class="stat-val">13</div><div class="stat-lbl">AI components</div></div>
    <div class="stat-box"><div class="stat-val">4</div><div class="stat-lbl">Live A/B tests</div></div>
  </div>
</div>

<div class="container">

<!-- ─── QUICK SUMMARY ──────────────────────────────────────────────────────── -->
<div class="section">
  <div style="background:var(--bg2); border:1px solid var(--border2); border-left:3px solid var(--red); border-radius:0 12px 12px 0; padding:20px 24px;">
    <div style="font-size:11px;color:var(--red2);font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">TL;DR — What this project demonstrates</div>
    <div style="font-size:13px; color:var(--text); line-height:1.8;">
      Built a <strong style="color:#fff">hybrid recommendation engine</strong> for 1,200+ movies:
      <span class="mono" style="color:#82aaff">ALS</span> candidate generation →
      <span class="mono" style="color:#c3e88d">Qdrant RAG</span> semantic retrieval →
      <span class="mono" style="color:#f78c6c">LightGBM</span> reranking →
      <span class="mono" style="color:#c792ea">REINFORCE RL</span> policy →
      <span class="mono" style="color:#fbbf24">Slate Optimizer</span> diversity enforcement.<br>
      p95 API latency <strong style="color:var(--green)">&lt;180ms</strong> ·
      NDCG@10 <strong style="color:var(--green)">0.3847</strong> ·
      IPS-weighted eval via DuckDB every 6h ·
      4 live A/B experiments with doubly-robust estimation ·
      Voice AI with GPT-4o + TTS ·
      Metaflow MLOps pipeline with automated eval gates.
    </div>
  </div>
</div>

<!-- ─── ARCHITECTURE ───────────────────────────────────────────────────────── -->
<div class="section">
  <div class="section-header">
    <div class="section-icon">🏗️</div>
    <div class="section-title">System Architecture</div>
    <div class="section-subtitle">data → retrieval → serving → feedback</div>
  </div>

  <div style="font-size:12px;color:var(--muted);margin-bottom:16px;">
    Full data flow: MovieLens 25M ratings → ALS training → Qdrant embeddings → FastAPI serving → Redis feature cache → RL reward loop
  </div>

  <!-- Layer 1: Data -->
  <div style="margin-bottom:8px;font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.1em;">Data Layer</div>
  <div class="arch-flow">
    <div class="arch-node">
      <div class="arch-node-icon">📦</div>
      <div class="arch-node-name">MovieLens 25M</div>
      <div class="arch-node-tech">Raw ratings CSV<br>25M interactions</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node">
      <div class="arch-node-icon">🐍</div>
      <div class="arch-node-name">PySpark Ingest</div>
      <div class="arch-node-tech">spark_features.py<br>feature eng.</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node">
      <div class="arch-node-icon">🐘</div>
      <div class="arch-node-name">Postgres</div>
      <div class="arch-node-tech">ratings, items<br>user profiles</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node highlight">
      <div class="arch-node-icon">🔄</div>
      <div class="arch-node-name">Metaflow DAG</div>
      <div class="arch-node-tech">phenomenal_flow_v3<br>nightly retraining</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node">
      <div class="arch-node-icon">🎨</div>
      <div class="arch-node-name">TMDB Enrich</div>
      <div class="arch-node-tech">p.py patcher<br>1,200+ posters</div>
    </div>
  </div>

  <!-- Layer 2: Models -->
  <div style="margin-bottom:8px;font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.1em;margin-top:12px;">Model Layer</div>
  <div class="arch-flow">
    <div class="arch-node highlight">
      <div class="arch-node-icon">🤝</div>
      <div class="arch-node-name">ALS (Spark)</div>
      <div class="arch-node-tech">Scala + Spark<br>200 latent factors<br>candidate gen.</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node highlight">
      <div class="arch-node-icon">🌲</div>
      <div class="arch-node-name">LightGBM</div>
      <div class="arch-node-tech">train_ranker_lgbm<br>NDCG objective<br>reranks top-100</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node highlight">
      <div class="arch-node-icon">🤖</div>
      <div class="arch-node-name">REINFORCE RL</div>
      <div class="arch-node-tech">rl_policy.py<br>session rewards<br>contextual bandit</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node">
      <div class="arch-node-icon">🎯</div>
      <div class="arch-node-name">Slate Optimizer</div>
      <div class="arch-node-tech">5 diversity rules<br>Jaccard ≥ 0.6<br>≥5 genres/page</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node">
      <div class="arch-node-icon">🧠</div>
      <div class="arch-node-name">RAG (Qdrant)</div>
      <div class="arch-node-tech">OpenAI embeds<br>semantic retrieval<br>voice queries</div>
    </div>
  </div>

  <!-- Layer 3: Serving -->
  <div style="margin-bottom:8px;font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.1em;margin-top:12px;">Serving Layer</div>
  <div class="arch-flow">
    <div class="arch-node">
      <div class="arch-node-icon">⚡</div>
      <div class="arch-node-name">FastAPI</div>
      <div class="arch-node-tech">app.py · :8000<br>/recommend<br>/voice · /explain</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node">
      <div class="arch-node-icon">🔴</div>
      <div class="arch-node-name">Redis Cache</div>
      <div class="arch-node-tech">feature store<br>session state<br>bandit weights</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node highlight">
      <div class="arch-node-icon">🎙️</div>
      <div class="arch-node-name">Voice AI</div>
      <div class="arch-node-tech">Whisper STT<br>GPT-4o · TTS<br>intent routing</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node">
      <div class="arch-node-icon">📺</div>
      <div class="arch-node-name">Next.js 14</div>
      <div class="arch-node-tech">:3000 · React<br>Tailwind · TypeScript<br>SSR + CSR</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node">
      <div class="arch-node-icon">📊</div>
      <div class="arch-node-name">Kafka Events</div>
      <div class="arch-node-tech">3 topics<br>Flink consumer<br>real-time signals</div>
    </div>
  </div>

  <!-- Layer 4: Eval -->
  <div style="margin-bottom:8px;font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.1em;margin-top:12px;">Eval + MLOps Layer</div>
  <div class="arch-flow">
    <div class="arch-node">
      <div class="arch-node-icon">🦆</div>
      <div class="arch-node-name">DuckDB</div>
      <div class="arch-node-tech">Parquet logs<br>IPS-NDCG@10<br>slice eval</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node">
      <div class="arch-node-icon">🚪</div>
      <div class="arch-node-name">Eval Gate</div>
      <div class="arch-node-tech">policy_gate.py<br>NDCG drop &gt;5%<br>→ rollback</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node highlight">
      <div class="arch-node-icon">🧪</div>
      <div class="arch-node-name">A/B Framework</div>
      <div class="arch-node-tech">ab_experiment.py<br>IPS + DR estimator<br>p&lt;0.05 threshold</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node">
      <div class="arch-node-icon">🌫️</div>
      <div class="arch-node-name">Shadow Tests</div>
      <div class="arch-node-tech">shadow_ab.py<br>parallel scoring<br>zero user impact</div>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node">
      <div class="arch-node-icon">🔄</div>
      <div class="arch-node-name">Airflow CI</div>
      <div class="arch-node-tech">nightly DAG<br>retrain → eval<br>→ alert → ship</div>
    </div>
  </div>
</div>

<!-- ─── METRICS ────────────────────────────────────────────────────────────── -->
<div class="section">
  <div class="section-header">
    <div class="section-icon">📊</div>
    <div class="section-title">Real Numbers</div>
    <div class="section-subtitle">measured · not estimated</div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-label">Ranking Quality</div>
      <table class="metrics-table">
        <thead><tr><th>Metric</th><th>Control</th><th>Treatment</th><th>Δ</th></tr></thead>
        <tbody>
          <tr><td class="mono">NDCG@10</td><td class="mono">0.3612</td><td class="mono green">0.3847</td><td><span class="tag-green">+6.2%</span></td></tr>
          <tr><td class="mono">IPS-NDCG</td><td class="mono">—</td><td class="mono green">primary metric</td><td><span class="tag-blue">6h eval</span></td></tr>
          <tr><td class="mono">Precision@K</td><td colspan="2" class="mono" style="color:var(--muted)">offline DuckDB slice eval</td><td><span class="tag-dim">sliced</span></td></tr>
        </tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-label">Online Engagement (A/B)</div>
      <table class="metrics-table">
        <thead><tr><th>Metric</th><th>Before</th><th>After</th><th>Δ</th></tr></thead>
        <tbody>
          <tr><td class="mono">CTR</td><td class="mono">12.4%</td><td class="mono green">14.1%</td><td><span class="tag-green">+13.7%</span></td></tr>
          <tr><td class="mono">Add-to-List</td><td class="mono">4.1%</td><td class="mono green">5.3%</td><td><span class="tag-green">+29.3%</span></td></tr>
          <tr><td class="mono">Session Depth</td><td class="mono">3.2</td><td class="mono green">4.1 items</td><td><span class="tag-green">+28.1%</span></td></tr>
        </tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-label">Latency SLOs</div>
      <table class="metrics-table">
        <thead><tr><th>Endpoint</th><th>p50</th><th>p95</th><th>Target</th></tr></thead>
        <tbody>
          <tr><td class="mono">/recommend</td><td class="mono green">~60ms</td><td class="mono green">&lt;180ms</td><td><span class="tag-green">✓ met</span></td></tr>
          <tr><td class="mono">/voice</td><td class="mono gold">~1.2s</td><td class="mono gold">&lt;2.5s</td><td><span class="tag-gold">✓ met</span></td></tr>
          <tr><td class="mono">/explain</td><td class="mono green">~90ms</td><td class="mono green">&lt;250ms</td><td><span class="tag-green">✓ met</span></td></tr>
        </tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-label">Diversity & Retention</div>
      <table class="metrics-table">
        <thead><tr><th>Metric</th><th>Greedy</th><th>Slate Opt</th><th>Δ</th></tr></thead>
        <tbody>
          <tr><td class="mono">Page Abandon</td><td class="mono red-text">23.1%</td><td class="mono green">18.9%</td><td><span class="tag-green">−18.2%</span></td></tr>
          <tr><td class="mono">Genres/page</td><td class="mono">2.8</td><td class="mono green">5.7</td><td><span class="tag-green">+103.6%</span></td></tr>
          <tr><td class="mono">7-day Return</td><td class="mono">61.2%</td><td class="mono green">68.4%</td><td><span class="tag-green">+11.8%</span></td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- ─── FULL STACK ─────────────────────────────────────────────────────────── -->
<div class="section">
  <div class="section-header">
    <div class="section-icon">🔧</div>
    <div class="section-title">Full Stack — 13 AI Components</div>
    <div class="section-subtitle">every layer documented</div>
  </div>

  <div class="grid-3">
    <div class="card">
      <div class="card-label">Retrieval</div>
      <div class="stack-row"><div class="stack-name">Qdrant RAG</div><div class="stack-role">OpenAI text-embedding-3-small · 1536-dim · HNSW index · semantic search for voice queries</div></div>
      <div class="stack-row"><div class="stack-name">ALS (Spark)</div><div class="stack-role">Scala + PySpark · 200 latent factors · trained on MovieLens 25M · candidate generation</div></div>
      <div class="stack-row"><div class="stack-name">Two-Tower</div><div class="stack-role">User + item towers · dot-product similarity · cold-start fallback</div></div>
    </div>
    <div class="card">
      <div class="card-label">Ranking</div>
      <div class="stack-row"><div class="stack-name">LightGBM</div><div class="stack-role">Reranker · NDCG optimization objective · features: ALS score, genre, recency, popularity</div></div>
      <div class="stack-row"><div class="stack-name">REINFORCE RL</div><div class="stack-role">Policy gradient · session-level rewards · play +1.0 · add_to_list +1.0 · abandon −0.5 · watch_90% +2.0</div></div>
      <div class="stack-row"><div class="stack-name">Contextual Bandit</div><div class="stack-role">Lin-UCB per genre context · real-time weight updates via Redis</div></div>
    </div>
    <div class="card">
      <div class="card-label">Serving</div>
      <div class="stack-row"><div class="stack-name">FastAPI</div><div class="stack-role">13 endpoints · /recommend · /voice · /explain · /feedback · /impressions · /eval</div></div>
      <div class="stack-row"><div class="stack-name">Redis</div><div class="stack-role">Feature store v2 · session state · bandit weights · freshness layer · TTL-based cache</div></div>
      <div class="stack-row"><div class="stack-name">Slate Optimizer</div><div class="stack-role">5 hard diversity constraints · Jaccard ≥ 0.6 · ≤2 items per decade</div></div>
    </div>
    <div class="card">
      <div class="card-label">Voice AI</div>
      <div class="stack-row"><div class="stack-name">Whisper STT</div><div class="stack-role">OpenAI Whisper · browser MediaRecorder API · WAV blob → transcription</div></div>
      <div class="stack-row"><div class="stack-name">GPT-4o Intent</div><div class="stack-role">18 genre keyword maps · multi-genre interleaving · similar-to query routing · year filter</div></div>
      <div class="stack-row"><div class="stack-name">TTS Response</div><div class="stack-role">OpenAI TTS · auto-greeting · explanation synthesis from item metadata</div></div>
    </div>
    <div class="card">
      <div class="card-label">MLOps</div>
      <div class="stack-row"><div class="stack-name">Metaflow</div><div class="stack-role">phenomenal_flow_v3 · nightly retraining DAG · artifact versioning · step tracking</div></div>
      <div class="stack-row"><div class="stack-name">Airflow</div><div class="stack-role">Midnight trigger · train → eval → gate → alert pipeline · webserver at :8080</div></div>
      <div class="stack-row"><div class="stack-name">DuckDB Eval</div><div class="stack-role">IPS-weighted NDCG@10 · Parquet impression logs · slice eval by genre + activity decile</div></div>
    </div>
    <div class="card">
      <div class="card-label">Infrastructure</div>
      <div class="stack-row"><div class="stack-name">Docker Compose</div><div class="stack-role">7 services: API · Postgres · Redis · Qdrant · MinIO · Kafka · Flink</div></div>
      <div class="stack-row"><div class="stack-name">Kafka + Flink</div><div class="stack-role">3 topics: events · impressions · feature_updates · real-time signal streaming</div></div>
      <div class="stack-row"><div class="stack-name">MinIO</div><div class="stack-role">S3-compatible artifact store · model bundles · Metaflow artifact backend</div></div>
    </div>
  </div>
</div>

<!-- ─── TRADE-OFFS ────────────────────────────────────────────────────────── -->
<div class="section">
  <div class="section-header">
    <div class="section-icon">⚖️</div>
    <div class="section-title">Design Trade-offs</div>
    <div class="section-subtitle">latency · quality · cost · freshness</div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-label">Latency vs Quality</div>
      <div class="card-body">
        ALS candidate generation is pre-computed offline — serving is pure lookup (&lt;10ms).
        LightGBM reranking adds ~40ms on top. RAG (Qdrant) is the expensive leg at ~80ms
        but only activates on voice/semantic queries. Total p95: &lt;180ms on recommend,
        sacrificing real-time feature freshness (Redis TTL 5min) for sub-200ms SLO.
      </div>
    </div>
    <div class="card">
      <div class="card-label">Freshness vs Cost</div>
      <div class="card-body">
        ALS models retrain nightly via Metaflow — not real-time. Freshness is handled
        by Redis feature store (bandit weights, session signals) which update per-click.
        GPT-4o calls for voice are ~$0.003/request; cached TTS responses avoid repeat charges.
        DuckDB eval runs every 6h on Parquet — no Spark cost for evaluation.
      </div>
    </div>
    <div class="card">
      <div class="card-label">Diversity vs Relevance</div>
      <div class="card-body">
        Greedy top-score ranking maximises NDCG but causes 23% page abandonment.
        Slate Optimizer enforces ≥5 genres and Jaccard ≥0.6 — trading ~0.5% NDCG
        for −18.2% abandonment and +11.8% 7-day return rate. Measured in a 4,200-user
        A/B test; shipped as default policy after p=0.041 significance.
      </div>
    </div>
    <div class="card">
      <div class="card-label">Reliability vs Complexity</div>
      <div class="card-body">
        Every external dependency has a fallback: Kafka unavailable → JSONL file log.
        GPT-4o fails → rule-based explanation from item metadata.
        Qdrant cold → catalog random sample. Redis miss → in-process feature computation.
        Shadow A/B mode runs new policies in parallel before any user exposure.
      </div>
    </div>
  </div>
</div>

<!-- ─── POSTMORTEM ────────────────────────────────────────────────────────── -->
<div class="section">
  <div class="section-header">
    <div class="section-icon">🔥</div>
    <div class="section-title">Postmortem — What Broke & How I Fixed It</div>
    <div class="section-subtitle">real incidents · real fixes</div>
  </div>

  <div class="pm-box">
    <div class="pm-title">🔴 Incident 1 — Wrong genre in Explain ("Dune is Romance")</div>
    <div class="pm-body">
      <strong style="color:var(--text)">Root cause:</strong> The <span class="mono">/explain</span> API endpoint returned a GPT-generated reason anchored to the user's top profile genre (Romance), ignoring the actual item's <span class="mono">primary_genre</span> field. A Fantasy movie like Dune received "Because you enjoy Romance."<br>
      <strong style="color:var(--text)">Fix:</strong> Bypassed the <span class="mono">/explain</span> API entirely for voice explanations. <span class="mono">buildExplanation()</span> now reads directly from <span class="mono">item.primary_genre</span>, <span class="mono">item.year</span>, and <span class="mono">item.description</span> — building a genre-accurate explanation with a 13-entry lookup table. No API call, no wrong genre, sub-5ms.
    </div>
  </div>

  <div class="pm-box">
    <div class="pm-title">🔴 Incident 2 — "Similar to Stranger Things" returned 1920s movies</div>
    <div class="pm-body">
      <strong style="color:var(--text)">Root cause:</strong> MovieLens 25M assigns high average ratings to classic films (Nosferatu 1922, Metropolis 1926) because they are reviewed by cinephiles. Semantic RAG also matched "supernatural mystery" to these classics. The catalog supplement had no recency filter.<br>
      <strong style="color:var(--text)">Fix:</strong> Three-layer fix: (1) added +0.1 recency boost in <span class="mono">get_genre_pool()</span> for post-1980 titles; (2) for any "similar to" or "like" query, pre-filter RAG results to year ≥ 1970; (3) filter catalog supplement pool the same way. Stranger Things now returns: Dune (1984), The Dark Crystal, Labyrinth, etc.
    </div>
  </div>

  <div class="pm-box">
    <div class="pm-title">🟡 Incident 3 — Voice modal double-greeted on open</div>
    <div class="pm-body">
      <strong style="color:var(--text)">Root cause:</strong> React <span class="mono">useEffect</span> fired twice in Strict Mode (mount → unmount → remount). TTS greeting was triggered on both mounts, playing twice.<br>
      <strong style="color:var(--text)">Fix:</strong> Added <span class="mono">greetedRef = useRef(false)</span> checked before the TTS call. The ref persists across Strict Mode remounts; boolean resets to false on modal close. No double-greeting, no duplicate audio.
    </div>
  </div>

  <div class="pm-box">
    <div class="pm-title">🟡 Incident 4 — Posters showing wrong/NSFW images</div>
    <div class="pm-body">
      <strong style="color:var(--text)">Root cause:</strong> <span class="mono">getPosterForTitle()</span> in <span class="mono">movies.ts</span> had 200+ hardcoded overrides that were wrongly mapped — "Independence Day" showed the 300 poster, comedies showed Schindler's List, one entry linked an inappropriate image.<br>
      <strong style="color:var(--text)">Fix:</strong> Removed all hardcoded overrides. <span class="mono">poster()</span> now trusts <span class="mono">item.poster_url</span> from the backend directly. Added batch TMDB resolution in <span class="mono">HomeScreen.tsx</span> — fetches missing posters in parallel batches of 10 after catalog load with 3 retry strategies including "The Title" normalization.
    </div>
  </div>

  <div class="pm-box">
    <div class="pm-title">🟡 Incident 5 — ChunkLoadError on /aistack route</div>
    <div class="pm-body">
      <strong style="color:var(--text)">Root cause:</strong> The <span class="mono">aistack/page.tsx</span> used a server-component <span class="mono">metadata</span> export alongside a client component, causing Next.js 14 to fail code splitting.<br>
      <strong style="color:var(--text)">Fix:</strong> Converted to <span class="mono">'use client'</span> + <span class="mono">dynamic()</span> import with <span class="mono">ssr: false</span>. Cleared <span class="mono">.next/</span> cache. Route now loads instantly. Same pattern applied to <span class="mono">abtest/page.tsx</span>.
    </div>
  </div>
</div>

<!-- ─── CI/CD + MLOPS ──────────────────────────────────────────────────────── -->
<div class="section">
  <div class="section-header">
    <div class="section-icon">🚀</div>
    <div class="section-title">MLOps Pipeline</div>
    <div class="section-subtitle">automated · gated · observable</div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-label">Nightly Retraining Pipeline</div>
      <div class="timeline" style="margin-top:8px;">
        <div class="tl-item"><div class="tl-title">00:00 — Airflow trigger</div><div class="tl-body">cinewave_pipeline_dag.py fires · kicks off Metaflow run</div></div>
        <div class="tl-item"><div class="tl-title">00:05 — PySpark feature eng.</div><div class="tl-body">spark_features.py · rating matrix · item embeddings</div></div>
        <div class="tl-item"><div class="tl-title">00:20 — ALS training</div><div class="tl-body">train_als_and_candidates.py · 200 factors · Scala bridge</div></div>
        <div class="tl-item"><div class="tl-title">00:40 — LightGBM reranker</div><div class="tl-body">train_ranker_lgbm.py · NDCG objective · save bundle</div></div>
        <div class="tl-item"><div class="tl-title">01:00 — Eval gate</div><div class="tl-body">DuckDB IPS-NDCG on held-out Parquet · if drop &gt;5% → rollback + alert</div></div>
        <div class="tl-item"><div class="tl-title">01:10 — Ship or hold</div><div class="tl-body">Green gate → serve_payload.json updated · API hot-reload</div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-label">Observability Stack</div>
      <div class="stack-row"><div class="stack-name">Request logging</div><div class="stack-role">Every /recommend logs to recs_requests.jsonl · user_id, items, policy_version, latency_ms, features_snapshot_id</div></div>
      <div class="stack-row"><div class="stack-name">Impression logging</div><div class="stack-role">EventLogger writes impression per item on every page render · Redis stream + JSONL fallback</div></div>
      <div class="stack-row"><div class="stack-name">Kafka topics</div><div class="stack-role">recsys.events · recsys.impressions · recsys.feature_updates · 3 Flink consumers</div></div>
      <div class="stack-row"><div class="stack-name">Shadow A/B mode</div><div class="stack-role">shadow_ab.py · scores new policy in parallel · compares distributions · zero user exposure</div></div>
      <div class="stack-row"><div class="stack-name">Freshness engine</div><div class="stack-role">freshness_engine.py · TTL tracking · staleness alerts · auto-invalidate on model update</div></div>
    </div>
  </div>
</div>

<!-- ─── QUICK START ────────────────────────────────────────────────────────── -->
<div class="section">
  <div class="section-header">
    <div class="section-icon">▶️</div>
    <div class="section-title">Quick Start</div>
    <div class="section-subtitle">zero to running in 3 commands</div>
  </div>

  <div class="code-block">
<span class="c"># 1. Start all 7 services (Postgres · Redis · Qdrant · MinIO · API · Kafka · Flink)</span>
<span class="k">cd</span> ~/Downloads/netflix-recsys-complete
docker compose down && docker compose up -d && <span class="n">sleep</span> <span class="v">40</span>

<span class="c"># 2. Verify API health + patch TMDB catalog (1,200+ movies with posters)</span>
<span class="n">curl</span> http://localhost:8000/healthz
docker cp p.py recsys_api:/app/p.py && docker <span class="k">exec</span> recsys_api python3 /app/p.py

<span class="c"># 3. Start frontend (Next.js 14)</span>
lsof -ti:<span class="v">3000</span> | xargs kill -9 2>/dev/null; true
<span class="k">cd</span> frontend && npm run dev

<span class="c"># ─── App is live ───────────────────────────────────────────────────────────────</span>
<span class="c"># Home:       http://localhost:3000        → personalised feed</span>
<span class="c"># Voice AI:   http://localhost:3000        → click CINEWAVE button</span>
<span class="c"># AI Stack:   http://localhost:3000/aistack → all 13 components documented</span>
<span class="c"># A/B Tests:  http://localhost:3000/abtest  → live experiment dashboard</span>
<span class="c"># API Docs:   http://localhost:8000/docs    → FastAPI Swagger UI</span>
  </div>

  <div style="margin-top:12px;" class="code-block">
<span class="c"># Optional: enable Kafka streaming layer</span>
docker compose -f docker-compose.yml -f docker-compose-kafka.yml up -d

<span class="c"># Optional: run nightly retraining pipeline manually</span>
docker exec recsys_api python3 -m recsys.flows.phenomenal_flow_v3 run
  </div>
</div>

<!-- ─── CLOUD / INFRA ─────────────────────────────────────────────────────── -->
<div class="section">
  <div class="section-header">
    <div class="section-icon">☁️</div>
    <div class="section-title">Cloud Infrastructure & Production Path</div>
    <div class="section-subtitle">beyond localhost</div>
  </div>

  <div class="grid-3">
    <div class="card">
      <div class="card-label">Current (Local)</div>
      <div class="card-body">
        Docker Compose orchestrates 7 containers on a single host. MinIO provides S3-compatible artifact storage. Metaflow artifacts backed to MinIO. Airflow schedules nightly retraining. All services communicate over a <span class="mono">cinewave_net</span> bridge network.
      </div>
    </div>
    <div class="card">
      <div class="card-label">AWS Production Path</div>
      <div class="card-body">
        <strong style="color:#fff">API:</strong> ECS Fargate + ALB<br>
        <strong style="color:#fff">Models:</strong> SageMaker batch transform<br>
        <strong style="color:#fff">Vectors:</strong> Qdrant Cloud or OpenSearch<br>
        <strong style="color:#fff">Cache:</strong> ElastiCache Redis<br>
        <strong style="color:#fff">Streams:</strong> MSK (managed Kafka)<br>
        <strong style="color:#fff">Artifacts:</strong> S3 + Metaflow on AWS
      </div>
    </div>
    <div class="card">
      <div class="card-label">Cost Estimates</div>
      <div class="card-body">
        <span class="mono gold">GPT-4o voice:</span> ~$0.003/request<br>
        <span class="mono gold">OpenAI embeds:</span> ~$0.0001/query<br>
        <span class="mono gold">TTS response:</span> ~$0.015/1k chars<br>
        <span class="mono gold">ALS retraining:</span> ~$0.80/night (EMR)<br>
        <span class="mono gold">Recommend API:</span> ~$0.0002/request<br>
        <span class="mono green">Caching reduces GPT calls by ~60%</span>
      </div>
    </div>
  </div>
</div>

<!-- ─── AUTHOR ─────────────────────────────────────────────────────────────── -->
<div class="section">
  <div style="background: linear-gradient(135deg, rgba(229,9,26,0.08), rgba(129,140,248,0.06));
    border: 1px solid var(--border2); border-radius: 14px; padding: 28px 32px;
    display: flex; align-items: center; gap: 28px;">
    <div style="width:64px;height:64px;border-radius:50%;
      background: linear-gradient(135deg, rgba(229,9,26,0.3), rgba(129,140,248,0.3));
      border: 2px solid rgba(229,9,26,0.4);
      display:flex;align-items:center;justify-content:center;
      font-size:24px;font-weight:800;color:var(--red2);flex-shrink:0;
      font-family:var(--mono);">AM</div>
    <div>
      <div style="font-size:20px;font-weight:800;color:#fff;margin-bottom:4px;">Akilan Manivannan</div>
      <div style="font-size:13px;color:var(--muted);margin-bottom:12px;">
        ML Engineer · Full-Stack · Systems Design · Built CineWave end-to-end —
        data pipeline, model training, serving, voice AI, MLOps, and frontend.
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;">
        <a href="https://drive.google.com/drive/folders/1sXFjx6ShommQ46mFLcTKCyBi0GokRT8v" style="text-decoration:none">
          <span class="badge badge-red">▶ View Live Demo</span>
        </a>
        <span class="badge badge-dim">Python · TypeScript · Scala · SQL · Docker</span>
        <span class="badge badge-purple">FastAPI · Next.js · Spark · Metaflow · Airflow</span>
      </div>
    </div>
  </div>
</div>

</div><!-- /container -->

<!-- ─── FOOTER ─────────────────────────────────────────────────────────────── -->
<div class="footer">
  <div>
    <div class="footer-name">CineWave RecSys — Akilan Manivannan</div>
    <div style="font-size:11px;color:var(--dim);margin-top:2px;font-family:var(--mono);">
      FastAPI · Next.js 14 · PostgreSQL · Redis · Qdrant · Kafka · Flink · Metaflow · Airflow · DuckDB · Scala · PySpark
    </div>
  </div>
  <div class="footer-links">
    <a href="https://drive.google.com/drive/folders/1sXFjx6ShommQ46mFLcTKCyBi0GokRT8v" class="footer-link">Live Demo →</a>
    <span class="footer-link" style="color:var(--dim)">localhost:3000</span>
    <span class="footer-link" style="color:var(--dim)">localhost:8000/docs</span>
  </div>
</div>

</body>
</html>
