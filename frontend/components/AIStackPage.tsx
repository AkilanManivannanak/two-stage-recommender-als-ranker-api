'use client'

import { useState, useEffect } from 'react'
import Navbar from './Navbar'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

interface StackItem {
  id: string
  name: string
  category: string
  color: string
  icon: string
  what: string
  how: string
  why: string
  file: string
  status: 'live' | 'pipeline' | 'scheduled'
}

const STACK: StackItem[] = [
  {
    id: 'rag',
    name: 'RAG Semantic Search',
    category: 'Retrieval',
    color: '#c084fc',
    icon: '🔍',
    what: 'Retrieval-Augmented Generation — converts user queries into vector embeddings and finds semantically similar movies.',
    how: 'User query → OpenAI text-embedding-3-small → 1536-dim vector → cosine similarity search over 500+ indexed titles in Qdrant.',
    why: 'Keyword search misses "something dark and mind-bending" — semantic search understands meaning, not just words.',
    file: 'backend/src/recsys/serving/rag_engine.py',
    status: 'live',
  },
  {
    id: 'als',
    name: 'ALS Collaborative Filtering',
    category: 'Retrieval',
    color: '#4dabf7',
    icon: '🔀',
    what: 'Alternating Least Squares matrix factorisation — learns user and item embeddings from 800k+ ratings.',
    how: 'Decomposes the user-item rating matrix into two low-rank factor matrices (rank=64). Dot product of user and item factors = predicted rating.',
    why: 'Users who liked Movie A and B tend to like Movie C — even without explicit ratings for C.',
    file: 'backend/src/recsys/serving/scala/FeaturePipeline.scala',
    status: 'pipeline',
  },
  {
    id: 'lgbm',
    name: 'LightGBM Ranker',
    category: 'Ranking',
    color: '#46d369',
    icon: '🏆',
    what: 'Gradient-boosted tree ranker (LambdaMART) that re-ranks ALS candidates using rich features.',
    how: 'Input features: ALS score, genre match, item popularity, recency, user activity. Trained with NDCG as the optimisation objective.',
    why: 'ALS scores alone don\'t capture freshness, diversity, or genre fit. LightGBM combines all signals into one final ranking.',
    file: 'backend/scripts/train_ranker_lgbm.py',
    status: 'pipeline',
  },
  {
    id: 'rl',
    name: 'REINFORCE Policy + LinUCB Bandit',
    category: 'Reinforcement Learning',
    color: '#facc15',
    icon: '🧠',
    what: 'Two-layer RL system: LinUCB selects which genre arm to explore; REINFORCE learns a slate-ordering policy from user feedback.',
    how: 'LinUCB: Thompson sampling over genre arms with confidence bounds. REINFORCE: 8-feature state vector → softmax policy → Gumbel-max ordering → Monte Carlo returns → weight updates.',
    why: 'Static ranking optimises for average-case. RL adapts to each user\'s session and maximises long-term engagement, not just immediate clicks.',
    file: 'backend/src/recsys/serving/rl_policy.py + bandit_v2.py',
    status: 'live',
  },
  {
    id: 'pyspark',
    name: 'PySpark Feature Engineering',
    category: 'Feature Store',
    color: '#f97316',
    icon: '⚡',
    what: 'Distributed feature computation — replaces Python loops over 800k ratings with parallelised Spark DataFrame operations.',
    how: 'df.groupBy("user_id", "genre").agg(avg("rating"), count("*")) — runs on local[*] Spark in the container, mirrors EMR/Databricks pattern.',
    why: 'Python for-loops over 800k rows take ~40s. Spark columnar groupBy takes ~4s. 10× speedup on feature extraction for user genre profiles.',
    file: 'backend/src/recsys/serving/spark_features.py',
    status: 'pipeline',
  },
  {
    id: 'scala',
    name: 'Scala ALS Pipeline',
    category: 'Feature Store',
    color: '#e11d48',
    icon: '☕',
    what: 'Native Spark MLlib ALS in Scala — eliminates JVM↔Python serialisation overhead for each ALS iteration.',
    how: 'spark-submit FeaturePipeline.jar → trains ALS (rank=64, 20 iterations, alpha=40) → writes user/item factors + co-occurrence Parquet → Python scala_bridge.py reads for serving.',
    why: 'Python PySpark adds ~15ms per ALS iteration for JVM bridge. Native Scala runs the same model 2-4× faster on large rating matrices.',
    file: 'backend/scala/src/main/scala/com/cinewave/recsys/FeaturePipeline.scala',
    status: 'pipeline',
  },
  {
    id: 'metaflow',
    name: 'Metaflow ML Pipeline',
    category: 'Orchestration',
    color: '#818cf8',
    icon: '🌊',
    what: 'End-to-end ML pipeline: data ingestion → feature engineering → ALS training → LightGBM training → evaluation → artifact packaging.',
    how: 'phenomenal_flow_v3.py: 12 steps with automatic checkpointing, parameter versioning, and artifact storage. Each run is reproducible and diff-able.',
    why: 'Without Metaflow, retraining means re-running ad-hoc scripts with no tracking. Metaflow gives experiment history, rollback, and A/B model comparison.',
    file: 'backend/flows/phenomenal_flow_v3.py',
    status: 'pipeline',
  },
  {
    id: 'kafka',
    name: 'Kafka Event Streaming',
    category: 'Real-Time',
    color: '#10b981',
    icon: '📡',
    what: 'Real-time event bus — every click, like, and impression is streamed to Kafka topics for immediate feature updates.',
    how: 'FastAPI /feedback → KafkaEventProducer.send_event() → recsys.events topic → Flink consumer → Postgres + Redis updates. Fallback to JSONL if Kafka is down.',
    why: 'Without streaming, user feedback only affects recommendations after the next daily training run. Kafka makes feedback visible in seconds.',
    file: 'backend/src/recsys/serving/kafka_producer.py + metaflow_integration.py',
    status: 'live',
  },
  {
    id: 'vlm',
    name: 'VLM Poster Analysis',
    category: 'GenAI',
    color: '#f59e0b',
    icon: '👁',
    what: 'Vision-Language Model (GPT-4o Vision) analyses movie posters to extract visual signals: mood, colour palette, setting, tone.',
    how: 'Poster image → base64 → GPT-4o /v1/chat/completions with vision → structured JSON: {mood, setting, tone, visual_genre}. Used to surface visually similar movies.',
    why: 'Two sci-fi movies might have completely different visual feels — one cold/dystopian, one warm/adventurous. VLM captures what text metadata misses.',
    file: 'backend/src/recsys/serving/vlm_engine.py',
    status: 'live',
  },
  {
    id: 'llm',
    name: 'LLM Explanations (GPT-4o)',
    category: 'GenAI',
    color: '#6366f1',
    icon: '💬',
    what: 'GPT-4o generates personalised, one-sentence explanations for every recommendation — not generic descriptions.',
    how: 'Semantic Sidecar: title + genre + user top genres → GPT-4o Structured Outputs → {reason, top_feature, confidence}. Cached in Redis per user-item pair.',
    why: '"Because you like Crime dramas" is useful. "A gripping political thriller that matches your taste for morally complex stories" is compelling.',
    file: 'backend/src/recsys/serving/semantic_sidecar.py + smart_explain.py',
    status: 'live',
  },
  {
    id: 'slate',
    name: 'Slate Optimizer',
    category: 'Ranking',
    color: '#06b6d4',
    icon: '📐',
    what: 'Post-ranker diversity enforcement — ensures the final page slate follows 5 hard diversity rules.',
    how: 'Rules: ≥5 distinct genres on page, ≤3 items from same genre per row, ≥1 item from each top user genre, ≤2 items from same decade, Jaccard diversity ≥ 0.6.',
    why: 'Pure ranking by score produces a page of 10 crime dramas. Slate Optimizer ensures the page feels varied and serves different moods.',
    file: 'backend/src/recsys/serving/slate_optimizer_v2.py',
    status: 'live',
  },
  {
    id: 'duckdb',
    name: 'DuckDB Offline Evaluation',
    category: 'Evaluation',
    color: '#84cc16',
    icon: '📊',
    what: 'In-process analytical engine — runs offline evaluation (NDCG, precision, coverage) on logged impressions without a database server.',
    how: 'DuckDB reads Parquet impression logs → SQL GROUP BY policy_version → IPS-weighted NDCG@10 per slice (genre, user activity decile, year).',
    why: 'PostgreSQL is optimised for OLTP. Analytical queries over millions of impression events need columnar storage. DuckDB is 50-100× faster for this workload.',
    file: 'backend/infra/duckdb/run_offline_eval.py',
    status: 'scheduled',
  },
  {
    id: 'airflow',
    name: 'Airflow Scheduling',
    category: 'Orchestration',
    color: '#f43f5e',
    icon: '⏰',
    what: 'Workflow scheduler — triggers the daily Metaflow retraining pipeline and monitors SLAs.',
    how: 'DAG: midnight trigger → phenomenal_flow_v3 run → DuckDB eval → if NDCG drops >5% → Slack alert + hold deployment. Airflow webserver at :8080.',
    why: 'Without scheduling, model retraining is manual. Airflow ensures the model trains daily, eval runs automatically, and degradation is caught before users feel it.',
    file: 'backend/airflow/dags/cinewave_pipeline_dag.py',
    status: 'scheduled',
  },
]

const CATEGORIES = [...new Set(STACK.map(s => s.category))]

const STATUS_LABELS: Record<string, { label: string; color: string; dot: string }> = {
  live:      { label: 'Live in API',    color: 'rgba(34,197,94,0.15)',  dot: '#22c55e' },
  pipeline:  { label: 'In Pipeline',   color: 'rgba(250,204,21,0.15)', dot: '#facc15' },
  scheduled: { label: 'Scheduled',     color: 'rgba(99,102,241,0.15)', dot: '#818cf8' },
}

function FeatureCard({ item, onClick, selected }: { item: StackItem; onClick: () => void; selected: boolean }) {
  const s = STATUS_LABELS[item.status]
  return (
    <div
      onClick={onClick}
      className="cursor-pointer rounded-2xl p-4 transition-all duration-200 border"
      style={{
        background: selected ? item.color + '15' : 'rgba(255,255,255,0.03)',
        borderColor: selected ? item.color + '60' : 'rgba(255,255,255,0.08)',
        boxShadow: selected ? `0 0 20px ${item.color}20` : 'none',
      }}
    >
      <div className="flex items-start justify-between mb-2">
        <span className="text-2xl">{item.icon}</span>
        <span className="text-[9px] px-2 py-0.5 rounded-full flex items-center gap-1"
          style={{ background: s.color, color: s.dot }}>
          <span className="w-1.5 h-1.5 rounded-full inline-block" style={{ background: s.dot }} />
          {s.label}
        </span>
      </div>
      <p className="text-xs font-bold text-white mb-0.5">{item.name}</p>
      <p className="text-[10px] font-mono" style={{ color: item.color + 'cc' }}>{item.category}</p>
      <p className="text-[10px] text-white/40 mt-1.5 line-clamp-2">{item.what}</p>
    </div>
  )
}

function DetailPanel({ item }: { item: StackItem }) {
  const s = STATUS_LABELS[item.status]
  return (
    <div className="rounded-2xl border p-6 h-full"
      style={{ background: 'rgba(255,255,255,0.03)', borderColor: item.color + '30' }}>
      <div className="flex items-center gap-3 mb-4">
        <div className="w-12 h-12 rounded-xl flex items-center justify-center text-2xl"
          style={{ background: item.color + '20' }}>
          {item.icon}
        </div>
        <div>
          <h2 className="text-lg font-bold text-white">{item.name}</h2>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-xs font-mono" style={{ color: item.color }}>{item.category}</span>
            <span className="text-[9px] px-2 py-0.5 rounded-full flex items-center gap-1"
              style={{ background: s.color, color: s.dot }}>
              <span className="w-1.5 h-1.5 rounded-full inline-block" style={{ background: s.dot }} />
              {s.label}
            </span>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="p-3 rounded-xl" style={{ background: 'rgba(255,255,255,0.04)' }}>
          <p className="text-[10px] font-bold uppercase tracking-widest mb-1.5" style={{ color: item.color }}>What it does</p>
          <p className="text-sm text-white/75 leading-relaxed">{item.what}</p>
        </div>

        <div className="p-3 rounded-xl" style={{ background: 'rgba(255,255,255,0.04)' }}>
          <p className="text-[10px] font-bold uppercase tracking-widest mb-1.5" style={{ color: item.color }}>How it works</p>
          <p className="text-sm text-white/75 leading-relaxed">{item.how}</p>
        </div>

        <div className="p-3 rounded-xl" style={{ background: 'rgba(255,255,255,0.04)' }}>
          <p className="text-[10px] font-bold uppercase tracking-widest mb-1.5" style={{ color: item.color }}>Why it matters</p>
          <p className="text-sm text-white/75 leading-relaxed">{item.why}</p>
        </div>

        <div className="p-3 rounded-xl" style={{ background: 'rgba(255,255,255,0.04)' }}>
          <p className="text-[10px] font-bold uppercase tracking-widest mb-1" style={{ color: item.color }}>Source file</p>
          <code className="text-[11px] font-mono text-white/50 break-all">{item.file}</code>
        </div>
      </div>
    </div>
  )
}

export default function AIStackPage() {
  const [selected, setSelected] = useState<StackItem>(STACK[0])
  const [activeCategory, setActiveCategory] = useState<string | null>(null)
  const [health, setHealth] = useState<any>(null)

  useEffect(() => {
    fetch(`${API_BASE}/healthz`).then(r => r.json()).then(setHealth).catch(() => {})
  }, [])

  const filtered = activeCategory ? STACK.filter(s => s.category === activeCategory) : STACK

  const liveCounts = {
    live:      STACK.filter(s => s.status === 'live').length,
    pipeline:  STACK.filter(s => s.status === 'pipeline').length,
    scheduled: STACK.filter(s => s.status === 'scheduled').length,
  }

  return (
    <div className="min-h-screen bg-cine-bg text-white">
      <Navbar />
      <div className="pt-20 pb-16 max-w-screen-xl mx-auto px-4 md:px-8">

        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-3xl">🎬</span>
            <h1 className="text-3xl font-black tracking-tight">
              <span style={{ color: '#e5091a' }}>CINE</span>
              <span className="text-white">WAVE</span>
              <span className="text-white/40 ml-3 font-light text-2xl">AI Stack</span>
            </h1>
          </div>
          <p className="text-white/50 text-sm max-w-2xl">
            A production-grade, Netflix-inspired recommendation system built with 13 AI/ML components.
            Every feature is real, wired, and running — not a demo.
          </p>

          {/* Live status bar */}
          <div className="flex items-center gap-4 mt-4 flex-wrap">
            {[
              { k: 'live',      c: '#22c55e', label: `${liveCounts.live} Live in API` },
              { k: 'pipeline',  c: '#facc15', label: `${liveCounts.pipeline} In ML Pipeline` },
              { k: 'scheduled', c: '#818cf8', label: `${liveCounts.scheduled} Scheduled` },
            ].map(({ c, label }) => (
              <div key={label} className="flex items-center gap-1.5 text-xs text-white/50">
                <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: c }} />
                {label}
              </div>
            ))}
            {health && (
              <div className="flex items-center gap-1.5 text-xs ml-auto">
                <span className="w-2 h-2 rounded-full bg-green-400" />
                <span className="text-green-400/70">API healthy · v5 · {health.model_version || 'cinewave'}</span>
              </div>
            )}
          </div>
        </div>

        {/* Architecture flow */}
        <div className="mb-8 p-4 rounded-2xl border border-white/8" style={{ background: 'rgba(255,255,255,0.02)' }}>
          <p className="text-[10px] font-mono text-white/30 uppercase tracking-widest mb-3">Data Flow</p>
          <div className="flex items-center gap-2 overflow-x-auto pb-1 flex-wrap gap-y-2">
            {[
              { label: 'User Query', color: '#e5091a' },
              { label: '→' },
              { label: 'RAG / ALS', color: '#c084fc' },
              { label: '→' },
              { label: 'LightGBM', color: '#46d369' },
              { label: '→' },
              { label: 'RL Policy', color: '#facc15' },
              { label: '→' },
              { label: 'Slate Opt', color: '#06b6d4' },
              { label: '→' },
              { label: 'GPT-4o', color: '#6366f1' },
              { label: '→' },
              { label: 'Response', color: '#22c55e' },
            ].map((step, i) => (
              'label' in step && step.label === '→'
                ? <span key={i} className="text-white/20 text-sm">→</span>
                : <span key={i} className="text-[11px] font-mono px-2.5 py-1 rounded-full"
                    style={{ background: (step as any).color + '20', color: (step as any).color, border: `1px solid ${(step as any).color}40` }}>
                    {step.label}
                  </span>
            ))}
          </div>
        </div>

        {/* Category filter */}
        <div className="flex gap-2 mb-6 flex-wrap">
          <button
            onClick={() => setActiveCategory(null)}
            className="px-3 py-1.5 rounded-full text-xs font-mono transition-all"
            style={{
              background: !activeCategory ? 'rgba(229,9,20,0.2)' : 'rgba(255,255,255,0.05)',
              color: !activeCategory ? '#ff4d57' : 'rgba(255,255,255,0.4)',
              border: `1px solid ${!activeCategory ? 'rgba(229,9,20,0.4)' : 'rgba(255,255,255,0.1)'}`,
            }}>
            All ({STACK.length})
          </button>
          {CATEGORIES.map(cat => {
            const items = STACK.filter(s => s.category === cat)
            const active = activeCategory === cat
            return (
              <button key={cat} onClick={() => setActiveCategory(active ? null : cat)}
                className="px-3 py-1.5 rounded-full text-xs font-mono transition-all"
                style={{
                  background: active ? items[0].color + '20' : 'rgba(255,255,255,0.05)',
                  color: active ? items[0].color : 'rgba(255,255,255,0.4)',
                  border: `1px solid ${active ? items[0].color + '40' : 'rgba(255,255,255,0.1)'}`,
                }}>
                {cat} ({items.length})
              </button>
            )
          })}
        </div>

        {/* Main layout: grid + detail */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Cards grid */}
          <div className="lg:col-span-2 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2 gap-3 content-start">
            {filtered.map(item => (
              <FeatureCard
                key={item.id}
                item={item}
                selected={selected?.id === item.id}
                onClick={() => setSelected(item)}
              />
            ))}
          </div>

          {/* Detail panel */}
          <div className="lg:col-span-3 sticky top-24">
            {selected && <DetailPanel item={selected} />}
          </div>
        </div>

        {/* Tech summary table */}
        <div className="mt-12">
          <h2 className="text-lg font-bold text-white mb-4">Full Tech Stack</h2>
          <div className="overflow-x-auto rounded-2xl border border-white/8">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/8" style={{ background: 'rgba(255,255,255,0.03)' }}>
                  {['Layer', 'Technology', 'Purpose', 'Status'].map(h => (
                    <th key={h} className="text-left px-4 py-3 text-[10px] font-mono text-white/40 uppercase tracking-widest">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  ['Frontend',       'Next.js 14 + TypeScript',         'Streaming UI with SSR and app router', 'live'],
                  ['API',            'FastAPI + Python 3.11',           'Serving layer — all recommendation endpoints', 'live'],
                  ['Retrieval',      'ALS (rank=64) + RAG (Qdrant)',    'Candidate generation — ~500 items → top 30', 'live'],
                  ['Ranking',        'LightGBM LambdaMART',             'Re-ranks 30 candidates into final slate', 'live'],
                  ['RL',             'REINFORCE + LinUCB',              'Adapts ordering to session feedback', 'live'],
                  ['Feature Eng.',   'PySpark + Scala MLlib',           'Distributed feature computation', 'pipeline'],
                  ['Pipeline',       'Metaflow (12 steps)',             'Daily model retraining + versioning', 'scheduled'],
                  ['Streaming',      'Kafka + Flink',                   'Real-time event ingestion', 'live'],
                  ['Vector DB',      'Qdrant',                          'Semantic embedding search', 'live'],
                  ['Cache',          'Redis',                           'Feature store + session cache', 'live'],
                  ['Storage',        'PostgreSQL + MinIO',              'Ratings, artifacts, embeddings', 'live'],
                  ['Scheduler',      'Airflow 2.9',                     'Pipeline scheduling + SLA monitoring', 'scheduled'],
                  ['Analytics',      'DuckDB',                          'Offline NDCG evaluation on Parquet', 'scheduled'],
                  ['GenAI',          'GPT-4o + GPT-4o Vision',         'Explanations + poster analysis', 'live'],
                  ['Voice',          'Whisper + TTS nova',              'Voice search and spoken results', 'live'],
                ].map(([layer, tech, purpose, status], i) => {
                  const s = STATUS_LABELS[status as keyof typeof STATUS_LABELS]
                  return (
                    <tr key={i} className="border-b border-white/5 hover:bg-white/3 transition-colors">
                      <td className="px-4 py-3 font-mono text-xs text-white/50">{layer}</td>
                      <td className="px-4 py-3 text-xs text-white font-medium">{tech}</td>
                      <td className="px-4 py-3 text-xs text-white/50">{purpose}</td>
                      <td className="px-4 py-3">
                        <span className="text-[9px] px-2 py-0.5 rounded-full flex items-center gap-1 w-fit"
                          style={{ background: s.color, color: s.dot }}>
                          <span className="w-1.5 h-1.5 rounded-full" style={{ background: s.dot }} />
                          {s.label}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
