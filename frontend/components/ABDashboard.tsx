'use client'

import { useState, useEffect, useCallback } from 'react'
import Navbar from './Navbar'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

// ── Types ─────────────────────────────────────────────────────────────────────

interface Metric {
  label: string
  value: number
  unit: string
  delta?: number   // vs control
  better?: boolean // higher is better
}

interface Variant {
  id: string
  name: string
  description: string
  color: string
  users: number
  pct: number
  metrics: Metric[]
  winner?: boolean
}

interface Experiment {
  id: string
  name: string
  status: 'running' | 'stopped' | 'completed'
  started: string
  hypothesis: string
  variants: Variant[]
  significance: number  // p-value
  confident: boolean
  recommendation: string
}

// ── Simulated live experiments ────────────────────────────────────────────────
// In production these come from /agent/experiment_summary or AB_STORE

function buildExperiments(): Experiment[] {
  const now = Date.now()
  const rng = (seed: number, min: number, max: number) => {
    const x = Math.sin(seed + now / 86400000) * 10000
    return min + ((x - Math.floor(x)) * (max - min))
  }

  return [
    {
      id: 'exp_rl_vs_als',
      name: 'RL Policy vs Pure ALS Ranking',
      status: 'running',
      started: '2026-03-28',
      hypothesis: 'REINFORCE policy reranking increases session depth by ≥5% vs baseline ALS-only ranking',
      significance: 0.032,
      confident: true,
      recommendation: '✅ Ship RL Policy — statistically significant improvement in all primary metrics',
      variants: [
        {
          id: 'control',
          name: 'Control: ALS Only',
          description: 'Pure ALS collaborative filtering, no RL reranking',
          color: '#6366f1',
          users: 1243,
          pct: 50,
          metrics: [
            { label: 'NDCG@10',        value: 0.3612, unit: '',   delta: 0,     better: true },
            { label: 'Click-Through',  value: 12.4,   unit: '%',  delta: 0,     better: true },
            { label: 'Session Depth',  value: 3.2,    unit: 'items', delta: 0,  better: true },
            { label: 'Add-to-List',    value: 4.1,    unit: '%',  delta: 0,     better: true },
            { label: 'Diversity',      value: 0.61,   unit: '',   delta: 0,     better: true },
          ],
        },
        {
          id: 'treatment',
          name: 'Treatment: RL Policy',
          description: 'REINFORCE policy reranks ALS slate using session feedback',
          color: '#e5091a',
          users: 1251,
          pct: 50,
          winner: true,
          metrics: [
            { label: 'NDCG@10',        value: 0.3847, unit: '',      delta: +6.2,  better: true },
            { label: 'Click-Through',  value: 14.1,   unit: '%',     delta: +13.7, better: true },
            { label: 'Session Depth',  value: 4.1,    unit: 'items', delta: +28.1, better: true },
            { label: 'Add-to-List',    value: 5.3,    unit: '%',     delta: +29.3, better: true },
            { label: 'Diversity',      value: 0.68,   unit: '',      delta: +11.5, better: true },
          ],
        },
      ],
    },
    {
      id: 'exp_llm_explain',
      name: 'GPT Explanations vs Rule-Based',
      status: 'running',
      started: '2026-03-29',
      hypothesis: 'GPT-4o personalised explanations increase add-to-list rate vs static rule-based text',
      significance: 0.018,
      confident: true,
      recommendation: '✅ Ship GPT Explanations — strong lift on engagement, cost justified by +29% add-to-list',
      variants: [
        {
          id: 'control',
          name: 'Control: Rule-Based',
          description: 'Static genre-based explanation templates',
          color: '#6366f1',
          users: 987,
          pct: 50,
          metrics: [
            { label: 'Add-to-List',    value: 3.8,  unit: '%',  delta: 0,    better: true },
            { label: 'Play Rate',      value: 11.2, unit: '%',  delta: 0,    better: true },
            { label: 'Dwell Time',     value: 42,   unit: 's',  delta: 0,    better: true },
            { label: 'Satisfaction',   value: 3.6,  unit: '/5', delta: 0,    better: true },
          ],
        },
        {
          id: 'treatment',
          name: 'Treatment: GPT-4o',
          description: 'Personalised one-sentence explanation using user history + item features',
          color: '#e5091a',
          users: 1014,
          pct: 50,
          winner: true,
          metrics: [
            { label: 'Add-to-List',    value: 4.9,  unit: '%',  delta: +28.9, better: true },
            { label: 'Play Rate',      value: 13.8, unit: '%',  delta: +23.2, better: true },
            { label: 'Dwell Time',     value: 67,   unit: 's',  delta: +59.5, better: true },
            { label: 'Satisfaction',   value: 4.2,  unit: '/5', delta: +16.7, better: true },
          ],
        },
      ],
    },
    {
      id: 'exp_slate_diversity',
      name: 'Slate Optimizer vs Greedy Ranking',
      status: 'completed',
      started: '2026-03-20',
      hypothesis: 'Diversity-enforced slate (≥5 genres) reduces abandonment vs greedy top-score ranking',
      significance: 0.041,
      confident: true,
      recommendation: '✅ Shipped — Slate Optimizer reduced abandonment by 18%, now default policy',
      variants: [
        {
          id: 'control',
          name: 'Control: Greedy',
          description: 'Top-N by score, no diversity constraints',
          color: '#6366f1',
          users: 2100,
          pct: 50,
          metrics: [
            { label: 'Page Abandonment', value: 23.1, unit: '%', delta: 0,     better: false },
            { label: 'Genres / Page',    value: 2.8,  unit: '',  delta: 0,     better: true  },
            { label: 'Return Rate',      value: 61.2, unit: '%', delta: 0,     better: true  },
          ],
        },
        {
          id: 'treatment',
          name: 'Treatment: Slate Opt',
          description: '5 hard diversity rules: ≥5 genres, ≤3 per genre, ≤2 per decade',
          color: '#22c55e',
          users: 2089,
          pct: 50,
          winner: true,
          metrics: [
            { label: 'Page Abandonment', value: 18.9, unit: '%', delta: -18.2, better: false },
            { label: 'Genres / Page',    value: 5.7,  unit: '',  delta: +103.6, better: true },
            { label: 'Return Rate',      value: 68.4, unit: '%', delta: +11.8,  better: true },
          ],
        },
      ],
    },
    {
      id: 'exp_voice_search',
      name: 'Voice Search vs Text-Only',
      status: 'running',
      started: '2026-03-30',
      hypothesis: 'Voice-enabled discovery increases session length and add-to-list vs text-only',
      significance: 0.067,
      confident: false,
      recommendation: '⏳ Not yet significant (p=0.067) — need 3 more days of data',
      variants: [
        {
          id: 'control',
          name: 'Control: Text Only',
          description: 'Standard text search bar, no voice',
          color: '#6366f1',
          users: 445,
          pct: 50,
          metrics: [
            { label: 'Discovery Rate',  value: 8.2,  unit: '%', delta: 0,    better: true },
            { label: 'Add-to-List',     value: 3.9,  unit: '%', delta: 0,    better: true },
            { label: 'Query Success',   value: 71.3, unit: '%', delta: 0,    better: true },
          ],
        },
        {
          id: 'treatment',
          name: 'Treatment: Voice + Text',
          description: 'Voice assistant with RAG + LLM + TTS response',
          color: '#e5091a',
          users: 438,
          pct: 50,
          metrics: [
            { label: 'Discovery Rate',  value: 11.4, unit: '%', delta: +39.0, better: true },
            { label: 'Add-to-List',     value: 5.1,  unit: '%', delta: +30.8, better: true },
            { label: 'Query Success',   value: 84.7, unit: '%', delta: +18.8, better: true },
          ],
        },
      ],
    },
  ]
}

// ── Subcomponents ─────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: string }) {
  const cfg = {
    running:   { bg: 'rgba(34,197,94,0.15)',  color: '#22c55e', dot: true,  label: 'Live' },
    stopped:   { bg: 'rgba(239,68,68,0.15)',  color: '#ef4444', dot: false, label: 'Stopped' },
    completed: { bg: 'rgba(99,102,241,0.15)', color: '#818cf8', dot: false, label: 'Completed' },
  }[status] || { bg: 'rgba(255,255,255,0.1)', color: '#fff', dot: false, label: status }

  return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold"
      style={{ background: cfg.bg, color: cfg.color }}>
      {cfg.dot && <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: cfg.color }} />}
      {cfg.label}
    </span>
  )
}

function PValue({ p, confident }: { p: number; confident: boolean }) {
  return (
    <div className="text-center">
      <div className="text-xs text-white/40 mb-0.5">p-value</div>
      <div className="text-lg font-black font-mono" style={{ color: confident ? '#22c55e' : '#facc15' }}>
        {p.toFixed(3)}
      </div>
      <div className="text-[9px] font-mono" style={{ color: confident ? '#22c55e' : '#facc15' }}>
        {confident ? '95% confident' : 'not significant'}
      </div>
    </div>
  )
}

function MetricRow({ m, isControl }: { m: any; isControl: boolean }) {
  const showDelta = !isControl && m.delta !== 0
  const positive  = m.better ? m.delta > 0 : m.delta < 0
  return (
    <div className="flex items-center justify-between py-2 border-b border-white/5">
      <span className="text-xs text-white/60 w-32">{m.label}</span>
      <span className="text-sm font-bold text-white font-mono">
        {m.value}{m.unit}
      </span>
      {showDelta && (
        <span className="text-xs font-bold font-mono px-2 py-0.5 rounded"
          style={{
            background: positive ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)',
            color:      positive ? '#22c55e' : '#ef4444',
          }}>
          {m.delta > 0 ? '+' : ''}{m.delta.toFixed(1)}%
        </span>
      )}
      {isControl && <span className="text-[10px] text-white/20 px-2">baseline</span>}
    </div>
  )
}

function VariantCard({ v, isControl }: { v: Variant; isControl: boolean }) {
  return (
    <div className="rounded-2xl border p-4 flex-1 min-w-0"
      style={{
        background:   isControl ? 'rgba(255,255,255,0.03)' : v.color + '08',
        borderColor:  v.winner  ? v.color + '60' : 'rgba(255,255,255,0.08)',
        boxShadow:    v.winner  ? `0 0 20px ${v.color}15` : 'none',
      }}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ background: v.color }} />
            <p className="text-sm font-bold text-white">{v.name}</p>
            {v.winner && (
              <span className="text-[9px] px-2 py-0.5 rounded-full font-bold"
                style={{ background: v.color + '25', color: v.color }}>
                WINNER
              </span>
            )}
          </div>
          <p className="text-[10px] text-white/40 mt-0.5 ml-5">{v.description}</p>
        </div>
        <div className="text-right flex-shrink-0 ml-4">
          <div className="text-lg font-black text-white">{v.users.toLocaleString()}</div>
          <div className="text-[10px] text-white/40">users ({v.pct}%)</div>
        </div>
      </div>
      <div className="mt-2">
        {v.metrics.map(m => (
          <MetricRow key={m.label} m={m} isControl={isControl} />
        ))}
      </div>
    </div>
  )
}

function ExperimentCard({ exp, expanded, onToggle }: {
  exp: Experiment; expanded: boolean; onToggle: () => void
}) {
  const control   = exp.variants[0]
  const treatment = exp.variants[1]

  return (
    <div className="rounded-2xl border border-white/8 overflow-hidden mb-4"
      style={{ background: 'rgba(255,255,255,0.02)' }}>
      {/* Header */}
      <button
        className="w-full text-left px-6 py-4 flex items-center justify-between hover:bg-white/3 transition-colors"
        onClick={onToggle}>
        <div className="flex items-center gap-3 min-w-0">
          <StatusBadge status={exp.status} />
          <div className="min-w-0">
            <p className="text-sm font-bold text-white truncate">{exp.name}</p>
            <p className="text-[10px] text-white/35 mt-0.5 truncate">{exp.hypothesis}</p>
          </div>
        </div>
        <div className="flex items-center gap-4 flex-shrink-0 ml-4">
          <div className="text-right hidden sm:block">
            <div className="text-xs text-white/40">Total users</div>
            <div className="text-sm font-bold text-white">
              {exp.variants.reduce((s, v) => s + v.users, 0).toLocaleString()}
            </div>
          </div>
          <PValue p={exp.significance} confident={exp.confident} />
          <span className="text-white/30 text-lg ml-2">{expanded ? '▲' : '▼'}</span>
        </div>
      </button>

      {expanded && (
        <div className="px-6 pb-6 border-t border-white/5">
          {/* Recommendation */}
          <div className="mt-4 mb-4 px-4 py-3 rounded-xl text-sm"
            style={{ background: exp.confident ? 'rgba(34,197,94,0.08)' : 'rgba(250,204,21,0.08)',
                     border: `1px solid ${exp.confident ? 'rgba(34,197,94,0.2)' : 'rgba(250,204,21,0.2)'}`,
                     color: exp.confident ? '#86efac' : '#fef08a' }}>
            {exp.recommendation}
          </div>

          {/* Metric bars comparison */}
          {control && treatment && (
            <div className="mb-4">
              <p className="text-[10px] font-mono text-white/30 uppercase tracking-widest mb-3">Key Metric Lifts</p>
              <div className="space-y-2">
                {treatment.metrics.map((m, i) => {
                  const ctrl = control.metrics[i]
                  if (!ctrl || m.delta === 0) return null
                  const positive = m.better ? m.delta > 0 : m.delta < 0
                  const barPct   = Math.min(Math.abs(m.delta) / 50 * 100, 100)
                  return (
                    <div key={m.label} className="flex items-center gap-3">
                      <span className="text-[10px] text-white/50 w-28 flex-shrink-0">{m.label}</span>
                      <div className="flex-1 h-5 rounded-full overflow-hidden bg-white/5 relative">
                        <div className="absolute inset-y-0 left-0 rounded-full transition-all duration-1000"
                          style={{ width: `${barPct}%`, background: positive ? '#22c55e' : '#ef4444', opacity: 0.7 }} />
                        <div className="absolute inset-0 flex items-center px-2">
                          <span className="text-[9px] font-mono font-bold"
                            style={{ color: positive ? '#86efac' : '#fca5a5' }}>
                            {m.delta > 0 ? '+' : ''}{m.delta.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                      <span className="text-[10px] text-white/30 w-20 text-right flex-shrink-0">
                        {ctrl.value}{ctrl.unit} → {m.value}{m.unit}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Variant cards */}
          <div className="flex gap-4 flex-wrap md:flex-nowrap">
            {exp.variants.map((v, i) => (
              <VariantCard key={v.id} v={v} isControl={i === 0} />
            ))}
          </div>

          <div className="mt-3 text-[10px] text-white/25 font-mono">
            Started {exp.started} · IPS-weighted NDCG evaluation · doubly-robust estimator · 95% CI
          </div>
        </div>
      )}
    </div>
  )
}

// ── Summary stats ─────────────────────────────────────────────────────────────

function SummaryBar({ experiments }: { experiments: Experiment[] }) {
  const running   = experiments.filter(e => e.status === 'running').length
  const confident = experiments.filter(e => e.confident).length
  const totalUsers = experiments.reduce((s, e) => s + e.variants.reduce((ss, v) => ss + v.users, 0), 0)

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
      {[
        { label: 'Active Experiments', value: running,    color: '#22c55e', icon: '🧪' },
        { label: 'Significant Results', value: confident, color: '#e5091a', icon: '✅' },
        { label: 'Users in Experiments', value: totalUsers.toLocaleString(), color: '#818cf8', icon: '👥' },
        { label: 'Evaluation Method', value: 'IPS-NDCG', color: '#facc15', icon: '📊' },
      ].map(s => (
        <div key={s.label} className="rounded-2xl border border-white/8 p-4"
          style={{ background: 'rgba(255,255,255,0.03)' }}>
          <div className="text-xl mb-1">{s.icon}</div>
          <div className="text-2xl font-black" style={{ color: s.color }}>{s.value}</div>
          <div className="text-[10px] text-white/40 mt-0.5">{s.label}</div>
        </div>
      ))}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function ABDashboard() {
  const [experiments] = useState<Experiment[]>(buildExperiments)
  const [expanded, setExpanded]   = useState<string>(experiments[0]?.id || '')
  const [liveTime, setLiveTime]   = useState(new Date())

  useEffect(() => {
    const t = setInterval(() => setLiveTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  const toggle = useCallback((id: string) => {
    setExpanded(prev => prev === id ? '' : id)
  }, [])

  return (
    <div className="min-h-screen bg-cine-bg text-white">
      <Navbar />
      <div className="pt-20 pb-16 max-w-screen-xl mx-auto px-4 md:px-8">

        {/* Header */}
        <div className="mb-8 flex items-start justify-between flex-wrap gap-4">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <span className="text-3xl">🧪</span>
              <h1 className="text-3xl font-black">
                <span style={{ color: '#e5091a' }}>A/B</span>
                <span className="text-white ml-2">Experiment Dashboard</span>
              </h1>
            </div>
            <p className="text-white/50 text-sm max-w-2xl">
              Live experiment tracking with IPS-weighted NDCG evaluation, doubly-robust estimation,
              and automatic statistical significance testing. Every model change is tested before ship.
            </p>
          </div>
          <div className="text-right">
            <div className="text-[10px] font-mono text-white/30">Live updates</div>
            <div className="text-sm font-mono text-white/60">
              {liveTime.toLocaleTimeString()}
            </div>
          </div>
        </div>

        <SummaryBar experiments={experiments} />

        {/* Experiments */}
        <div>
          <h2 className="text-sm font-bold text-white/50 uppercase tracking-widest mb-4">
            Experiments ({experiments.length})
          </h2>
          {experiments.map(exp => (
            <ExperimentCard
              key={exp.id}
              exp={exp}
              expanded={expanded === exp.id}
              onToggle={() => toggle(exp.id)}
            />
          ))}
        </div>

        {/* How it works */}
        <div className="mt-10 rounded-2xl border border-white/8 p-6"
          style={{ background: 'rgba(255,255,255,0.02)' }}>
          <h3 className="text-sm font-bold text-white mb-4">How CineWave A/B Testing Works</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-white/50 leading-relaxed">
            <div>
              <p className="text-white/80 font-semibold mb-1">1. Assignment</p>
              User ID is hashed into experiment buckets. Each user always gets the same variant
              (sticky assignment). Split is configurable per experiment (50/50 default).
            </div>
            <div>
              <p className="text-white/80 font-semibold mb-1">2. Measurement</p>
              Every recommendation is logged with policy_id + features_snapshot_id. Kafka streams
              clicks/adds to Postgres. DuckDB computes IPS-weighted NDCG offline every 6 hours.
            </div>
            <div>
              <p className="text-white/80 font-semibold mb-1">3. Decision</p>
              Doubly-robust estimator corrects for selection bias. We require p&lt;0.05 AND
              minimum detectable effect of 5% lift before shipping. Airflow auto-alerts on regressions.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
