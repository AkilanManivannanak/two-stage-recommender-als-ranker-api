'use client'

import { useState, useEffect, useCallback } from 'react'
import { api } from '@/lib/api'
import { formatMs } from '@/lib/utils'
import { MetricCardSkeleton } from './Skeletons'

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [window, setWindow] = useState(3600)
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.metrics(window)
      setMetrics(data)
      setLastRefresh(new Date())
    } catch {
      setError('Could not load metrics. Make sure the /metrics/latency endpoint is available.')
      // Mock data for demo
      setMetrics({
        n: 1247,
        p50_ms: 45,
        p90_ms: 120,
        p95_ms: 185,
        p99_ms: 340,
        max_ms: 892,
        req_per_min: 12.4,
      })
    } finally {
      setLoading(false)
    }
  }, [window])

  useEffect(() => { load() }, [load])

  // Auto-refresh every 30s
  useEffect(() => {
    const interval = setInterval(load, 30_000)
    return () => clearInterval(interval)
  }, [load])

  const WINDOW_OPTIONS = [
    { label: '1h', value: 3600 },
    { label: '6h', value: 21600 },
    { label: '24h', value: 86400 },
  ]

  const statCards = metrics ? [
    { label: 'Total Requests', value: metrics.n?.toLocaleString(), sub: `last ${window / 3600}h`, accent: '#e50914' },
    { label: 'Throughput', value: `${metrics.req_per_min?.toFixed(1)}/min`, sub: 'requests per minute', accent: '#46d369' },
    { label: 'p50 Latency', value: formatMs(metrics.p50_ms), sub: 'median', accent: '#c084fc' },
    { label: 'p99 Latency', value: formatMs(metrics.p99_ms), sub: '99th percentile', accent: '#f97316' },
    { label: 'Max Latency', value: formatMs(metrics.max_ms), sub: 'worst case', accent: '#ff4d57' },
    { label: 'p95 Latency', value: formatMs(metrics.p95_ms), sub: '95th percentile', accent: '#facc15' },
  ] : []

  const latencyBuckets = metrics ? [
    { label: 'p50', value: metrics.p50_ms, color: '#c084fc' },
    { label: 'p90', value: metrics.p90_ms, color: '#e50914' },
    { label: 'p95', value: metrics.p95_ms, color: '#f97316' },
    { label: 'p99', value: metrics.p99_ms, color: '#facc15' },
    { label: 'max', value: metrics.max_ms, color: '#ff4d57' },
  ] : []

  const maxLatency = metrics?.max_ms ?? 1

  return (
    <div className="max-w-screen-xl mx-auto px-4 md:px-6 py-8 space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h2 className="text-3xl font-display tracking-wide text-cine-text"
            style={{ fontFamily: 'Bebas Neue, Georgia, serif' }}>
            Latency <span className="text-gradient-accent">Metrics</span>
          </h2>
          <p className="text-sm text-cine-text-dim mt-1">
            Real-time recommendation API performance
            {lastRefresh && (
              <span className="ml-2 text-cine-muted font-mono text-xs">
                Updated {lastRefresh.toLocaleTimeString()}
              </span>
            )}
          </p>
        </div>

        <div className="flex items-center gap-2">
          {WINDOW_OPTIONS.map(opt => (
            <button
              key={opt.value}
              onClick={() => setWindow(opt.value)}
              className={`px-3 py-1.5 rounded-md text-xs font-mono transition-all ${
                window === opt.value
                  ? 'bg-cine-accent/20 text-cine-accent border border-cine-accent/30'
                  : 'text-cine-muted border border-cine-border hover:text-cine-text'
              }`}
            >
              {opt.label}
            </button>
          ))}

          <button
            onClick={load}
            disabled={loading}
            className="p-1.5 rounded-md border border-cine-border text-cine-muted hover:text-cine-accent hover:border-cine-accent/40 transition-all disabled:opacity-50"
          >
            <svg className={`w-4 h-4 ${loading ? 'animate-spin-slow' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
      </div>

      {error && (
        <div className="px-4 py-3 rounded-lg border border-yellow-500/30 bg-yellow-500/10 text-yellow-400 text-xs font-mono">
          ⚠ {error} — showing mock data
        </div>
      )}

      {/* Stat cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {loading && !metrics ? (
          [...Array(6)].map((_, i) => <MetricCardSkeleton key={i} />)
        ) : statCards.map((card) => (
          <div key={card.label} className="cine-card rounded-xl p-4 space-y-2">
            <p className="text-xs font-mono text-cine-muted uppercase tracking-wider">{card.label}</p>
            <p className="text-2xl font-display tracking-wide" style={{ color: card.accent, fontFamily: 'Bebas Neue, Georgia, serif' }}>
              {card.value}
            </p>
            <p className="text-xs text-cine-muted">{card.sub}</p>
          </div>
        ))}
      </div>

      {/* Latency distribution chart */}
      {metrics && (
        <div className="cine-card rounded-2xl p-6">
          <h3 className="text-sm font-mono text-cine-text-dim uppercase tracking-wider mb-6">
            Latency Distribution
          </h3>
          <div className="space-y-4">
            {latencyBuckets.map(({ label, value, color }) => (
              <div key={label} className="flex items-center gap-4">
                <span className="text-xs font-mono text-cine-muted w-8 flex-shrink-0">{label}</span>
                <div className="flex-1 h-5 bg-cine-bg border border-cine-border rounded-md overflow-hidden relative">
                  <div
                    className="h-full rounded-md transition-all duration-1000"
                    style={{
                      width: `${Math.max(2, (value / maxLatency) * 100)}%`,
                      background: `linear-gradient(90deg, ${color}80, ${color}40)`,
                      borderRight: `2px solid ${color}`,
                    }}
                  />
                  <span
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-xs font-mono"
                    style={{ color }}
                  >
                    {formatMs(value)}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* Summary */}
          <div className="mt-6 pt-4 border-t border-cine-border flex flex-wrap gap-6 text-xs font-mono">
            <div>
              <span className="text-cine-muted">SLA p99 ≤ 500ms: </span>
              <span className={metrics.p99_ms <= 500 ? 'text-green-400' : 'text-red-400'}>
                {metrics.p99_ms <= 500 ? '✓ Passing' : '✗ Failing'}
              </span>
            </div>
            <div>
              <span className="text-cine-muted">SLA p95 ≤ 200ms: </span>
              <span className={metrics.p95_ms <= 200 ? 'text-green-400' : 'text-red-400'}>
                {metrics.p95_ms <= 200 ? '✓ Passing' : '✗ Failing'}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
