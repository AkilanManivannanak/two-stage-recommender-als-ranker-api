'use client'

import { useState, useEffect } from 'react'
import { useAppState, useAppDispatch } from '@/lib/store'
import { api, API_BASE } from '@/lib/api'
import { formatMs } from '@/lib/utils'

export default function DevOverlay() {
  const { devOverlayOpen, activeUser, shadowEnabled, sessionItemIds, apiHealthy, apiBundle } = useAppState()
  const dispatch = useAppDispatch()
  const [activeTab, setActiveTab] = useState<'config' | 'request' | 'metrics' | 'version'>('config')
  const [metrics, setMetrics] = useState<any>(null)
  const [version, setVersion] = useState<any>(null)
  const [lastRequest, setLastRequest] = useState<any>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!devOverlayOpen) return
    api.metrics().then(setMetrics).catch(() => setMetrics({ error: 'unavailable' }))
    api.version().then(setVersion).catch(() => setVersion({ error: 'unavailable' }))
  }, [devOverlayOpen])

  const getLastRequestBody = () => ({
    user_id: activeUser?.user_id ?? null,
    k: 20,
    session_item_ids: sessionItemIds.length ? sessionItemIds : null,
    shadow: shadowEnabled,
  })

  const copyToClipboard = async (text: string) => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (!devOverlayOpen) return null

  const tabs = ['config', 'request', 'metrics', 'version'] as const

  return (
    <div className="fixed inset-0 z-[200] flex items-end justify-end p-4 pointer-events-none">
      <div
        className="pointer-events-auto w-full max-w-md cine-glass rounded-2xl border border-cine-border shadow-overlay animate-slide-up overflow-hidden"
        style={{ maxHeight: '80vh' }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-cine-border bg-cine-bg/40">
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 text-cine-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
            <span className="text-xs font-mono font-semibold text-cine-text uppercase tracking-wider">Dev Overlay</span>
            <span className="text-xs font-mono text-cine-muted">· Press ` to toggle</span>
          </div>
          <button
            onClick={() => dispatch({ type: 'TOGGLE_DEV_OVERLAY' })}
            className="text-cine-muted hover:text-cine-text transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 px-3 py-2 border-b border-cine-border">
          {tabs.map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-2.5 py-1 rounded text-xs font-mono capitalize transition-all ${
                activeTab === tab
                  ? 'bg-cine-accent/20 text-cine-accent'
                  : 'text-cine-muted hover:text-cine-text'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="overflow-y-auto p-4" style={{ maxHeight: 'calc(80vh - 110px)' }}>
          {activeTab === 'config' && (
            <div className="space-y-4">
              <ConfigRow label="API Base" value={API_BASE} accent />
              <ConfigRow label="API Status" value={apiHealthy === null ? 'checking...' : apiHealthy ? '✓ Online' : '✗ Offline'} ok={apiHealthy} />
              <ConfigRow label="Bundle" value={apiBundle ?? 'unknown'} />
              <ConfigRow label="Active User" value={activeUser ? `user_id=${activeUser.user_id}` : 'none'} />
              <ConfigRow label="Session Items" value={sessionItemIds.length ? sessionItemIds.join(', ') : 'none'} />

              <div className="border-t border-cine-border pt-4 space-y-3">
                <p className="text-xs font-mono text-cine-muted uppercase tracking-wider">Toggles</p>

                <ToggleRow
                  label="Shadow Compare"
                  description="Routes request to shadow model for A/B comparison"
                  value={shadowEnabled}
                  onChange={() => dispatch({ type: 'TOGGLE_SHADOW' })}
                  accent="gold"
                />
              </div>
            </div>
          )}

          {activeTab === 'request' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-xs font-mono text-cine-muted">POST /recommend</p>
                <button
                  onClick={() => copyToClipboard(JSON.stringify(getLastRequestBody(), null, 2))}
                  className="text-xs font-mono text-cine-accent hover:underline"
                >
                  {copied ? '✓ copied' : 'copy'}
                </button>
              </div>
              <pre className="text-xs font-mono text-cine-text bg-cine-bg/80 border border-cine-border rounded-lg p-3 overflow-x-auto">
                {JSON.stringify(getLastRequestBody(), null, 2)}
              </pre>

              <div className="mt-4">
                <p className="text-xs font-mono text-cine-muted mb-2">POST /explain</p>
                <pre className="text-xs font-mono text-cine-text bg-cine-bg/80 border border-cine-border rounded-lg p-3 overflow-x-auto">
                  {JSON.stringify({
                    user_id: activeUser?.user_id ?? null,
                    item_ids: ['<selected_item_ids>'],
                  }, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {activeTab === 'metrics' && (
            <div className="space-y-3">
              {!metrics ? (
                <div className="space-y-2">
                  {[...Array(6)].map((_, i) => <div key={i} className="h-6 skeleton rounded" />)}
                </div>
              ) : metrics.error ? (
                <p className="text-xs text-cine-muted font-mono">{metrics.error}</p>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { label: 'Requests', value: metrics.n },
                      { label: 'Req/min', value: metrics.req_per_min?.toFixed(1) },
                      { label: 'p50', value: formatMs(metrics.p50_ms) },
                      { label: 'p90', value: formatMs(metrics.p90_ms) },
                      { label: 'p95', value: formatMs(metrics.p95_ms) },
                      { label: 'p99', value: formatMs(metrics.p99_ms) },
                      { label: 'max', value: formatMs(metrics.max_ms) },
                    ].map(({ label, value }) => (
                      <div key={label} className="bg-cine-bg/60 border border-cine-border rounded-lg p-2.5">
                        <p className="text-xs text-cine-muted font-mono">{label}</p>
                        <p className="text-sm font-mono text-cine-accent font-medium">{value}</p>
                      </div>
                    ))}
                  </div>

                  {/* Latency bar chart mini */}
                  <div className="mt-4 space-y-2">
                    {[
                      { label: 'p50', value: metrics.p50_ms, max: metrics.max_ms },
                      { label: 'p90', value: metrics.p90_ms, max: metrics.max_ms },
                      { label: 'p95', value: metrics.p95_ms, max: metrics.max_ms },
                      { label: 'p99', value: metrics.p99_ms, max: metrics.max_ms },
                    ].map(({ label, value, max }) => (
                      <div key={label} className="flex items-center gap-2">
                        <span className="text-xs font-mono text-cine-muted w-6">{label}</span>
                        <div className="flex-1 h-1.5 bg-cine-border rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-cine-accent to-cine-accent/60"
                            style={{ width: `${Math.min(100, (value / max) * 100)}%` }}
                          />
                        </div>
                        <span className="text-xs font-mono text-cine-text w-14 text-right">{formatMs(value)}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}

          {activeTab === 'version' && (
            <div className="space-y-3">
              {!version ? (
                <div className="space-y-2">
                  {[...Array(4)].map((_, i) => <div key={i} className="h-6 skeleton rounded" />)}
                </div>
              ) : version.error ? (
                <p className="text-xs text-cine-muted font-mono">{version.error}</p>
              ) : (
                <>
                  <ConfigRow label="Bundle Dir" value={version.bundle_dir ?? 'unknown'} />
                  <div>
                    <p className="text-xs font-mono text-cine-muted mb-2">Manifest</p>
                    <pre className="text-xs font-mono text-cine-text bg-cine-bg/80 border border-cine-border rounded-lg p-3 overflow-x-auto">
                      {JSON.stringify(version.manifest ?? {}, null, 2)}
                    </pre>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function ConfigRow({ label, value, accent, ok }: { label: string; value: string; accent?: boolean; ok?: boolean | null }) {
  return (
    <div className="flex items-start justify-between gap-4">
      <span className="text-xs font-mono text-cine-muted flex-shrink-0">{label}</span>
      <span className={`text-xs font-mono text-right break-all ${
        ok === true ? 'text-green-400' :
        ok === false ? 'text-red-400' :
        accent ? 'text-cine-accent' : 'text-cine-text'
      }`}>{value}</span>
    </div>
  )
}

function ToggleRow({ label, description, value, onChange, accent = 'accent' }: {
  label: string; description: string; value: boolean; onChange: () => void; accent?: string
}) {
  return (
    <div className="flex items-start justify-between gap-3">
      <div>
        <p className="text-xs font-mono text-cine-text">{label}</p>
        <p className="text-xs text-cine-muted mt-0.5">{description}</p>
      </div>
      <button
        onClick={onChange}
        className={`relative w-10 h-5 rounded-full flex-shrink-0 mt-0.5 transition-colors duration-200 ${
          value
            ? accent === 'gold' ? 'bg-cine-gold' : 'bg-cine-accent'
            : 'bg-cine-border'
        }`}
      >
        <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200 ${
          value ? 'translate-x-5' : 'translate-x-0.5'
        }`} />
      </button>
    </div>
  )
}
