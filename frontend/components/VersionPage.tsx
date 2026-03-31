'use client'

import { useState, useEffect } from 'react'
import { api } from '@/lib/api'

export default function VersionPage() {
  const [version, setVersion] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  useEffect(() => {
    api.version()
      .then(setVersion)
      .catch(() => {
        setError(true)
        setVersion({
          bundle_dir: '/models/rec-bundle-v2.3.1',
          manifest: {
            version: '2.3.1',
            als_model: 'als_rank64_reg0.01_iter20',
            ranker_model: 'lgbm_ranker_v3',
            n_users: 138493,
            n_items: 26744,
            als_factors: 64,
            als_regularization: 0.01,
            als_iterations: 20,
            ranker_estimators: 300,
            ranker_max_depth: 6,
            train_date: '2025-01-15',
            eval_ndcg_at_10: 0.3847,
            eval_map_at_10: 0.2913,
            eval_hr_at_10: 0.6241,
          }
        })
      })
      .finally(() => setLoading(false))
  }, [])

  const manifest = version?.manifest ?? {}

  const modelSections = [
    {
      title: 'ALS Collaborative Filter',
      icon: '🔮',
      color: '#c084fc',
      fields: [
        { key: 'als_model', label: 'Model ID' },
        { key: 'als_factors', label: 'Latent Factors' },
        { key: 'als_regularization', label: 'Regularization' },
        { key: 'als_iterations', label: 'Iterations' },
        { key: 'n_users', label: 'Users Trained' },
        { key: 'n_items', label: 'Items Indexed' },
      ]
    },
    {
      title: 'LightGBM Ranker',
      icon: '⚡',
      color: '#facc15',
      fields: [
        { key: 'ranker_model', label: 'Model ID' },
        { key: 'ranker_estimators', label: 'Estimators' },
        { key: 'ranker_max_depth', label: 'Max Depth' },
      ]
    },
    {
      title: 'Evaluation Metrics',
      icon: '📊',
      color: '#e50914',
      fields: [
        { key: 'eval_ndcg_at_10', label: 'NDCG@10' },
        { key: 'eval_map_at_10', label: 'MAP@10' },
        { key: 'eval_hr_at_10', label: 'HitRate@10' },
        { key: 'train_date', label: 'Train Date' },
      ]
    }
  ]

  if (loading) {
    return (
      <div className="max-w-screen-xl mx-auto px-4 md:px-6 py-8 space-y-6">
        <div className="h-8 skeleton rounded w-48" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="rounded-xl p-5 space-y-3 border border-cine-border">
              <div className="h-5 skeleton rounded w-32" />
              {[...Array(4)].map((_, j) => <div key={j} className="h-4 skeleton rounded" />)}
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-screen-xl mx-auto px-4 md:px-6 py-8 space-y-8 animate-fade-in">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-display tracking-wide"
          style={{ fontFamily: 'Bebas Neue, Georgia, serif' }}>
          Model <span className="text-gradient-gold">Version</span>
        </h2>
        <div className="flex items-center gap-3 mt-2">
          <span className="text-sm text-cine-text-dim">Bundle:</span>
          <code className="text-sm font-mono text-cine-accent bg-cine-accent/10 px-2 py-0.5 rounded border border-cine-accent/20">
            {version?.bundle_dir ?? 'unknown'}
          </code>
          {manifest.version && (
            <span className="text-xs font-mono text-cine-muted bg-cine-surface border border-cine-border px-2 py-0.5 rounded">
              v{manifest.version}
            </span>
          )}
        </div>
        {error && (
          <p className="mt-2 text-xs text-yellow-400 font-mono">⚠ Using mock data — connect /version endpoint</p>
        )}
      </div>

      {/* Model sections */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {modelSections.map(section => (
          <div
            key={section.title}
            className="cine-card rounded-2xl p-5 space-y-4"
            style={{ borderColor: section.color + '30' }}
          >
            <div className="flex items-center gap-2">
              <span className="text-xl">{section.icon}</span>
              <h3 className="text-sm font-semibold text-cine-text">{section.title}</h3>
            </div>
            <div className="space-y-2.5">
              {section.fields.map(({ key, label }) => (
                <div key={key} className="flex items-center justify-between gap-2">
                  <span className="text-xs text-cine-muted">{label}</span>
                  <span
                    className="text-xs font-mono font-medium text-right"
                    style={{ color: manifest[key] !== undefined ? section.color : '#4a5568' }}
                  >
                    {manifest[key] !== undefined
                      ? typeof manifest[key] === 'number' && String(manifest[key]).includes('.')
                        ? Number(manifest[key]).toFixed(4)
                        : String(manifest[key])
                      : '—'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Full manifest JSON */}
      <div className="cine-card rounded-2xl overflow-hidden">
        <div className="px-5 py-3 border-b border-cine-border flex items-center gap-2">
          <svg className="w-4 h-4 text-cine-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span className="text-xs font-mono text-cine-text-dim">Full Manifest (GET /version)</span>
        </div>
        <pre className="p-5 text-xs font-mono text-cine-text-dim overflow-x-auto">
          {JSON.stringify(manifest, null, 2)}
        </pre>
      </div>
    </div>
  )
}
