'use client'

import { useState, useEffect, useCallback } from 'react'
import { api } from '@/lib/api'
import { useAppState, useAppDispatch } from '@/lib/store'
import { placeholderPoster, parseGenres, genreColor, formatScore } from '@/lib/utils'
import type { ItemDetail, RecommendItem } from '@/lib/api'
import { DetailModalSkeleton } from './Skeletons'

interface TitleDetailModalProps {
  item_id: number | null
  recData?: RecommendItem
  onClose: () => void
}

export default function TitleDetailModal({ item_id, recData, onClose }: TitleDetailModalProps) {
  const { activeUser, sessionItemIds } = useAppState()
  const dispatch = useAppDispatch()
  const [detail, setDetail] = useState<ItemDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [explaining, setExplaining] = useState(false)
  const [explanation, setExplanation] = useState<string | null>(null)
  const [imgError, setImgError] = useState(false)
  const [feedbackState, setFeedbackState] = useState<'like' | 'dislike' | 'add_to_list' | null>(null)

  useEffect(() => {
    if (!item_id) return
    setDetail(null)
    setExplanation(null)
    setLoading(true)
    setImgError(false)

    api.item(item_id)
      .then(setDetail)
      .catch(() => {
        // Mock fallback
        setDetail({
          item_id,
          title: `Title #${item_id}`,
          genres: 'Drama|Thriller',
          description: 'An AI-powered recommendation. Connect your backend to see real title metadata.',
          poster_url: null,
        })
      })
      .finally(() => setLoading(false))
  }, [item_id])

  const handleExplain = useCallback(async () => {
    if (!activeUser || !item_id) return
    setExplaining(true)
    try {
      const resp = await api.explain({ user_id: activeUser.user_id, item_ids: [item_id] })
      const ex = resp.explanations.find(e => e.item_id === item_id)
      setExplanation(ex?.reason ?? 'No explanation available for this item.')
    } catch {
      setExplanation('Explanation service unavailable. Connect the /explain endpoint to enable this feature.')
    } finally {
      setExplaining(false)
    }
  }, [activeUser, item_id])

  const handleAddToSession = () => {
    if (item_id) {
      dispatch({ type: 'ADD_SESSION_ITEM', payload: item_id })
    }
  }

  const handleFeedback = async (event: 'like' | 'dislike' | 'add_to_list') => {
    if (!activeUser || !item_id) return
    setFeedbackState(event)
    try {
      await api.feedback({ user_id: activeUser.user_id, item_id, event })
    } catch { /* best-effort */ }
  }

  // Keyboard close
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  if (!item_id) return null

  const genres = parseGenres(detail?.genres ?? '')
  const posterSrc = (!detail?.poster_url || imgError)
    ? placeholderPoster(item_id, detail?.title ?? `#${item_id}`)
    : detail.poster_url
  const inSession = sessionItemIds.includes(item_id)

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
      onClick={onClose}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-cine-bg/80 backdrop-blur-md animate-fade-in" />

      {/* Modal */}
      <div
        className="relative cine-glass rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-overlay animate-slide-up"
        onClick={e => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-10 w-8 h-8 flex items-center justify-center rounded-full bg-cine-bg/60 border border-cine-border text-cine-text-dim hover:text-cine-text hover:border-cine-accent/40 transition-all"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {loading ? (
          <DetailModalSkeleton />
        ) : detail ? (
          <>
            {/* Header with poster */}
            <div className="relative">
              {/* Banner/backdrop */}
              <div className="h-48 overflow-hidden rounded-t-2xl">
                <img
                  src={posterSrc}
                  alt={detail.title}
                  className="w-full h-full object-cover blur-sm scale-110 opacity-50"
                  onError={() => setImgError(true)}
                />
                <div className="absolute inset-0 bg-gradient-to-b from-transparent to-cine-surface" />
              </div>

              {/* Poster + title overlay */}
              <div className="absolute bottom-0 left-0 right-0 px-6 pb-4 flex items-end gap-4">
                <div className="w-28 h-40 rounded-lg overflow-hidden border-2 border-cine-border shadow-card flex-shrink-0 -mb-12 relative z-10">
                  <img
                    src={posterSrc}
                    alt={detail.title}
                    className="w-full h-full object-cover"
                    onError={() => setImgError(true)}
                  />
                </div>
                <div className="pb-1 flex-1 min-w-0">
                  <h2 className="text-xl font-bold text-cine-text leading-tight line-clamp-2">
                    {detail.title}
                  </h2>
                  {recData && (
                    <p className="text-xs font-mono text-cine-accent mt-1">
                      Score: {formatScore(recData.score)}
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Body */}
            <div className="px-6 pt-16 pb-6 space-y-5">
              {/* Genres */}
              <div className="flex flex-wrap gap-2">
                {genres.map(g => (
                  <span
                    key={g}
                    className="text-xs px-2.5 py-1 rounded-full font-medium"
                    style={{ background: genreColor(g) + '20', color: genreColor(g), border: `1px solid ${genreColor(g)}30` }}
                  >
                    {g}
                  </span>
                ))}
                <span className="text-xs px-2.5 py-1 rounded-full font-mono text-cine-muted border border-cine-border">
                  ID: {detail.item_id}
                </span>
              </div>

              {/* Description */}
              {detail.description && (
                <p className="text-sm text-cine-text-dim leading-relaxed">
                  {detail.description}
                </p>
              )}

              {/* Score breakdown */}
              {recData && (
                <div className="bg-cine-bg/60 rounded-xl p-4 border border-cine-border space-y-3">
                  <h3 className="text-xs font-mono text-cine-text-dim uppercase tracking-widest">Score Breakdown</h3>
                  {[
                    { label: 'Final Score', value: recData.score, accent: '#e50914' },
                    { label: 'ALS Score', value: recData.als_score, accent: '#c084fc' },
                    { label: 'Ranker Score', value: recData.ranker_score, accent: '#facc15' },
                  ].map(({ label, value, accent }) => (
                    <div key={label} className="space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-cine-text-dim">{label}</span>
                        <span className="text-xs font-mono" style={{ color: accent }}>{(value ?? 0).toFixed(6)}</span>
                      </div>
                      <div className="h-1.5 bg-cine-border rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-700"
                          style={{
                            width: `${Math.min(100, Math.max(0, (Math.abs(value ?? 0)) * 100))}%`,
                            background: accent,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Explanation section */}
              <div className="border border-cine-border rounded-xl overflow-hidden">
                <div className="px-4 py-3 bg-cine-bg/40 border-b border-cine-border flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <svg className="w-4 h-4 text-cine-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                    <span className="text-xs font-mono text-cine-text-dim uppercase tracking-wider">Why Recommended</span>
                  </div>
                  {!explanation && (
                    <button
                      onClick={handleExplain}
                      disabled={explaining}
                      className="flex items-center gap-1.5 px-3 py-1 rounded-md text-xs bg-cine-accent/15 text-cine-accent border border-cine-accent/30 hover:bg-cine-accent/25 transition-colors disabled:opacity-50"
                    >
                      {explaining ? (
                        <>
                          <div className="w-3 h-3 border border-cine-accent border-t-transparent rounded-full animate-spin" />
                          Explaining...
                        </>
                      ) : 'Explain'}
                    </button>
                  )}
                </div>
                <div className="px-4 py-3">
                  {explanation ? (
                    <p className="text-sm text-cine-text leading-relaxed">{explanation}</p>
                  ) : (
                    <p className="text-xs text-cine-muted italic">
                      Click &quot;Explain&quot; to see why this was recommended for you using the /explain API.
                    </p>
                  )}
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-3 flex-wrap">
                <button className="flex items-center gap-2 px-4 py-2.5 bg-cine-accent text-cine-bg font-semibold text-sm rounded-lg hover:bg-cine-accent/90 transition-colors">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                  Play
                </button>

                <button
                  onClick={() => handleFeedback('add_to_list')}
                  className={`flex items-center gap-2 px-4 py-2.5 font-medium text-sm rounded-lg border transition-all ${
                    feedbackState === 'add_to_list'
                      ? 'bg-cine-gold/20 border-cine-gold text-cine-gold'
                      : 'border-cine-border text-cine-text-dim hover:border-cine-border/80 hover:text-cine-text'
                  }`}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                  </svg>
                  {feedbackState === 'add_to_list' ? 'Saved' : 'Watchlist'}
                </button>

                <button
                  onClick={() => handleFeedback('like')}
                  className={`p-2.5 rounded-lg border transition-all ${
                    feedbackState === 'like'
                      ? 'bg-green-500/20 border-green-500 text-green-400'
                      : 'border-cine-border text-cine-text-dim hover:text-green-400 hover:border-green-500/40'
                  }`}
                  title="Like"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M14 10V3a1 1 0 00-1.707-.707L6 8.586V20a2 2 0 002 2h7.172a2 2 0 001.978-1.714l.85-6A2 2 0 0016 12h-2z" />
                  </svg>
                </button>

                <button
                  onClick={() => handleFeedback('dislike')}
                  className={`p-2.5 rounded-lg border transition-all ${
                    feedbackState === 'dislike'
                      ? 'bg-red-500/20 border-red-500 text-red-400'
                      : 'border-cine-border text-cine-text-dim hover:text-red-400 hover:border-red-500/40'
                  }`}
                  title="Not for me"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M10 14v7a1 1 0 001.707.707L18 15.414V4a2 2 0 00-2-2H8.828a2 2 0 00-1.978 1.714l-.85 6A2 2 0 008 12h2z" />
                  </svg>
                </button>

                <button
                  onClick={handleAddToSession}
                  className={`p-2.5 rounded-lg border transition-all ml-auto ${
                    inSession
                      ? 'bg-cine-accent/20 border-cine-accent text-cine-accent'
                      : 'border-cine-border text-cine-text-dim hover:text-cine-accent hover:border-cine-accent/40'
                  }`}
                  title={inSession ? 'In session context' : 'Add to session context'}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    {inSession
                      ? <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      : <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                    }
                  </svg>
                </button>
              </div>
            </div>
          </>
        ) : null}
      </div>
    </div>
  )
}
