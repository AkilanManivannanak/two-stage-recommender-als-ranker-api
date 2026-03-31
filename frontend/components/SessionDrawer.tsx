'use client'

import { useEffect, useState } from 'react'
import { useAppState, useAppDispatch } from '@/lib/store'
import { api } from '@/lib/api'
import { placeholderPoster } from '@/lib/utils'
import type { ItemDetail } from '@/lib/api'

export default function SessionDrawer() {
  const { sessionDrawerOpen, sessionItemIds, activeUser } = useAppState()
  const dispatch = useAppDispatch()
  const [items, setItems] = useState<ItemDetail[]>([])

  // Fetch details for session items
  useEffect(() => {
    if (!sessionItemIds.length) { setItems([]); return }
    Promise.all(
      sessionItemIds.map(id =>
        api.item(id).catch(() => ({
          item_id: id,
          title: `Item #${id}`,
          genres: '',
          description: null,
          poster_url: null,
        }))
      )
    ).then(setItems)
  }, [sessionItemIds])

  const handleRefreshWithSession = async () => {
    if (!activeUser || !sessionItemIds.length) return
    dispatch({ type: 'SET_RECS_LOADING', payload: true })
    try {
      const resp = await api.recommend({
        user_id: activeUser.user_id,
        k: 20,
        session_item_ids: sessionItemIds,
      })
      dispatch({
        type: 'SET_RECS',
        payload: resp.items.map(item => ({
          ...item,
          title: `Item #${item.item_id}`,
          genres: '',
          poster_url: null,
        }))
      })
    } catch (e) {
      console.error(e)
    } finally {
      dispatch({ type: 'SET_RECS_LOADING', payload: false })
      dispatch({ type: 'TOGGLE_SESSION_DRAWER' })
    }
  }

  if (!sessionDrawerOpen) return null

  return (
    <div className="fixed inset-0 z-[90] flex justify-end" onClick={() => dispatch({ type: 'TOGGLE_SESSION_DRAWER' })}>
      <div className="absolute inset-0 bg-cine-bg/50 backdrop-blur-sm" />

      <div
        className="relative w-full max-w-sm h-full cine-glass border-l border-cine-border flex flex-col animate-slide-in-right"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-cine-border">
          <div>
            <h3 className="text-sm font-semibold text-cine-text">Session Context</h3>
            <p className="text-xs text-cine-muted mt-0.5">
              Items used as session signal for recommendations
            </p>
          </div>
          <button
            onClick={() => dispatch({ type: 'TOGGLE_SESSION_DRAWER' })}
            className="w-7 h-7 flex items-center justify-center rounded-md hover:bg-cine-surface text-cine-muted hover:text-cine-text transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-5 py-4">
          {sessionItemIds.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-48 text-center">
              <div className="text-4xl mb-3">📂</div>
              <p className="text-sm text-cine-text-dim">No items in session yet</p>
              <p className="text-xs text-cine-muted mt-1">
                Click + on any title card to add it to your session context
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {items.map(item => (
                <div
                  key={item.item_id}
                  className="flex items-center gap-3 p-3 rounded-lg bg-cine-bg/60 border border-cine-border"
                >
                  <div className="w-10 h-14 rounded overflow-hidden flex-shrink-0">
                    <img
                      src={item.poster_url ?? placeholderPoster(item.item_id, item.title)}
                      alt={item.title}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-cine-text truncate">{item.title}</p>
                    <p className="text-xs font-mono text-cine-muted">ID: {item.item_id}</p>
                  </div>
                  <button
                    onClick={() => {
                      // Remove from session (would need dispatch action)
                    }}
                    className="text-cine-muted hover:text-red-400 transition-colors p-1"
                    title="Remove"
                  >
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-cine-border space-y-2">
          <div className="flex items-center justify-between text-xs text-cine-muted mb-3">
            <span className="font-mono">session_item_ids</span>
            <span className="font-mono text-cine-accent">[{sessionItemIds.join(', ')}]</span>
          </div>

          <button
            onClick={handleRefreshWithSession}
            disabled={sessionItemIds.length === 0}
            className="w-full py-2.5 bg-cine-accent text-cine-bg font-semibold text-sm rounded-lg hover:bg-cine-accent/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Re-rank with Session Context
          </button>

          <button
            onClick={() => dispatch({ type: 'CLEAR_SESSION' })}
            disabled={sessionItemIds.length === 0}
            className="w-full py-2 text-cine-muted text-sm hover:text-red-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Clear Session
          </button>
        </div>
      </div>
    </div>
  )
}
