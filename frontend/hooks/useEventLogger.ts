// hooks/useEventLogger.ts
// Client-side event logging with full event schema
// Sends events to the backend /impressions/log and /feedback endpoints

import { useCallback, useRef } from 'react'
import { api, API_BASE } from '@/lib/api'

interface EventPayload {
  user_id: number
  item_id: number
  event_type: string
  session_id?: string
  row_id?: string
  position?: number
  page_position?: number
  surface?: string
  duration_s?: number
  features_snapshot_id?: string
  policy_id?: string
}

interface ImpressionItem {
  item_id: number
  row_id: string
  position: number
  page_position: number
}

export function useEventLogger(userId: number) {
  const sessionId = useRef(`sess_${userId}_${Math.floor(Date.now() / 1800000)}`)

  const logEvent = useCallback(async (payload: Omit<EventPayload, 'user_id'>) => {
    // Map event_type to the /feedback endpoint format
    const feedbackEvents = ['play', 'like', 'dislike', 'add_to_list', 'not_interested']
    const eventType = payload.event_type

    // Use feedback endpoint for interaction events
    if (feedbackEvents.includes(eventType)) {
      try {
        await api.feedback({
          user_id: userId,
          item_id: payload.item_id,
          event: eventType as any,
        })
      } catch (e) {
        // Non-blocking — log failure but don't surface to user
      }
    }

    // Always send full event schema to event log endpoint
    try {
      await fetch(`${API_BASE}/events/log`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          session_id: sessionId.current,
          ...payload,
          surface: payload.surface || 'home',
          policy_id: payload.policy_id || 'v4.0.0',
        }),
      })
    } catch (e) {
      // Non-blocking
    }
  }, [userId])

  const logImpressions = useCallback(async (items: ImpressionItem[]) => {
    if (!items.length) return
    try {
      const itemIds = items.map(i => i.item_id)
      await fetch(`${API_BASE}/impressions/log?user_id=${userId}&row_name=home&model_version=v4.0.0`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ item_ids: itemIds, user_id: userId, row_name: 'home', model_version: 'v4.0.0' }),
      })
    } catch (e) {
      // Non-blocking
    }
  }, [userId])

  const logPlay = useCallback((itemId: number, rowId: string, position: number) =>
    logEvent({ item_id: itemId, event_type: 'play', row_id: rowId, position }),
    [logEvent])

  const logLike = useCallback((itemId: number) =>
    logEvent({ item_id: itemId, event_type: 'like' }), [logEvent])

  const logDislike = useCallback((itemId: number) =>
    logEvent({ item_id: itemId, event_type: 'dislike' }), [logEvent])

  const logAddToList = useCallback((itemId: number) =>
    logEvent({ item_id: itemId, event_type: 'add_to_list' }), [logEvent])

  return { logEvent, logImpressions, logPlay, logLike, logDislike, logAddToList }
}
