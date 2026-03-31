'use client'

import { useEffect } from 'react'
import { useAppState, useAppDispatch } from '@/lib/store'
import { api } from '@/lib/api'
import ProfilePicker from '@/components/ProfilePicker'
import HomeScreen from '@/components/HomeScreen'
import DevOverlay from '@/components/DevOverlay'
import SessionDrawer from '@/components/SessionDrawer'

export default function Page() {
  const state    = useAppState()
  const dispatch = useAppDispatch()

  useEffect(() => {
    async function init() {
      try {
        const [health, users] = await Promise.all([
          api.health(),
          api.demoUsers(),
        ])
        dispatch({
          type: 'SET_API_HEALTH',
          payload: {
            healthy: health.ok,
            bundle: health.bundle,
            // v4: read whether GenAI is (incorrectly) in the request path
          },
        })
        dispatch({ type: 'SET_USERS', payload: users.users })
      } catch {
        dispatch({ type: 'SET_API_HEALTH', payload: { healthy: false, bundle: null } })
        dispatch({
          type: 'SET_USERS', payload: [
            { user_id: 1,   recent_titles: ['Stranger Things', 'Dark', 'Black Mirror'],           recent_item_ids: [1, 6, 12] },
            { user_id: 7,   recent_titles: ['Peaky Blinders', 'Narcos', 'The Irishman'],          recent_item_ids: [10, 3, 25] },
            { user_id: 42,  recent_titles: ['Ozark', 'Mindhunter', 'Squid Game'],                 recent_item_ids: [2, 11, 7] },
            { user_id: 99,  recent_titles: ['Bridgerton', 'Never Have I Ever', 'Sex Education'],  recent_item_ids: [15, 19, 14] },
            { user_id: 256, recent_titles: ['BoJack Horseman', 'Russian Doll'],                   recent_item_ids: [9, 20] },
          ],
        })
      }
    }
    init()
  }, [dispatch])

  return (
    <main className="relative min-h-screen overflow-hidden">
      {/* Ambient background glows */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute -top-40 -left-40 w-96 h-96 rounded-full bg-red-900/10 blur-3xl" />
        <div className="absolute top-1/3 -right-40 w-80 h-80 rounded-full bg-red-800/8 blur-3xl" />
        <div className="absolute -bottom-40 left-1/3 w-96 h-96 rounded-full bg-red-900/8 blur-3xl" />
      </div>

      <div className="relative z-10">
        {!state.activeUser ? <ProfilePicker /> : <HomeScreen />}
      </div>

      <DevOverlay />
      <SessionDrawer />
    </main>
  )
}
