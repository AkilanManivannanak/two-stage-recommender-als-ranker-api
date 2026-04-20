'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { useAppState, useAppDispatch } from '@/lib/store'
import FilmReelLogo from './FilmReelLogo'
import dynamic from 'next/dynamic'

// Lazy load VoiceModal — only loads when opened
const VoiceModal = dynamic(() => import('./VoiceModal'), { ssr: false })

export default function Navbar() {
  const { activeUser, sessionItemIds, apiHealthy, shadowEnabled } = useAppState()
  const dispatch = useAppDispatch()
  const [scrolled, setScrolled] = useState(false)
  const [voiceOpen, setVoiceOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 10)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // Keyboard shortcut: backtick for dev overlay, V for voice
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === '`') dispatch({ type: 'TOGGLE_DEV_OVERLAY' })
      if (e.key === 'v' && !e.ctrlKey && !e.metaKey && (e.target as HTMLElement)?.tagName !== 'INPUT') {
        setVoiceOpen(v => !v)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [dispatch])

  return (
    <>
      <header className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? 'cine-glass shadow-overlay' : 'bg-gradient-to-b from-cine-bg/90 to-transparent'
      }`}>
        <div className="max-w-screen-xl mx-auto px-4 md:px-6 h-16 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-2.5">
            <FilmReelLogo size={30} />
            <span className="text-2xl font-display tracking-widest hidden sm:block"
              style={{ fontFamily: 'Bebas Neue, Georgia, serif' }}>
              <span className="text-gradient-accent">CINE</span>
              <span style={{ color: '#f5f5f1' }}>WAVE</span>
            </span>
          </div>

          {/* Right controls */}
          <div className="flex items-center gap-2 md:gap-3">

            {/* ── VOICE BUTTON (Siri-like) ── */}
            <button
              onClick={() => setVoiceOpen(true)}
              className="relative flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-mono border transition-all duration-200 group"
              style={{
                background: voiceOpen ? 'rgba(229,9,20,0.15)' : 'rgba(229,9,20,0.05)',
                border: `1px solid ${voiceOpen ? 'rgba(229,9,20,0.5)' : 'rgba(229,9,20,0.2)'}`,
                color: voiceOpen ? '#ff4d57' : 'rgba(255,255,255,0.5)',
              }}
              title="Voice assistant (V)"
            >
              <svg width={14} height={14} fill="none" stroke="currentColor" viewBox="0 0 24 24"
                strokeWidth={1.8} strokeLinecap="round">
                <rect x="9" y="2" width="6" height="11" rx="3" />
                <path d="M5 10a7 7 0 0014 0" />
                <line x1="12" y1="19" x2="12" y2="22" />
                <line x1="9" y1="22" x2="15" y2="22" />
              </svg>
              <span className="hidden sm:inline">CINEWAVE</span>
              <span className="text-[9px] opacity-40 hidden md:inline">V</span>
            </button>

            {/* A/B Dashboard link */}
            <Link href="/abtest"
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-mono border border-cine-border hover:border-green-400/40 text-cine-text-dim hover:text-green-300 transition-all duration-200"
              title="A/B Experiment Dashboard">
              <span>🧪</span>
              <span className="hidden sm:inline">A/B</span>
            </Link>

            {/* AI Stack page link */}
            <Link href="/aistack"
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-mono border border-cine-border hover:border-purple-400/40 text-cine-text-dim hover:text-purple-300 transition-all duration-200"
              title="AI Stack — all features explained">
              <span>🧠</span>
              <span className="hidden sm:inline">AI STACK</span>
            </Link>

            {/* ML Intelligence */}
            <Link href="/diffusion"
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-mono border border-cine-border hover:border-pink-400/40 text-cine-text-dim hover:text-pink-300 transition-all duration-200"
              title="Diffusion Model — DDPM + DALL-E 3">
              <span>🎨</span>
              <span className="hidden sm:inline">DIFFUSION</span>
            </Link>

            <Link href="/ml"
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-mono border border-cine-border hover:border-red-400/40 text-cine-text-dim hover:text-red-300 transition-all duration-200"
              title="ML Intelligence Dashboard">
              <span>⚡</span>
              <span className="hidden sm:inline">ML</span>
            </Link>

            {/* Author name */}
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-mono border border-cine-border/50 text-cine-text-dim select-none">
              <svg className="w-3 h-3 opacity-60" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z"/>
              </svg>
              <span>Akilan Manivannan</span>
            </div>

            {/* API status */}
            <div className="flex items-center gap-1.5 pl-1 border-l border-cine-border">
              <span className={`w-2 h-2 rounded-full ${
                apiHealthy === null ? 'bg-yellow-400 animate-pulse' :
                apiHealthy ? 'bg-green-400' : 'bg-red-400'
              }`} />
            </div>

            {/* Profile avatar */}
            {activeUser && (
              <button
                onClick={() => dispatch({ type: 'SET_ACTIVE_USER', payload: null })}
                className="w-8 h-8 rounded-full bg-gradient-to-br from-cine-accent/20 to-red-900/40 border border-cine-border flex items-center justify-center text-xs font-mono text-cine-accent hover:border-cine-accent/50 transition-colors"
                title="Switch profile"
              >
                {activeUser.user_id}
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Voice Modal */}
      {voiceOpen && (
        <VoiceModal
          open={voiceOpen}
          onClose={() => setVoiceOpen(false)}
        />
      )}
    </>
  )
}
