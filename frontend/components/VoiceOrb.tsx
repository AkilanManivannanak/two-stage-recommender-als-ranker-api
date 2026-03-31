'use client'

import { useEffect, useRef } from 'react'
import type { VoiceState } from '@/hooks/useVoiceAssistant'

interface VoiceOrbProps {
  state: VoiceState
  transcript: string
  spokenText: string
  onTap: () => void
  onCancel: () => void
  micPermission: 'unknown' | 'granted' | 'denied'
}

const CFG: Record<VoiceState, { label: string; color: string; ring: string; pulse: boolean }> = {
  idle:           { label: 'Tap to speak',    color: 'rgba(229,9,20,0.2)',   ring: 'rgba(229,9,20,0.3)',  pulse: false },
  wake_listening: { label: 'Say "CineWave"',  color: 'rgba(229,9,20,0.1)',   ring: 'rgba(229,9,20,0.15)', pulse: true  },
  listening:      { label: 'Listening…',      color: 'rgba(229,9,20,0.95)',  ring: 'rgba(229,9,20,0.7)',  pulse: true  },
  processing:     { label: 'Thinking…',       color: 'rgba(168,85,247,0.7)', ring: 'rgba(168,85,247,0.5)',pulse: true  },
  speaking:       { label: 'Speaking…',       color: 'rgba(34,197,94,0.7)',  ring: 'rgba(34,197,94,0.5)', pulse: true  },
  clarifying:     { label: 'Confirm?',        color: 'rgba(249,115,22,0.7)', ring: 'rgba(249,115,22,0.5)',pulse: false },
  error:          { label: 'Try again',       color: 'rgba(239,68,68,0.4)',  ring: 'rgba(239,68,68,0.3)', pulse: false },
}

export default function VoiceOrb({ state, transcript, spokenText, onTap, onCancel, micPermission }: VoiceOrbProps) {
  const cfg = CFG[state] || CFG.idle
  const isListening = state === 'listening'
  const isActive = state !== 'idle' && state !== 'error'

  const handleClick = () => {
    if (state === 'idle' || state === 'error') onTap()
    else if (state === 'speaking' || state === 'processing') onCancel()
  }

  return (
    <div className="flex flex-col items-center gap-4">
      {/* Orb container */}
      <div className="relative flex items-center justify-center" style={{ width: 96, height: 96 }}>
        {/* Pulse rings */}
        {cfg.pulse && (
          <>
            <div className="absolute inset-0 rounded-full animate-ping" style={{ background: cfg.ring, opacity: 0.4, animationDuration: '1.2s' }} />
            <div className="absolute rounded-full animate-ping" style={{ inset: -8, background: cfg.ring, opacity: 0.2, animationDuration: '1.8s', animationDelay: '0.4s' }} />
          </>
        )}

        {/* Main orb button */}
        <button
          onClick={handleClick}
          className="relative z-10 flex items-center justify-center rounded-full transition-all duration-300 select-none"
          style={{
            width: 80, height: 80,
            background: cfg.color,
            border: `2px solid ${isListening ? 'rgba(229,9,20,1)' : 'rgba(255,255,255,0.15)'}`,
            boxShadow: isActive
              ? `0 0 0 4px ${cfg.ring}, 0 0 40px ${cfg.color}, 0 8px 30px rgba(0,0,0,0.5)`
              : '0 4px 20px rgba(0,0,0,0.4)',
            transform: isListening ? 'scale(1.1)' : 'scale(1)',
          }}
          aria-label={cfg.label}
        >
          {/* Icon */}
          {(state === 'idle' || state === 'wake_listening') && (
            <svg width={28} height={28} fill="none" stroke="white" viewBox="0 0 24 24" strokeWidth={1.8} strokeLinecap="round" opacity={state === 'wake_listening' ? 0.5 : 1}>
              <rect x="9" y="2" width="6" height="11" rx="3" />
              <path d="M5 10a7 7 0 0014 0" />
              <line x1="12" y1="19" x2="12" y2="22" />
              <line x1="9" y1="22" x2="15" y2="22" />
            </svg>
          )}
          {state === 'listening' && (
            <svg width={28} height={28} fill="none" stroke="white" viewBox="0 0 24 24" strokeWidth={2} strokeLinecap="round">
              <rect x="9" y="2" width="6" height="11" rx="3" fill="rgba(255,255,255,0.3)" />
              <path d="M5 10a7 7 0 0014 0" />
              <line x1="12" y1="19" x2="12" y2="22" />
              <line x1="9" y1="22" x2="15" y2="22" />
            </svg>
          )}
          {state === 'processing' && (
            <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          )}
          {state === 'speaking' && (
            <svg width={28} height={28} fill="none" stroke="white" viewBox="0 0 24 24" strokeWidth={1.8} strokeLinecap="round">
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" fill="rgba(255,255,255,0.3)" />
              <path d="M15.54 8.46a5 5 0 010 7.07M19.07 4.93a10 10 0 010 14.14" />
            </svg>
          )}
          {(state === 'clarifying' || state === 'error') && (
            <svg width={28} height={28} fill="none" stroke="white" viewBox="0 0 24 24" strokeWidth={2} strokeLinecap="round">
              <circle cx="12" cy="12" r="10" />
              <path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3M12 17h.01" />
            </svg>
          )}
        </button>
      </div>

      {/* State label */}
      <div className="text-center space-y-1">
        <p className="text-xs font-mono" style={{ color: isListening ? '#ff4d57' : state === 'speaking' ? '#22c55e' : 'rgba(255,255,255,0.45)' }}>
          {cfg.label}
        </p>
        {micPermission === 'denied' && state === 'idle' && (
          <p className="text-[10px] text-red-400/70 max-w-[140px] text-center leading-tight">
            Allow mic in browser settings
          </p>
        )}
      </div>

      {/* Live transcript */}
      {transcript && (
        <div className="max-w-[260px] px-3 py-2 rounded-xl text-center"
          style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)' }}>
          <p className="text-xs text-white/70 leading-relaxed">&ldquo;{transcript}&rdquo;</p>
        </div>
      )}

      {/* Spoken response */}
      {spokenText && (state === 'speaking' || (state === 'idle' && transcript)) && (
        <div className="max-w-[260px] px-3 py-2 rounded-xl text-center"
          style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)' }}>
          <p className="text-[10px] font-mono text-green-400/60 mb-0.5">CineWave says</p>
          <p className="text-xs text-white/70 leading-relaxed">{spokenText}</p>
        </div>
      )}
    </div>
  )
}
