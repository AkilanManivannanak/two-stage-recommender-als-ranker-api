'use client'

import { useRef } from 'react'
import { type VoiceState } from '@/hooks/useVoiceAssistant'

interface VoiceButtonProps {
  state: VoiceState
  onStart: () => void
  onStop: () => void
  onCancel: () => void
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const STATE_COLORS: Record<VoiceState, string> = {
  idle:           'rgba(229,9,20,0.15)',
  wake_listening: 'rgba(229,9,20,0.08)',
  listening:      'rgba(229,9,20,0.9)',
  processing:     'rgba(192,132,252,0.7)',
  speaking:       'rgba(70,211,105,0.7)',
  clarifying:     'rgba(251,146,60,0.7)',
  error:          'rgba(239,68,68,0.4)',
}

const STATE_LABELS: Record<VoiceState, string> = {
  idle:           'Hold to speak',
  wake_listening: 'Say "CineWave"',
  listening:      'Listening…',
  processing:     'Processing…',
  speaking:       'Speaking…',
  clarifying:     'Needs clarification',
  error:          'Try again',
}

export default function VoiceButton({ state, onStart, onStop, onCancel, size = 'md', className = '' }: VoiceButtonProps) {
  const pressRef  = useRef(false)
  const isListening = state === 'listening'
  const isActive    = state !== 'idle' && state !== 'error'
  const sizes = { sm: { btn: 40, icon: 16 }, md: { btn: 56, icon: 22 }, lg: { btn: 72, icon: 28 } }
  const { btn, icon } = sizes[size]
  const color = STATE_COLORS[state]
  const label = STATE_LABELS[state]

  const handlePointerDown = (e: React.PointerEvent) => {
    e.preventDefault()
    if (state !== 'idle') return
    pressRef.current = true
    onStart()
  }
  const handlePointerUp = (e: React.PointerEvent) => {
    e.preventDefault()
    if (pressRef.current && isListening) { pressRef.current = false; onStop() }
  }

  return (
    <div className={`flex flex-col items-center gap-2 ${className}`}>
      <button
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        onClick={() => { if (isActive && !isListening) onCancel() }}
        className="relative flex items-center justify-center rounded-full transition-all duration-200 select-none touch-none"
        style={{
          width: btn, height: btn,
          background: color,
          border: `2px solid ${isListening ? 'rgba(229,9,20,1)' : 'rgba(255,255,255,0.12)'}`,
          boxShadow: isListening ? '0 0 0 8px rgba(229,9,20,0.2),0 0 30px rgba(229,9,20,0.4)' : '0 4px 16px rgba(0,0,0,0.4)',
          transform: isListening ? 'scale(1.1)' : 'scale(1)',
          cursor: 'pointer',
        }}
        title={label} aria-label={label}
      >
        {isListening && (
          <span className="absolute inset-0 rounded-full animate-ping" style={{ background: 'rgba(229,9,20,0.3)' }} />
        )}
        {state === 'processing' && (
          <span className="absolute inset-0 rounded-full border-2 border-transparent animate-spin" style={{ borderTopColor: color }} />
        )}
        <svg width={icon} height={icon} fill="none" stroke="white" viewBox="0 0 24 24" strokeWidth={1.8} strokeLinecap="round">
          {(state === 'idle' || state === 'listening' || state === 'wake_listening') && <>
            <rect x="9" y="2" width="6" height="11" rx="3" fill={isListening ? 'rgba(255,255,255,0.3)' : 'none'} />
            <path d="M5 10a7 7 0 0014 0" /><line x1="12" y1="19" x2="12" y2="22" /><line x1="9" y1="22" x2="15" y2="22" />
          </>}
          {state === 'processing' && <path d="M2 12h3M7 6v12M12 4v16M17 8v8M22 12h-3" />}
          {state === 'speaking' && <>
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" fill="rgba(255,255,255,0.3)" />
            <path d="M15.54 8.46a5 5 0 010 7.07" />
          </>}
          {(state === 'clarifying' || state === 'error') && <>
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </>}
        </svg>
      </button>
      <span className="text-[10px] font-mono text-center" style={{ color: isListening ? '#ff4d57' : 'rgba(255,255,255,0.4)', minWidth: 80 }}>
        {label}
      </span>
    </div>
  )
}
