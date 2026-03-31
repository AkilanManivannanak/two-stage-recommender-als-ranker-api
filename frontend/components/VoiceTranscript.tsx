'use client'

import type { VoiceState } from '@/hooks/useVoiceAssistant'

interface VoiceTranscriptProps {
  transcript: string
  state: VoiceState
  intent?: any
  confidence?: number
  showIntent?: boolean
}

const STATE_MESSAGES: Partial<Record<VoiceState, string>> = {
  listening:      '● Recording…',
  processing:     'Thinking…',
  speaking:       'Speaking…',
  clarifying:     'Waiting for your answer…',
}

export default function VoiceTranscript({ transcript, state, intent, confidence, showIntent = false }: VoiceTranscriptProps) {
  const statusMsg = STATE_MESSAGES[state]
  if (!transcript && !statusMsg) return null

  const intentColor: Record<string, string> = {
    discover: '#46d369', refine: '#4dabf7', explain: '#c084fc',
    navigate: '#facc15', control: '#f97316', compare: '#38bdf8', unknown: '#6b7280',
  }
  const color = intentColor[intent?.intent as string] || '#6b7280'

  return (
    <div className="w-full space-y-2">
      {statusMsg && (
        <div className="flex items-center gap-2">
          {state === 'listening' && <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />}
          <span className="text-xs font-mono" style={{ color: state === 'listening' ? '#ff4d57' : 'rgba(255,255,255,0.4)' }}>
            {statusMsg}
          </span>
        </div>
      )}
      {transcript && (
        <div>
          <p className="text-sm text-white/80 leading-relaxed">&ldquo;{transcript}&rdquo;</p>
          {confidence !== undefined && (
            <div className="mt-1.5 flex items-center gap-2">
              <div className="flex-1 h-0.5 bg-white/10 rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${(confidence * 100).toFixed(0)}%`,
                    background: confidence > 0.7 ? '#46d369' : confidence > 0.4 ? '#facc15' : '#ef4444',
                  }} />
              </div>
              <span className="text-[9px] font-mono text-white/25 w-8 text-right">{(confidence * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>
      )}
      {showIntent && intent && !intent.needs_clarification && (
        <div className="flex flex-wrap gap-1.5">
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full border"
            style={{ color, borderColor: color + '40', background: color + '10' }}>
            {intent.intent}
          </span>
          {intent.filters?.genres?.map((g: string) => (
            <span key={g} className="text-[10px] font-mono px-2 py-0.5 rounded-full border border-white/10 text-white/40">{g}</span>
          ))}
          {intent.filters?.moods?.map((m: string) => (
            <span key={m} className="text-[10px] font-mono px-2 py-0.5 rounded-full border border-white/10 text-white/30">{m}</span>
          ))}
          {intent.filters?.max_runtime_minutes && (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full border border-white/10 text-white/30">
              ≤{intent.filters.max_runtime_minutes}min
            </span>
          )}
        </div>
      )}
    </div>
  )
}
