'use client'

import { useState } from 'react'
import { useVoiceAssistant, type VoiceResult } from '@/hooks/useVoiceAssistant'
import VoiceButton from './VoiceButton'
import { useAppDispatch } from '@/lib/store'

interface VoicePanelProps {
  onItemsFound?: (items: any[]) => void
  onNavigate?: (rank: number) => void
  compact?: boolean
}

export default function VoicePanel({ onItemsFound, onNavigate, compact = false }: VoicePanelProps) {
  const dispatch = useAppDispatch()
  const [lastResult, setLastResult] = useState<VoiceResult | null>(null)

  const voice = useVoiceAssistant({
    autoSpeak: true,
    onResult: (result) => {
      setLastResult(result)
      const items = result.items || []
      if (items.length > 0) onItemsFound?.(items)
      if (result.intent?.target_rank) onNavigate?.(result.intent.target_rank)
      if (result.intent?.action === 'clear_session') dispatch({ type: 'CLEAR_SESSION' })
      if (result.intent?.action === 'add_to_list' && result.intent?.reference_item_id)
        dispatch({ type: 'ADD_SESSION_ITEM', payload: result.intent.reference_item_id })
    },
    onItemsFound,
    onDispatch: (action, payload) => {
      if (action === 'CLEAR_SESSION') dispatch({ type: 'CLEAR_SESSION' })
      if (action === 'ADD_SESSION_ITEM') dispatch({ type: 'ADD_SESSION_ITEM', payload })
    },
  })

  const stateColor: Record<string, string> = {
    idle:           'text-white/40',
    wake_listening: 'text-white/30',
    listening:      'text-red-400',
    processing:     'text-yellow-400',
    speaking:       'text-green-400',
    clarifying:     'text-orange-400',
    error:          'text-red-500',
  }

  if (compact) {
    return (
      <div className="flex items-center gap-3">
        <VoiceButton
          state={voice.state}
          onStart={voice.startListening}
          onStop={voice.stopListening}
          onCancel={voice.cancel}
          size="sm"
        />
        {voice.transcript && (
          <span className="text-xs font-mono text-white/60 truncate max-w-[160px]">
            {voice.transcript}
          </span>
        )}
      </div>
    )
  }

  return (
    <div className="w-full max-w-md mx-auto">
      <div className="flex flex-col items-center gap-6 py-6">
        <VoiceButton
          state={voice.state}
          onStart={voice.startListening}
          onStop={voice.stopListening}
          onCancel={voice.cancel}
          size="lg"
        />

        {voice.transcript && (
          <div className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10">
            <p className="text-xs font-mono text-white/40 mb-1">You said</p>
            <p className="text-sm text-white leading-relaxed">{voice.transcript}</p>
            {lastResult?.confidence !== undefined && (
              <div className="mt-2 flex items-center gap-2">
                <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full rounded-full transition-all"
                    style={{
                      width: `${(lastResult.confidence * 100).toFixed(0)}%`,
                      background: lastResult.confidence > 0.7 ? '#46d369' : lastResult.confidence > 0.4 ? '#facc15' : '#ef4444',
                    }} />
                </div>
                <span className="text-[10px] font-mono text-white/40">
                  {(lastResult.confidence * 100).toFixed(0)}%
                </span>
              </div>
            )}
          </div>
        )}

        {voice.pendingConfirm && (
          <div className="w-full px-4 py-3 rounded-xl bg-orange-500/10 border border-orange-500/30">
            <p className="text-xs font-mono text-orange-400 mb-1">Confirm</p>
            <p className="text-sm text-white mb-3">{voice.pendingConfirm.question}</p>
            <div className="flex gap-2">
              <button onClick={() => voice.confirm(true)}
                className="flex-1 py-2 rounded-lg bg-green-500/20 border border-green-500/40 text-green-400 text-xs font-semibold">
                Yes
              </button>
              <button onClick={() => voice.confirm(false)}
                className="flex-1 py-2 rounded-lg bg-red-500/20 border border-red-500/40 text-red-400 text-xs font-semibold">
                No
              </button>
            </div>
          </div>
        )}

        {lastResult?.spoken && !voice.pendingConfirm && (
          <div className="w-full px-4 py-3 rounded-xl bg-green-500/8 border border-green-500/20">
            <p className="text-xs font-mono text-green-400 mb-1">CineWave</p>
            <p className="text-sm text-white/80 leading-relaxed">{lastResult.spoken}</p>
          </div>
        )}

        {voice.error && (
          <div className="w-full px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/30">
            <p className="text-xs text-red-400">{voice.error}</p>
          </div>
        )}

        {lastResult?.items && lastResult.items.length > 0 && (
          <div className="w-full">
            <p className="text-xs font-mono text-white/30 mb-2 uppercase tracking-wider">
              {lastResult.items.length} results
            </p>
            <div className="space-y-1.5">
              {lastResult.items.slice(0, 4).map((item: any, i: number) => (
                <div key={item.item_id || i}
                  className="flex items-center gap-3 px-3 py-2 rounded-lg bg-white/5 border border-white/8">
                  <span className="text-xs font-mono text-white/30 w-4">{i + 1}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-white truncate">{item.title || `Item #${item.item_id}`}</p>
                    <p className="text-[10px] text-white/40">{item.primary_genre || item.genres || ''}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
