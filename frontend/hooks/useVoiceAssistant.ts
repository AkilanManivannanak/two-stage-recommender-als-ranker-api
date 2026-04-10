'use client'

import { useState, useRef, useCallback } from 'react'

export type VoiceState =
  | 'idle'
  | 'wake_listening'
  | 'listening'
  | 'processing'
  | 'speaking'
  | 'clarifying'
  | 'error'

export interface VoiceResult {
  transcript: string
  items: any[]
  intent?: {
    action?: string
    target_rank?: number
    reference_item_id?: number
    [key: string]: any
  }
  response_text?: string
  audio_url?: string
}

interface UseVoiceAssistantOptions {
  autoSpeak?: boolean
  onResult?: (result: VoiceResult) => void
  onItemsFound?: (items: any[]) => void
  onDispatch?: (action: string, payload?: any) => void
}

interface UseVoiceAssistantReturn {
  state: VoiceState
  transcript: string
  startListening: () => void
  stopListening: () => void
  cancel: () => void
}

export function useVoiceAssistant(
  options: UseVoiceAssistantOptions = {}
): UseVoiceAssistantReturn {
  const { autoSpeak = true, onResult, onItemsFound, onDispatch } = options
  const [state, setState] = useState<VoiceState>('idle')
  const [transcript, setTranscript] = useState('')
  const recognitionRef = useRef<any>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const cancel = useCallback(() => {
    recognitionRef.current?.stop()
    audioRef.current?.pause()
    audioRef.current = null
    setState('idle')
    setTranscript('')
  }, [])

  const startListening = useCallback(() => {
    if (state !== 'idle') return
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRecognition) { setState('error'); return }
    const recognition = new SpeechRecognition()
    recognition.lang = 'en-US'
    recognition.interimResults = true
    recognitionRef.current = recognition
    setState('listening')
    recognition.onresult = (event: any) => {
      const text = Array.from(event.results).map((r: any) => r[0].transcript).join('')
      setTranscript(text)
    }
    recognition.onend = async () => {
      if (state === 'idle') return
      setState('processing')
      try {
        const res = await fetch('/api/voice', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ transcript }),
        })
        if (!res.ok) throw new Error('Voice API error')
        const result: VoiceResult = await res.json()
        onResult?.(result)
        if (result.items?.length) onItemsFound?.(result.items)
        if (result.intent?.action) onDispatch?.(result.intent.action, result.intent.reference_item_id)
        if (autoSpeak && result.audio_url) {
          setState('speaking')
          const audio = new Audio(result.audio_url)
          audioRef.current = audio
          audio.onended = () => { setState('idle'); audioRef.current = null }
          audio.onerror = () => { setState('idle'); audioRef.current = null }
          audio.play().catch(() => setState('idle'))
        } else { setState('idle') }
      } catch { setState('error'); setTimeout(() => setState('idle'), 2000) }
      setTranscript('')
    }
    recognition.onerror = () => { setState('error'); setTimeout(() => setState('idle'), 2000) }
    recognition.start()
  }, [state, transcript, autoSpeak, onResult, onItemsFound, onDispatch])

  const stopListening = useCallback(() => { recognitionRef.current?.stop() }, [])

  return { state, transcript, startListening, stopListening, cancel }
}

export default useVoiceAssistant
