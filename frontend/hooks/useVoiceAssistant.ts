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
  confidence?: number
  spoken?: string
  intent?: {
    action?: string
    target_rank?: number
    reference_item_id?: number
    [key: string]: any
  }
  response_text?: string
  audio_url?: string
}

interface PendingConfirm {
  question: string
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
  pendingConfirm: PendingConfirm | null
  error: string | null
  startListening: () => void
  stopListening: () => void
  cancel: () => void
  confirm: (yes: boolean) => void
}

export function useVoiceAssistant(
  options: UseVoiceAssistantOptions = {}
): UseVoiceAssistantReturn {
  const { autoSpeak = true, onResult, onItemsFound, onDispatch } = options

  const [state, setState] = useState<VoiceState>('idle')
  const [transcript, setTranscript] = useState('')
  const [pendingConfirm, setPendingConfirm] = useState<PendingConfirm | null>(null)
  const [error, setError] = useState<string | null>(null)
  const recognitionRef = useRef<any>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const transcriptRef = useRef('')

  const cancel = useCallback(() => {
    recognitionRef.current?.stop()
    audioRef.current?.pause()
    audioRef.current = null
    setPendingConfirm(null)
    setError(null)
    setState('idle')
    setTranscript('')
    transcriptRef.current = ''
  }, [])

  const confirm = useCallback((yes: boolean) => {
    setPendingConfirm(null)
    if (!yes) { setState('idle'); return }
    setState('idle')
  }, [])

  const processTranscript = useCallback(async (text: string) => {
    setState('processing')
    setError(null)
    try {
      const res = await fetch('/api/voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript: text }),
      })
      if (!res.ok) throw new Error(`Voice API error ${res.status}`)
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
      } else {
        setState('idle')
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setError(msg)
      setState('error')
      setTimeout(() => { setState('idle'); setError(null) }, 3000)
    }
    setTranscript('')
    transcriptRef.current = ''
  }, [autoSpeak, onResult, onItemsFound, onDispatch])

  const startListening = useCallback(() => {
    if (state !== 'idle') return
    setError(null)
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRecognition) {
      setError('Speech recognition not supported')
      setState('error')
      setTimeout(() => { setState('idle'); setError(null) }, 3000)
      return
    }
    const recognition = new SpeechRecognition()
    recognition.lang = 'en-US'
    recognition.interimResults = true
    recognitionRef.current = recognition
    setState('listening')
    recognition.onresult = (event: any) => {
      const text = Array.from(event.results).map((r: any) => r[0].transcript).join('')
      setTranscript(text)
      transcriptRef.current = text
    }
    recognition.onend = () => {
      const text = transcriptRef.current
      if (!text.trim()) { setState('idle'); return }
      processTranscript(text)
    }
    recognition.onerror = (event: any) => {
      const msg = event.error === 'no-speech' ? 'No speech detected' : `Error: ${event.error}`
      setError(msg)
      setState('error')
      setTimeout(() => { setState('idle'); setError(null) }, 2000)
    }
    recognition.start()
  }, [state, processTranscript])

  const stopListening = useCallback(() => { recognitionRef.current?.stop() }, [])

  return { state, transcript, pendingConfirm, error, startListening, stopListening, cancel, confirm }
}

export default useVoiceAssistant
