'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { useAppState, useAppDispatch } from '@/lib/store'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

interface MovieResult {
  item_id: number
  title: string
  poster_url?: string | null
  primary_genre?: string
  year?: number
  description?: string
  score?: number
  reason?: string
}

interface ChatMessage {
  role: 'user' | 'cinewave'
  text: string
  items?: MovieResult[]
  timestamp: number
}

const GREETINGS = [
  "Hey! I'm CineWave 🎬 Your personal movie guide. What are you in the mood for tonight?",
  "Hi there! CineWave here. Tell me what you want to watch — a genre, a vibe, or a movie you love!",
  "Hey! Ask me anything — 'something like Inception', 'feel-good romance', 'dark thriller'. What sounds good?",
]

const FOLLOW_UPS = [
  "Want me to explain why I picked any of these?",
  "Would you like something different — maybe a different genre or mood?",
  "Shall I find something more specific, or are any of these catching your eye?",
  "Do you want more picks like these, or something completely different?",
]

const NO_RESULTS = [
  "I couldn't find a great match for that. Try something like 'action movies like John Wick' or 'feel-good comedies'!",
  "Hmm, nothing perfect came up. Could you describe the vibe? Like 'dark and suspenseful' or 'light and funny'?",
  "I didn't find what you're looking for — maybe try naming a movie you loved and I'll find similar ones!",
]

const CLARIFY = [
  "Sorry, I didn't quite catch that! Could you say it again?",
  "Hmm, I didn't understand that. Try something like 'dark thriller' or 'movies like Interstellar'.",
  "I missed that — say it again? Or just type it below!",
]

const THANK_YOU_RESPONSES = [
  "You're very welcome! Enjoy the movie 🎬 Come back anytime if you need more recommendations!",
  "Happy to help! Hope you find something great to watch tonight. I'm here whenever you need more picks!",
  "Of course! That's what I'm here for. Enjoy your movie night! 🍿",
  "Glad I could help! Let me know anytime you need more recommendations!",
]

const SOCIAL_RESPONSES: Record<string, string> = {
  hi:    "Hey there! What are you in the mood to watch tonight?",
  hello: "Hello! Looking for something to watch? Tell me a genre or a movie you love!",
  hey:   "Hey! What kind of movie are you feeling — action, romance, thriller?",
  bye:   "Goodbye! Come back anytime you need movie recommendations! 🎬",
  ok:    "Great! Is there anything else I can help you find?",
  okay:  "Perfect! Want me to find you something else to watch?",
  yes:   "Great! What would you like to know more about?",
  no:    "No problem! Want me to try a different genre or mood instead?",
}

const EXAMPLES = [
  'Something similar to Stranger Things',
  'Feel-good romantic movies',
  'Dark psychological thriller',
  'Movies like Interstellar',
  'Korean crime dramas',
  'Funny family movies',
  'Mind-bending sci-fi',
  'Scary horror films',
]

let _catalog: Record<number, MovieResult> = {}
async function getCatalog() {
  if (Object.keys(_catalog).length > 0) return _catalog
  try {
    const r = await fetch(`${API_BASE}/catalog/popular?k=1300`)
    const d = await r.json()
    for (const item of d.items || []) _catalog[item.item_id] = item
  } catch {}
  return _catalog
}

// Fetch real TMDB poster when backend has none
async function fetchTMDBPoster(title: string): Promise<string | null> {
  const TMDB_KEY = '191853b81cda0419b8fb4e79f32bddb8'
  try {
    const q = encodeURIComponent(title)
    const r = await fetch(
      `https://api.themoviedb.org/3/search/movie?api_key=${TMDB_KEY}&query=${q}&language=en-US&page=1`,
      { signal: AbortSignal.timeout(4000) }
    )
    if (r.ok) {
      const d = await r.json()
      const hit = d.results?.[0]
      if (hit?.poster_path) return `https://image.tmdb.org/t/p/w500${hit.poster_path}`
    }
  } catch {}
  return null
}

// In-memory poster cache so we don't re-fetch
const _posterCache: Record<string, string> = {}
async function resolveposter(item: MovieResult): Promise<string | null> {
  if (item.poster_url && item.poster_url.startsWith('http') && !item.poster_url.includes('NUDE')) {
    return item.poster_url
  }
  if (_posterCache[item.title]) return _posterCache[item.title]
  const url = await fetchTMDBPoster(item.title)
  if (url) _posterCache[item.title] = url
  return url
}

async function fetchReason(userId: number, item: MovieResult): Promise<string> {
  if (item.reason && item.reason.length > 15) return item.reason
  try {
    const r = await fetch(`${API_BASE}/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, item_ids: [item.item_id] }),
    })
    if (r.ok) {
      const d = await r.json()
      const reason = d.explanations?.[0]?.reason || ''
      if (reason.length > 5) return reason
    }
  } catch {}
  const genre = item.primary_genre || 'this genre'
  const pct = item.score ? Math.min(Math.round(item.score * 100), 99) : null
  return pct && pct > 75
    ? `${pct}% match based on your viewing history.`
    : `Top-rated in ${genre} — loved by viewers with similar taste.`
}

function pick<T>(arr: T[]): T { return arr[Math.floor(Math.random() * arr.length)] }

type VoiceState = 'idle' | 'listening' | 'processing' | 'speaking' | 'done' | 'error'

export default function VoiceModal({ open, onClose, onItemsFound }: {
  open: boolean; onClose: () => void; onItemsFound?: (items: any[]) => void
}) {
  const { activeUser, sessionItemIds } = useAppState()
  const dispatch = useAppDispatch()
  const userId = activeUser?.user_id ?? 1

  const [voiceState, setVoiceState] = useState<VoiceState>('idle')
  const [chat, setChat]             = useState<ChatMessage[]>([])
  const [liveText, setLiveText]     = useState('')
  const [textInput, setTextInput]   = useState('')
  const [exIdx, setExIdx]           = useState(0)
  const [isSpeaking, setIsSpeaking] = useState(false)

  const recRef     = useRef<any>(null)
  const audioRef   = useRef<HTMLAudioElement | null>(null)
  const chatEndRef      = useRef<HTMLDivElement>(null)
  const lastItemsRef    = useRef<MovieResult[]>([])
  const greetedRef      = useRef(false)   // prevents double greeting TTS   // remembers last shown results for explain queries

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [chat])

  useEffect(() => {
    if (!open) return
    const t = setInterval(() => setExIdx(i => (i + 1) % EXAMPLES.length), 3000)
    return () => clearInterval(t)
  }, [open])

  // Greeting when modal opens — greetedRef prevents double TTS
  useEffect(() => {
    if (open && chat.length === 0 && !greetedRef.current) {
      greetedRef.current = true
      const g = pick(GREETINGS)
      setChat([{ role: 'cinewave', text: g, timestamp: Date.now() }])
      setTimeout(() => speakText(g), 600)
    }
    if (!open) {
      greetedRef.current = false  // reset when modal closes so next open greets again
    }
  }, [open])

  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') { stopAll(); onClose() } }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [onClose])

  const stopAll = useCallback(() => {
    if (recRef.current) { try { recRef.current.stop() } catch {}; recRef.current = null }
    if (audioRef.current) { try { audioRef.current.pause() } catch {}; audioRef.current = null }
    setIsSpeaking(false)
    setVoiceState('idle')
    setLiveText('')
  }, [])

  // TTS via /voice/respond
  const speakText = useCallback(async (text: string) => {
    if (!text) return
    if (audioRef.current) { try { audioRef.current.pause() } catch {}; audioRef.current = null }
    setIsSpeaking(true)
    setVoiceState('speaking')
    try {
      const resp = await fetch(`${API_BASE}/voice/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.slice(0, 380), voice: 'nova', speed: 1.0 }),
      })
      if (resp.ok) {
        const d = await resp.json()
        const b64 = d.audio_base64 || d.audio_b64
        if (b64) {
          const audio = new Audio(`data:audio/mp3;base64,${b64}`)
          audioRef.current = audio
          audio.onended = () => { setIsSpeaking(false); setVoiceState('done'); audioRef.current = null }
          audio.onerror = () => { setIsSpeaking(false); setVoiceState('done'); audioRef.current = null }
          await audio.play()
          return
        }
      }
    } catch {}
    setIsSpeaking(false)
    setVoiceState('done')
  }, [])

  const addMessage = useCallback((text: string, items?: MovieResult[]) => {
    setChat(prev => [...prev, { role: 'cinewave', text, items, timestamp: Date.now() }])
  }, [])

  const buildSpokenText = (query: string, items: MovieResult[], genres: string[]): string => {
    const count = items.length
    const top = items.slice(0, 3).map(i => i.title).filter(Boolean)
    const topStr = top.length >= 3 ? `${top[0]}, ${top[1]}, and ${top[2]}` : top.join(' and ')
    const q = query.toLowerCase()
    const follow = pick(FOLLOW_UPS)

    if (q.includes('similar to') || (q.includes(' like ') && q.indexOf(' like ') < 30)) {
      const ref = query.replace(/.*(?:similar to| like )/i, '').replace(/['"]/g, '').trim()
      return `Great taste! I found ${count} titles with a similar feel to ${ref}. Starting with ${topStr}. ${follow}`
    }
    if (genres.length > 0) {
      const g = genres.slice(0,2).map(s => s[0].toUpperCase() + s.slice(1)).join(' and ')
      return `Here are ${count} ${g} picks for you! Leading with ${top[0] || 'the top pick'}. You will also love ${topStr}. ${follow}`
    }
    return `I found ${count} great matches! Check out ${topStr}. ${follow}`
  }

  // Build a detailed explanation for a specific movie
  const buildExplanation = useCallback(async (item: MovieResult, _uid: number): Promise<string> => {
    const title  = item.title || 'this film'
    const genre  = item.primary_genre || ''
    const year   = item.year ? ` (${item.year})` : ''
    const desc   = item.description ? item.description.slice(0, 180) : ''

    const genreExplains: Record<string, string> = {
      'thriller':    `a suspenseful ${genre} that keeps you on edge with twists and tension`,
      'horror':      `a genuinely scary ${genre} with atmosphere and chills`,
      'sci-fi':      `a mind-bending ${genre} with thought-provoking ideas and world-building`,
      'fantasy':     `a rich ${genre} world full of imagination and adventure`,
      'drama':       `an emotionally powerful ${genre} with outstanding performances`,
      'comedy':      `a feel-good ${genre} that delivers exactly the laughs you need`,
      'romance':     `a heartfelt ${genre} love story you won't forget`,
      'action':      `a high-octane ${genre} packed with spectacular sequences`,
      'crime':       `a gripping ${genre} with complex characters and twisting plot`,
      'documentary': `a compelling real-world story that will stay with you`,
      'animation':   `a visually stunning ${genre} that works for all ages`,
      'adventure':   `an exciting ${genre} journey from start to finish`,
      'mystery':     `a clever ${genre} with a satisfying puzzle to solve`,
    }

    const g = genre.toLowerCase()
    const genreDesc = Object.entries(genreExplains).find(([k]) => g.includes(k))?.[1]
      || `a highly-rated ${genre || 'film'} that matches your taste`

    const descPart = desc ? ` ${desc.trim()}` : ''

    return `${title}${year} is ${genreDesc}.${descPart} Would you like to add it to your watchlist, or find more movies like this?`
  }, [])

  // Main search
  const runSearch = useCallback(async (query: string) => {
    const q = query.trim()
    if (!q) return
    setVoiceState('processing')
    setLiveText('')
    setChat(prev => [...prev, { role: 'user', text: q, timestamp: Date.now() }])

    // ── Handle social/meta queries without API call ───────────────────────
    const qLower = q.toLowerCase().replace(/[^a-z ]/g, '').trim()

    // Thank you / goodbye
    if (/thank|thanks|thx|cheers/.test(qLower)) {
      const msg = pick(THANK_YOU_RESPONSES)
      addMessage(msg)
      await speakText(msg)
      setVoiceState('done')
      return
    }

    // Simple social words
    const socialKey = Object.keys(SOCIAL_RESPONSES).find(k => qLower === k || qLower.startsWith(k + ' '))
    if (socialKey && q.length < 15) {
      const msg = SOCIAL_RESPONSES[socialKey]
      addMessage(msg)
      await speakText(msg)
      setVoiceState('done')
      return
    }

    // ── Handle "explain [movie]" — look up from last shown results ────────
    const isExplainQuery = /^(explain|tell me about|what is|what about|describe|more about|info on|information about|details about)/i.test(q)
    if (isExplainQuery && lastItemsRef.current.length > 0) {
      // Find which movie they're asking about from last results
      const afterVerb = q.replace(/^(explain|tell me about|what is|what about|describe|more about|info on|information about|details about)\s*/i, '').toLowerCase().trim()
      // Score each candidate: exact > prefix > substring > word match
      let best = null as MovieResult | null
      let bestScore = 0
      for (const item of lastItemsRef.current) {
        const t = item.title.toLowerCase()
        let score = 0
        if (t === afterVerb) score = 100
        else if (t.startsWith(afterVerb) || afterVerb.startsWith(t)) score = 80
        else if (t.includes(afterVerb) || afterVerb.includes(t)) score = 60
        else {
          const words = afterVerb.split(' ').filter((w: string) => w.length > 2)
          const hits = words.filter((w: string) => t.includes(w)).length
          score = (hits / Math.max(words.length, 1)) * 40
        }
        if (score > bestScore) { bestScore = score; best = item }
      }
      if (best && bestScore > 20) {
        const explainMsg = await buildExplanation(best, userId)
        addMessage(explainMsg, [best])   // show poster alongside explanation
        await speakText(explainMsg)
        setVoiceState('done')
        return
      }
    }

    try {
      const catalog = await getCatalog()
      let items: any[] = []
      let genresUsed: string[] = []

      try {
        const resp = await fetch(`${API_BASE}/voice/assist`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            transcript: q, user_id: userId,
            context_item_ids: sessionItemIds.length ? sessionItemIds : [],
            speak_response: false,
          }),
        })
        if (resp.ok) {
          const d = await resp.json()
          items      = d.tool_result?.items || d.items || []
          genresUsed = d.tool_result?.genres_used || []
        }
      } catch {}

      if (!items.length) {
        try {
          const r = await fetch(`${API_BASE}/ux/mood?q=${encodeURIComponent(q)}&user_id=${userId}`)
          if (r.ok) { const d = await r.json(); items = d.items || [] }
        } catch {}
      }

      if (!items.length) {
        const msg = pick(NO_RESULTS)
        addMessage(msg)
        await speakText(msg)
        setVoiceState('done')
        return
      }

      // Enrich all 8
      const enriched: MovieResult[] = items.slice(0, 8).map((item: any) => {
        const cat = catalog[item.item_id] || {}
        return {
          item_id:       item.item_id,
          title:         cat.title || item.title || `Movie #${item.item_id}`,
          poster_url:    cat.poster_url || item.poster_url || null,
          primary_genre: cat.primary_genre || item.primary_genre || '',
          year:          cat.year || item.year,
          description:   cat.description || item.description || '',
          score:         item.score || item.semantic_score || 0,
          reason:        item.reason || item.llm_reasoning || item.rag_reason || '',
        }
      })

      // All 8 get a reason
      const withReasons = await Promise.all(
        enriched.map(item => fetchReason(userId, item).then(r => ({ ...item, reason: r })))
      )

      // Resolve real TMDB posters for items missing them
      const withPosters = await Promise.all(
        withReasons.map(async item => {
          if (!item.poster_url || !item.poster_url.startsWith('http') || item.poster_url.includes('NUDE')) {
            const realPoster = await resolveposter(item)
            if (realPoster) return { ...item, poster_url: realPoster }
          }
          return item
        })
      )
      lastItemsRef.current = withPosters   // remember for explain queries
      onItemsFound?.(withPosters)

      const spokenText = buildSpokenText(q, withReasons, genresUsed)
      addMessage(spokenText, withPosters)
      await speakText(spokenText)
      setVoiceState('done')

    } catch {
      const err = "Sorry, something went wrong. Could you try again?"
      addMessage(err)
      await speakText(err)
      setVoiceState('error')
    }
  }, [userId, sessionItemIds, onItemsFound, addMessage, speakText])

  // Mic button
  const startListening = useCallback(() => {
    if (voiceState === 'processing' || voiceState === 'speaking') return
    if (audioRef.current) { try { audioRef.current.pause() } catch {}; audioRef.current = null; setIsSpeaking(false) }

    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SR) { addMessage("Voice isn't supported here — just type below!"); return }

    const rec = new SR()
    rec.continuous = false; rec.interimResults = true; rec.lang = 'en-US'
    recRef.current = rec
    rec.onstart  = () => { setVoiceState('listening'); setLiveText('') }
    rec.onresult = (e: any) => {
      const t = Array.from(e.results).map((r: any) => r[0].transcript).join('')
      setLiveText(t)
      if (e.results[e.results.length - 1].isFinal) {
        recRef.current = null; setLiveText('')
        if (t.trim()) runSearch(t)
      }
    }
    rec.onerror = (e: any) => {
      setVoiceState('idle'); setLiveText(''); recRef.current = null
      if (e.error === 'no-speech') { const m = pick(CLARIFY); addMessage(m); speakText(m) }
    }
    rec.onend = () => { if (voiceState === 'listening') setVoiceState('idle'); setLiveText('') }
    rec.start()
  }, [voiceState, runSearch, addMessage, speakText])

  const stopListening = () => {
    if (recRef.current) { try { recRef.current.stop() } catch {}; recRef.current = null }
    setVoiceState('idle'); setLiveText('')
  }

  if (!open) return null

  const isListening  = voiceState === 'listening'
  const isProcessing = voiceState === 'processing'
  const isSpeakingNow = voiceState === 'speaking'
  const orbColor = isListening ? '#e5091a' : isProcessing ? '#a855f7' : isSpeakingNow ? '#22c55e' : '#c5071a'
  const statusText = isListening ? '🎤 Listening...' : isProcessing ? '🔍 Searching...' : isSpeakingNow ? '🔊 Speaking...' : voiceState === 'done' ? '✨ Ask another!' : 'Tap mic to speak'

  return (
    <div className="fixed inset-0 z-[300] flex items-end justify-center">
      <div className="absolute inset-0 bg-black/80 backdrop-blur-md" onClick={() => { stopAll(); onClose() }} />

      <div className="relative w-full max-w-lg mx-auto rounded-t-3xl overflow-hidden"
        style={{ background: 'linear-gradient(180deg,#0f0f0f 0%,#080808 100%)', border: '1px solid rgba(229,9,20,0.1)', borderBottom:'none', boxShadow:'0 -24px 80px rgba(0,0,0,0.95)', maxHeight:'92vh', animation:'slideUp 0.3s cubic-bezier(0.23,1,0.32,1)' }}>

        {/* Drag handle */}
        <div className="flex justify-center pt-3 pb-0.5">
          <div className="w-10 h-1 rounded-full bg-white/10" />
        </div>

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-2 border-b border-white/5">
          <div className="flex items-center gap-2">
            <span className="text-sm">🎬</span>
            <span className="text-[10px] font-bold text-white/50 uppercase tracking-widest">CineWave AI</span>
            {isSpeaking && (
              <div className="flex gap-0.5 items-end h-3">
                {[1,2,3,4].map(i => (
                  <div key={i} className="w-0.5 rounded-full bg-green-400 animate-bounce"
                    style={{ height:`${4+(i%3)*4}px`, animationDelay:`${i*0.1}s` }} />
                ))}
              </div>
            )}
          </div>
          <button onClick={() => { stopAll(); onClose() }}
            className="w-7 h-7 rounded-full bg-white/5 flex items-center justify-center text-white/30 hover:text-white/70 hover:bg-white/10 transition text-sm">
            ✕
          </button>
        </div>

        {/* Chat area */}
        <div className="overflow-y-auto px-4 py-3" style={{ maxHeight:'calc(92vh - 165px)' }}>
          <div className="space-y-3">

            {chat.map((msg, idx) => (
              <div key={idx}>
                {msg.role === 'user' && (
                  <div className="flex justify-end">
                    <div className="px-4 py-2.5 rounded-2xl rounded-br-sm max-w-[82%]"
                      style={{ background:'rgba(229,9,20,0.16)', border:'1px solid rgba(229,9,20,0.3)' }}>
                      <p className="text-sm text-white">{msg.text}</p>
                    </div>
                  </div>
                )}

                {msg.role === 'cinewave' && (
                  <div className="flex justify-start">
                    <div className="max-w-[97%] w-full">
                      <div className="px-4 py-3 rounded-2xl rounded-bl-sm"
                        style={{ background:'rgba(255,255,255,0.035)', border:'1px solid rgba(255,255,255,0.07)' }}>
                        <p className="text-[9px] font-bold text-red-400/40 uppercase tracking-widest mb-1">CineWave</p>
                        <p className="text-sm text-white/85 leading-relaxed">{msg.text}</p>
                      </div>

                      {msg.items && msg.items.length > 0 && (
                        <div className="mt-1.5 space-y-1.5">
                          {msg.items.map((item, i) => (
                            <div key={item.item_id}
                              onClick={() => dispatch({ type: 'ADD_SESSION_ITEM', payload: item.item_id })}
                              className="flex items-start gap-3 p-2.5 rounded-xl cursor-pointer transition-colors"
                              style={{ background:'rgba(255,255,255,0.03)', border:'1px solid rgba(255,255,255,0.06)' }}
                              onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.07)')}
                              onMouseLeave={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.03)')}>
                              {/* Poster */}
                              <div className="w-11 h-16 rounded-lg overflow-hidden flex-shrink-0 bg-white/5">
                                {item.poster_url
                                  ? <img src={item.poster_url} alt={item.title} className="w-full h-full object-cover" />
                                  : <div className="w-full h-full flex items-center justify-center">
                                      <span className="text-white/20 text-xl font-black">{(item.title||'?')[0]}</span>
                                    </div>
                                }
                              </div>
                              {/* Details */}
                              <div className="flex-1 min-w-0">
                                <div className="flex justify-between gap-2">
                                  <p className="text-xs font-bold text-white leading-snug">{item.title}</p>
                                  <span className="text-[9px] text-white/20 font-mono flex-shrink-0">#{i+1}</span>
                                </div>
                                <div className="flex items-center gap-1.5 mt-0.5 flex-wrap">
                                  {item.primary_genre && <span className="text-[9px] text-red-400/70 font-semibold">{item.primary_genre}</span>}
                                  {item.year && <span className="text-[9px] text-white/25">{item.year}</span>}
                                  {item.score && item.score > 0 && <span className="text-[9px] text-green-400/70 font-mono">{Math.min(Math.round(item.score*100),99)}% match</span>}
                                </div>
                                {item.reason && <p className="text-[10px] text-white/38 mt-0.5 leading-snug line-clamp-2">{item.reason}</p>}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Live transcript */}
            {liveText && (
              <div className="flex justify-end">
                <div className="px-4 py-2 rounded-2xl opacity-50"
                  style={{ background:'rgba(229,9,20,0.1)', border:'1px dashed rgba(229,9,20,0.25)' }}>
                  <p className="text-sm text-white/60 italic">{liveText}...</p>
                </div>
              </div>
            )}

            {/* Processing */}
            {isProcessing && (
              <div className="flex justify-start">
                <div className="px-4 py-2.5 rounded-xl"
                  style={{ background:'rgba(168,85,247,0.07)', border:'1px solid rgba(168,85,247,0.18)' }}>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 border-2 border-purple-400/30 border-t-purple-400 rounded-full animate-spin" />
                    <p className="text-xs text-purple-300/60">Searching with RAG + LLM + RL Policy...</p>
                  </div>
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {/* Example chips */}
          {chat.length <= 1 && voiceState === 'idle' && (
            <div className="mt-4">
              <p className="text-[9px] font-mono text-white/18 text-center uppercase tracking-widest mb-2">Try asking</p>
              <div className="flex flex-wrap gap-1.5 justify-center">
                {EXAMPLES.slice(0,6).map((ex,i) => (
                  <button key={ex} onClick={() => runSearch(ex)}
                    className="px-3 py-1.5 rounded-full text-xs border transition-all"
                    style={{
                      borderColor: i===exIdx%6 ? 'rgba(229,9,20,0.55)' : 'rgba(255,255,255,0.08)',
                      color:       i===exIdx%6 ? 'rgba(229,9,20,0.85)' : 'rgba(255,255,255,0.3)',
                      background:  i===exIdx%6 ? 'rgba(229,9,20,0.07)' : 'transparent',
                    }}>
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Input row */}
        <div className="px-4 pb-6 pt-3 border-t border-white/5" style={{ background:'rgba(6,6,6,0.98)' }}>
          <div className="flex items-center gap-2.5">
            {/* Mic orb */}
            <button
              onClick={isListening ? stopListening : startListening}
              disabled={isProcessing}
              className="flex-shrink-0 flex items-center justify-center rounded-full transition-all duration-200 disabled:opacity-40"
              style={{ width:50, height:50, background:orbColor, boxShadow: isListening ? `0 0 0 12px rgba(229,9,20,0.1), 0 0 30px ${orbColor}` : `0 0 12px ${orbColor}88` }}>
              {isProcessing
                ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                : isSpeakingNow
                ? <svg width={18} height={18} fill="white" viewBox="0 0 24 24"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/></svg>
                : <svg width={18} height={18} fill="none" stroke="white" viewBox="0 0 24 24" strokeWidth={isListening ? 2.5 : 2}>
                    <rect x="9" y="2" width="6" height="11" rx="3" fill={isListening ? 'rgba(255,255,255,0.3)' : 'none'} />
                    <path d="M5 10a7 7 0 0014 0" /><line x1="12" y1="19" x2="12" y2="22" /><line x1="9" y1="22" x2="15" y2="22" />
                  </svg>
              }
            </button>

            <input type="text" value={textInput} onChange={e => setTextInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && textInput.trim()) { setTextInput(''); runSearch(textInput) }}}
              placeholder={isListening ? 'Listening...' : 'Ask CineWave anything...'}
              disabled={isListening || isProcessing}
              className="flex-1 rounded-xl px-4 py-2.5 text-sm text-white placeholder-white/22 outline-none transition-all disabled:opacity-35"
              style={{ background:'rgba(255,255,255,0.055)', border:'1px solid rgba(255,255,255,0.09)' }} />

            <button onClick={() => { if (textInput.trim()) { const q = textInput; setTextInput(''); runSearch(q) }}}
              disabled={!textInput.trim() || isProcessing || isListening}
              className="px-4 py-2.5 rounded-xl text-white text-sm font-bold transition-all disabled:opacity-30"
              style={{ background:'rgba(229,9,20,0.85)' }}>
              Send
            </button>
          </div>
          <p className="text-[9px] text-white/16 text-center mt-2 font-mono">{statusText} · RAG + LLM + RL + Metaflow</p>
        </div>
      </div>

      <style>{`@keyframes slideUp { from { transform:translateY(100%); opacity:0 } to { transform:translateY(0); opacity:1 } }`}</style>
    </div>
  )
}
