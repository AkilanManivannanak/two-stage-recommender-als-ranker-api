'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { parseGenres, genreColor } from '@/lib/utils'
import { getBackdropForTitle, getPosterForTitle, getMovieByTitle } from '@/lib/movies'
import type { CatalogItem } from '@/lib/api'
import FilmReelLogo from './FilmReelLogo'

interface HeroItem extends CatalogItem { score?: number }

interface LiveWallpaperHeroProps {
  items: HeroItem[]
  onSelect: (item_id: number) => void
  onAddToSession: (item_id: number) => void
}

const INTERVAL = 5800

export default function LiveWallpaperHero({ items, onSelect, onAddToSession }: LiveWallpaperHeroProps) {
  const slides = items.slice(0, 5)
  const [cur, setCur] = useState(0)
  const [prev, setPrev] = useState<number | null>(null)
  const [animKey, setAnimKey] = useState(0)
  const [paused, setPaused] = useState(false)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const advance = useCallback((next: number) => {
    setCur(c => { setPrev(c); return next })
    setAnimKey(k => k + 1)
  }, [])

  useEffect(() => {
    if (paused || slides.length <= 1) return
    timerRef.current = setInterval(() => {
      setCur(c => {
        const n = (c + 1) % slides.length
        setPrev(c)
        setAnimKey(k => k + 1)
        return n
      })
    }, INTERVAL)
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [paused, slides.length])

  if (!slides.length) return null

  const item = slides[cur]
  const prevItem = prev !== null ? slides[prev] : null
  const movie = getMovieByTitle(item.title)
  const genres = parseGenres(item.genres)
  const poster = item.poster_url || getPosterForTitle(item.title, item.item_id)
  const backdrop = getBackdropForTitle(item.title, item.item_id)
  const prevBackdrop = prevItem ? getBackdropForTitle(prevItem.title, prevItem.item_id) : null

  return (
    <div
      className="relative w-full select-none"
      style={{ height: 'clamp(520px, 78vh, 820px)' }}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
    >

      {/* ══ BACKDROP LAYERS ══════════════════════════════════════ */}

      {/* Outgoing backdrop — cross-fade out */}
      {prevBackdrop && (
        <div key={`prev-${prev}`} className="absolute inset-0 z-1"
          style={{
            backgroundImage: `url(${prevBackdrop})`,
            backgroundSize: 'cover', backgroundPosition: 'center 30%',
            animation: 'heroFadeOut 1.1s ease-in-out forwards',
          }}
        />
      )}

      {/* Active backdrop — Ken Burns fade in */}
      <div key={`bg-${animKey}`} className="absolute inset-0 z-2"
        style={{
          backgroundImage: `url(${backdrop})`,
          backgroundSize: 'cover', backgroundPosition: 'center 30%',
          animation: 'heroFadeIn 1.1s ease-in-out forwards, kenBurns 14s ease-out forwards',
        }}
      />

      {/* Gradient overlays */}
      <div className="absolute inset-0 z-10" style={{
        background: 'linear-gradient(100deg, rgba(20,20,20,0.97) 0%, rgba(20,20,20,0.82) 28%, rgba(20,20,20,0.38) 58%, transparent 100%)',
      }}/>
      <div className="absolute inset-0 z-10" style={{
        background: 'linear-gradient(to top, #141414 0%, rgba(20,20,20,0.65) 20%, transparent 52%)',
      }}/>
      <div className="absolute inset-x-0 top-0 h-40 z-10" style={{
        background: 'linear-gradient(to bottom, rgba(20,20,20,0.88) 0%, transparent 100%)',
      }}/>
      <div className="absolute inset-0 z-10 pointer-events-none" style={{
        background: 'radial-gradient(ellipse 50% 65% at 12% 70%, rgba(229,9,20,0.08) 0%, transparent 60%)',
      }}/>

      {/* ══ RIGHT — POSTER FILMSTRIP ════════════════════════════ */}
      <div className="absolute right-6 md:right-14 top-1/2 z-30 -translate-y-1/2 hidden md:flex flex-col gap-2.5">
        {slides.map((slide, i) => {
          const p = slide.poster_url || getPosterForTitle(slide.title, slide.item_id)
          const active = i === cur
          return (
            <button
              key={slide.item_id}
              onClick={() => advance(i)}
              className="relative overflow-hidden rounded-xl group"
              style={{
                width:  active ? '140px' : '70px',
                height: active ? '210px' : '105px',
                flexShrink: 0,
                transition: 'all 0.65s cubic-bezier(0.23,1,0.32,1)',
                border: active ? '2px solid rgba(229,9,20,0.8)' : '1px solid rgba(255,255,255,0.1)',
                boxShadow: active
                  ? '0 20px 56px rgba(0,0,0,0.9), 0 0 0 1px rgba(229,9,20,0.25), 0 0 40px rgba(229,9,20,0.18)'
                  : '0 4px 14px rgba(0,0,0,0.55)',
                transform: active ? 'translateX(0) scale(1)' : 'translateX(5px) scale(0.95)',
                opacity: active ? 1 : 0.45,
              }}
            >
              <img src={p} alt={slide.title} className="w-full h-full object-cover"
                style={{
                  filter: active ? 'brightness(1) saturate(1.15)' : 'brightness(0.5) saturate(0.6)',
                  transform: 'scale(1.04)',
                  transition: 'filter 0.5s ease, transform 0.5s ease',
                }}
              />
              {/* Gloss on hover */}
              <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                style={{ background: 'linear-gradient(135deg, rgba(255,255,255,0.18) 0%, transparent 55%)' }}/>
              {/* Active pulse dot */}
              {active && (
                <div className="absolute top-2 right-2 w-2 h-2 rounded-full bg-red-500"
                  style={{ boxShadow: '0 0 8px rgba(229,9,20,1)', animation: 'pulse 2s ease-in-out infinite' }}/>
              )}
              {/* Progress bar */}
              {active && !paused && (
                <div className="absolute bottom-0 inset-x-0 h-0.5 bg-black/30">
                  <div key={animKey} className="h-full bg-red-500"
                    style={{ animation: `progress ${INTERVAL}ms linear forwards` }}/>
                </div>
              )}
              {/* Rank number */}
              <div className="absolute top-1.5 left-1.5 w-5 h-5 rounded-full bg-black/70 flex items-center justify-center text-[9px] font-bold text-white/60">
                {i + 1}
              </div>
            </button>
          )
        })}
      </div>

      {/* ══ MAIN CONTENT ════════════════════════════════════════ */}
      <div className="absolute inset-0 z-20 flex items-end pointer-events-none">
        <div className="w-full max-w-screen-xl mx-auto px-6 md:px-12 pb-16 md:pb-20">
          <div
            key={`txt-${animKey}`}
            className="max-w-2xl pointer-events-auto"
            style={{ animation: 'slideUp 0.7s cubic-bezier(0.23,1,0.32,1) forwards' }}
          >
            {/* Genre + meta row */}
            <div className="flex flex-wrap items-center gap-2 mb-4">
              {genres.slice(0, 3).map(g => (
                <span key={g} className="text-xs px-3 py-1 rounded-full font-semibold backdrop-blur-sm"
                  style={{
                    background: genreColor(g) + '28', color: genreColor(g),
                    border: `1px solid ${genreColor(g)}55`,
                  }}>
                  {g}
                </span>
              ))}
              {movie?.year && <span className="text-xs font-mono px-2 py-1 rounded-full bg-white/8 text-white/35 border border-white/10">{movie.year}</span>}
              {movie?.rating && <span className="text-xs font-mono px-2 py-1 rounded-full bg-white/8 text-white/35 border border-white/10">{movie.rating}</span>}
            </div>

            {/* Big title */}
            <h1 style={{
              fontFamily: 'Bebas Neue, Georgia, serif',
              fontSize: 'clamp(3.2rem, 8vw, 6.5rem)',
              lineHeight: 0.9,
              color: '#fff',
              textShadow: '0 2px 8px rgba(0,0,0,0.5), 0 12px 48px rgba(0,0,0,0.5)',
              letterSpacing: '0.025em',
              marginBottom: '14px',
            }}>
              {item.title}
            </h1>

            {/* Description */}
            {movie?.description && (
              <p className="text-sm text-white/60 mb-5 max-w-lg leading-relaxed" style={{ lineClamp: 2 }}>
                {movie.description.slice(0, 140)}{movie.description.length > 140 ? '…' : ''}
              </p>
            )}

            {/* Match badges */}
            <div className="flex items-center gap-3 mb-6">
              <span className="text-sm font-bold" style={{ color: '#46d369', textShadow: '0 0 14px rgba(70,211,105,0.6)' }}>
                {96 - cur * 2}% Match
              </span>
              {['4K', 'HDR', 'Dolby'].map(b => (
                <span key={b} className="text-[10px] font-mono px-1.5 py-0.5 border border-white/20 text-white/35 rounded">{b}</span>
              ))}
            </div>

            {/* CTA */}
            <div className="flex items-center gap-3">
              <button onClick={() => onSelect(item.item_id)}
                className="flex items-center gap-2.5 px-7 py-3.5 bg-white text-black font-bold text-sm rounded-xl transition-all active:scale-95 hover:bg-white/92"
                style={{ letterSpacing: '0.04em', boxShadow: '0 4px 24px rgba(0,0,0,0.45)' }}
              >
                <svg className="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                Play
              </button>
              <button onClick={() => onSelect(item.item_id)}
                className="flex items-center gap-2.5 px-7 py-3.5 text-white font-semibold text-sm rounded-xl transition-all active:scale-95 border border-white/25 backdrop-blur-md hover:bg-white/10"
                style={{ background: 'rgba(90,90,90,0.65)', letterSpacing: '0.03em' }}
              >
                <svg className="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                More Info
              </button>
              <button onClick={() => onAddToSession(item.item_id)}
                className="p-3.5 text-white rounded-xl transition-all active:scale-95 border border-white/18 backdrop-blur-md hover:border-red-500/55 hover:bg-red-500/12"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4"/>
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ══ DOT INDICATORS ══════════════════════════════════════ */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-30 flex gap-2 items-center">
        {slides.map((_, i) => (
          <button key={i} onClick={() => advance(i)}
            className="rounded-full transition-all duration-500 ease-out"
            style={{
              width: i === cur ? '28px' : '6px',
              height: '6px',
              background: i === cur ? '#e50914' : 'rgba(255,255,255,0.28)',
              boxShadow: i === cur ? '0 0 10px rgba(229,9,20,0.75)' : 'none',
            }}
          />
        ))}
      </div>

      {/* Subtle film reel watermark */}
      <div className="absolute bottom-5 right-5 z-20 opacity-8 pointer-events-none">
        <FilmReelLogo size={36} />
      </div>
    </div>
  )
}
