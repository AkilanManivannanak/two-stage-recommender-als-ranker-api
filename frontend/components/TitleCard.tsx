'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { placeholderPoster, parseGenres, genreColor, formatScore } from '@/lib/utils'
import type { RecommendItem } from '@/lib/api'


const _TMDB_KEY = '191853b81cda0419b8fb4e79f32bddb8'
const _PC: Record<string, string> = {}

function useTMDBPoster(title: string, url: string | null): string {
  const valid = (u: string) => !!u && u.startsWith('https://image.tmdb.org') && !u.includes('NUDE')
  const [src, setSrc] = useState<string>(valid(url || '') ? url! : '')

  useEffect(() => {
    // Prop changed to valid URL (e.g. after HomeScreen batch resolution)
    if (valid(url || '')) { setSrc(url!); return }
    // Already cached
    if (_PC[title]) { setSrc(_PC[title]); return }
    if (!title) return

    // Fetch immediately — module-level cache prevents duplicate requests
    const c = new AbortController()
    // Normalize "Title, The" → "The Title" for better TMDB match
    const clean = title.replace(/,\s*(The|A|An)$/i, (_, art) => '').replace(/^(.+),\s*(The|A|An)$/i, '$2 $1').trim()
    const queries = clean !== title ? [clean, title] : [title]

    const tryNext = (idx: number) => {
      if (idx >= queries.length) return
      fetch(
        `https://api.themoviedb.org/3/search/movie?api_key=${_TMDB_KEY}&query=${encodeURIComponent(queries[idx])}&language=en-US&page=1`,
        { signal: c.signal }
      )
        .then(r => r.json())
        .then(d => {
          const h = d.results?.[0]
          if (h?.poster_path) {
            const p = `https://image.tmdb.org/t/p/w500${h.poster_path}`
            _PC[title] = p
            setSrc(p)
          } else {
            tryNext(idx + 1)
          }
        })
        .catch(() => {})
    }
    tryNext(0)
    return () => c.abort()
  }, [title, url])

  return src
}

interface TitleCardProps {
  item_id: number
  title: string
  genres: string
  poster_url: string | null
  recData?: RecommendItem
  onSelect?: () => void
  onAddToSession?: () => void
  size?: 'sm' | 'md' | 'lg'
  showScores?: boolean
  rank?: number
}

export default function TitleCard({
  item_id, title, genres, poster_url, recData, onSelect, onAddToSession, size = 'md', showScores = false, rank
}: TitleCardProps) {
  const [imgError, setImgError] = useState(false)
  const [liked, setLiked] = useState<boolean | null>(null)
  const [addedToSession, setAddedToSession] = useState(false)
  const [isHovered, setIsHovered] = useState(false)
  const [tilt, setTilt] = useState({ rotX: 0, rotY: 0, shineX: 50, shineY: 50 })
  const cardRef = useRef<HTMLDivElement>(null)
  const rafRef = useRef<number | null>(null)

  const resolvedPoster = useTMDBPoster(title, poster_url)
  const posterSrc = (!resolvedPoster || imgError) ? placeholderPoster(item_id, title) : resolvedPoster
  const genreList = parseGenres(genres)
  const sizeClasses = { sm: 'w-32', md: 'w-40 md:w-44', lg: 'w-48 md:w-56' }
  const heightClasses = { sm: 'h-48', md: 'h-60 md:h-64', lg: 'h-72 md:h-80' }

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(() => {
      if (!cardRef.current) return
      const rect = cardRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      const cx = rect.width / 2
      const cy = rect.height / 2
      setTilt({
        rotY: ((x - cx) / cx) * 18,
        rotX: -((y - cy) / cy) * 14,
        shineX: (x / rect.width) * 100,
        shineY: (y / rect.height) * 100,
      })
    })
  }, [])

  const handleMouseEnter = () => setIsHovered(true)
  const handleMouseLeave = () => {
    setIsHovered(false)
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    setTilt({ rotX: 0, rotY: 0, shineX: 50, shineY: 50 })
  }

  const handleAddToSession = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!addedToSession) { setAddedToSession(true); onAddToSession?.() }
  }
  const handleLike = (e: React.MouseEvent, val: boolean) => {
    e.stopPropagation()
    setLiked(liked === val ? null : val)
  }

  const shineAngle = Math.atan2(tilt.shineY - 50, tilt.shineX - 50) * (180 / Math.PI)
  const shadowDepth = isHovered ? 20 + Math.abs(tilt.rotX) : 4
  const shadowSpread = isHovered ? 50 + Math.abs(tilt.rotX) * 2 : 20
  const shadowStyle = isHovered
    ? `0 ${shadowDepth}px ${shadowSpread}px rgba(0,0,0,0.75), 0 0 0 1px rgba(229,9,20,0.2)`
    : '0 4px 20px rgba(0,0,0,0.5)'

  const cardTransform = isHovered
    ? `rotateX(${tilt.rotX}deg) rotateY(${tilt.rotY}deg) scale3d(1.06,1.06,1.06) translateZ(10px)`
    : 'rotateX(0deg) rotateY(0deg) scale3d(1,1,1) translateZ(0px)'

  const cardTransition = isHovered
    ? 'transform 0.08s ease-out, box-shadow 0.08s ease-out'
    : 'transform 0.55s cubic-bezier(0.23,1,0.32,1), box-shadow 0.55s cubic-bezier(0.23,1,0.32,1)'

  return (
    <div
      className={`carousel-item ${sizeClasses[size]} cursor-pointer select-none`}
      style={{ perspective: '900px', perspectiveOrigin: 'center center' }}
      onMouseMove={handleMouseMove}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={onSelect}
    >
      <div
        ref={cardRef}
        style={{
          transform: cardTransform,
          transition: cardTransition,
          boxShadow: shadowStyle,
          borderRadius: '12px',
          transformStyle: 'preserve-3d',
          willChange: 'transform',
          position: 'relative',
        }}
      >
        {/* Main poster shell */}
        <div
          className={`relative ${heightClasses[size]} rounded-xl overflow-hidden bg-cine-card`}
          style={{ transformStyle: 'preserve-3d' }}
        >
          {/* Poster image */}
          <img
            src={posterSrc}
            alt={title}
            className="w-full h-full object-cover"
            style={{
              transform: isHovered ? 'scale(1.08)' : 'scale(1)',
              transition: isHovered ? 'transform 0.08s ease-out' : 'transform 0.55s cubic-bezier(0.23,1,0.32,1)',
            }}
            onError={() => setImgError(true)}
            loading="lazy"
          />

          {/* Inner depth vignette */}
          <div
            className="absolute inset-0 rounded-xl pointer-events-none"
            style={{ boxShadow: 'inset 0 0 50px rgba(0,0,0,0.5)', opacity: isHovered ? 1 : 0.4, transition: 'opacity 0.3s' }}
          />

          {/* Apple-style gloss highlight */}
          <div
            className="absolute inset-0 rounded-xl pointer-events-none"
            style={{
              background: `radial-gradient(ellipse at ${tilt.shineX}% ${tilt.shineY}%, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.06) 35%, transparent 65%)`,
              opacity: isHovered ? 1 : 0,
              transition: 'opacity 0.2s',
              mixBlendMode: 'overlay',
            }}
          />

          {/* Specular sweep */}
          <div
            className="absolute inset-0 rounded-xl pointer-events-none"
            style={{
              background: `linear-gradient(${shineAngle + 90}deg, rgba(255,255,255,0.1) 0%, transparent 45%, rgba(0,0,0,0.08) 100%)`,
              opacity: isHovered ? 0.8 : 0,
              transition: isHovered ? 'opacity 0.1s' : 'opacity 0.5s',
            }}
          />

          {/* Bottom gradient reveal */}
          <div
            className="absolute inset-0 rounded-xl pointer-events-none"
            style={{
              background: 'linear-gradient(180deg, transparent 50%, rgba(20,20,20,0.96) 100%)',
              opacity: isHovered ? 1 : 0,
              transition: 'opacity 0.3s',
            }}
          />

          {/* Rank badge — elevated in Z */}
          {rank !== undefined && (
            <div
              className="absolute top-2 left-2 bg-black/75 backdrop-blur-sm border border-white/10 rounded px-1.5 py-0.5 font-mono text-xs text-yellow-400"
              style={{
                transform: isHovered ? 'translateZ(20px)' : 'translateZ(0)',
                transition: isHovered ? 'transform 0.08s ease-out' : 'transform 0.55s cubic-bezier(0.23,1,0.32,1)',
              }}
            >
              #{rank}
            </div>
          )}

          {/* Action buttons — highest Z plane */}
          <div
            className="absolute inset-x-0 bottom-0 p-2 flex items-end justify-between pointer-events-none"
            style={{
              opacity: isHovered ? 1 : 0,
              transform: isHovered ? 'translateY(0) translateZ(30px)' : 'translateY(6px) translateZ(0)',
              transition: isHovered
                ? 'opacity 0.15s, transform 0.2s cubic-bezier(0.23,1,0.32,1)'
                : 'opacity 0.35s, transform 0.5s cubic-bezier(0.23,1,0.32,1)',
              pointerEvents: isHovered ? 'all' : 'none',
            }}
          >
            <div className="flex gap-1">
              <button
                className={`p-1.5 rounded-full backdrop-blur-sm border border-white/10 transition-colors ${
                  liked === true ? 'bg-green-500 text-white' : 'bg-black/60 text-gray-300 hover:text-green-400'
                }`}
                onClick={e => handleLike(e, true)}
              >
                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M14 10V3a1 1 0 00-1.707-.707L6 8.586V20a2 2 0 002 2h7.172a2 2 0 001.978-1.714l.85-6A2 2 0 0016 12h-2z" />
                </svg>
              </button>
              <button
                className={`p-1.5 rounded-full backdrop-blur-sm border border-white/10 transition-colors ${
                  liked === false ? 'bg-red-600 text-white' : 'bg-black/60 text-gray-300 hover:text-red-400'
                }`}
                onClick={e => handleLike(e, false)}
              >
                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M10 14v7a1 1 0 001.707.707L18 15.414V4a2 2 0 00-2-2H8.828a2 2 0 00-1.978 1.714l-.85 6A2 2 0 008 12h2z" />
                </svg>
              </button>
            </div>
            {onAddToSession && (
              <button
                className={`p-1.5 rounded-full backdrop-blur-sm border border-white/10 transition-colors ${
                  addedToSession ? 'bg-red-600 text-white' : 'bg-black/60 text-gray-300 hover:text-red-400'
                }`}
                onClick={handleAddToSession}
              >
                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                  {addedToSession
                    ? <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    : <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth={2} strokeLinecap="round" />
                  }
                </svg>
              </button>
            )}
          </div>
        </div>

        {/* Red glow puddle beneath card on hover */}
        <div
          className="absolute inset-x-6 pointer-events-none"
          style={{
            bottom: '-10px',
            height: '16px',
            background: 'radial-gradient(ellipse, rgba(229,9,20,0.55) 0%, transparent 70%)',
            filter: 'blur(8px)',
            opacity: isHovered ? 0.85 : 0,
            transition: 'opacity 0.4s',
          }}
        />
      </div>

      {/* Text below card */}
      <div
        className="mt-3 px-0.5"
        style={{
          transform: isHovered ? 'translateY(-3px)' : 'translateY(0)',
          transition: 'transform 0.4s cubic-bezier(0.23,1,0.32,1)',
        }}
      >
        <p
          className="text-xs font-medium leading-tight line-clamp-2 transition-colors duration-200"
          style={{ color: isHovered ? '#ff4d57' : '#e5e5e5' }}
        >
          {title}
        </p>
        <div className="flex flex-wrap gap-1 mt-1">
          {genreList.slice(0, 2).map(g => (
            <span
              key={g}
              className="text-[9px] px-1 py-0.5 rounded font-medium"
              style={{
                background: genreColor(g) + '20',
                color: genreColor(g),
                border: `1px solid ${genreColor(g)}30`,
              }}
            >
              {g}
            </span>
          ))}
        </div>
        {showScores && recData && (
          <div className="mt-2 space-y-1">
            {[
              { label: 'ALS', value: recData.als_score },
              { label: 'RNK', value: recData.ranker_score },
            ].map(({ label, value }) => (
              <div key={label} className="flex items-center gap-1.5">
                <span className="text-[9px] font-mono text-cine-muted w-7">{label}</span>
                <div className="flex-1 h-1 bg-cine-border rounded-full overflow-hidden">
                  <div className="score-bar-fill h-full" style={{ width: `${Math.min(100, Math.abs(value ?? 0) * 100)}%` }} />
                </div>
                <span className="text-[9px] font-mono text-cine-muted w-10 text-right">{(value ?? 0).toFixed(3)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
