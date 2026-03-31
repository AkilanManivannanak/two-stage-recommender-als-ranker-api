'use client'

import { useState, useRef, useCallback } from 'react'
import { useAppState, useAppDispatch } from '@/lib/store'
import { HERO_BACKDROPS } from '@/lib/utils'
import FilmReelLogo from './FilmReelLogo'
import type { DemoUser } from '@/lib/api'

const PROFILE_NAMES = ['Cinephile', 'Action Fan', 'Indie Lover', 'Blockbuster', 'Art House']

const PROFILE_IMAGES = [
  'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop&q=80',
  'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&h=400&fit=crop&q=80',
  'https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=400&h=400&fit=crop&q=80',
  'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&h=400&fit=crop&q=80',
  'https://images.unsplash.com/photo-1570295999919-56ceb5ecca61?w=400&h=400&fit=crop&q=80',
  'https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400&h=400&fit=crop&q=80',
  'https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=400&h=400&fit=crop&q=80',
  'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400&h=400&fit=crop&q=80',
]


export default function ProfilePicker() {
  const { users, apiHealthy, apiBundle } = useAppState()
  const dispatch = useAppDispatch()
  const [selecting, setSelecting] = useState<number | null>(null)

  const handleSelect = async (user: DemoUser) => {
    setSelecting(user.user_id)
    await new Promise(r => setTimeout(r, 500))
    dispatch({ type: 'SET_ACTIVE_USER', payload: user })
  }

  // Background collage using hero backdrops
  const bgImages = HERO_BACKDROPS.slice(0, 4)

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12 relative overflow-hidden">

      {/* ── Cinematic Background Collage ───────────────────────── */}
      <div className="absolute inset-0 z-0">
        {/* Grid of blurred cinematic images */}
        <div className="absolute inset-0 grid grid-cols-2 grid-rows-2 opacity-25">
          {bgImages.map((src, i) => (
            <div key={i} className="overflow-hidden">
              <img
                src={src}
                alt=""
                className="w-full h-full object-cover"
                style={{
                  filter: 'blur(2px) saturate(0.6)',
                  transform: 'scale(1.05)',
                }}
              />
            </div>
          ))}
        </div>

        {/* Deep vignette overlay */}
        <div
          className="absolute inset-0"
          style={{
            background: 'radial-gradient(ellipse 90% 80% at 50% 50%, rgba(20,20,20,0.55) 0%, rgba(20,20,20,0.85) 60%, #141414 100%)',
          }}
        />

        {/* Red cinematic center glow */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: 'radial-gradient(ellipse 60% 50% at 50% 50%, rgba(229,9,20,0.08) 0%, transparent 70%)',
          }}
        />

        {/* Top & bottom film-strip gradients */}
        <div className="absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-cine-bg to-transparent" />
        <div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-cine-bg to-transparent" />
      </div>

      {/* ── Hero Logo ────────────────────────────────────────────── */}
      <div className="relative z-10 mb-12 text-center animate-fade-in">
        <div className="flex items-center justify-center gap-4 mb-3">
          <FilmReelLogo size={56} />
          <h1
            className="text-7xl md:text-9xl tracking-widest"
            style={{ fontFamily: 'Bebas Neue, Georgia, serif', lineHeight: 1 }}
          >
            <span className="text-gradient-accent">CINE</span>
            <span style={{ color: '#f5f5f1' }}>WAVE</span>
          </h1>
        </div>
        <p className="text-cine-text-dim text-xs tracking-[0.35em] uppercase mt-1">
          AI-Powered Recommendations • Demo
        </p>

        {/* API pill */}
        <div className="mt-5 inline-flex items-center gap-2 px-4 py-2 rounded-full border border-cine-border bg-black/50 backdrop-blur-md">
          <span className={`w-1.5 h-1.5 rounded-full ${
            apiHealthy === null ? 'bg-yellow-400 animate-pulse' :
            apiHealthy ? 'bg-green-400' : 'bg-red-500'
          }`} />
          <span className="text-xs font-mono text-cine-text-dim">
            {apiHealthy === null ? 'Connecting...' :
             apiHealthy ? `API Online${apiBundle ? ` · ${apiBundle}` : ''}` : 'API Offline — mock mode'}
          </span>
        </div>
      </div>

      {/* ── Who's watching ─────────────────────────────────────── */}
      <h2
        className="relative z-10 mb-10 text-2xl font-light text-white/60 tracking-[0.25em] text-center animate-slide-up"
        style={{ animationDelay: '0.1s', animationFillMode: 'both' }}
      >
        Who&apos;s watching?
      </h2>

      {/* ── Profile Grid ──────────────────────────────────────── */}
      <div
        className="relative z-10 flex flex-wrap justify-center gap-6 md:gap-10 max-w-3xl animate-slide-up"
        style={{ animationDelay: '0.2s', animationFillMode: 'both' }}
      >
        {users.length === 0
          ? [...Array(5)].map((_, i) => (
              <div key={i} className="flex flex-col items-center gap-3">
                <div className="w-32 h-32 rounded-2xl skeleton" />
                <div className="w-20 h-3 rounded skeleton" />
              </div>
            ))
          : users.map((user, i) => (
              <ProfileCard3D
                key={user.user_id}
                user={user}
                image={PROFILE_IMAGES[i % PROFILE_IMAGES.length]}
                name={PROFILE_NAMES[i % PROFILE_NAMES.length]}
                isSelecting={selecting === user.user_id}
                onSelect={handleSelect}
                delay={i * 0.07}
              />
            ))
        }
      </div>

      {/* ── Footer ─────────────────────────────────────────────── */}
      <div
        className="relative z-10 mt-14 text-center animate-fade-in space-y-1.5"
        style={{ animationDelay: '0.45s', animationFillMode: 'both' }}
      >
        <p className="text-xs text-white/30">
          Each profile maps to a real <code className="font-mono text-red-400/60">user_id</code> in your recommender
        </p>
        <p className="text-xs text-white/20">
          Press <kbd className="px-1.5 py-0.5 bg-white/5 border border-white/10 rounded font-mono text-white/40 text-xs">`</kbd> for dev tools
        </p>
        <p className="text-xs text-white/20 tracking-wider mt-2">
          © 2026 <span className="text-red-400/50">Akilan Manivannan</span> · CineWave
        </p>
      </div>
    </div>
  )
}

// ── 3D Profile Card ──────────────────────────────────────────────────────────

function ProfileCard3D({ user, image, name, isSelecting, onSelect, delay }: {
  user: DemoUser
  image: string
  name: string
  isSelecting: boolean
  onSelect: (u: DemoUser) => void
  delay: number
}) {
  const [tilt, setTilt] = useState({ rotX: 0, rotY: 0, shineX: 50, shineY: 50 })
  const [hovered, setHovered] = useState(false)
  const [imgErr, setImgErr] = useState(false)
  const ref = useRef<HTMLButtonElement>(null)
  const raf = useRef<number | null>(null)

  const onMove = useCallback((e: React.MouseEvent) => {
    if (!ref.current) return
    if (raf.current) cancelAnimationFrame(raf.current)
    raf.current = requestAnimationFrame(() => {
      if (!ref.current) return
      const r = ref.current.getBoundingClientRect()
      const x = e.clientX - r.left, y = e.clientY - r.top
      setTilt({
        rotY: ((x - r.width / 2) / (r.width / 2)) * 25,
        rotX: -((y - r.height / 2) / (r.height / 2)) * 20,
        shineX: (x / r.width) * 100,
        shineY: (y / r.height) * 100,
      })
    })
  }, [])

  const transform = isSelecting
    ? 'scale3d(0.9,0.9,0.9)'
    : hovered
      ? `rotateX(${tilt.rotX}deg) rotateY(${tilt.rotY}deg) scale3d(1.12,1.12,1.12) translateZ(20px)`
      : 'rotateX(0deg) rotateY(0deg) scale3d(1,1,1)'

  return (
    <button
      ref={ref}
      className="flex flex-col items-center gap-4 outline-none group"
      style={{ perspective: '700px', animationDelay: `${delay}s` }}
      onMouseMove={onMove}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => { setHovered(false); setTilt({ rotX: 0, rotY: 0, shineX: 50, shineY: 50 }) }}
      onClick={() => !isSelecting && onSelect(user)}
      disabled={isSelecting}
    >
      {/* Card face */}
      <div
        className="relative w-36 h-36 rounded-2xl overflow-hidden"
        style={{
          transform,
          transition: hovered ? 'transform 0.08s ease-out' : 'transform 0.65s cubic-bezier(0.23,1,0.32,1)',
          boxShadow: hovered
            ? '0 28px 60px rgba(0,0,0,0.85), 0 0 0 2px rgba(229,9,20,0.5), 0 0 50px rgba(229,9,20,0.2)'
            : '0 8px 30px rgba(0,0,0,0.6), 0 0 0 1px rgba(51,51,51,0.4)',
          transformStyle: 'preserve-3d',
          willChange: 'transform',
        }}
      >
        {/* Profile image */}
        {!imgErr ? (
          <img
            src={image}
            alt={name}
            className="w-full h-full object-cover"
            style={{
              transform: hovered ? 'scale(1.12)' : 'scale(1)',
              transition: hovered ? 'transform 0.08s ease-out' : 'transform 0.65s cubic-bezier(0.23,1,0.32,1)',
              filter: 'saturate(0.85) brightness(0.9)',
            }}
            onError={() => setImgErr(true)}
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-br from-red-900/40 to-black flex items-center justify-center text-5xl">
            🎬
          </div>
        )}

        {/* Dark gradient bottom */}
        <div className="absolute inset-0" style={{
          background: 'linear-gradient(180deg, rgba(0,0,0,0.1) 0%, rgba(0,0,0,0.7) 100%)',
        }} />

        {/* Apple gloss */}
        <div className="absolute inset-0 pointer-events-none" style={{
          background: `radial-gradient(ellipse at ${tilt.shineX}% ${tilt.shineY}%, rgba(255,255,255,0.28) 0%, rgba(255,255,255,0.06) 35%, transparent 65%)`,
          opacity: hovered ? 1 : 0,
          transition: 'opacity 0.2s',
          mixBlendMode: 'overlay',
        }} />

        {/* Top gloss band */}
        <div className="absolute inset-x-0 top-0 h-12 pointer-events-none"
          style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.1) 0%, transparent 100%)', opacity: 0.7 }} />

        {/* Spinning reel watermark on hover */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none"
          style={{ opacity: hovered ? 0.15 : 0, transition: 'opacity 0.3s' }}>
          <FilmReelLogo size={72} />
        </div>

        {/* Selecting spinner */}
        {isSelecting && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60">
            <div className="w-8 h-8 border-2 border-red-500 border-t-transparent rounded-full animate-spin" />
          </div>
        )}

        {/* User ID badge — floats in 3D */}
        <div
          className="absolute bottom-2 right-2 text-xs font-mono px-2 py-0.5 rounded-lg"
          style={{
            background: 'rgba(0,0,0,0.75)',
            border: '1px solid rgba(229,9,20,0.4)',
            color: '#ff4d57',
            backdropFilter: 'blur(4px)',
            transform: hovered ? 'translateZ(28px)' : 'translateZ(0)',
            transition: hovered ? 'transform 0.08s ease-out' : 'transform 0.65s cubic-bezier(0.23,1,0.32,1)',
          }}
        >
          #{user.user_id}
        </div>

        {/* Glow ring */}
        <div className="absolute inset-0 rounded-2xl pointer-events-none" style={{
          border: hovered ? '2px solid rgba(229,9,20,0.5)' : '2px solid transparent',
          transition: 'border 0.3s',
        }} />
      </div>

      {/* Red glow puddle */}
      <div style={{
        height: '12px',
        marginTop: '-10px',
        background: 'radial-gradient(ellipse, rgba(229,9,20,0.55) 0%, transparent 70%)',
        filter: 'blur(8px)',
        opacity: hovered ? 1 : 0,
        transition: 'opacity 0.4s',
        width: '80%',
      }} />

      {/* Label */}
      <div className="text-center -mt-2">
        <p className="text-sm font-semibold transition-colors duration-200"
          style={{ color: hovered ? '#ff4d57' : '#e5e5e5' }}>
          {name}
        </p>
        <p className="text-xs text-white/30 font-mono">user_{user.user_id}</p>
      </div>
    </button>
  )
}
