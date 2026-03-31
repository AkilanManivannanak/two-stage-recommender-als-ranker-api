// lib/utils.ts — CineWave utilities

import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Generate a deterministic placeholder gradient poster for items without TMDB images */
export function placeholderPoster(itemId: number, title?: string | null): string {
  const colors = [
    ['#1a1a2e','#e94560'],['#0f3460','#16213e'],['#533483','#e94560'],
    ['#2b2d42','#ef233c'],['#1b1b2f','#e43f5a'],['#162447','#e43f5a'],
    ['#1f4068','#1b262c'],['#11052c','#4a47a3'],['#2d132c','#ee4540'],
  ]
  const safeId = (typeof itemId === 'number' && isFinite(itemId)) ? itemId : 0
  const [bg, accent] = colors[safeId % colors.length]
  const safeTitle = title || ''
  const initials = safeTitle.split(' ').slice(0,2).map(w=>w[0]?.toUpperCase()||'').join('')
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 600">
    <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="${bg}"/><stop offset="100%" stop-color="${accent}22"/>
    </linearGradient></defs>
    <rect width="400" height="600" fill="url(#g)"/>
    <rect x="0" y="0" width="400" height="4" fill="${accent}" opacity="0.8"/>
    <text x="200" y="290" font-family="Georgia,serif" font-size="80" fill="${accent}"
      opacity="0.9" text-anchor="middle" dominant-baseline="middle">${initials}</text>
    <text x="200" y="400" font-family="system-ui,sans-serif" font-size="18" fill="white"
      opacity="0.7" text-anchor="middle">${safeTitle.slice(0,22)}</text>
  </svg>`
  return `data:image/svg+xml;base64,${btoa(svg)}`
}

/** Format runtime minutes → "2h 15m" */
export function formatRuntime(min?: number): string {
  if (!min) return ''
  const h = Math.floor(min / 60)
  const m = min % 60
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}

/** Score to colour: green > 0.7, amber 0.4-0.7, red < 0.4 */
export function scoreColor(score: number): string {
  if (score >= 0.7) return '#46d369'
  if (score >= 0.4) return '#f5c518'
  return '#e50914'
}

/** Genre to emoji */
export function genreEmoji(genre?: string): string {
  const map: Record<string,string> = {
    Action:'⚡', Comedy:'😂', Drama:'🎭', Horror:'👻', 'Sci-Fi':'🚀',
    Romance:'💕', Thriller:'🔪', Documentary:'📽️', Animation:'✨', Crime:'🔫',
  }
  return map[genre||''] || '🎬'
}

/** TMDB poster URL helper */
export function tmdbPoster(path: string | null | undefined, size: 'w300'|'w500'|'original' = 'w500'): string | null {
  if (!path) return null
  if (path.startsWith('http')) return path
  return `https://image.tmdb.org/t/p/${size}${path}`
}

/** Intent badge colour */
export function intentColor(intent: string | null): string {
  const map: Record<string,string> = {
    binge:'#46d369', discovery:'#3b82f6', background:'#737373',
    social:'#f59e0b', mood_lift:'#ec4899', unknown:'#737373',
  }
  return map[intent||'unknown'] || '#737373'
}

/** Format milliseconds to readable string e.g. 42.3ms */
export function formatMs(ms?: number): string {
  if (ms == null) return '—'
  if (ms < 1) return '<1ms'
  return `${ms.toFixed(1)}ms`
}

/** Profile avatar placeholder images */
export const PROFILE_IMAGES = [
  'https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=120&h=120&fit=crop',
  'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=120&h=120&fit=crop',
  'https://images.unsplash.com/photo-1527980965255-d3b416303d12?w=120&h=120&fit=crop',
  'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=120&h=120&fit=crop',
  'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=120&h=120&fit=crop',
]

/** Hero backdrop images for profile picker */
export const HERO_BACKDROPS = [
  'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=1400&h=800&fit=crop',
  'https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=1400&h=800&fit=crop',
  'https://images.unsplash.com/photo-1524985069026-dd778a71c7b4?w=1400&h=800&fit=crop',
]


/** Parse genre string "Action|Comedy" or "Action, Comedy" into array */
export function parseGenres(genres?: string | null): string[] {
  if (!genres) return []
  return genres.split(/[|,]/).map(g => g.trim()).filter(Boolean)
}

/** Genre to colour for badge */
export function genreColor(genre: string): string {
  const map: Record<string, string> = {
    Action: '#ef4444', Comedy: '#f59e0b', Drama: '#8b5cf6',
    Horror: '#dc2626', 'Sci-Fi': '#3b82f6', Romance: '#ec4899',
    Thriller: '#f97316', Documentary: '#10b981', Animation: '#6366f1',
    Crime: '#ef4444', Adventure: '#f59e0b', Fantasy: '#8b5cf6',
    Mystery: '#0ea5e9', Biography: '#10b981', History: '#78716c',
    Music: '#ec4899', Sport: '#22c55e', War: '#6b7280',
  }
  return map[genre] || '#737373'
}

/** Format recommendation score 0-1 → "0.87" */
export function formatScore(score?: number | null): string {
  if (score == null || !isFinite(score)) return '—'
  return score.toFixed(2)
}

/** bundleVersion alias for state — returns apiBundle */
export function getBundleLabel(bundle: string | null): string {
  return bundle ?? 'demo'
}
