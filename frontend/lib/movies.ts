/**
 * movies.ts
 * =========
 * Poster and metadata resolver for CineWave.
 *
 * This file provides getMovieByTitle() and getPosterForTitle() which
 * resolve backend catalog items (which may have ML-1M style titles like
 * "Toy Story (1995)" or "Wizard of Oz, The") to TMDB poster URLs.
 *
 * Resolution order:
 *   1. Exact title match in MOVIE_DB
 *   2. Normalised title match (removes year, punctuation, articles)
 *   3. Reversed-article match  "Wizard of Oz, The" → "The Wizard of Oz"
 *   4. Prefix / suffix fuzzy match
 *   5. poster_url from backend item (already TMDB URL)
 *   6. Safe SVG placeholder — never "" and never "NUDE"
 */

import { MOVIE_DB as GENERATED_MOVIE_DB } from './movies.generated'

export interface MovieEntry {
  title:       string
  genres:      string
  year:        number
  rating:      string
  description: string
  poster:      string
  backdrop:    string
}

export const MOVIE_DB: Record<string, MovieEntry> = GENERATED_MOVIE_DB

// ── Title normalisation ────────────────────────────────────────────────────

function normalizeTitle(value: string | null | undefined): string {
  return String(value || '')
    .normalize('NFKC')
    .replace(/['']/g, "'")
    .replace(/[""]/g, '"')
    .replace(/&/g,    'and')
    .replace(/\s*\(\d{4}\)\s*$/, '')   // remove trailing year "(1995)"
    .replace(/\s*\(\d+\)\s*$/,   '')   // remove trailing number
    .replace(/[^a-zA-Z0-9]+/g,   ' ')
    .trim()
    .toLowerCase()
}

/** "Wizard of Oz, The" → "the wizard of oz" */
function reverseArticle(title: string): string {
  return title.replace(/^(.+),\s*(the|a|an)\s*$/i, '$2 $1').toLowerCase()
}

// ── Build lookup indexes ──────────────────────────────────────────────────

const NORMALIZED_INDEX: Record<string, MovieEntry> = {}
const REVERSED_INDEX:   Record<string, MovieEntry> = {}

for (const movie of Object.values(MOVIE_DB)) {
  const norm = normalizeTitle(movie.title)
  if (norm) NORMALIZED_INDEX[norm] = movie
  const rev = reverseArticle(norm)
  if (rev && rev !== norm) REVERSED_INDEX[rev] = movie
}

// ── Main resolver ─────────────────────────────────────────────────────────

export function getMovieByTitle(title: string | null | undefined): MovieEntry | null {
  if (!title) return null

  // 1. Exact match
  if (MOVIE_DB[title]) return MOVIE_DB[title]

  const norm = normalizeTitle(title)

  // 2. Normalised match
  if (NORMALIZED_INDEX[norm]) return NORMALIZED_INDEX[norm]

  // 3. Reversed-article match  "Wizard of Oz, The"
  const rev = reverseArticle(norm)
  if (REVERSED_INDEX[rev]) return REVERSED_INDEX[rev]

  // 4. Partial / prefix match (handles subtitle variants)
  const partial = Object.entries(NORMALIZED_INDEX).find(
    ([key]) => norm.length >= 4 && (norm.startsWith(key) || key.startsWith(norm))
  )
  if (partial) return partial[1]

  return null
}

// ── Poster / backdrop helpers ─────────────────────────────────────────────

const SAFE_PLACEHOLDER =
  'data:image/svg+xml,' + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="300" viewBox="0 0 200 300">' +
    '<rect width="200" height="300" fill="#1a1a2e"/>' +
    '</svg>'
  )
function isSafeUrl(url: string | null | undefined): boolean {
  if (!url) return false
  if (!url.startsWith('http')) return false
  if (url.includes('NUDE'))    return false
  if (url.includes('nude'))    return false
  return true
}

const POSTER_FALLBACKS    = Object.values(MOVIE_DB).filter(m => isSafeUrl(m.poster)).slice(0, 20).map(m => m.poster)
const BACKDROP_FALLBACKS  = Object.values(MOVIE_DB).filter(m => isSafeUrl(m.backdrop)).slice(0, 12).map(m => m.backdrop)

export function getPosterForTitle(
  title:   string | null | undefined,
  itemId?: number,
  backendPosterUrl?: string | null
): string {
  // 0. Backend already gave us a safe TMDB URL — use it directly
  if (isSafeUrl(backendPosterUrl)) return backendPosterUrl!

  // 1. Lookup in MOVIE_DB
  const movie = getMovieByTitle(title)
  if (movie?.poster && isSafeUrl(movie.poster)) return movie.poster

  // 2. Rotate through fallbacks by itemId
  if (POSTER_FALLBACKS.length > 0) {
    const idx = Math.abs(Number(itemId || 0)) % POSTER_FALLBACKS.length
    return POSTER_FALLBACKS[idx]
  }

  return SAFE_PLACEHOLDER
}

export function getBackdropForTitle(
  title:   string | null | undefined,
  itemId?: number,
  backendBackdropUrl?: string | null
): string {
  if (isSafeUrl(backendBackdropUrl)) return backendBackdropUrl!

  const movie = getMovieByTitle(title)
  if (movie?.backdrop && isSafeUrl(movie.backdrop)) return movie.backdrop

  // Fall back to poster (better than NUDE or blank)
  const poster = getPosterForTitle(title, itemId)
  if (poster !== SAFE_PLACEHOLDER) return poster

  if (BACKDROP_FALLBACKS.length > 0) {
    const idx = Math.abs(Number(itemId || 0)) % BACKDROP_FALLBACKS.length
    return BACKDROP_FALLBACKS[idx]
  }

  return SAFE_PLACEHOLDER
}

export const PROFILE_IMAGES = [
  'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop&q=80',
  'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&h=400&fit=crop&q=80',
  'https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=400&h=400&fit=crop&q=80',
  'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&h=400&fit=crop&q=80',
  'https://images.unsplash.com/photo-1570295999919-56ceb5ecca61?w=400&h=400&fit=crop&q=80',
  'https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400&h=400&fit=crop&q=80',
  'https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=400&h=400&fit=crop&q=80',
  'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400&h=400&fit=crop&q=80',
]
