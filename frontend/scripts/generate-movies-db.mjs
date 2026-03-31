import fs from 'node:fs/promises'
import path from 'node:path'

const API_KEY = process.env.TMDB_API_KEY || '191853b81cda0419b8fb4e79f32bddb8'

const TARGET_COUNT   = Math.max(200, Number(process.env.TARGET_COUNT   || 1200))
const MIN_VOTE_COUNT = Math.max(20,  Number(process.env.MIN_VOTE_COUNT  || 80))
const DISCOVER_PAGES = Math.max(2, Math.min(20, Number(process.env.DISCOVER_PAGES || 15)))
const CONCURRENCY    = Math.max(1, Math.min(12, Number(process.env.TMDB_CONCURRENCY || 8)))
const LANGUAGE       = process.env.TMDB_LANGUAGE || 'en-US'
const IMAGE_POSTER   = 'https://image.tmdb.org/t/p/w500'
const IMAGE_BACKDROP = 'https://image.tmdb.org/t/p/w1280'
const ROOT = process.cwd()

const DEFAULT_BACKEND_DIRS = [
  path.resolve(ROOT, 'backend/artifacts/bundle'),
  path.resolve(ROOT, '../backend/artifacts/bundle'),
  path.resolve(ROOT, '../../backend/artifacts/bundle'),
]

async function tmdb(pathname, query = {}) {
  const url = new URL(`https://api.themoviedb.org/3${pathname}`)
  for (const [key, value] of Object.entries({ api_key: API_KEY, language: LANGUAGE, ...query })) {
    if (value !== undefined && value !== null && value !== '') url.searchParams.set(key, String(value))
  }
  const res = await fetch(url)
  if (!res.ok) throw new Error(`TMDB ${res.status} for ${url.pathname}`)
  return res.json()
}

function normalizeTitle(value) {
  return String(value || '')
    .normalize('NFKC')
    .replace(/['']/g, "'")
    .replace(/[""]/g, '"')
    .replace(/&/g, 'and')
    .replace(/[^a-zA-Z0-9]+/g, ' ')
    .trim()
    .toLowerCase()
}

function sanitize(text) {
  return String(text || '')
    .replace(/\u0000/g, '')
    .replace(/\r/g, ' ')
    .replace(/\n+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function uniqGenres(detail, genreMap) {
  const ids   = Array.isArray(detail.genre_ids) ? detail.genre_ids : []
  const names = Array.isArray(detail.genres)
    ? detail.genres.map((g) => g.name)
    : ids.map((id) => genreMap.get(id)).filter(Boolean)
  return [...new Set(names.filter(Boolean))]
}

function ratingFromReleaseDates(payload) {
  const results = Array.isArray(payload?.results) ? payload.results : []
  const us = results.find((r) => r.iso_3166_1 === 'US')
  if (!us?.release_dates?.length) return 'NR'
  const cert = us.release_dates.map((r) => sanitize(r.certification)).find(Boolean)
  return cert || 'NR'
}

function movieKey(movie) {
  return `${normalizeTitle(movie.title)}::${movie.year}`
}

function qualityScore(movie) {
  const vote       = Number(movie.vote_average || 0)
  const voteCount  = Number(movie.vote_count   || 0)
  const popularity = Number(movie.popularity   || 0)
  const runtime    = Number(movie.runtime      || 0)
  const runtimeBoost = runtime >= 80 ? 2 : runtime >= 60 ? 1 : 0
  return (vote * 100) + Math.log10(Math.max(voteCount, 1)) * 30 + (popularity * 0.15) + runtimeBoost
}

function toFrontendMovie(detail, genreMap, rating) {
  const genres = uniqGenres(detail, genreMap)
  return {
    title:       sanitize(detail.title),
    genres:      genres.join('|') || 'Unknown',
    year:        Number(String(detail.release_date || '0').slice(0, 4)) || 0,
    rating,
    description: sanitize(detail.overview),
    poster:      `${IMAGE_POSTER}${detail.poster_path}`,
    backdrop:    `${IMAGE_BACKDROP}${detail.backdrop_path}`,
  }
}

function toBackendMovie(detail, genreMap, rating, itemId) {
  const genres  = uniqGenres(detail, genreMap)
  const primary = genres[0] || 'Unknown'
  const avgRating5 = Math.max(0, Math.min(5, Number((Number(detail.vote_average || 0) / 2).toFixed(1))))
  const matchPct   = Math.max(70, Math.min(99, Math.round(Number(detail.vote_average || 0) * 10)))
  return {
    item_id:        itemId,
    movieId:        itemId,
    title:          sanitize(detail.title),
    genres:         genres.join('|') || primary,
    primary_genre:  primary,
    description:    sanitize(detail.overview),
    poster_url:     `${IMAGE_POSTER}${detail.poster_path}`,
    backdrop_url:   `${IMAGE_BACKDROP}${detail.backdrop_path}`,
    tmdb_id:        detail.id,
    tmdb_rating:    Number(detail.vote_average || 0),
    maturity_rating: rating,
    year:           Number(String(detail.release_date || '0').slice(0, 4)) || 0,
    avg_rating:     avgRating5,
    popularity:     Number(detail.popularity || 0),
    runtime_min:    Number(detail.runtime || 100),
    match_pct:      matchPct,
  }
}

async function mapLimit(items, limit, fn) {
  const out  = new Array(items.length)
  let next = 0
  async function worker() {
    while (true) {
      const current = next++
      if (current >= items.length) return
      out[current] = await fn(items[current], current)
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker))
  return out
}

async function collectCandidates() {
  console.log('Fetching TMDB genres...')
  const genreResp = await tmdb('/genre/movie/list')
  const genreMap  = new Map((genreResp.genres || []).map((g) => [g.id, g.name]))
  const candidatesById = new Map()
  const perGenre       = new Map([...genreMap.keys()].map((id) => [id, []]))

  const discoverJobs = [
    { label: 'global:popularity', params: { sort_by: 'popularity.desc',          vote_count_gte: MIN_VOTE_COUNT } },
    { label: 'global:vote',       params: { sort_by: 'vote_average.desc',         vote_count_gte: 500 } },
    { label: 'global:release',    params: { sort_by: 'primary_release_date.desc', vote_count_gte: MIN_VOTE_COUNT } },
    { label: 'global:revenue',    params: { sort_by: 'revenue.desc',              vote_count_gte: MIN_VOTE_COUNT } },
    { label: '2020s',    params: { sort_by: 'popularity.desc', vote_count_gte: MIN_VOTE_COUNT, 'primary_release_date.gte': '2020-01-01' } },
    { label: '2010s',    params: { sort_by: 'popularity.desc', vote_count_gte: MIN_VOTE_COUNT, 'primary_release_date.gte': '2010-01-01', 'primary_release_date.lte': '2019-12-31' } },
    { label: '2000s',    params: { sort_by: 'popularity.desc', vote_count_gte: MIN_VOTE_COUNT, 'primary_release_date.gte': '2000-01-01', 'primary_release_date.lte': '2009-12-31' } },
    { label: '1990s',    params: { sort_by: 'popularity.desc', vote_count_gte: MIN_VOTE_COUNT, 'primary_release_date.gte': '1990-01-01', 'primary_release_date.lte': '1999-12-31' } },
    { label: 'classics', params: { sort_by: 'vote_average.desc', vote_count_gte: 1000, 'primary_release_date.lte': '1989-12-31' } },
  ]

  for (const genreId of genreMap.keys()) {
    discoverJobs.push({ label: `genre:${genreId}:popular`, params: { with_genres: genreId, sort_by: 'popularity.desc',           vote_count_gte: MIN_VOTE_COUNT } })
    discoverJobs.push({ label: `genre:${genreId}:vote`,    params: { with_genres: genreId, sort_by: 'vote_average.desc',          vote_count_gte: 100 } })
    discoverJobs.push({ label: `genre:${genreId}:new`,     params: { with_genres: genreId, sort_by: 'primary_release_date.desc',  vote_count_gte: 50 } })
  }

  console.log(`Discovery: ${discoverJobs.length} jobs × ${DISCOVER_PAGES} pages...`)

  for (const job of discoverJobs) {
    for (let page = 1; page <= DISCOVER_PAGES; page++) {
      try {
        const data = await tmdb('/discover/movie', {
          include_adult: false,
          include_video: false,
          region: 'US',
          page,
          ...job.params,
        })
        for (const item of data.results || []) {
          if (!item?.id || !item?.title || !item?.poster_path || !item?.backdrop_path || !item?.overview || !item?.release_date) continue
          if (Number(item.vote_count || 0) < MIN_VOTE_COUNT) continue
          const year = Number(String(item.release_date).slice(0, 4)) || 0
          if (!year) continue
          if (!candidatesById.has(item.id)) candidatesById.set(item.id, item)
          for (const genreId of item.genre_ids || []) {
            const arr = perGenre.get(genreId)
            if (arr && !arr.find((x) => x.id === item.id)) arr.push(item)
          }
        }
      } catch { /* skip failed pages */ }
    }
    process.stdout.write(`\r  Candidates so far: ${candidatesById.size}    `)
  }

  const all = [...candidatesById.values()].sort((a, b) => (b.popularity || 0) - (a.popularity || 0))
  console.log(`\nCollected ${all.length} unique candidates.`)

  const chosen  = []
  const seen    = new Set()
  const perGenreQuota = Math.max(20, Math.floor(TARGET_COUNT / Math.max(genreMap.size, 1)))

  for (const [, items] of perGenre.entries()) {
    const sorted = [...items].sort((a, b) => (b.popularity || 0) - (a.popularity || 0))
    let taken = 0
    for (const item of sorted) {
      if (taken >= perGenreQuota) break
      if (seen.has(item.id)) continue
      chosen.push(item); seen.add(item.id); taken++
    }
  }
  for (const item of all) {
    if (chosen.length >= TARGET_COUNT) break
    if (seen.has(item.id)) continue
    chosen.push(item); seen.add(item.id)
  }

  console.log(`Selected ${chosen.length} movies for detail fetch.`)
  return { chosen, genreMap }
}

async function buildCatalog() {
  const { chosen, genreMap } = await collectCandidates()
  console.log(`Fetching full details (concurrency=${CONCURRENCY})...`)

  let done = 0
  const details = await mapLimit(chosen, CONCURRENCY, async (candidate) => {
    const [detail, releaseDates] = await Promise.all([
      tmdb(`/movie/${candidate.id}`),
      tmdb(`/movie/${candidate.id}/release_dates`).catch(() => ({ results: [] })),
    ])
    done++
    if (done % 100 === 0) process.stdout.write(`\r  ${done}/${chosen.length} fetched...    `)
    if (!detail?.id || !detail?.title || !detail?.poster_path || !detail?.backdrop_path || !detail?.overview || !detail?.release_date) return null
    return { detail, rating: ratingFromReleaseDates(releaseDates) }
  })
  console.log(`\nDetails complete.`)

  const uniqueByTitle = new Map()
  for (const row of details.filter(Boolean)) {
    const movie = toFrontendMovie(row.detail, genreMap, row.rating)
    const key   = movieKey(movie)
    const prev  = uniqueByTitle.get(key)
    if (!prev || qualityScore(row.detail) > qualityScore(prev.detail)) {
      uniqueByTitle.set(key, row)
    }
  }

  const finalRows = [...uniqueByTitle.values()]
    .sort((a, b) => qualityScore(b.detail) - qualityScore(a.detail))
    .slice(0, TARGET_COUNT)

  const frontendMap = {}
  const backendRows = []
  finalRows.forEach((row, index) => {
    const m = toFrontendMovie(row.detail, genreMap, row.rating)
    frontendMap[m.title] = m
    backendRows.push(toBackendMovie(row.detail, genreMap, row.rating, index + 1))
  })

  return { frontendMap, backendRows }
}

function asTypeScript(frontendMap) {
  return [
    `// Auto-generated from TMDB. Do not edit by hand.`,
    `// Generated: ${new Date().toISOString()}`,
    `// Movies: ${Object.keys(frontendMap).length}`,
    `import type { MovieEntry } from './movies'`,
    ``,
    `export const MOVIE_DB: Record<string, MovieEntry> = ${JSON.stringify(frontendMap, null, 2)}`,
    ``,
  ].join('\n')
}

async function ensureDir(dir) { await fs.mkdir(dir, { recursive: true }) }

async function main() {
  console.log(`Target: ${TARGET_COUNT} movies | Pages: ${DISCOVER_PAGES} | Min votes: ${MIN_VOTE_COUNT}`)
  const { frontendMap, backendRows } = await buildCatalog()

  // Write frontend — works from either frontend/ or project root
  let frontendLib
  try {
    await fs.access(path.resolve(ROOT, 'lib'))
    frontendLib = path.resolve(ROOT, 'lib/movies.generated.ts')
  } catch {
    frontendLib = path.resolve(ROOT, 'frontend/lib/movies.generated.ts')
  }
  await ensureDir(path.dirname(frontendLib))
  await fs.writeFile(frontendLib, asTypeScript(frontendMap), 'utf8')
  console.log(`Wrote ${Object.keys(frontendMap).length} frontend entries to ${path.relative(ROOT, frontendLib)}.`)

  // Write backend
  let resolvedBackendDir = null
  if (process.env.BACKEND_BUNDLE_DIR) {
    resolvedBackendDir = path.resolve(ROOT, process.env.BACKEND_BUNDLE_DIR)
  } else {
    for (const candidate of DEFAULT_BACKEND_DIRS) {
      try { await fs.access(path.dirname(candidate)); resolvedBackendDir = candidate; break } catch {}
    }
  }

  if (resolvedBackendDir) {
    await ensureDir(resolvedBackendDir)
    await fs.writeFile(path.join(resolvedBackendDir, 'movies.json'), JSON.stringify(backendRows, null, 2), 'utf8')
    await fs.writeFile(
      path.join(resolvedBackendDir, 'serve_payload.json'),
      JSON.stringify({ generated_by: 'generate-movies-db.mjs', source: 'TMDB', generated_at: new Date().toISOString(), metrics: {}, feature_importance: {}, feature_cols: [], movie_count: backendRows.length }, null, 2),
      'utf8'
    )
    console.log(`Wrote ${backendRows.length} backend entries to ${resolvedBackendDir}/movies.json`)
  }

  const titles = Object.keys(frontendMap)
  console.log(`\nDone. ${titles.length} unique movies. No duplicates: ${titles.length === new Set(titles).size}`)
}

main().catch((err) => { console.error(err); process.exit(1) })
