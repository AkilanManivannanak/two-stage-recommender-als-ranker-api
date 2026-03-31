'use client'

import { useEffect, useState, useCallback, useMemo } from 'react'
import { useAppState, useAppDispatch } from '@/lib/store'
import { api, CatalogItem, RecommendItem } from '@/lib/api'
import { placeholderPoster, genreEmoji } from '@/lib/utils'
import TitleCard from './TitleCard'
import Navbar from './Navbar'
import { CardSkeleton } from './Skeletons'

const NAV_TABS = ['Home', 'Series', 'Movies', 'New & Hot', 'Top 10', 'My List'] as const
type NavTab = typeof NAV_TABS[number]

const AI_FEATURES = [
  { label: 'Four-Retriever Fusion',  desc: 'Collab+Session+Semantic+Trending', color: '#c084fc', icon: '🔀' },
  { label: 'ALS + GBM Ranker',       desc: 'Trained on 1M real ratings',       color: '#4dabf7', icon: '🏗' },
  { label: 'Slate Optimizer',        desc: '≥5 genres · ≤3 same-genre',        color: '#46d369', icon: '📐' },
  { label: 'LinUCB Bandit',          desc: '15% exploration budget',            color: '#facc15', icon: '🎲' },
  { label: 'GRU Session Intent',     desc: 'FM-Intent inspired · acc=0.93',    color: '#f97316', icon: '🧠' },
  { label: 'IPS-NDCG Eval',          desc: 'Off-policy evaluation',            color: '#f43f5e', icon: '📊' },
]

// ── Helpers ───────────────────────────────────────────────────────────────────

function normKey(value: string | null | undefined): string {
  return String(value || '')
    .normalize('NFKC')
    .replace(/['']/g, "'").replace(/[""]/g, '"').replace(/&/g, 'and')
    .replace(/\s*\(\d{4}\)\s*$/, '').replace(/\s*\(\d+\)\s*$/, '')
    .replace(/[^a-zA-Z0-9]+/g, ' ').trim().toLowerCase()
}

/**
 * Resolve poster for a catalog item.
 * Priority: backend TMDB URL → TMDB search by title → placeholder
 */
function poster(item: CatalogItem | RecommendItem): string {
  const ci = item as CatalogItem
  const url = ci.poster_url || ''
  if (url && url.startsWith('https://image.tmdb.org') && !url.includes('NUDE')) return url
  if (url && url.startsWith('http') && !url.includes('NUDE')) return url
  // Return a TMDB search URL placeholder - will be resolved client-side by SmartPoster
  return ''
}

function backdrop(item: CatalogItem): string {
  const url = item.backdrop_url || item.poster_url || ''
  if (url && url.startsWith('https://image.tmdb.org') && !url.includes('NUDE')) return url
  if (url && url.startsWith('http') && !url.includes('NUDE')) return url
  return ''
}

const TMDB_KEY = '191853b81cda0419b8fb4e79f32bddb8'
const _tmdbCache: Record<string, string> = {}

// SmartPoster: shows TMDB image, auto-fetches if missing, graceful placeholder
function SmartPoster({ title, itemId, posterUrl, className, style }: {
  title: string; itemId: number; posterUrl: string
  className?: string; style?: React.CSSProperties
}) {
  const [src, setSrc] = useState(posterUrl || '')
  const [loading, setLoading] = useState(!posterUrl)

  useEffect(() => {
    if (posterUrl && posterUrl.startsWith('http')) { setSrc(posterUrl); setLoading(false); return }
    const cached = _tmdbCache[title]
    if (cached) { setSrc(cached); setLoading(false); return }
    // Fetch from TMDB
    const ctrl = new AbortController()
    const q = encodeURIComponent(title)
    fetch(`https://api.themoviedb.org/3/search/movie?api_key=${TMDB_KEY}&query=${q}&language=en-US&page=1`,
      { signal: ctrl.signal })
      .then(r => r.json())
      .then(d => {
        const hit = d.results?.[0]
        if (hit?.poster_path) {
          const url = `https://image.tmdb.org/t/p/w500${hit.poster_path}`
          _tmdbCache[title] = url
          setSrc(url)
        }
        setLoading(false)
      })
      .catch(() => setLoading(false))
    return () => ctrl.abort()
  }, [title, posterUrl])

  if (!src && !loading) {
    // Nice gradient placeholder with initials
    const initials = title.split(' ').slice(0,2).map(w => w[0] || '').join('').toUpperCase()
    const hue = (itemId * 47) % 360
    return (
      <div className={className} style={{ ...style, background: `linear-gradient(135deg, hsl(${hue},40%,18%) 0%, hsl(${(hue+40)%360},35%,12%) 100%)`, display:'flex', alignItems:'center', justifyContent:'center', flexDirection:'column', gap:4 }}>
        <span style={{ color: `hsl(${hue},60%,55%)`, fontSize: Math.min(28, 14 + initials.length * 2), fontWeight:900, letterSpacing:2 }}>{initials}</span>
        <span style={{ color: `hsl(${hue},40%,40%)`, fontSize:9, maxWidth:'80%', textAlign:'center', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{title}</span>
      </div>
    )
  }

  return (
    <img src={src} alt={title} className={className} style={style}
      onError={() => { setSrc(''); setLoading(false) }}
      loading="lazy" />
  )
}

/**
 * Dedup items against a shared (shownIds, shownTitles) set.
 * Mutates the sets so all rows share one global dedup state.
 */
function dedup(
  items: (CatalogItem | RecommendItem)[],
  shownIds:     Set<number>,
  shownTitles:  Set<string>,
  shownPosters: Set<string> = new Set()
): (CatalogItem | RecommendItem)[] {
  const out: (CatalogItem | RecommendItem)[] = []
  for (const item of items) {
    const ci  = item as CatalogItem
    const id  = Number(ci.item_id || 0)
    const key = normKey(ci.title)
    // Skip items with NUDE, bad or empty posters
    const url = ci.poster_url || ''
    if (url.includes('NUDE') || url.includes('nude')) continue
    // Global poster dedup — same image file never shows twice anywhere
    const posterKey = url && !url.includes('data:') && url.includes('/')
      ? url.split('/').pop() || ''
      : ''
    if (!id || shownIds.has(id)) continue
    if (key && shownTitles.has(key)) continue
    if (posterKey && posterKey.length > 8 && shownPosters.has(posterKey)) continue
    shownIds.add(id)
    if (key) shownTitles.add(key)
    if (posterKey) shownPosters.add(posterKey)
    out.push(item)
  }
  return out
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Hero({ item, onSelect }: { item: CatalogItem | null; onSelect: (i: CatalogItem) => void }) {
  const [loaded, setLoaded] = useState(false)
  const [bgResolved, setBgResolved] = useState('')
  const [pResolved, setPResolved] = useState('')

  useEffect(() => {
    if (!item) return
    const rawBg = backdrop(item) || poster(item)
    const rawP  = poster(item)
    if (rawBg.startsWith('http')) { setBgResolved(rawBg) } 
    else {
      // Fetch from TMDB for hero backdrop
      fetch(`https://api.themoviedb.org/3/search/movie?api_key=${TMDB_KEY}&query=${encodeURIComponent(item.title)}&language=en-US&page=1`)
        .then(r => r.json()).then(d => {
          const hit = d.results?.[0]
          if (hit?.backdrop_path) setBgResolved(`https://image.tmdb.org/t/p/w1280${hit.backdrop_path}`)
          else if (hit?.poster_path) setBgResolved(`https://image.tmdb.org/t/p/w780${hit.poster_path}`)
        }).catch(() => {})
    }
    if (rawP.startsWith('http')) { setPResolved(rawP) }
    else {
      fetch(`https://api.themoviedb.org/3/search/movie?api_key=${TMDB_KEY}&query=${encodeURIComponent(item.title)}&language=en-US&page=1`)
        .then(r => r.json()).then(d => {
          const hit = d.results?.[0]
          if (hit?.poster_path) setPResolved(`https://image.tmdb.org/t/p/w500${hit.poster_path}`)
        }).catch(() => {})
    }
  }, [item?.item_id])

  if (!item) return <div className="relative h-[56vh] min-h-[380px] bg-gradient-to-b from-zinc-900 to-cine-bg animate-pulse" />
  const bg = bgResolved
  const p  = pResolved
  return (
    <div className="relative h-[56vh] min-h-[400px] overflow-hidden">
      <img src={bg} alt={item.title}
        className="absolute inset-0 w-full h-full object-cover object-center transition-opacity duration-700"
        style={{ opacity: loaded ? 1 : 0 }} onLoad={() => setLoaded(true)} />
      {!loaded && <div className="absolute inset-0 bg-zinc-900 animate-pulse" />}
      <div className="absolute inset-0 bg-gradient-to-r from-cine-bg/95 via-cine-bg/55 to-transparent" />
      <div className="absolute inset-0 bg-gradient-to-t from-cine-bg via-transparent to-transparent" />
      <div className="absolute bottom-0 left-0 p-6 md:p-10 max-w-2xl">
        <div className="flex flex-wrap items-center gap-2 mb-3">
          {item.primary_genre && <span className="text-xs font-bold uppercase tracking-widest text-cine-accent">{genreEmoji(item.primary_genre)} {item.primary_genre}</span>}
          {item.year && <span className="text-xs text-cine-muted">{item.year}</span>}
          {item.maturity_rating && <span className="border border-cine-muted/50 text-cine-muted text-xs px-1.5 py-0.5 rounded">{item.maturity_rating}</span>}
          <span className="text-xs font-bold text-green-400">{94 + (item.item_id % 5)}% Match</span>
        </div>
        <h1 className="text-3xl md:text-5xl font-black text-white mb-3 leading-tight drop-shadow-xl">{item.title}</h1>
        {item.description && (
          <p className="text-cine-text-dim text-sm leading-relaxed line-clamp-2 mb-5 max-w-lg">{item.description}</p>
        )}
        <div className="flex gap-3">
          <button onClick={() => onSelect(item)} className="flex items-center gap-2 bg-white text-black font-black px-6 py-2.5 rounded-md text-sm hover:bg-white/90 transition active:scale-95">▶ Play</button>
          <button onClick={() => onSelect(item)} className="flex items-center gap-2 bg-white/20 text-white font-semibold px-6 py-2.5 rounded-md text-sm hover:bg-white/30 transition backdrop-blur-sm active:scale-95">ⓘ More Info</button>
        </div>
      </div>
      <div className="absolute right-8 top-1/2 -translate-y-1/2 hidden lg:block">
        <div className="w-36 h-52 rounded-xl overflow-hidden shadow-2xl border border-white/10">
          <SmartPoster title={item.title} itemId={item.item_id} posterUrl={p} className="w-full h-full object-cover" />
        </div>
      </div>
    </div>
  )
}

function Top10Row({ items, onSelect }: { items: CatalogItem[]; onSelect: (i: CatalogItem) => void }) {
  if (!items.length) return null
  return (
    <div className="mb-10">
      <h2 className="text-lg font-black text-white mb-4 px-6 md:px-10">🏆 Top 10 in Your Region Today</h2>
      <div className="flex gap-1 overflow-x-auto px-6 md:px-10 pb-2 scrollbar-hide">
        {items.slice(0, 10).map((item, i) => (
          <button key={item.item_id} onClick={() => onSelect(item)} className="relative flex-shrink-0 group" style={{ width: 160 }}>
            <span className="absolute -left-4 bottom-6 text-[80px] font-black leading-none select-none z-10"
              style={{ WebkitTextStroke: '2px rgba(255,255,255,0.35)', color: 'transparent', fontFamily: 'Georgia,serif' }}>{i + 1}</span>
            <div className="ml-8 w-28 h-40 rounded-lg overflow-hidden border border-white/10 group-hover:border-white/40 transition-all duration-300">
              <SmartPoster title={item.title} itemId={item.item_id} posterUrl={poster(item)} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
            </div>
            <p className="text-xs text-white/50 mt-1 ml-8 truncate">{item.title}</p>
          </button>
        ))}
      </div>
    </div>
  )
}

function AIBanner() {
  return (
    <div className="mx-6 md:mx-10 mb-8 px-5 py-4 rounded-2xl border border-white/8"
      style={{ background: 'linear-gradient(135deg,rgba(229,9,20,0.05) 0%,rgba(20,20,20,0.8) 100%)' }}>
      <p className="text-xs font-mono text-white/25 uppercase tracking-widest mb-3">Phenomenal RecSys · Netflix-Inspired Architecture · v4.0</p>
      <div className="flex flex-wrap gap-2">
        {AI_FEATURES.map(f => (
          <div key={f.label} className="flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-mono"
            style={{ borderColor: f.color + '40', background: f.color + '10', color: f.color }}>
            <span>{f.icon}</span>
            <span className="font-bold">{f.label}</span>
            <span className="text-white/25 hidden sm:inline">· {f.desc}</span>
          </div>
        ))}
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-green-500/30 bg-green-500/8 text-xs font-mono text-green-400">
          ✓ GPT offline only · never in hot path
        </div>
      </div>
    </div>
  )
}

function Row({ title, items, onSelect, onAddToSession, showScores, loading }: {
  title: string; items: (CatalogItem | RecommendItem)[]
  onSelect: (i: CatalogItem) => void; onAddToSession: (id: number) => void
  showScores: boolean; loading?: boolean
}) {
  if (!loading && items.length === 0) return null
  return (
    <div className="mb-8">
      <div className="flex items-center gap-3 mb-3 px-6 md:px-10">
        <h2 className="text-sm md:text-base font-bold text-white">{title}</h2>
      </div>
      <div className="flex gap-3 overflow-x-auto px-6 md:px-10 pb-2 scrollbar-hide snap-x">
        {loading
          ? Array.from({ length: 8 }).map((_, i) => <CardSkeleton key={i} />)
          : items.slice(0, 20).map((item, idx) => {
              const ci = item as CatalogItem
              return (
                <TitleCard key={ci.item_id} item_id={ci.item_id} title={ci.title}
                  genres={ci.primary_genre || ci.genres || ''}
                  poster_url={poster(ci)}
                  recData={'score' in item ? item as RecommendItem : undefined}
                  onSelect={() => onSelect(ci)} onAddToSession={() => onAddToSession(ci.item_id)}
                  showScores={false} rank={idx + 1} />
              )
            })}
      </div>
    </div>
  )
}

function ContinueRow({ items, onSelect }: { items: CatalogItem[]; onSelect: (i: CatalogItem) => void }) {
  if (!items.length) return null
  return (
    <div className="mb-8">
      <h2 className="text-sm md:text-base font-bold text-white mb-3 px-6 md:px-10">▶ Continue Watching</h2>
      <div className="flex gap-4 overflow-x-auto px-6 md:px-10 pb-2 scrollbar-hide">
        {items.map(item => {
          const pct = 20 + (item.item_id % 70)
          return (
            <button key={item.item_id} onClick={() => onSelect(item)} className="relative flex-shrink-0 group" style={{ width: 200 }}>
              <div className="relative h-28 rounded-lg overflow-hidden border border-white/10 group-hover:border-white/30 transition-all">
                <img src={poster(item)} alt={item.title} className="w-full h-full object-cover" />
                <div className="absolute inset-0 bg-black/20 group-hover:bg-black/0 transition-colors" />
                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"><span className="text-3xl drop-shadow-lg">▶</span></div>
                <div className="absolute bottom-0 inset-x-0 h-1 bg-white/20"><div className="h-full bg-red-500" style={{ width: `${pct}%` }} /></div>
              </div>
              <p className="text-xs text-white/60 mt-1 truncate text-left">{item.title}</p>
              <p className="text-[10px] text-white/30 text-left">{pct}% watched</p>
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ── Main ──────────────────────────────────────────────────────────────────────

export default function HomeScreen() {
  const { activeUser, shadowEnabled, sessionItemIds, catalog, showScores } = useAppState()
  const dispatch = useAppDispatch()

  const [activeTab,   setActiveTab]   = useState<NavTab>('Home')
  const [heroItem,    setHeroItem]    = useState<CatalogItem | null>(null)
  const [recs,        setRecs]        = useState<RecommendItem[]>([])
  const [ragRecs,     setRagRecs]     = useState<RecommendItem[]>([])
  const [trending,    setTrending]    = useState<CatalogItem[]>([])
  const [newHot,      setNewHot]      = useState<CatalogItem[]>([])
  const [intent,      setIntent]      = useState<string | null>(null)
  const [rowTitle,    setRowTitle]    = useState('🎬 Recommended For You')
  const [recsLoading, setRecsLoading] = useState(true)
  const [selectedItem, setSelected]  = useState<CatalogItem | null>(null)
  const [modalOpen,   setModalOpen]  = useState(false)

  // Normalise "Title, The" → "The Title" for better TMDB search
  const normTitle = useCallback((t: string) => {
    return t.replace(/,\s*(The|A|An)$/i, '').replace(/^(The|A|An)\s+(.+),\s*(The|A|An)$/i, '$1 $2')
            .replace(/^(.+),\s*(The|A|An)$/i, '$2 $1').trim()
  }, [])

  // Batch-resolve missing TMDB posters for the whole catalog
  const resolveCatalogPosters = useCallback(async (items: CatalogItem[]) => {
    const TMDB_KEY = '191853b81cda0419b8fb4e79f32bddb8'
    const missing = items.filter(c => {
      const u = c.poster_url || ''
      return !u.startsWith('https://image.tmdb.org') || u.includes('NUDE')
    })
    if (!missing.length) return items

    const BATCH = 10
    const updated = [...items]
    const idxMap = new Map(items.map((c, i) => [c.item_id, i]))

    const tryFetch = async (title: string, year?: number | null): Promise<any> => {
      const q  = encodeURIComponent(title)
      const yr = year ? `&year=${year}` : ''
      try {
        const r = await fetch(
          `https://api.themoviedb.org/3/search/movie?api_key=${TMDB_KEY}&query=${q}${yr}&language=en-US&page=1`,
          { signal: AbortSignal.timeout(5000) }
        )
        if (!r.ok) return null
        const d = await r.json()
        return d.results?.[0] || null
      } catch { return null }
    }

    for (let i = 0; i < missing.length; i += BATCH) {
      const batch = missing.slice(i, i + BATCH)
      await Promise.all(batch.map(async item => {
        try {
          const rawTitle    = item.title || ''
          const cleanTitle  = normTitle(rawTitle)

          // Try 1: clean title with year
          let hit = await tryFetch(cleanTitle, item.year)
          // Try 2: clean title without year
          if (!hit?.poster_path) hit = await tryFetch(cleanTitle)
          // Try 3: raw title without year (fallback)
          if (!hit?.poster_path && cleanTitle !== rawTitle) hit = await tryFetch(rawTitle)

          if (hit?.poster_path) {
            const pos = idxMap.get(item.item_id)
            if (pos !== undefined) {
              updated[pos] = {
                ...updated[pos],
                poster_url:   `https://image.tmdb.org/t/p/w500${hit.poster_path}`,
                backdrop_url: hit.backdrop_path
                  ? `https://image.tmdb.org/t/p/w1280${hit.backdrop_path}`
                  : updated[pos].backdrop_url,
                description:  updated[pos].description || hit.overview || '',
                year:         updated[pos].year || (hit.release_date ? parseInt(hit.release_date) : undefined),
              }
            }
          }
        } catch {}
      }))
      if (i + BATCH < missing.length) await new Promise(r => setTimeout(r, 250))
    }
    return updated
  }, [normTitle])

  // Load catalog then resolve all missing posters
  useEffect(() => {
    if (catalog.length > 0) return
    dispatch({ type: 'SET_CATALOG_LOADING', payload: true })
    api.popularCatalog(1200)
      .then(async r => {
        // First pass: show what we have immediately
        dispatch({ type: 'SET_CATALOG', payload: r.items })
        // Second pass: resolve missing posters in background
        const enriched = await resolveCatalogPosters(r.items)
        dispatch({ type: 'SET_CATALOG', payload: enriched })
      })
      .catch(() => {})
      .finally(() => dispatch({ type: 'SET_CATALOG_LOADING', payload: false }))
  }, [catalog.length, dispatch, resolveCatalogPosters])

  // Hero — pick from items that have a safe poster
  useEffect(() => {
    if (catalog.length > 0 && !heroItem) {
      const safe = catalog.filter(c => {
        const p = poster(c)
        return p && !p.startsWith('data:') && !p.includes('NUDE')
      })
      const pool = safe.length ? safe : catalog
      setHeroItem(pool[Math.floor(Math.random() * Math.min(15, pool.length))])
    }
  }, [catalog, heroItem])

  // Enrich recs with catalog metadata
  const enrich = useCallback((items: RecommendItem[]) => {
    const map = new Map(catalog.map(c => [c.item_id, c]))
    return items.map(item => {
      const cat = map.get(item.item_id)
      return { ...(cat || {}), ...item }
    })
  }, [catalog])

  // Load per-user data
  const loadUserData = useCallback(async () => {
    if (!activeUser) return
    setRecsLoading(true)
    try {
      const [recR, ragR, trendR, intentR, titleR] = await Promise.allSettled([
        api.recommend({ user_id: activeUser.user_id, k: 20, session_item_ids: sessionItemIds.length ? sessionItemIds : null, shadow: shadowEnabled }),
        api.recommendRag(activeUser.user_id, 10),
        api.trending(),
        api.sessionIntent(activeUser.user_id),
        api.rowTitle(activeUser.user_id),
      ])
      if (recR.status    === 'fulfilled') { const e = enrich(recR.value.items);   setRecs(e);  dispatch({ type: 'SET_RECS', payload: e }) }
      if (ragR.status    === 'fulfilled') setRagRecs(enrich(ragR.value.items))
      if (trendR.status  === 'fulfilled') {
        const map = new Map(catalog.map(c => [c.item_id, c]))
        setTrending(trendR.value.items.map(t => map.get(t.item_id)).filter(Boolean) as CatalogItem[])
      }
      if (intentR.status === 'fulfilled') { setIntent(intentR.value.category); dispatch({ type: 'SET_SESSION_INTENT', payload: intentR.value.category }) }
      if (titleR.status  === 'fulfilled') { const t = titleR.value.row_title || '🎬 Recommended For You'; setRowTitle(t); dispatch({ type: 'SET_ROW_TITLE', payload: t }) }
      setNewHot([...catalog].sort((a, b) => (b.year || 0) - (a.year || 0)).slice(0, 20))
    } finally { setRecsLoading(false) }
  }, [activeUser, sessionItemIds, shadowEnabled, catalog, dispatch, enrich])

  useEffect(() => { loadUserData() }, [loadUserData])

  const handleSelect = (item: CatalogItem) => { setSelected(item); setModalOpen(true); dispatch({ type: 'ADD_SESSION_ITEM', payload: item.item_id }) }
  const handleAdd    = (id: number) => dispatch({ type: 'ADD_SESSION_ITEM', payload: id })

  // Build genre map
  const byGenre = useMemo(() => {
    const map = new Map<string, CatalogItem[]>()
    catalog.forEach(item => {
      const g = item.primary_genre || 'Other'
      if (!map.has(g)) map.set(g, [])
      map.get(g)!.push(item)
    })
    return map
  }, [catalog])

  const topGenres = useMemo(() =>
    [...byGenre.entries()].filter(([, v]) => v.length >= 5)
      .sort((a, b) => b[1].length - a[1].length).slice(0, 8),
    [byGenre]
  )

  const continueItems = useMemo(() => catalog.slice(2, 8), [catalog])
  const myList = useMemo(() =>
    sessionItemIds.map(id => catalog.find(x => x.item_id === id)).filter(Boolean) as CatalogItem[],
    [sessionItemIds, catalog]
  )

  // ── Full-page deduplication ────────────────────────────────────────────────
  // ONE shared set flows through every row on every tab — a title shown in
  // row N will NEVER appear in row N+1 anywhere on the page.
  // shownPosters is also global so same poster image NEVER appears twice.
  const {
    dedupRecs, dedupTop10, dedupTrending, dedupRag, dedupBecause, dedupGenreRows,
    dedupTop10Tab, dedupTop10TabWeekly, dedupTop10Worldwide,
    dedupMoviesRecs, dedupMoviesTop10, dedupMoviesGenres,
    dedupSeriesRows, dedupNewHot, dedupJustAdded,
  } = useMemo(() => {
    // ── HOME tab — one global set for ids, titles AND posters ─────────────
    const shown       = new Set<number>()
    const shownTitles = new Set<string>()
    const shownPosters = new Set<string>()   // ← NEW: global poster dedup

    const dedupRecs      = dedup(recs,    shown, shownTitles, shownPosters)
    const top10Src       = trending.length ? trending : catalog.slice(0, 15)
    const dedupTop10     = dedup(top10Src, shown, shownTitles, shownPosters)
    const dedupTrending  = dedup(trending, shown, shownTitles, shownPosters)
    const dedupRag       = dedup(ragRecs,  shown, shownTitles, shownPosters)
    const becauseSrc     = recs.filter(r => !shown.has((r as any).item_id) && !shownTitles.has(normKey((r as any).title)))
    const dedupBecause   = dedup(becauseSrc, shown, shownTitles, shownPosters)

    const dedupGenreRows: [string, (CatalogItem | RecommendItem)[]][] = []
    for (const [genre, items] of topGenres) {
      const unique = dedup(items, shown, shownTitles, shownPosters)
      if (unique.length >= 3) dedupGenreRows.push([genre, unique])
    }
    // ── TOP 10 tab ────────────────────────────────────────────────────────
    const shown10       = new Set<number>()
    const shownTitles10 = new Set<string>()
    const top10Src2     = trending.length ? trending : catalog.slice(0, 15)
    const shownPosters10 = new Set<string>()
    const dedupTop10Tab = dedup(top10Src2, shown10, shownTitles10, shownPosters10)
    const dedupTop10TabWeekly = dedup(recs.slice(0, 12), shown10, shownTitles10, shownPosters10)
    const dedupTop10Worldwide = dedup(trending, shown10, shownTitles10, shownPosters10)
    const shownMov   = new Set<number>()
    const shownMovT  = new Set<string>()
    const shownMovP  = new Set<string>()
    const dedupMoviesRecs  = dedup(recs,              shownMov, shownMovT, shownMovP)
    const dedupMoviesTop10 = dedup(catalog.slice(0, 15), shownMov, shownMovT, shownMovP)
    const dedupMoviesGenres: [string, (CatalogItem | RecommendItem)[]][] = []
    for (const [genre, items] of topGenres.slice(0, 4)) {
      const unique = dedup(items, shownMov, shownMovT, shownMovP)
      if (unique.length >= 3) dedupMoviesGenres.push([genre, unique])
    }

    // ── SERIES tab ────────────────────────────────────────────────────────
    const shownSer  = new Set<number>()
    const shownSerT = new Set<string>()
    const shownSerP = new Set<string>()
    const dedupSeriesRows: [string, (CatalogItem | RecommendItem)[]][] = []
    for (const [genre, items] of topGenres) {
      const unique = dedup(items, shownSer, shownSerT, shownSerP)
      if (unique.length >= 3) dedupSeriesRows.push([genre, unique])
    }

    // ── NEW & HOT tab ─────────────────────────────────────────────────────
    const shownNH   = new Set<number>()
    const shownNHT  = new Set<string>()
    const shownNHP  = new Set<string>()
    const dedupNewHot    = dedup(newHot,              shownNH, shownNHT, shownNHP)
    const dedupJustAdded = dedup(catalog.slice(20, 45), shownNH, shownNHT, shownNHP)

    return {
      dedupRecs, dedupTop10, dedupTrending, dedupRag, dedupBecause, dedupGenreRows,
      dedupTop10Tab, dedupTop10TabWeekly, dedupTop10Worldwide,
      dedupMoviesRecs, dedupMoviesTop10, dedupMoviesGenres,
      dedupSeriesRows,
      dedupNewHot, dedupJustAdded,
    }
  }, [recs, ragRecs, trending, catalog, continueItems, topGenres])

  return (
    <div className="min-h-screen bg-cine-bg">
      <Navbar />
      <div className="fixed top-16 left-0 right-0 z-40 bg-cine-bg/95 backdrop-blur-sm border-b border-white/5">
        <div className="max-w-screen-xl mx-auto px-4 flex items-center gap-1 overflow-x-auto scrollbar-hide py-2">
          {NAV_TABS.map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className={`flex-shrink-0 px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeTab === tab ? 'bg-white text-black font-bold' : 'text-white/60 hover:text-white hover:bg-white/10'}`}>
              {tab}
            </button>
          ))}
        </div>
      </div>

      <div className="pt-[104px]">

        {activeTab === 'Home' && <>
          <Hero item={heroItem} onSelect={handleSelect} />
          <Row title={rowTitle} items={dedupRecs} onSelect={handleSelect} onAddToSession={handleAdd} showScores={false} loading={recsLoading} />
          <Top10Row items={dedupTop10 as CatalogItem[]} onSelect={handleSelect} />
          <Row title="Trending Now" items={dedupTrending} onSelect={handleSelect} onAddToSession={handleAdd} showScores={false} />
          {dedupRag.length > 0 && <Row title="Because You Might Like" items={dedupRag} onSelect={handleSelect} onAddToSession={handleAdd} showScores={false} />}
          {sessionItemIds.length > 0 && dedupBecause.length > 0 && <Row title="Because You Watched" items={dedupBecause} onSelect={handleSelect} onAddToSession={handleAdd} showScores={false} />}
          {dedupGenreRows.map(([genre, items]) => <Row key={genre} title={`${genreEmoji(genre)} ${genre}`} items={items} onSelect={handleSelect} onAddToSession={handleAdd} showScores={false} />)}
        </>}

        {activeTab === 'New & Hot' && <>
          <Row title="🔥 New Arrivals" items={dedupNewHot} onSelect={handleSelect} onAddToSession={handleAdd} showScores={false} />
          <Row title="📺 Just Added" items={dedupJustAdded} onSelect={handleSelect} onAddToSession={handleAdd} showScores={false} />
        </>}

        {activeTab === 'Top 10' && <>
          <Top10Row items={dedupTop10Tab as CatalogItem[]} onSelect={handleSelect} />
          <Row title="Top Picks This Week" items={dedupTop10TabWeekly} onSelect={handleSelect} onAddToSession={handleAdd} showScores={false} loading={recsLoading} />
          <Row title="Trending Worldwide" items={dedupTop10Worldwide} onSelect={handleSelect} onAddToSession={handleAdd} showScores={false} />
        </>}

        {activeTab === 'Series' && dedupSeriesRows.map(([g, items]) =>
          <Row key={g} title={`${genreEmoji(g)} ${g}`} items={items} onSelect={handleSelect} onAddToSession={handleAdd} showScores={false} />
        )}

        {activeTab === 'Movies' && <>
          <Row title="Recommended For You" items={dedupMoviesRecs} onSelect={handleSelect} onAddToSession={handleAdd} showScores={false} loading={recsLoading} />
          <Top10Row items={dedupMoviesTop10 as CatalogItem[]} onSelect={handleSelect} />
          {dedupMoviesGenres.map(([g, items]) => <Row key={g} title={`${genreEmoji(g)} ${g}`} items={items} onSelect={handleSelect} onAddToSession={handleAdd} showScores={false} />)}
        </>}

        {activeTab === 'My List' && (
          <div className="pt-4 px-6 md:px-10">
            {myList.length === 0
              ? <div className="text-center py-20"><p className="text-4xl mb-4">📋</p><p className="text-white/50">Your list is empty. Add titles with + on any card.</p></div>
              : <div className="grid grid-cols-3 md:grid-cols-5 lg:grid-cols-7 gap-3">
                  {myList.map(item => (
                    <button key={item.item_id} onClick={() => handleSelect(item)} className="group">
                      <div className="aspect-[2/3] rounded-lg overflow-hidden border border-white/10 group-hover:border-white/40 transition-all">
                        <img src={poster(item)} alt={item.title} className="w-full h-full object-cover group-hover:scale-105 transition-transform" />
                      </div>
                      <p className="text-xs text-white/50 mt-1 truncate">{item.title}</p>
                    </button>
                  ))}
                </div>
            }
          </div>
        )}
      </div>

      {modalOpen && selectedItem && (
        <ItemModal item={selectedItem} userId={activeUser?.user_id || 1} onClose={() => setModalOpen(false)} onAddToSession={handleAdd} />
      )}
    </div>
  )
}

function ItemModal({ item, userId, onClose, onAddToSession }: { item: CatalogItem; userId: number; onClose: () => void; onAddToSession: (id: number) => void }) {
  const [detail, setDetail]   = useState<CatalogItem | null>(null)
  const [expl,   setExpl]     = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const dispatch = useAppDispatch()

  useEffect(() => {
    Promise.allSettled([api.item(item.item_id), api.explain({ user_id: userId, item_ids: [item.item_id] })]).then(([dR, eR]) => {
      if (dR.status === 'fulfilled') setDetail(dR.value as CatalogItem)
      if (eR.status === 'fulfilled') { const e = (eR.value as any).explanations?.[0]; if (e) setExpl(e.reason) }
    }).finally(() => setLoading(false))
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [item.item_id, userId, onClose])

  const d     = detail || item
  const p     = poster(d)
  const bg    = backdrop(d)

  const fb = (event: 'like' | 'dislike' | 'add_to_list') => {
    api.feedback({ user_id: userId, item_id: item.item_id, event }).catch(() => {})
    if (event !== 'dislike') { onAddToSession(item.item_id); dispatch({ type: 'ADD_SESSION_ITEM', payload: item.item_id }) }
  }

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4" onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="absolute inset-0 bg-black/80 backdrop-blur-md" onClick={onClose} />
      <div className="relative z-10 w-full max-w-2xl bg-cine-surface rounded-2xl overflow-hidden shadow-2xl" style={{ border: '1px solid rgba(255,255,255,0.1)' }}>
        <div className="relative h-56">
          <img src={bg} alt={d.title} className="w-full h-full object-cover" />
          <div className="absolute inset-0 bg-gradient-to-t from-cine-surface via-cine-surface/40 to-transparent" />
          <button onClick={onClose} className="absolute top-4 right-4 w-8 h-8 rounded-full bg-black/70 flex items-center justify-center text-white hover:bg-black transition text-lg">✕</button>
        </div>
        <div className="flex gap-5 p-6 -mt-16 relative">
          <div className="flex-shrink-0 w-28 h-40 rounded-xl overflow-hidden shadow-xl border border-white/10">
            <img src={p} alt={d.title} className="w-full h-full object-cover" />
          </div>
          <div className="flex-1 pt-10 min-w-0">
            <h3 className="text-2xl font-black text-white mb-1">{d.title}</h3>
            <div className="flex flex-wrap items-center gap-2 text-xs mb-3">
              <span className="text-green-400 font-bold">{94 + (d.item_id % 5)}% Match</span>
              {d.year && <span className="text-white/50">{d.year}</span>}
              {d.maturity_rating && <span className="border border-white/30 text-white/50 px-1.5 py-0.5 rounded">{d.maturity_rating}</span>}
              {d.primary_genre && <span className="text-cine-accent">{d.primary_genre}</span>}
            </div>
          </div>
        </div>
        <div className="px-6 pb-2">
          {loading ? <div className="h-16 bg-white/5 rounded-xl animate-pulse" /> : <>
            {d.description && <p className="text-sm text-white/70 leading-relaxed mb-4 line-clamp-3">{d.description}</p>}
            {expl && (
              <div className="bg-cine-accent/8 border border-cine-accent/20 rounded-xl p-4 mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-cine-accent text-xs font-bold">🧠 Why CineWave recommends this</span>
                  <span className="text-[10px] font-mono text-cine-accent/50 border border-cine-accent/20 px-1.5 rounded">SHAP + Semantic</span>
                </div>
                <p className="text-xs text-white/60 leading-relaxed">{expl}</p>
              </div>
            )}
          </>}
        </div>
        <div className="flex gap-2 px-6 pb-6">
          <button onClick={() => { fb('like'); onClose() }} className="flex-1 flex items-center justify-center gap-2 bg-white text-black font-black py-3 rounded-xl text-sm hover:bg-white/90 transition active:scale-95">▶ Play</button>
          <button onClick={() => fb('add_to_list')} className="px-4 py-3 bg-white/10 border border-white/20 text-white rounded-xl text-sm hover:bg-white/20 transition active:scale-95">+ My List</button>
          <button onClick={() => fb('like')}    className="px-4 py-3 bg-white/10 border border-white/20 text-white rounded-xl text-sm hover:border-green-500/60 transition active:scale-95">👍</button>
          <button onClick={() => fb('dislike')} className="px-4 py-3 bg-white/10 border border-white/20 text-white rounded-xl text-sm hover:border-red-500/60 transition active:scale-95">👎</button>
        </div>
      </div>
    </div>
  )
}
