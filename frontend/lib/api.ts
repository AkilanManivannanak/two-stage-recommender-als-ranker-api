// lib/api.ts — CineWave × Netflix-Inspired RecSys v4

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://127.0.0.1:8000'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface HealthResponse {
  ok: boolean
  bundle: string
  bundle_loaded: boolean
  tmdb_enabled: boolean
  openai_enabled: boolean
  /** v4: always false — GenAI is pre-computed, never in live path */
  genai_in_request_path: boolean
  /** v4: number of currently stale features */
  stale_features: number
  ts: string
}

export interface VersionResponse {
  bundle_dir: string
  manifest: Record<string, unknown>
}

export interface RecommendItem {
  item_id: number
  title?: string
  primary_genre?: string
  genres?: string
  poster_url?: string | null
  backdrop_url?: string | null
  description?: string | null
  year?: number
  avg_rating?: number
  score: number
  als_score?: number
  ranker_score?: number
  final_score?: number
  lts_score?: number
  exploration_slot?: boolean
  explanation?: string
  rag_reason?: string
  session_boosted?: boolean
  llm_reasoning?: string
  two_tower_score?: number
}

export interface FeatureFreshness {
  age_s: number
  status: 'fresh' | 'warn' | 'stale' | 'fallback' | 'missing'
}

export interface RecommendResponse {
  user_id: number
  k: number
  items: RecommendItem[]
  model_version?: Record<string, unknown>
  method?: string
  exploration_slots?: number
  diversity_score?: number
  freshness_watermark?: Record<string, FeatureFreshness>
}

export interface ExplainResponse {
  user_id: number
  model?: Record<string, unknown>
  explanations: Array<{
    item_id: number; reason: string
    method?: string; attribution_method?: string
  }>
}

export interface DemoUser {
  user_id: number
  recent_titles: string[]
  recent_item_ids: number[]
  primary_genre?: string
  avg_rating?: number
  n_interactions?: number
}

export interface DemoUsersResponse {
  users: DemoUser[]
}

export interface CatalogItem {
  item_id: number
  title: string
  primary_genre?: string
  genres?: string
  poster_url: string | null
  backdrop_url?: string | null
  description?: string | null
  year?: number
  avg_rating?: number
  runtime_min?: number
  maturity_rating?: string
}

export interface CatalogResponse {
  items: CatalogItem[]
}

export interface ItemDetail extends CatalogItem {
  themes?: string[]
  moods?: string[]
  semantic_tags?: string[]
  spoiler_summary?: string
  why_label?: string
  lts_score?: number
}

export interface MetricsResponse {
  n: number
  p50_ms: number
  p90_ms: number
  p95_ms: number
  p99_ms: number
  max_ms: number
  req_per_min: number
}

export interface FeedbackRequest {
  user_id: number
  item_id: number
  event: 'play' | 'like' | 'dislike' | 'add_to_list' | 'not_interested'
}

export interface SessionIntentResponse {
  user_id: number
  category: string
  confidence: number
  intent_probs?: Record<string, number>
  blend_weight: number
  session_momentum: number
  genre_shift: boolean
  session_features?: Record<string, number>
  honest_note?: string
  model?: string
}

export interface TrendingResponse {
  items: Array<{
    item_id: number; trending_score: number
    title?: string; poster_url?: string | null
  }>
}

export interface MoodResponse {
  mood: string
  items: RecommendItem[]
}

export interface RowTitleResponse {
  user_id: number
  row_title: string
  /** 'precomputed_cache' | 'rule_based_fallback' */
  source: string
  freshness: string
  plane: string
}

export interface FreshnessReport {
  feature_store: Record<string, unknown>
  fresh_store: Record<string, { age_s: number; status: string; tier: string }>
  recent_alerts: Array<{ key: string; age_s: number; action: string; ts: number }>
  launch_detector: { tracked_items: number; in_launch_window: number }
  slas: Record<string, string>
}

export interface ModelTrainMetrics {
  session_intent?: {
    trained: boolean; final_acc?: number
    final_loss?: number; epochs?: number; loss_history?: number[]
  }
  two_tower?: {
    trained: boolean; final_loss?: number
    loss_history?: number[]; training?: string
  }
  honest_note?: string
}

// ─── Helper ────────────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`)
  return res.json()
}

// ─── API surface ──────────────────────────────────────────────────────────────

export const api = {
  // Core
  health:  () => apiFetch<HealthResponse>('/healthz'),
  version: () => apiFetch<VersionResponse>('/version'),

  // Recommendations
  recommend: (body: {
    user_id: number; k: number
    session_item_ids?: number[] | null; shadow?: boolean
  }) => apiFetch<RecommendResponse>('/recommend', { method: 'POST', body: JSON.stringify(body) }),

  recommendRag:      (userId: number, k = 10) =>
    apiFetch<RecommendResponse>(`/recommend/rag/${userId}?k=${k}`),
  recommendTwoTower: (userId: number, k = 10) =>
    apiFetch<RecommendResponse>(`/recommend/two_tower/${userId}?k=${k}`),
  recommendLlm: (body: { user_id: number; k: number; session_item_ids?: number[] | null },
                 sessionContext = '', timeOfDay = 'evening') =>
    apiFetch<RecommendResponse>(
      `/recommend/llm?session_context=${encodeURIComponent(sessionContext)}&time_of_day=${timeOfDay}`,
      { method: 'POST', body: JSON.stringify(body) }
    ),

  // Explain
  explain: (body: { user_id: number; item_ids: number[] }) =>
    apiFetch<ExplainResponse>('/explain', { method: 'POST', body: JSON.stringify(body) }),

  // Users & catalog
  demoUsers:      ()           => apiFetch<DemoUsersResponse>('/users/demo'),
  popularCatalog: (k = 1200)    => apiFetch<CatalogResponse>(`/catalog/popular?k=${k}`),
  item:           (id: number) => apiFetch<ItemDetail>(`/item/${id}`),

  // Real-time
  trending:      ()            => apiFetch<TrendingResponse>('/trending'),
  sessionIntent: (uid: number) => apiFetch<SessionIntentResponse>(`/session/intent/${uid}`),

  // GenAI UX — all pre-computed/cached, never blocking
  mood:         (mood: string, userId: number) =>
    apiFetch<MoodResponse>(`/ux/mood?mood=${encodeURIComponent(mood)}&user_id=${userId}`),
  rowTitle:     (userId: number) =>
    apiFetch<RowTitleResponse>(`/ux/row_title/${userId}`),
  titleSummary: (itemId: number) =>
    apiFetch<{ item_id: number; title: string; spoiler_safe_summary: string }>(`/ux/summary/${itemId}`),

  // Page assembly
  page: (userId: number) =>
    apiFetch<{
      user_id: number
      rows: Array<{ row_title: string; items: RecommendItem[] }>
      freshness_watermark?: Record<string, unknown>
    }>(`/page/${userId}`),

  // Metrics
  metrics:         (window_sec = 3600) =>
    apiFetch<MetricsResponse>(`/metrics/latency?window_sec=${window_sec}`),
  pipelineMetrics: () =>
    apiFetch<Record<string, unknown>>('/metrics/pipeline'),

  // v4: Freshness & model diagnostics
  freshness:         () => apiFetch<FreshnessReport>('/features/freshness'),
  modelTrainMetrics: () => apiFetch<ModelTrainMetrics>('/model/train_metrics'),

  // Feedback
  feedback: (body: FeedbackRequest) =>
    apiFetch<{ ok: boolean }>('/feedback', { method: 'POST', body: JSON.stringify(body) }),

  // Eval
  evalGate:  () => apiFetch<any>('/eval/gate', { method: 'POST', body: '{}' }),
  shadow:    (userId: number, k = 10) =>
    apiFetch<Record<string, unknown>>(`/shadow/${userId}?k=${k}`),
  sliceNdcg: (sliceKey = 'primary_genre') =>
    apiFetch<Record<string, unknown>>(`/eval/slice_ndcg?slice_key=${sliceKey}`),

  // Semantic search
  semanticSearch: (q: string, topK = 10) =>
    apiFetch<{ query: string; results: RecommendItem[] }>(
      `/search/semantic?q=${encodeURIComponent(q)}&top_k=${topK}`
    ),
  similar: (itemId: number, topK = 10) =>
    apiFetch<{ item_id: number; similar: RecommendItem[] }>(`/similar/${itemId}?top_k=${topK}`),

  // Architecture
  architecture: () => apiFetch<Record<string, unknown>>('/architecture'),
}
