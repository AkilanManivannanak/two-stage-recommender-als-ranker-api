'use client'

import { createContext, useContext, useReducer, ReactNode } from 'react'
import type { DemoUser, CatalogItem, RecommendItem } from './api'

// ─── State ────────────────────────────────────────────────────────────────────

interface AppState {
  users: DemoUser[]
  activeUser: DemoUser | null
  catalog: CatalogItem[]
  catalogLoading: boolean
  recs: RecommendItem[]
  recsLoading: boolean
  apiHealthy: boolean
  apiBundle: string | null
  sessionItemIds: number[]
  shadowEnabled: boolean
  showScores: boolean
  devOverlayOpen: boolean
  sessionDrawerOpen: boolean
  sessionIntent: string | null
  rowTitle: string | null
}

const initialState: AppState = {
  users: [],
  activeUser: null,
  catalog: [],
  catalogLoading: false,
  recs: [],
  recsLoading: false,
  apiHealthy: false,
  apiBundle: null,
  sessionItemIds: [],
  shadowEnabled: false,
  showScores: false,
  devOverlayOpen: false,
  sessionDrawerOpen: false,
  sessionIntent: null,
  rowTitle: null,
}

// ─── Actions ──────────────────────────────────────────────────────────────────

type Action =
  | { type: 'SET_API_HEALTH'; payload: { healthy: boolean; bundle: string | null } }
  | { type: 'SET_USERS'; payload: DemoUser[] }
  | { type: 'SET_ACTIVE_USER'; payload: DemoUser | null }
  | { type: 'SET_CATALOG'; payload: CatalogItem[] }
  | { type: 'SET_CATALOG_LOADING'; payload: boolean }
  | { type: 'SET_RECS'; payload: RecommendItem[] }
  | { type: 'SET_RECS_LOADING'; payload: boolean }
  | { type: 'ADD_SESSION_ITEM'; payload: number }
  | { type: 'CLEAR_SESSION' }
  | { type: 'TOGGLE_SHADOW' }
  | { type: 'TOGGLE_SCORES' }
  | { type: 'TOGGLE_DEV_OVERLAY' }
  | { type: 'TOGGLE_SESSION_DRAWER' }
  | { type: 'SET_SESSION_INTENT'; payload: string | null }
  | { type: 'SET_ROW_TITLE'; payload: string | null }

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_API_HEALTH':
      return { ...state, apiHealthy: action.payload.healthy, apiBundle: action.payload.bundle }
    case 'SET_USERS':
      return { ...state, users: action.payload }
    case 'SET_ACTIVE_USER':
      return { ...state, activeUser: action.payload, sessionItemIds: [], recs: [] }
    case 'SET_CATALOG':
      return { ...state, catalog: action.payload }
    case 'SET_CATALOG_LOADING':
      return { ...state, catalogLoading: action.payload }
    case 'SET_RECS':
      return { ...state, recs: action.payload }
    case 'SET_RECS_LOADING':
      return { ...state, recsLoading: action.payload }
    case 'ADD_SESSION_ITEM':
      if (state.sessionItemIds.includes(action.payload)) return state
      return { ...state, sessionItemIds: [...state.sessionItemIds.slice(-19), action.payload] }
    case 'CLEAR_SESSION':
      return { ...state, sessionItemIds: [] }
    case 'TOGGLE_SHADOW':
      return { ...state, shadowEnabled: !state.shadowEnabled }
    case 'TOGGLE_SCORES':
      return { ...state, showScores: !state.showScores }
    case 'TOGGLE_DEV_OVERLAY':
      return { ...state, devOverlayOpen: !state.devOverlayOpen }
    case 'TOGGLE_SESSION_DRAWER':
      return { ...state, sessionDrawerOpen: !state.sessionDrawerOpen }
    case 'SET_SESSION_INTENT':
      return { ...state, sessionIntent: action.payload }
    case 'SET_ROW_TITLE':
      return { ...state, rowTitle: action.payload }
    default:
      return state
  }
}

// ─── Context ──────────────────────────────────────────────────────────────────

const StateCtx = createContext<AppState>(initialState)
const DispatchCtx = createContext<React.Dispatch<Action>>(() => {})

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState)
  return (
    <StateCtx.Provider value={state}>
      <DispatchCtx.Provider value={dispatch}>
        {children}
      </DispatchCtx.Provider>
    </StateCtx.Provider>
  )
}

export const useAppState    = () => useContext(StateCtx)
export const useAppDispatch = () => useContext(DispatchCtx)
