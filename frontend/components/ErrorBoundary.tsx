'use client'

import { Component, ReactNode, ErrorInfo } from 'react'

interface Props { children: ReactNode; fallback?: ReactNode }
interface State { error: Error | null }

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[CineWave ErrorBoundary]', error, info)
  }

  render() {
    if (this.state.error) {
      return this.props.fallback ?? (
        <div className="flex items-center justify-center min-h-40 p-6">
          <div className="text-center space-y-3 max-w-md">
            <div className="text-3xl">⚠️</div>
            <p className="text-sm font-semibold text-cine-text">Something went wrong</p>
            <p className="text-xs text-cine-muted font-mono">{this.state.error.message}</p>
            <button
              onClick={() => this.setState({ error: null })}
              className="px-3 py-1.5 text-xs bg-cine-accent/20 text-cine-accent border border-cine-accent/30 rounded-lg hover:bg-cine-accent/30 transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
