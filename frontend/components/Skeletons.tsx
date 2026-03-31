'use client'

export function CardSkeleton({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  const heights = { sm: 'h-48', md: 'h-60 md:h-64', lg: 'h-72 md:h-80' }
  const widths = { sm: 'w-32', md: 'w-40 md:w-44', lg: 'w-48 md:w-56' }

  return (
    <div className={`${widths[size]} flex-shrink-0`}>
      <div className={`${heights[size]} rounded-xl skeleton`} />
      <div className="mt-2 space-y-1.5">
        <div className="h-3 skeleton rounded w-3/4" />
        <div className="h-2.5 skeleton rounded w-1/2" />
      </div>
    </div>
  )
}

export function CarouselSkeleton({ count = 8, size = 'md' }: { count?: number; size?: 'sm' | 'md' | 'lg' }) {
  return (
    <div className="space-y-4">
      <div className="h-5 skeleton rounded w-48" />
      <div className="flex gap-3 overflow-hidden">
        {[...Array(count)].map((_, i) => (
          <CardSkeleton key={i} size={size} />
        ))}
      </div>
    </div>
  )
}

export function HeroSkeleton() {
  return (
    <div className="relative h-[55vh] min-h-72 w-full skeleton rounded-2xl overflow-hidden">
      <div className="absolute bottom-8 left-8 space-y-3">
        <div className="h-10 skeleton rounded w-64" />
        <div className="h-4 skeleton rounded w-48" />
        <div className="h-4 skeleton rounded w-80" />
        <div className="flex gap-3 mt-4">
          <div className="h-10 w-28 skeleton rounded-lg" />
          <div className="h-10 w-28 skeleton rounded-lg" />
        </div>
      </div>
    </div>
  )
}

export function MetricCardSkeleton() {
  return (
    <div className="cine-card rounded-xl p-5 space-y-3">
      <div className="h-3 skeleton rounded w-24" />
      <div className="h-8 skeleton rounded w-20" />
      <div className="h-2 skeleton rounded w-full" />
    </div>
  )
}

export function DetailModalSkeleton() {
  return (
    <div className="flex gap-6 p-6">
      <div className="w-40 h-60 skeleton rounded-xl flex-shrink-0" />
      <div className="flex-1 space-y-3">
        <div className="h-7 skeleton rounded w-3/4" />
        <div className="h-4 skeleton rounded w-1/2" />
        <div className="h-4 skeleton rounded w-full" />
        <div className="h-4 skeleton rounded w-4/5" />
        <div className="h-4 skeleton rounded w-full" />
      </div>
    </div>
  )
}
