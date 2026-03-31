'use client'

// Animated SVG Film Reel — replaces the wave emoji in the logo

export default function FilmReelLogo({ size = 48, className = '' }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 80 80"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      style={{ filter: 'drop-shadow(0 0 12px rgba(229,9,20,0.7))' }}
    >
      {/* Outer spinning ring with dashes */}
      <circle
        cx="40" cy="40" r="36"
        stroke="rgba(229,9,20,0.4)"
        strokeWidth="1.5"
        strokeDasharray="6 4"
        fill="none"
      >
        <animateTransform
          attributeName="transform"
          type="rotate"
          from="0 40 40"
          to="360 40 40"
          dur="8s"
          repeatCount="indefinite"
        />
      </circle>

      {/* Main reel circle */}
      <circle cx="40" cy="40" r="30" fill="#1a0505" stroke="#e50914" strokeWidth="2" />

      {/* Spinning reel body group */}
      <g>
        <animateTransform
          attributeName="transform"
          type="rotate"
          from="0 40 40"
          to="360 40 40"
          dur="2.4s"
          repeatCount="indefinite"
        />

        {/* 3 sprocket holes arranged in triangle, offset 120° each */}
        {[0, 120, 240].map((deg, i) => {
          const rad = (deg * Math.PI) / 180
          const cx = 40 + 16 * Math.cos(rad)
          const cy = 40 + 16 * Math.sin(rad)
          return (
            <g key={i}>
              <circle cx={cx} cy={cy} r="6.5" fill="#e50914" opacity="0.9" />
              <circle cx={cx} cy={cy} r="3.5" fill="#141414" />
            </g>
          )
        })}

        {/* 6 spokes */}
        {[0, 60, 120, 180, 240, 300].map((deg, i) => {
          const rad = (deg * Math.PI) / 180
          const x1 = 40 + 8 * Math.cos(rad)
          const y1 = 40 + 8 * Math.sin(rad)
          const x2 = 40 + 24 * Math.cos(rad)
          const y2 = 40 + 24 * Math.sin(rad)
          return (
            <line
              key={i}
              x1={x1} y1={y1} x2={x2} y2={y2}
              stroke="rgba(229,9,20,0.35)"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          )
        })}
      </g>

      {/* Center hub — static */}
      <circle cx="40" cy="40" r="6" fill="#e50914" />
      <circle cx="40" cy="40" r="2.5" fill="#141414" />

      {/* Film strip notches on outer edge — spinning */}
      <g>
        <animateTransform
          attributeName="transform"
          type="rotate"
          from="0 40 40"
          to="360 40 40"
          dur="2.4s"
          repeatCount="indefinite"
        />
        {Array.from({ length: 16 }, (_, i) => {
          const deg = (i / 16) * 360
          const rad = (deg * Math.PI) / 180
          const cx = 40 + 27.5 * Math.cos(rad)
          const cy = 40 + 27.5 * Math.sin(rad)
          return <rect key={i} x={cx - 1.5} y={cy - 1} width="3" height="2" rx="0.5" fill="rgba(229,9,20,0.5)" transform={`rotate(${deg} ${cx} ${cy})`} />
        })}
      </g>

      {/* Inner glow pulse */}
      <circle cx="40" cy="40" r="30" fill="none" stroke="rgba(229,9,20,0.15)" strokeWidth="6">
        <animate
          attributeName="stroke-opacity"
          values="0.15;0.4;0.15"
          dur="2s"
          repeatCount="indefinite"
        />
      </circle>
    </svg>
  )
}
