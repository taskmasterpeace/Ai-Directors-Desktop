interface LtxLogoProps {
  className?: string
}

export function LtxLogo({ className = "h-6" }: LtxLogoProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 128 128"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
    >
      {/* Palette shape */}
      <path
        d="M64 12C34 12 10 36 10 64c0 28 24 52 54 52 6 0 10-4 10-9 0-2.5-1-4.5-2-6-1-1.5-2-3.5-2-6 0-5 4-9 9-9h11c17 0 30-13 30-30C120 32 96 12 64 12Z"
        fill="#8B4513"
      />
      {/* Clapperboard stripe on top */}
      <rect x="20" y="8" width="88" height="16" rx="4" fill="#2A2A2A" />
      <rect x="28" y="10" width="8" height="12" rx="1" fill="white" transform="rotate(-15 32 16)" />
      <rect x="44" y="10" width="8" height="12" rx="1" fill="white" transform="rotate(-15 48 16)" />
      <rect x="60" y="10" width="8" height="12" rx="1" fill="white" transform="rotate(-15 64 16)" />
      <rect x="76" y="10" width="8" height="12" rx="1" fill="white" transform="rotate(-15 80 16)" />
      <rect x="92" y="10" width="8" height="12" rx="1" fill="white" transform="rotate(-15 96 16)" />
      {/* Paint blobs */}
      <circle cx="38" cy="42" r="8" fill="#E74C3C" />
      <circle cx="58" cy="36" r="7" fill="#F39C12" />
      <circle cx="78" cy="36" r="7" fill="#3498DB" />
      <circle cx="94" cy="46" r="7" fill="#2ECC71" />
      <circle cx="98" cy="66" r="6" fill="#9B59B6" />
      <circle cx="34" cy="62" r="6" fill="#E67E22" />
      {/* Thumb hole */}
      <circle cx="56" cy="80" r="12" fill="#1a1a1a" />
      <circle cx="56" cy="80" r="9" fill="#111" />
    </svg>
  )
}
