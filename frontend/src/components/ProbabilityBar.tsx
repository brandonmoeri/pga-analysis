interface Props {
  label: string
  probability: number
  color: 'green' | 'yellow' | 'red'
}

const COLOR_MAP = {
  green: {
    fill: 'bg-green-500',
    text: 'text-green-700',
    track: 'bg-green-100',
    badge: 'bg-green-50 text-green-800 border border-green-200',
  },
  yellow: {
    fill: 'bg-yellow-400',
    text: 'text-yellow-700',
    track: 'bg-yellow-100',
    badge: 'bg-yellow-50 text-yellow-800 border border-yellow-200',
  },
  red: {
    fill: 'bg-red-500',
    text: 'text-red-700',
    track: 'bg-red-100',
    badge: 'bg-red-50 text-red-800 border border-red-200',
  },
}

export default function ProbabilityBar({ label, probability, color }: Props) {
  const pct = Math.min(100, Math.max(0, probability * 100))
  const c = COLOR_MAP[color]

  // Terminal-style block bar: 20 chars wide
  const filled = Math.round(pct / 5)
  const empty = 20 - filled
  const blockBar = '█'.repeat(filled) + '░'.repeat(empty)

  return (
    <div className="font-mono">
      {/* Label row */}
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm text-gray-500 w-24">{label}</span>
        <span className={`text-xs px-2 py-0.5 rounded font-semibold ${c.badge}`}>
          {pct.toFixed(1)}%
        </span>
      </div>

      {/* Block-character bar row */}
      <div className={`text-sm tracking-widest mb-2 ${c.text}`}>{blockBar}</div>

      {/* Visual fill bar */}
      <div className={`h-2 rounded-full ${c.track} mb-4 overflow-hidden`}>
        <div
          className={`h-full rounded-full ${c.fill} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
