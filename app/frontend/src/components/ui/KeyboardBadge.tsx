interface KeyboardBadgeProps {
  keys: string[]
  className?: string
}

export function KeyboardBadge({ keys, className = '' }: KeyboardBadgeProps) {
  return (
    <span className={`inline-flex items-center gap-0.5 ${className}`}>
      {keys.map((k, i) => (
        <kbd
          key={i}
          className="inline-flex min-h-[18px] min-w-[18px] items-center justify-center rounded border border-line-strong/60 bg-raised/80 px-1 font-mono text-[10px] font-semibold text-muted shadow-sm"
        >
          {k}
        </kbd>
      ))}
    </span>
  )
}
