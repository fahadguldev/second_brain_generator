import type { ChatMessage } from '../../types'

export function MetaRow({ info }: { info: NonNullable<ChatMessage['info']> }) {
  const parts: string[] = []
  if (typeof info.latency === 'number') parts.push(`${info.latency.toFixed(1)}s`)
  if (info.model) parts.push(info.model)
  if (parts.length === 0) return null

  return (
    <span className="font-mono text-[10px] text-muted/60">
      {parts.join(' · ')}
    </span>
  )
}
