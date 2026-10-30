import { useState } from 'react'
import { CaretDown, FileText } from '@phosphor-icons/react'
import { AnimatePresence, motion } from 'motion/react'
import type { ChatMessage, SourceItem } from '../../types'

function sourceMetadata(item: SourceItem) {
  return item.payload?.metadata ?? item.metadata
}

function sourceLabel(item: SourceItem): string {
  const type = sourceMetadata(item)?.source?.type
  if (!type) return 'archive record'
  const short = type.replace(/_/g, ' ')
  return short.length > 28 ? short.slice(0, 28) + '…' : short
}

function scorePct(score?: number): string {
  if (typeof score !== 'number') return ''
  return `${Math.round(score * 100)}%`
}

export function SourceList({
  sources,
}: {
  sources: NonNullable<ChatMessage['info']>['sources']
}) {
  const [open, setOpen] = useState(false)
  if (!sources || sources.length === 0) return null

  return (
    <div className="mt-2.5">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="inline-flex items-center gap-1.5 text-[11px] font-medium text-muted transition-colors hover:text-ink"
      >
        <FileText size={12} />
        <span>{sources.length} sources cited</span>
        <CaretDown
          size={11}
          className={`transition-transform duration-200 ${
            open ? 'rotate-180' : ''
          }`}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="mt-2 flex flex-wrap gap-1.5 overflow-hidden"
          >
            {sources.map((s, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1.5 rounded-md border border-white/8 bg-raised/60 px-2 py-1 text-[11px] text-muted"
              >
                <span>{sourceLabel(s)}</span>
                {s.score && (
                  <span className="font-mono text-[9px] text-accent">
                    {scorePct(s.score)}
                  </span>
                )}
              </span>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
