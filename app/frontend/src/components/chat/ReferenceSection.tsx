import { useState } from 'react'
import {
  ArrowSquareOut,
  CaretDown,
  Clock,
  Play,
  Quotes,
  VideoCamera,
} from '@phosphor-icons/react'
import { AnimatePresence, motion } from 'motion/react'
import type { SourceCitation, SourceItem } from '../../types'

interface NormalizedCitation {
  index: number
  episode: string
  title: string
  timestamp?: string
  videoUrl?: string
  scorePct?: number
  excerpt?: string
}

function normalizeSource(source: SourceItem, fallbackIndex: number): NormalizedCitation {
  const c: SourceCitation = source.citation || source.payload?.citation || {}
  const payload = source.payload || {}
  const meta = payload.metadata || source.metadata || {}
  const rawText = payload.text || source.text || ''

  // Fallback parsing if backend returned raw point without citation
  let episode = c.episode
  let title = c.title
  let timestamp = c.timestamp
  let videoUrl = c.video_url || meta.source?.url

  if (!title || !episode) {
    const fn = meta.source?.file || meta.source?.file_name || ''
    const match = fn.match(/(?:Question|Episode)_(\d+)_(.+?)_([A-Za-z0-9_-]{11})\.txt/i)
    if (match) {
      if (!episode) episode = `Question #${parseInt(match[1], 10)}`
      if (!title) title = match[2].replace(/_/g, ' ')
      if (!videoUrl) videoUrl = `https://www.youtube.com/watch?v=${match[3]}`
    }
  }

  const scoreVal = c.score_pct ?? (source.score ? Math.round(source.score * 100) : undefined)

  return {
    index: c.index ?? fallbackIndex,
    episode: episode || `Episode #${fallbackIndex}`,
    title: title || 'Ask Ganjiswag Discussion',
    timestamp: timestamp || undefined,
    videoUrl: videoUrl || undefined,
    scorePct: scoreVal,
    excerpt: c.excerpt || rawText.slice(0, 300),
  }
}

export function ReferenceSection({
  sources,
  defaultOpen = true,
}: {
  sources: SourceItem[]
  defaultOpen?: boolean
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen)
  const [expandedExcerpts, setExpandedExcerpts] = useState<Record<number, boolean>>({})

  if (!sources || sources.length === 0) return null

  const citations = sources.map((s, idx) => normalizeSource(s, idx + 1))

  const toggleExcerpt = (idx: number) => {
    setExpandedExcerpts(prev => ({ ...prev, [idx]: !prev[idx] }))
  }

  return (
    <div className="mt-5 border-t border-white/8 pt-4">
      {/* Header with expand/collapse */}
      <button
        type="button"
        onClick={() => setIsOpen(o => !o)}
        className="group flex w-full items-center justify-between rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-white/[0.03]"
      >
        <div className="flex items-center gap-2">
          <div className="flex h-5 w-5 items-center justify-center rounded-md bg-blue-500/10 text-blue-400">
            <VideoCamera size={13} weight="bold" />
          </div>
          <span className="text-xs font-semibold tracking-wide text-ink uppercase">
            Sources & References
          </span>
          <span className="rounded-full bg-blue-500/15 px-2 py-0.5 text-[10px] font-mono font-semibold text-blue-400 border border-blue-500/25">
            {citations.length} {citations.length === 1 ? 'source' : 'sources'}
          </span>
        </div>
        <CaretDown
          size={14}
          className={`text-muted transition-transform duration-200 group-hover:text-ink ${
            isOpen ? 'rotate-180' : ''
          }`}
        />
      </button>

      {/* Citations List */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            className="mt-2.5 space-y-2 overflow-hidden"
          >
            {citations.map((item) => {
              const isExcerptOpen = Boolean(expandedExcerpts[item.index])

              return (
                <div
                  key={item.index}
                  id={`source-ref-${item.index}`}
                  className="group relative rounded-xl border border-white/8 bg-raised/40 p-3 transition-all duration-300 hover:border-blue-500/30 hover:bg-raised/70"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-start gap-2.5">
                      {/* Blue Citation Number Badge */}
                      <span className="flex h-6 min-w-6 items-center justify-center rounded-md bg-blue-500/15 border border-blue-500/30 text-[11px] font-mono font-bold text-blue-400 shadow-xs">
                        {item.index}
                      </span>

                      {/* Reference Details */}
                      <div className="space-y-1">
                        <div className="flex flex-wrap items-center gap-2">
                          {/* Episode tag */}
                          <span className="rounded bg-white/6 px-1.5 py-0.5 text-[10.5px] font-medium text-ink/80">
                            {item.episode}
                          </span>

                          {/* Timestamp badge */}
                          {item.timestamp && (
                            <span className="inline-flex items-center gap-1 rounded bg-white/4 px-1.5 py-0.5 font-mono text-[10px] text-muted">
                              <Clock size={11} className="text-muted" />
                              <span>{item.timestamp}</span>
                            </span>
                          )}

                          {/* Relevance Score */}
                          {item.scorePct !== undefined && (
                            <span className="font-mono text-[10px] text-accent/90">
                              {item.scorePct}% match
                            </span>
                          )}
                        </div>

                        {/* Title / Question */}
                        <h4 className="text-[13px] font-medium leading-snug text-ink sm:text-[13.5px]">
                          {item.title}
                        </h4>
                      </div>
                    </div>

                    {/* Actions: Watch on YouTube */}
                    {item.videoUrl && (
                      <a
                        href={item.videoUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-red-500/25 bg-red-500/10 px-2.5 py-1 text-[11px] font-medium text-red-400 transition-all hover:bg-red-500/20 hover:border-red-500/40 hover:text-red-300"
                        title="Watch video at referenced timestamp"
                      >
                        <Play size={10} weight="fill" className="text-red-400" />
                        <span>Watch</span>
                        <ArrowSquareOut size={11} className="opacity-70" />
                      </a>
                    )}
                  </div>

                  {/* Transcript Excerpt Toggle */}
                  {item.excerpt && (
                    <div className="mt-2.5 pt-2 border-t border-white/5">
                      <button
                        type="button"
                        onClick={() => toggleExcerpt(item.index)}
                        className="inline-flex items-center gap-1.5 text-[11px] text-muted transition-colors hover:text-ink/90"
                      >
                        <Quotes size={11} className="text-blue-400" />
                        <span>
                          {isExcerptOpen ? 'Hide transcript snippet' : 'Show transcript snippet'}
                        </span>
                        <CaretDown
                          size={10}
                          className={`transition-transform duration-200 ${
                            isExcerptOpen ? 'rotate-180' : ''
                          }`}
                        />
                      </button>

                      <AnimatePresence>
                        {isExcerptOpen && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            transition={{ duration: 0.2 }}
                            className="mt-2 rounded-lg bg-surface/80 p-2.5 text-[12px] leading-relaxed text-muted font-normal italic border border-white/5 overflow-hidden"
                          >
                            "{item.excerpt}..."
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  )}
                </div>
              )
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
