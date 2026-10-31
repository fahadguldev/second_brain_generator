import React, { useState } from 'react'
import {
  CaretDown,
  CircleNotch,
  Pause,
  SpeakerHigh,
  VideoCamera,
  WarningCircle,
} from '@phosphor-icons/react'
import { motion, useReducedMotion } from 'motion/react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage, SourceItem } from '../types'
import { ThinkingDots } from './ThinkingDots'
import { AudioWaveform } from './ui/AudioWaveform'
import { CopyButton } from './ui/CopyButton'
import { MetaRow } from './chat/MetaRow'
import { ReferenceSection } from './chat/ReferenceSection'
import { VideoRecommendations } from './chat/VideoRecommendations'

export interface MessageBubbleProps {
  msg: ChatMessage
  onPlayVoice?: (messageId: string, text: string) => void
  isPlayingVoice?: boolean
  isVoiceLoading?: boolean
  voiceError?: string | null
}

function renderNodeWithCitations(
  node: React.ReactNode,
  sources?: SourceItem[],
  keyPrefix = 'c',
  onCitationClick?: (idx: number) => void
): React.ReactNode {
  if (typeof node === 'string') {
    const citationRegex = /\[(\d+(?:\s*,\s*\d+)*)\]/g
    const parts: React.ReactNode[] = []
    let lastIndex = 0
    let match: RegExpExecArray | null

    while ((match = citationRegex.exec(node)) !== null) {
      if (match.index > lastIndex) {
        parts.push(node.slice(lastIndex, match.index))
      }

      const indices = match[1]
        .split(',')
        .map(s => parseInt(s.trim(), 10))
        .filter(n => !isNaN(n))

      indices.forEach((idx, i) => {
        const matchedSource = sources?.find(
          s => (s.citation?.index ?? s.id) === idx
        )
        const c = matchedSource?.citation
        const tooltip = c
          ? `${c.episode ? `${c.episode}: ` : ''}${c.title || ''}${c.timestamp ? ` (${c.timestamp})` : ''}`
          : `Citation [${idx}]`

        parts.push(
          <a
            key={`${keyPrefix}-cite-${match!.index}-${idx}-${i}`}
            href={`#source-ref-${idx}`}
            onClick={(e) => {
              e.preventDefault()
              if (onCitationClick) {
                onCitationClick(idx)
              }
            }}
            title={tooltip}
            className="inline-flex items-center justify-center mx-0.5 px-1.5 py-0.2 rounded text-[11.5px] font-mono font-bold text-blue-400 bg-blue-500/15 border border-blue-500/30 hover:bg-blue-500/25 hover:border-blue-400 hover:text-blue-300 transition-all cursor-pointer align-baseline select-none no-underline shadow-xs"
          >
            {idx}
          </a>
        )
      })

      lastIndex = match.index + match[0].length
    }

    if (lastIndex < node.length) {
      parts.push(node.slice(lastIndex))
    }

    return parts.length > 0 ? parts : node
  }

  if (Array.isArray(node)) {
    return node.map((child, i) =>
      renderNodeWithCitations(child, sources, `${keyPrefix}-${i}`, onCitationClick)
    )
  }

  if (React.isValidElement(node) && node.props?.children) {
    return React.cloneElement(node, {
      ...node.props,
      children: renderNodeWithCitations(
        node.props.children,
        sources,
        `${keyPrefix}-el`,
        onCitationClick
      ),
    })
  }

  return node
}

export function MessageBubble({
  msg,
  onPlayVoice,
  isPlayingVoice = false,
  isVoiceLoading = false,
  voiceError = null,
}: MessageBubbleProps) {
  const [showSources, setShowSources] = useState(false)
  const reduce = useReducedMotion()
  const isUser = msg.role === 'user'

  const handleCitationClick = (idx: number) => {
    setShowSources(true)
    setTimeout(() => {
      const el = document.getElementById(`source-ref-${idx}`)
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' })
        el.classList.add('ring-2', 'ring-blue-400', 'bg-blue-500/10')
        setTimeout(() => {
          el.classList.remove('ring-2', 'ring-blue-400', 'bg-blue-500/10')
        }, 2000)
      }
    }, 120)
  }

  if (msg.pending && !msg.text) {
    return (
      <motion.div
        initial={reduce ? false : { opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center gap-2.5 py-2 text-muted"
      >
        <ThinkingDots />
        <span className="font-mono text-xs">Retrieving knowledge…</span>
      </motion.div>
    )
  }

  // User Message
  if (isUser) {
    return (
      <motion.div
        initial={reduce ? false : { opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
        className="group flex justify-end py-1.5"
      >
        <div className="flex max-w-[85%] items-end gap-2">
          <div className="rounded-2xl rounded-tr-xs border border-white/10 bg-raised px-4 py-2.5 text-[14px] leading-relaxed text-ink shadow-xs sm:text-[15px]">
            <p className="whitespace-pre-wrap">{msg.text}</p>
          </div>
          <div className="opacity-0 transition-opacity group-hover:opacity-100">
            <CopyButton text={msg.text} title="Copy" />
          </div>
        </div>
      </motion.div>
    )
  }

  // Assistant Message (Markdown reading canvas with inline blue citations)
  const sources = msg.info?.sources

  return (
    <motion.div
      initial={reduce ? false : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className="group flex w-full flex-col py-3"
    >
      {/* Error state */}
      {msg.error && (
        <div className="mb-2 flex items-center gap-1.5 text-xs text-danger">
          <WarningCircle size={14} weight="duotone" />
          <span>Error retrieving response</span>
        </div>
      )}

      {/* Main Response Typography with Markdown and Inline Citations */}
      <div className="text-[15px] leading-relaxed text-ink/90 sm:text-[15.5px]">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            h1: ({ children }) => (
              <h1 className="mt-4 mb-2 text-xl font-bold tracking-tight text-ink sm:text-2xl border-b border-white/8 pb-1">
                {renderNodeWithCitations(children, sources, 'h1', handleCitationClick)}
              </h1>
            ),
            h2: ({ children }) => (
              <h2 className="mt-3.5 mb-2 text-lg font-bold tracking-tight text-ink sm:text-xl">
                {renderNodeWithCitations(children, sources, 'h2', handleCitationClick)}
              </h2>
            ),
            h3: ({ children }) => (
              <h3 className="mt-3 mb-1.5 text-[15.5px] font-semibold text-accent sm:text-[16.5px]">
                {renderNodeWithCitations(children, sources, 'h3', handleCitationClick)}
              </h3>
            ),
            p: ({ children }) => (
              <p className="my-2 leading-relaxed text-ink/90">
                {renderNodeWithCitations(children, sources, 'p', handleCitationClick)}
              </p>
            ),
            ul: ({ children }) => (
              <ul className="my-2 ml-4 list-disc space-y-1 text-ink/90">
                {children}
              </ul>
            ),
            ol: ({ children }) => (
              <ol className="my-2 ml-4 list-decimal space-y-1 text-ink/90">
                {children}
              </ol>
            ),
            li: ({ children }) => (
              <li className="leading-relaxed">
                {renderNodeWithCitations(children, sources, 'li', handleCitationClick)}
              </li>
            ),
            strong: ({ children }) => (
              <strong className="font-semibold text-ink">
                {children}
              </strong>
            ),
            blockquote: ({ children }) => (
              <blockquote className="my-2.5 border-l-2 border-accent/40 pl-3 italic text-muted">
                {children}
              </blockquote>
            ),
            code: ({ children }) => (
              <code className="rounded bg-raised/80 px-1.5 py-0.5 font-mono text-[13px] text-accent">
                {children}
              </code>
            ),
            hr: () => <hr className="my-3 border-white/8" />,
          }}
        >
          {msg.text}
        </ReactMarkdown>

        {/* Real-time Streaming Pulse Indicator */}
        {msg.isStreaming && (
          <span className="inline-flex items-center ml-1 align-baseline">
            <span className="inline-block h-4 w-1.5 rounded-xs bg-accent animate-pulse" />
          </span>
        )}
      </div>

      {/* Embedded Sources & References (Hidden until clicked) */}
      {!msg.isStreaming && sources && sources.length > 0 && (
        <div className="mt-4 pt-1">
          {!showSources ? (
            <motion.button
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              type="button"
              onClick={() => setShowSources(true)}
              className="inline-flex items-center gap-2 rounded-xl border border-line bg-surface/80 hover:bg-raised px-3.5 py-2 text-xs font-medium text-muted hover:text-ink transition-all cursor-pointer group shadow-xs"
            >
              <div className="flex h-5 w-5 items-center justify-center rounded-md bg-blue-500/15 text-blue-400 group-hover:bg-blue-500/25 transition-colors">
                <VideoCamera size={13} weight="bold" />
              </div>
              <span>View {sources.length} {sources.length === 1 ? 'Source & Reference' : 'Sources & References'}</span>
              <span className="rounded-md bg-blue-500/10 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-blue-400 border border-blue-500/20">
                [1..{sources.length}]
              </span>
              <CaretDown size={13} className="text-muted/60 transition-transform group-hover:translate-y-0.5 group-hover:text-ink" />
            </motion.button>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className="space-y-4"
            >
              <div className="flex items-center justify-between border-b border-white/8 pb-2">
                <div className="flex items-center gap-2 text-xs font-semibold text-ink">
                  <VideoCamera size={14} className="text-blue-400" />
                  <span>Referenced Sources ({sources.length})</span>
                </div>
                <button
                  type="button"
                  onClick={() => setShowSources(false)}
                  className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-muted hover:bg-raised hover:text-ink transition-colors cursor-pointer"
                >
                  <CaretDown size={13} className="rotate-180" />
                  <span>Hide sources</span>
                </button>
              </div>

              <VideoRecommendations sources={sources} />
              <ReferenceSection sources={sources} defaultOpen={true} />
            </motion.div>
          )}
        </div>
      )}

      {/* Bottom Action Bar */}
      <div className="mt-3 flex items-center gap-3 text-muted">
        {onPlayVoice && (
          <button
            type="button"
            onClick={() => onPlayVoice(msg.id, msg.text)}
            disabled={isVoiceLoading}
            aria-label={
              isPlayingVoice ? 'Pause voice' : 'Listen to response'
            }
            className={`inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs transition-colors ${
              isPlayingVoice
                ? 'bg-accent/15 text-accent'
                : 'text-muted hover:bg-raised hover:text-ink'
            }`}
          >
            {isVoiceLoading ? (
              <>
                <CircleNotch size={12} className="animate-spin text-accent" />
                <span className="text-[11px]">Synthesizing…</span>
              </>
            ) : isPlayingVoice ? (
              <>
                <Pause size={12} weight="fill" className="text-accent" />
                <span className="text-[11px]">Pause</span>
                <AudioWaveform isPlaying={true} barCount={4} size="sm" />
              </>
            ) : (
              <>
                <SpeakerHigh size={12} weight="bold" />
                <span className="text-[11px]">Listen</span>
              </>
            )}
          </button>
        )}

        <CopyButton text={msg.text} title="Copy response" />

        {msg.info && <MetaRow info={msg.info} />}

        {voiceError && (
          <span className="max-w-[200px] truncate text-[10px] text-danger" title={voiceError}>
            {voiceError}
          </span>
        )}
      </div>
    </motion.div>
  )
}
