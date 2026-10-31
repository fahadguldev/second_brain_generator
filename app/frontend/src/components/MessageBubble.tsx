import React from 'react'
import {
  CircleNotch,
  Pause,
  SpeakerHigh,
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
  keyPrefix = 'c'
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
              const el = document.getElementById(`source-ref-${idx}`)
              if (el) {
                el.scrollIntoView({ behavior: 'smooth', block: 'center' })
                el.classList.add('ring-2', 'ring-blue-400', 'bg-blue-500/10')
                setTimeout(() => {
                  el.classList.remove('ring-2', 'ring-blue-400', 'bg-blue-500/10')
                }, 2000)
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
      renderNodeWithCitations(child, sources, `${keyPrefix}-${i}`)
    )
  }

  if (React.isValidElement(node) && node.props?.children) {
    return React.cloneElement(node, {
      ...node.props,
      children: renderNodeWithCitations(
        node.props.children,
        sources,
        `${keyPrefix}-el`
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
  const reduce = useReducedMotion()
  const isUser = msg.role === 'user'

  if (msg.pending) {
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
                {renderNodeWithCitations(children, sources)}
              </h1>
            ),
            h2: ({ children }) => (
              <h2 className="mt-3.5 mb-2 text-lg font-bold tracking-tight text-ink sm:text-xl">
                {renderNodeWithCitations(children, sources)}
              </h2>
            ),
            h3: ({ children }) => (
              <h3 className="mt-3 mb-1.5 text-[15.5px] font-semibold text-accent sm:text-[16.5px]">
                {renderNodeWithCitations(children, sources)}
              </h3>
            ),
            p: ({ children }) => (
              <p className="my-2 leading-relaxed text-ink/90">
                {renderNodeWithCitations(children, sources)}
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
                {renderNodeWithCitations(children, sources)}
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
      </div>

      {/* Embedded Sources & References */}
      {sources && sources.length > 0 && (
        <>
          <VideoRecommendations sources={sources} />
          <ReferenceSection sources={sources} />
        </>
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
