import {
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import { ArrowUp, Microphone } from '@phosphor-icons/react'
import { motion } from 'motion/react'
import { useVoiceInput } from '../hooks/useVoiceInput'
import { AudioWaveform } from './ui/AudioWaveform'

export interface ComposerProps {
  onSend: (text: string) => void
  disabled: boolean
}

export function Composer({ onSend, disabled }: ComposerProps) {
  const [value, setValue] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)

  const handleTranscript = useCallback((transcript: string) => {
    setValue(prev => {
      const separator = prev && !prev.endsWith(' ') ? ' ' : ''
      return `${prev}${separator}${transcript}`.trimStart()
    })
  }, [])

  const { isListening, isSupported, toggleListening } = useVoiceInput({
    onTranscript: handleTranscript,
  })

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }, [value])

  const submit = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
    ref.current?.focus()
  }

  const hasText = Boolean(value.trim())

  return (
    <div className="relative border-t border-line/40 bg-bg/60 px-4 pb-4 pt-3 backdrop-blur-xl sm:px-6">
      <div className="mx-auto max-w-2xl">
        {/* Floating Obsidian Dock */}
        <div
          className={`flex items-end gap-2 rounded-2xl border bg-surface/90 p-2 shadow-lift backdrop-blur-2xl transition-all ${
            isListening
              ? 'border-danger ring-2 ring-danger/20'
              : disabled
              ? 'border-white/5 opacity-60'
              : 'border-white/10 hover:border-white/20 focus-within:border-accent/40 focus-within:ring-1 focus-within:ring-accent/20'
          }`}
        >
          <textarea
            ref={ref}
            value={value}
            onChange={e => setValue(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
            rows={1}
            placeholder={
              isListening ? 'Listening… speak now' : 'Ask Fahad anything…'
            }
            aria-label="Ask a question"
            disabled={disabled}
            className="max-h-40 min-h-[40px] flex-1 resize-none bg-transparent px-3 py-2 text-sm leading-relaxed text-ink outline-none placeholder:text-muted/60"
          />

          {/* Voice Microphone Input */}
          {isSupported && (
            <motion.button
              type="button"
              onClick={toggleListening}
              disabled={disabled}
              whileTap={{ scale: 0.92 }}
              aria-label={
                isListening ? 'Stop voice recording' : 'Start voice recording'
              }
              className={`grid h-9 w-9 shrink-0 place-items-center rounded-xl transition-all ${
                isListening
                  ? 'bg-danger text-white'
                  : 'bg-raised/70 text-muted hover:bg-raised hover:text-ink'
              } disabled:cursor-not-allowed disabled:opacity-40`}
            >
              {isListening ? (
                <AudioWaveform
                  isPlaying={true}
                  barCount={3}
                  size="sm"
                  color="bg-white"
                />
              ) : (
                <Microphone size={16} weight="bold" />
              )}
            </motion.button>
          )}

          {/* Send Button */}
          <motion.button
            type="button"
            onClick={submit}
            disabled={disabled || !hasText}
            whileTap={{ scale: 0.92 }}
            aria-label="Send message"
            className={`grid h-9 w-9 shrink-0 place-items-center rounded-xl transition-all ${
              hasText && !disabled
                ? 'bg-accent text-accent-ink shadow-xs hover:bg-accent/90'
                : 'bg-raised/50 text-muted/40 cursor-not-allowed'
            }`}
          >
            <ArrowUp size={16} weight="bold" />
          </motion.button>
        </div>

        {/* Minimalist Footnote */}
        <p className="mt-2 text-center font-mono text-[10px] text-muted/60">
          {isListening
            ? 'voice recording active'
            : 'Enter to send · Shift+Enter for new line'}
        </p>
      </div>
    </div>
  )
}
