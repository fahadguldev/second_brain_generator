import { useEffect, useRef, useState } from 'react'
import { ArrowUp, Sparkle, X } from '@phosphor-icons/react'
import { AnimatePresence, motion } from 'motion/react'
import type { CallMode, CallState } from '../hooks/useVoiceCall'
import { VoiceVisualizer } from './voice/VoiceVisualizer'
import { VoiceControlsDock } from './voice/VoiceControlsDock'
import { ConvAIAgentCall } from './voice/ConvAIAgentCall'

export interface VoiceCallModalProps {
  isOpen: boolean
  callState: CallState
  duration: number
  isMuted: boolean
  currentTranscript: string
  lastAIResponse: string
  error: string | null
  provider?: string
  hasConvAIAgent?: boolean
  callMode?: CallMode
  onSwitchMode?: (mode: CallMode) => void
  onEndCall: () => void
  onToggleMute: () => void
  onInterrupt: () => void
  onSendTextMessage: (text: string) => void
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

export function VoiceCallModal({
  isOpen,
  callState,
  duration,
  isMuted,
  currentTranscript,
  lastAIResponse,
  error,
  provider = 'elevenlabs',
  hasConvAIAgent = false,
  callMode = 'brain',
  onSwitchMode,
  onEndCall,
  onToggleMute,
  onInterrupt,
  onSendTextMessage,
}: VoiceCallModalProps) {
  const [showSubtitles, setShowSubtitles] = useState(true)
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [inputText, setInputText] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (isChatOpen) {
      setTimeout(() => inputRef.current?.focus(), 60)
    }
  }, [isChatOpen])

  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => {
      const targetTag = (e.target as HTMLElement).tagName
      if (e.key === 'Escape') {
        if (isChatOpen) {
          setIsChatOpen(false)
        } else {
          onEndCall()
        }
      } else if (
        e.code === 'Space' &&
        targetTag !== 'TEXTAREA' &&
        targetTag !== 'INPUT'
      ) {
        e.preventDefault()
        onToggleMute()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isOpen, isChatOpen, onEndCall, onToggleMute])

  if (!isOpen) return null

  const handleSendMessage = (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    const trimmed = inputText.trim()
    if (!trimmed) return
    onSendTextMessage(trimmed)
    setInputText('')
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex flex-col justify-between bg-black/92 p-6 text-white backdrop-blur-2xl select-none sm:p-10"
      >
        {/* Top Header Bar */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-2xl bg-white/10 shadow-soft">
              <Sparkle size={20} weight="fill" className="text-accent" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-bold tracking-tight">Fahad Gul</h2>
                <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 font-mono text-[10px] text-emerald-300">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                  {provider === 'elevenlabs' ? 'ElevenLabs Voice' : 'Voice Mode'}
                </span>
              </div>
              <p className="font-mono text-[11px] tracking-tight text-white/50">
                1-on-1 Real-time Call
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Optional Mode Switcher if ConvAI Agent is configured */}
            {hasConvAIAgent && onSwitchMode && (
              <div className="hidden items-center rounded-full border border-white/10 bg-white/5 p-0.5 sm:flex">
                <button
                  type="button"
                  onClick={() => onSwitchMode('brain')}
                  className={`rounded-full px-2.5 py-1 text-[11px] font-medium transition-colors ${
                    callMode === 'brain' ? 'bg-white/15 text-white' : 'text-white/40 hover:text-white'
                  }`}
                >
                  Second Brain
                </button>
                <button
                  type="button"
                  onClick={() => onSwitchMode('convai')}
                  className={`rounded-full px-2.5 py-1 text-[11px] font-medium transition-colors ${
                    callMode === 'convai' ? 'bg-white/15 text-white' : 'text-white/40 hover:text-white'
                  }`}
                >
                  ElevenLabs Agent
                </button>
              </div>
            )}

            <span className="font-mono text-xs font-semibold tracking-wider text-white/70">
              {formatDuration(duration)}
            </span>

            <button
              type="button"
              onClick={() => setShowSubtitles(prev => !prev)}
              className={`rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors ${
                showSubtitles
                  ? 'border-white/30 bg-white/15 text-white'
                  : 'border-white/10 text-white/40 hover:text-white'
              }`}
            >
              Captions
            </button>

            <button
              type="button"
              onClick={onEndCall}
              aria-label="Close call"
              className="grid h-8 w-8 place-items-center rounded-full bg-white/10 text-white/70 transition-colors hover:bg-white/20 hover:text-white"
            >
              <X size={16} weight="bold" />
            </button>
          </div>
        </div>

        {/* Center Visualizer & Content */}
        {callMode === 'convai' ? (
          <ConvAIAgentCall onEndCall={onEndCall} />
        ) : (
          <>
            <div className="flex flex-1 flex-col items-center justify-center">
              <VoiceVisualizer callState={callState} isMuted={isMuted} />

              {error && (
                <p className="mt-3 text-center text-xs font-medium text-red-400">{error}</p>
              )}

              {/* Real-time Subtitles / Captions Card */}
              {showSubtitles && (
                <div className="mt-8 min-h-[64px] max-w-xl px-4 text-center">
                  {callState === 'speaking' && lastAIResponse ? (
                    <motion.p
                      key="response"
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="line-clamp-3 text-sm font-medium leading-relaxed text-white/90"
                    >
                      <span className="mr-1.5 font-semibold text-emerald-400">Fahad:</span>
                      {lastAIResponse}
                    </motion.p>
                  ) : currentTranscript ? (
                    <motion.p
                      key="transcript"
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="line-clamp-2 text-sm font-medium leading-relaxed text-white/90"
                    >
                      <span className="mr-1.5 font-semibold text-accent">You:</span>
                      {currentTranscript}
                    </motion.p>
                  ) : (
                    <p className="font-mono text-xs text-white/40">
                      {callState === 'listening'
                        ? 'Speak clearly into your microphone or tap the keyboard to type…'
                        : ''}
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Bottom Controls & In-Call Chat Drawer */}
            <div className="mx-auto w-full max-w-md">
              <AnimatePresence>
                {isChatOpen && (
                  <motion.form
                    initial={{ opacity: 0, y: 12, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 12, scale: 0.98 }}
                    transition={{ duration: 0.16 }}
                    onSubmit={handleSendMessage}
                    className="mb-3 flex items-center gap-2 rounded-2xl border border-white/20 bg-white/10 p-2 shadow-2xl backdrop-blur-xl"
                  >
                    <input
                      ref={inputRef}
                      type="text"
                      value={inputText}
                      onChange={e => setInputText(e.target.value)}
                      placeholder="Type a question directly..."
                      className="flex-1 bg-transparent px-3 py-2 text-sm text-white placeholder-white/40 outline-none"
                    />
                    <button
                      type="submit"
                      disabled={!inputText.trim()}
                      aria-label="Send message"
                      className="grid h-9 w-9 place-items-center rounded-xl bg-accent text-accent-ink transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      <ArrowUp size={18} weight="bold" />
                    </button>
                  </motion.form>
                )}
              </AnimatePresence>

              <VoiceControlsDock
                callState={callState}
                isMuted={isMuted}
                isChatOpen={isChatOpen}
                onToggleMute={onToggleMute}
                onToggleChat={() => setIsChatOpen(prev => !prev)}
                onInterrupt={onInterrupt}
                onEndCall={onEndCall}
              />
            </div>
          </>
        )}
      </motion.div>
    </AnimatePresence>
  )
}
