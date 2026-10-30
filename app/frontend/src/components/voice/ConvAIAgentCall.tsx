import { useCallback, useEffect, useState } from 'react'
import { useConversation } from '@elevenlabs/react'
import { CircleNotch, WarningCircle } from '@phosphor-icons/react'
import { VoiceVisualizer } from './VoiceVisualizer'
import { VoiceControlsDock } from './VoiceControlsDock'
import type { CallState } from '../../hooks/useVoiceCall'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

interface ConvAIAgentCallProps {
  onEndCall: () => void
}

export function ConvAIAgentCall({ onEndCall }: ConvAIAgentCallProps) {
  const [error, setError] = useState<string | null>(null)
  const [isChatOpen, setIsChatOpen] = useState(false)

  const conversation = useConversation({
    onError: err => {
      console.error('[ConvAI] Error:', err)
      setError(typeof err === 'string' ? err : 'Connection error occurred')
    },
    onDisconnect: () => {
      onEndCall()
    },
  })

  const { status, isSpeaking, startSession, endSession, setMuted, isMuted, sendUserMessage } =
    conversation

  // Start ConvAI session with signed URL on mount
  useEffect(() => {
    let cancelled = false
    const initSession = async () => {
      try {
        const res = await fetch(`${API_URL}/api/voice/convai/signed-url`)
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}))
          throw new Error(errData.detail || 'Could not retrieve signed URL')
        }
        const data = await res.json()
        if (cancelled) return
        await startSession({
          signedUrl: data.signed_url,
        })
      } catch (err: any) {
        if (!cancelled) {
          console.error('[ConvAI] Init error:', err)
          setError(err?.message || 'Failed to start ElevenLabs session')
        }
      }
    }
    void initSession()
    return () => {
      cancelled = true
      void endSession()
    }
  }, [])

  const handleEnd = useCallback(() => {
    void endSession()
    onEndCall()
  }, [endSession, onEndCall])

  const callState: CallState =
    status === 'connecting'
      ? 'connecting'
      : isSpeaking
      ? 'speaking'
      : 'listening'

  return (
    <div className="flex flex-1 flex-col items-center justify-between">
      <div className="flex flex-1 flex-col items-center justify-center">
        {status === 'connecting' ? (
          <div className="flex flex-col items-center gap-3">
            <CircleNotch size={56} className="animate-spin text-accent" />
            <span className="font-mono text-xs text-white/60">Connecting to ElevenLabs Agent…</span>
          </div>
        ) : (
          <VoiceVisualizer callState={callState} isMuted={isMuted} />
        )}

        {error && (
          <div className="mt-4 flex items-center gap-1.5 rounded-full border border-red-500/30 bg-red-500/10 px-3 py-1 text-xs text-red-300">
            <WarningCircle size={14} />
            <span>{error}</span>
          </div>
        )}
      </div>

      <div className="mx-auto w-full max-w-md">
        <VoiceControlsDock
          callState={callState}
          isMuted={isMuted}
          isChatOpen={isChatOpen}
          onToggleMute={() => setMuted(!isMuted)}
          onToggleChat={() => setIsChatOpen(prev => !prev)}
          onInterrupt={() => {}}
          onEndCall={handleEnd}
        />
      </div>
    </div>
  )
}
