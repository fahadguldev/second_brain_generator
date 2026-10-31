import { useCallback, useEffect, useRef, useState } from 'react'
import type { VoiceStatus } from '../types'
import { useSpeechRecognizer } from './voice/useSpeechRecognizer'
import { useTTSAudio } from './voice/useTTSAudio'

import { API_URL } from '../lib/api'

export type CallState = 'connecting' | 'listening' | 'processing' | 'speaking'
export type CallMode = 'brain' | 'convai'

interface UseVoiceCallProps {
  onSendMessage: (text: string) => Promise<string | null>
  isThinking: boolean
}

export function useVoiceCall({ onSendMessage }: UseVoiceCallProps) {
  const [isCallActive, setIsCallActive] = useState(false)
  const [callState, setCallState] = useState<CallState>('connecting')
  const [callMode, setCallMode] = useState<CallMode>('brain')
  const [duration, setDuration] = useState(0)
  const [isMuted, setIsMuted] = useState(false)
  const [currentTranscript, setCurrentTranscript] = useState('')
  const [lastAIResponse, setLastAIResponse] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [voiceStatus, setVoiceStatus] = useState<VoiceStatus | null>(null)

  const isCallActiveRef = useRef(false)
  const callStateRef = useRef<CallState>('connecting')
  const audioStreamRef = useRef<MediaStream | null>(null)
  const timerIntervalRef = useRef<any>(null)

  const updateCallState = useCallback((newState: CallState) => {
    callStateRef.current = newState
    setCallState(newState)
  }, [])

  // Fetch voice status on mount
  useEffect(() => {
    fetch(`${API_URL}/api/voice/status`)
      .then(res => (res.ok ? res.json() : null))
      .then(data => {
        if (data) setVoiceStatus(data)
      })
      .catch(() => {})
  }, [])

  // Call duration timer
  useEffect(() => {
    if (isCallActive) {
      setDuration(0)
      timerIntervalRef.current = setInterval(() => {
        setDuration(prev => prev + 1)
      }, 1000)
    } else {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current)
    }
    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current)
    }
  }, [isCallActive])

  // TTS audio playback hook
  const { playTTS, stopAudio } = useTTSAudio({
    onEnded: () => {
      if (isCallActiveRef.current && callStateRef.current === 'speaking') {
        updateCallState('listening')
        setCurrentTranscript('')
        recognizer.start()
      }
    },
    onError: err => {
      console.error('[VoiceCall] Audio error:', err)
      setError(err)
      if (isCallActiveRef.current) {
        updateCallState('listening')
        recognizer.start()
      }
    },
  })

  // Submit speech query to Second Brain RAG
  const processSpeech = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || !isCallActiveRef.current) return
      if (callStateRef.current === 'processing') return

      updateCallState('processing')
      setError(null)

      try {
        const answer = await onSendMessage(trimmed)
        if (!isCallActiveRef.current || callStateRef.current !== 'processing') return

        if (answer) {
          updateCallState('speaking')
          setLastAIResponse(answer)
          await playTTS(answer)
        } else {
          updateCallState('listening')
          setCurrentTranscript('')
          recognizer.start()
        }
      } catch (err: any) {
        console.error('[VoiceCall] Error processing message:', err)
        setError('Failed to process message')
        updateCallState('listening')
        recognizer.start()
      }
    },
    [onSendMessage, playTTS, updateCallState],
  )

  // Speech recognizer hook with natural barge-in
  const recognizer = useSpeechRecognizer({
    onTranscriptChange: transcript => {
      if (callStateRef.current === 'listening') {
        setCurrentTranscript(transcript)
      }
    },
    onSpeechEnd: text => {
      if (isCallActiveRef.current && callStateRef.current === 'listening') {
        void processSpeech(text)
      }
    },
    onBargeIn: () => {
      if (callStateRef.current === 'speaking') {
        stopAudio()
        updateCallState('listening')
        setLastAIResponse('')
      }
    },
  })

  // In-call text drawer
  const sendTextMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || !isCallActiveRef.current) return
      if (callStateRef.current === 'processing') return

      stopAudio()
      recognizer.clearSilenceTimer()
      setCurrentTranscript(trimmed)
      updateCallState('processing')
      setError(null)

      try {
        const answer = await onSendMessage(trimmed)
        if (!isCallActiveRef.current || callStateRef.current !== 'processing') return

        if (answer) {
          updateCallState('speaking')
          setLastAIResponse(answer)
          await playTTS(answer)
        } else {
          updateCallState('listening')
          setCurrentTranscript('')
          recognizer.start()
        }
      } catch (err: any) {
        console.error('[VoiceCall] Typed message error:', err)
        setError('Failed to process message')
        updateCallState('listening')
        recognizer.start()
      }
    },
    [onSendMessage, playTTS, recognizer, stopAudio, updateCallState],
  )

  const startCall = useCallback(async () => {
    setError(null)
    setIsCallActive(true)
    isCallActiveRef.current = true
    updateCallState('connecting')
    setCurrentTranscript('')
    setLastAIResponse('')

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
      audioStreamRef.current = stream
      updateCallState('listening')
      recognizer.start()
    } catch (err: any) {
      console.error('[VoiceCall] Microphone permission error:', err)
      setError('Microphone access denied. Please allow microphone permissions.')
      setIsCallActive(false)
      isCallActiveRef.current = false
      updateCallState('connecting')
    }
  }, [recognizer, updateCallState])

  const endCall = useCallback(() => {
    isCallActiveRef.current = false
    recognizer.stop()
    stopAudio()

    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach(t => t.stop())
      audioStreamRef.current = null
    }

    if (timerIntervalRef.current) clearInterval(timerIntervalRef.current)
    setIsCallActive(false)
    updateCallState('connecting')
    setCurrentTranscript('')
    setLastAIResponse('')
    setError(null)
  }, [recognizer, stopAudio, updateCallState])

  const toggleMute = useCallback(() => {
    setIsMuted(prev => {
      const next = !prev
      recognizer.setMuted(next)
      return next
    })
  }, [recognizer])

  const interrupt = useCallback(() => {
    stopAudio()
    updateCallState('listening')
    setCurrentTranscript('')
    recognizer.start()
  }, [recognizer, stopAudio, updateCallState])

  return {
    isCallActive,
    callState,
    callMode,
    duration,
    isMuted,
    currentTranscript,
    lastAIResponse,
    error,
    voiceStatus,
    hasConvAIAgent: Boolean(voiceStatus?.has_convai_agent),
    provider: voiceStatus?.provider ?? 'elevenlabs',
    startCall,
    endCall,
    toggleMute,
    interrupt,
    sendTextMessage,
    setCallMode,
  }
}
