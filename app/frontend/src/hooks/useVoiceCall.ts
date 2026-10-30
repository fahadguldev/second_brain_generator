import { useCallback, useEffect, useRef, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

export type CallState = 'connecting' | 'listening' | 'processing' | 'speaking'

interface IWindow extends Window {
  webkitSpeechRecognition?: any
  SpeechRecognition?: any
}

interface UseVoiceCallProps {
  onSendMessage: (text: string) => Promise<string | null>
  isThinking: boolean
}

export function useVoiceCall({ onSendMessage, isThinking: _isThinking }: UseVoiceCallProps) {
  const [isCallActive, setIsCallActive] = useState(false)
  const [callState, setCallState] = useState<CallState>('connecting')
  const [duration, setDuration] = useState(0)
  const [isMuted, setIsMuted] = useState(false)
  const [currentTranscript, setCurrentTranscript] = useState('')
  const [lastAIResponse, setLastAIResponse] = useState('')
  const [error, setError] = useState<string | null>(null)

  const recognitionRef = useRef<any>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const silenceTimeoutRef = useRef<any>(null)
  const timerIntervalRef = useRef<any>(null)
  const audioStreamRef = useRef<MediaStream | null>(null)

  const isCallActiveRef = useRef(false)
  const isMutedRef = useRef(false)
  const callStateRef = useRef<CallState>('connecting')

  // Synchronous state updater to avoid stale state in race conditions
  const updateCallState = useCallback((newState: CallState) => {
    callStateRef.current = newState
    setCallState(newState)
  }, [])

  useEffect(() => {
    isCallActiveRef.current = isCallActive
  }, [isCallActive])

  useEffect(() => {
    isMutedRef.current = isMuted
  }, [isMuted])

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

  const stopActiveAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }
  }, [])

  const playAIResponse = useCallback(
    async (text: string) => {
      if (!isCallActiveRef.current) return
      updateCallState('speaking')
      setLastAIResponse(text)

      try {
        console.log('[VoiceCall] Requesting TTS for answer (length:', text.length, ')')
        const res = await fetch(`${API_URL}/api/tts`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        })

        if (!res.ok) {
          let detail = 'Voice playback failed'
          try {
            const errJson = await res.json()
            detail = errJson.detail || detail
          } catch {
            // ignore
          }
          throw new Error(detail)
        }

        const blob = await res.blob()
        if (!isCallActiveRef.current || callStateRef.current !== 'speaking') {
          console.log('[VoiceCall] Barge-in or end occurred while TTS was downloading; skipping.')
          return
        }

        const objectUrl = URL.createObjectURL(blob)
        let audio = audioRef.current
        if (!audio) {
          audio = new Audio()
          audioRef.current = audio
        }
        audio.src = objectUrl

        audio.onended = () => {
          URL.revokeObjectURL(objectUrl)
          // Audio finished: return smoothly to listening for the user!
          if (isCallActiveRef.current && callStateRef.current === 'speaking') {
            updateCallState('listening')
            setCurrentTranscript('')
          }
        }

        audio.onerror = (e) => {
          console.error('[VoiceCall] Audio error:', e)
          URL.revokeObjectURL(objectUrl)
          if (isCallActiveRef.current) {
            updateCallState('listening')
          }
        }

        await audio.play()
      } catch (err: any) {
        console.error('[VoiceCall] Voice playback failed:', err)
        setError(err?.message || 'Could not synthesize voice')
        setTimeout(() => {
          if (isCallActiveRef.current && callStateRef.current === 'speaking') {
            updateCallState('listening')
          }
        }, 1500)
      }
    },
    [updateCallState],
  )

  const processUserSpeech = useCallback(
    async (textToSubmit: string) => {
      const trimmed = textToSubmit.trim()
      if (!trimmed || !isCallActiveRef.current) return
      // Prevent duplicate submissions if already thinking
      if (callStateRef.current === 'processing') return

      if (silenceTimeoutRef.current) {
        clearTimeout(silenceTimeoutRef.current)
        silenceTimeoutRef.current = null
      }

      updateCallState('processing')
      setError(null)

      try {
        console.log('[VoiceCall] Submitting spoken query to RAG:', trimmed)
        const answer = await onSendMessage(trimmed)
        if (!isCallActiveRef.current) return

        // If user hung up or explicitly interrupted, don't speak
        if ((callStateRef.current as CallState) !== 'processing') {
          console.log('[VoiceCall] State changed during RAG generation:', callStateRef.current)
          return
        }

        if (answer) {
          console.log('[VoiceCall] RAG answered, starting TTS playback...')
          await playAIResponse(answer)
        } else {
          console.warn('[VoiceCall] RAG returned empty or null answer')
          updateCallState('listening')
          setCurrentTranscript('')
        }
      } catch (err: any) {
        console.error('[VoiceCall] Processing speech error:', err)
        setError('Failed to process message')
        updateCallState('listening')
      }
    },
    [onSendMessage, playAIResponse, updateCallState],
  )

  // Direct In-Call Text Input: skips STT completely!
  const sendTextMessage = useCallback(
    async (textToSubmit: string) => {
      const trimmed = textToSubmit.trim()
      if (!trimmed || !isCallActiveRef.current) return
      if (callStateRef.current === 'processing') return

      // Stop any speech playing currently (immediate barge-in)
      stopActiveAudio()

      // Cancel any pending speech recognition silence timer
      if (silenceTimeoutRef.current) {
        clearTimeout(silenceTimeoutRef.current)
        silenceTimeoutRef.current = null
      }

      // Update transcript and enter processing state (bypassing STT entirely!)
      setCurrentTranscript(trimmed)
      updateCallState('processing')
      setError(null)

      try {
        console.log('[VoiceCall] Submitting typed query to RAG (STT bypassed):', trimmed)
        const answer = await onSendMessage(trimmed)
        if (!isCallActiveRef.current) return
        if ((callStateRef.current as CallState) !== 'processing') return

        if (answer) {
          console.log('[VoiceCall] RAG answered typed query, starting TTS playback...')
          await playAIResponse(answer)
        } else {
          updateCallState('listening')
          setCurrentTranscript('')
        }
      } catch (err: any) {
        console.error('[VoiceCall] Processing typed message error:', err)
        setError('Failed to process message')
        updateCallState('listening')
      }
    },
    [onSendMessage, playAIResponse, stopActiveAudio, updateCallState],
  )

  // Voice recognition loop (remains continuously hot to enable natural voice barge-in)
  const startRecognition = useCallback(() => {
    if (!isCallActiveRef.current || isMutedRef.current) return
    const win = typeof window !== 'undefined' ? (window as unknown as IWindow) : null
    const SpeechRecognition = win?.SpeechRecognition || win?.webkitSpeechRecognition

    if (!SpeechRecognition) {
      setError('Browser speech recognition not supported.')
      return
    }

    try {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort()
        } catch {
          // ignore
        }
      }

      const recognition = new SpeechRecognition()
      recognition.continuous = true
      recognition.interimResults = true
      recognition.lang = 'en-US'

      recognition.onstart = () => {
        // Recognition started
      }

      recognition.onresult = (event: any) => {
        if (isMutedRef.current || !isCallActiveRef.current) return

        // While Fahad is thinking ('processing'), IGNORE incoming audio to prevent double-submitting
        if (callStateRef.current === 'processing') {
          return
        }

        let interimText = ''
        let finalText = ''

        for (let i = 0; i < event.results.length; i++) {
          const res = event.results[i]
          if (res.isFinal) {
            finalText += res[0].transcript + ' '
          } else {
            interimText += res[0].transcript
          }
        }

        const combined = (finalText + interimText).trim()

        if (combined) {
          // NATURAL BARGE-IN: If AI is speaking and user speaks, cut off immediately!
          if (callStateRef.current === 'speaking') {
            stopActiveAudio()
            updateCallState('listening')
            setLastAIResponse('')
          }

          if (callStateRef.current === 'listening') {
            setCurrentTranscript(combined)

            // Reset silence timer on every spoken word
            if (silenceTimeoutRef.current) clearTimeout(silenceTimeoutRef.current)
            silenceTimeoutRef.current = setTimeout(() => {
              // User paused for 1.2s: auto-submit to the brain!
              if (combined.trim() && isCallActiveRef.current && callStateRef.current === 'listening') {
                void processUserSpeech(combined.trim())
              }
            }, 1200)
          }
        }
      }

      recognition.onerror = (event: any) => {
        if (event.error === 'no-speech') return
        console.warn('Speech recognition notice:', event.error)
      }

      recognition.onend = () => {
        // Auto-restart recognition so mic never drops out during the call
        if (isCallActiveRef.current && !isMutedRef.current) {
          try {
            recognition.start()
          } catch {
            // ignore collision
          }
        }
      }

      recognitionRef.current = recognition
      recognition.start()
    } catch (err) {
      console.warn('Could not start recognition:', err)
    }
  }, [processUserSpeech, stopActiveAudio, updateCallState])

  const stopRecognition = useCallback(() => {
    if (silenceTimeoutRef.current) {
      clearTimeout(silenceTimeoutRef.current)
      silenceTimeoutRef.current = null
    }
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop()
      } catch {
        // ignore
      }
    }
  }, [])

  const startCall = useCallback(async () => {
    setError(null)
    setIsCallActive(true)
    isCallActiveRef.current = true
    updateCallState('connecting')
    setCurrentTranscript('')
    setLastAIResponse('')

    try {
      // Hardware Acoustic Echo Cancellation + Noise Suppression
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      audioStreamRef.current = stream

      // Pre-warm audio element for mobile browser autoplay unlock
      const dummyAudio = new Audio()
      dummyAudio.play().catch(() => {})

      updateCallState('listening')
      startRecognition()
    } catch (err: any) {
      console.error('Microphone access denied:', err)
      setError('Microphone access denied. Please allow microphone permissions.')
      setIsCallActive(false)
      isCallActiveRef.current = false
      updateCallState('connecting')
    }
  }, [startRecognition, updateCallState])

  const endCall = useCallback(() => {
    isCallActiveRef.current = false
    stopRecognition()
    stopActiveAudio()

    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach(t => t.stop())
      audioStreamRef.current = null
    }

    if (silenceTimeoutRef.current) {
      clearTimeout(silenceTimeoutRef.current)
      silenceTimeoutRef.current = null
    }
    if (timerIntervalRef.current) clearInterval(timerIntervalRef.current)

    setIsCallActive(false)
    updateCallState('connecting')
    setCurrentTranscript('')
    setLastAIResponse('')
    setError(null)
  }, [stopActiveAudio, stopRecognition, updateCallState])

  const toggleMute = useCallback(() => {
    setIsMuted(prev => {
      const next = !prev
      isMutedRef.current = next
      if (next) {
        stopRecognition()
      } else {
        startRecognition()
      }
      return next
    })
  }, [startRecognition, stopRecognition])

  const interrupt = useCallback(() => {
    stopActiveAudio()
    updateCallState('listening')
    setCurrentTranscript('')
    startRecognition()
  }, [startRecognition, stopActiveAudio, updateCallState])

  return {
    isCallActive,
    callState,
    duration,
    isMuted,
    currentTranscript,
    lastAIResponse,
    error,
    startCall,
    endCall,
    toggleMute,
    interrupt,
    sendTextMessage,
  }
}
