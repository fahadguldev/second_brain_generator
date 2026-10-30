import { useCallback, useRef } from 'react'

interface IWindow extends Window {
  webkitSpeechRecognition?: any
  SpeechRecognition?: any
}

interface UseSpeechRecognizerOptions {
  onTranscriptChange: (transcript: string) => void
  onSpeechEnd: (transcript: string) => void
  onBargeIn?: () => void
  silenceDelayMs?: number
}

export function useSpeechRecognizer({
  onTranscriptChange,
  onSpeechEnd,
  onBargeIn,
  silenceDelayMs = 1200,
}: UseSpeechRecognizerOptions) {
  const recognitionRef = useRef<any>(null)
  const silenceTimeoutRef = useRef<any>(null)
  const isRunningRef = useRef(false)
  const isMutedRef = useRef(false)

  const clearSilenceTimer = useCallback(() => {
    if (silenceTimeoutRef.current) {
      clearTimeout(silenceTimeoutRef.current)
      silenceTimeoutRef.current = null
    }
  }, [])

  const stop = useCallback(() => {
    isRunningRef.current = false
    clearSilenceTimer()
    if (recognitionRef.current) {
      try {
        recognitionRef.current.abort()
      } catch {
        // ignore
      }
      recognitionRef.current = null
    }
  }, [clearSilenceTimer])

  const start = useCallback(() => {
    stop()
    if (isMutedRef.current) return

    const win = typeof window !== 'undefined' ? (window as unknown as IWindow) : null
    const SpeechRecognition = win?.SpeechRecognition || win?.webkitSpeechRecognition
    if (!SpeechRecognition) return

    try {
      const recognition = new SpeechRecognition()
      recognition.continuous = true
      recognition.interimResults = true
      recognition.lang = 'en-US'

      recognition.onresult = (event: any) => {
        if (isMutedRef.current || !isRunningRef.current) return

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
          if (onBargeIn) onBargeIn()
          onTranscriptChange(combined)

          clearSilenceTimer()
          silenceTimeoutRef.current = setTimeout(() => {
            if (isRunningRef.current && combined.trim()) {
              onSpeechEnd(combined.trim())
            }
          }, silenceDelayMs)
        }
      }

      recognition.onerror = (event: any) => {
        if (event.error === 'no-speech') return
        console.warn('[SpeechRecognizer] Notice:', event.error)
      }

      recognition.onend = () => {
        if (isRunningRef.current && !isMutedRef.current) {
          try {
            recognition.start()
          } catch {
            // ignore collision
          }
        }
      }

      recognitionRef.current = recognition
      isRunningRef.current = true
      recognition.start()
    } catch (err) {
      console.warn('[SpeechRecognizer] Start error:', err)
    }
  }, [clearSilenceTimer, onBargeIn, onSpeechEnd, onTranscriptChange, silenceDelayMs, stop])

  const setMuted = useCallback(
    (muted: boolean) => {
      isMutedRef.current = muted
      if (muted) {
        stop()
      } else {
        start()
      }
    },
    [start, stop],
  )

  return {
    start,
    stop,
    setMuted,
    clearSilenceTimer,
  }
}
