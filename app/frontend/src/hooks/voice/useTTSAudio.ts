import { useCallback, useEffect, useRef, useState } from 'react'

import { API_URL } from '../../lib/api'

interface UseTTSAudioOptions {
  onEnded: () => void
  onError: (error: string) => void
}

export function useTTSAudio({ onEnded, onError }: UseTTSAudioOptions) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const currentUrlRef = useRef<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)

  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }
    if (currentUrlRef.current) {
      URL.revokeObjectURL(currentUrlRef.current)
      currentUrlRef.current = null
    }
    setIsPlaying(false)
  }, [])

  useEffect(() => {
    return () => {
      stopAudio()
    }
  }, [stopAudio])

  const playTTS = useCallback(
    async (text: string) => {
      stopAudio()
      try {
        const res = await fetch(`${API_URL}/api/tts`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        })

        if (!res.ok) {
          let detail = 'Voice synthesis failed'
          try {
            const errJson = await res.json()
            detail = errJson.detail || detail
          } catch {
            // ignore
          }
          throw new Error(detail)
        }

        const blob = await res.blob()
        const objectUrl = URL.createObjectURL(blob)
        currentUrlRef.current = objectUrl

        let audio = audioRef.current
        if (!audio) {
          audio = new Audio()
          audioRef.current = audio
        }
        audio.src = objectUrl

        audio.onended = () => {
          stopAudio()
          onEnded()
        }

        audio.onerror = () => {
          stopAudio()
          onError('Playback error occurred')
        }

        setIsPlaying(true)
        await audio.play()
      } catch (err: any) {
        stopAudio()
        onError(err?.message || 'Failed to play voice')
      }
    },
    [onEnded, onError, stopAudio],
  )

  return {
    playTTS,
    stopAudio,
    isPlaying,
  }
}
