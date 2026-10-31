import { useCallback, useEffect, useRef, useState } from 'react'

import { API_URL } from '../lib/api'

interface IWindow extends Window {
  webkitSpeechRecognition?: any
  SpeechRecognition?: any
}

export function useVoiceInput({
  onTranscript,
}: {
  onTranscript: (text: string) => void
}) {
  const [isListening, setIsListening] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [isSupported, setIsSupported] = useState(false)

  const recognitionRef = useRef<any>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  useEffect(() => {
    const win = typeof window !== 'undefined' ? (window as unknown as IWindow) : null
    const SpeechRecognition = win?.SpeechRecognition || win?.webkitSpeechRecognition
    const hasMediaRecorder = typeof navigator !== 'undefined' && Boolean(navigator.mediaDevices?.getUserMedia)

    setIsSupported(Boolean(SpeechRecognition || hasMediaRecorder))

    if (SpeechRecognition) {
      const recognition = new SpeechRecognition()
      recognition.continuous = false
      recognition.interimResults = true
      recognition.lang = 'en-US'

      recognition.onresult = (event: any) => {
        let currentTranscript = ''
        for (let i = event.resultIndex; i < event.results.length; i++) {
          currentTranscript += event.results[i][0].transcript
        }
        if (currentTranscript.trim()) {
          onTranscript(currentTranscript)
        }
      }

      recognition.onerror = (event: any) => {
        console.warn('Speech recognition fallback needed:', event.error)
        setIsListening(false)
      }

      recognition.onend = () => {
        setIsListening(false)
      }

      recognitionRef.current = recognition
    }
  }, [onTranscript])

  // Record audio through MediaRecorder and send to Fish Audio ASR (/api/stt)
  const transcribeWithFishAudio = useCallback(async (audioBlob: Blob) => {
    setIsTranscribing(true)
    try {
      const formData = new FormData()
      formData.append('file', audioBlob, 'speech.webm')

      const res = await fetch(`${API_URL}/api/stt`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) throw new Error('Fish Audio ASR failed')
      const data = await res.json()
      if (data.text?.trim()) {
        onTranscript(data.text.trim())
      }
    } catch (err) {
      console.warn('Backend STT failed:', err)
    } finally {
      setIsTranscribing(false)
    }
  }, [onTranscript])

  const startMediaRecording = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) return
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      audioChunksRef.current = []
      const recorder = new MediaRecorder(stream)

      recorder.ondataavailable = e => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        stream.getTracks().forEach(t => t.stop())
        // If web speech wasn't available, transcribe via Fish Audio ASR
        if (!recognitionRef.current && audioBlob.size > 0) {
          void transcribeWithFishAudio(audioBlob)
        }
      }

      mediaRecorderRef.current = recorder
      recorder.start()
      setIsListening(true)
    } catch (err) {
      console.warn('Microphone permission denied:', err)
      setIsListening(false)
    }
  }, [transcribeWithFishAudio])

  const stopMediaRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current = null
    }
    setIsListening(false)
  }, [])

  const toggleListening = useCallback(() => {
    if (isListening) {
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
      stopMediaRecording()
      setIsListening(false)
    } else {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.start()
          setIsListening(true)
        } catch {
          // If browser speech recognition failed to start, use MediaRecorder
          void startMediaRecording()
        }
      } else {
        void startMediaRecording()
      }
    }
  }, [isListening, startMediaRecording, stopMediaRecording])

  return {
    isListening,
    isTranscribing,
    isSupported,
    toggleListening,
  }
}
