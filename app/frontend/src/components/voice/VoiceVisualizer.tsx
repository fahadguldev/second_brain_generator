import { CircleNotch, MicrophoneSlash, Waveform } from '@phosphor-icons/react'
import { motion } from 'motion/react'
import type { CallState } from '../../hooks/useVoiceCall'
import { AudioWaveform } from '../ui/AudioWaveform'

interface VoiceVisualizerProps {
  callState: CallState
  isMuted: boolean
}

export function VoiceVisualizer({ callState, isMuted }: VoiceVisualizerProps) {
  const getStatusLabel = () => {
    switch (callState) {
      case 'connecting':
        return 'Connecting to GanjiSwag…'
      case 'listening':
        return isMuted ? 'Microphone muted' : 'Listening to you…'
      case 'processing':
        return 'GanjiSwag is thinking…'
      case 'speaking':
        return 'GanjiSwag is speaking…'
    }
  }

  const getStatusColor = () => {
    switch (callState) {
      case 'connecting':
        return 'text-white/60 border-white/20'
      case 'listening':
        return isMuted ? 'text-red-400 border-red-400/40' : 'text-accent border-accent/40'
      case 'processing':
        return 'text-amber-400 border-amber-400/40'
      case 'speaking':
        return 'text-emerald-400 border-emerald-400/40'
    }
  }

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="relative flex h-60 w-60 items-center justify-center sm:h-76 sm:w-76">
        {/* Outer Pulsating Ambient Glow Aura */}
        <motion.div
          animate={{
            scale:
              callState === 'speaking'
                ? [1, 1.25, 1.05]
                : callState === 'listening' && !isMuted
                ? [1, 1.15, 1]
                : 1,
            opacity:
              callState === 'speaking'
                ? [0.35, 0.7, 0.35]
                : callState === 'listening' && !isMuted
                ? [0.25, 0.55, 0.25]
                : 0.15,
          }}
          transition={{
            repeat: Infinity,
            duration: callState === 'speaking' ? 1.5 : 2.4,
            ease: 'easeInOut',
          }}
          className={`absolute inset-0 rounded-full blur-3xl transition-colors duration-500 ${
            callState === 'speaking'
              ? 'bg-emerald-500/35'
              : callState === 'processing'
              ? 'bg-amber-500/30'
              : isMuted
              ? 'bg-red-500/25'
              : 'bg-accent/30'
          }`}
        />

        {/* Concentric Ripple Ring */}
        <motion.div
          animate={{
            scale: callState === 'speaking' ? [1, 1.14, 1] : [1, 1.05, 1],
            opacity: [0.3, 0.7, 0.3],
          }}
          transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}
          className={`absolute inset-4 rounded-full border shadow-2xl transition-colors duration-500 ${
            callState === 'speaking'
              ? 'border-emerald-400/40 bg-emerald-500/10'
              : callState === 'processing'
              ? 'border-amber-400/40 bg-amber-500/10'
              : 'border-accent/40 bg-accent/10'
          }`}
        />

        {/* Core Center Floating Orb */}
        <motion.div
          whileHover={{ scale: 1.04 }}
          transition={{ type: 'spring', stiffness: 350, damping: 20 }}
          className="relative grid h-36 w-36 place-items-center rounded-full border border-white/30 bg-white/10 shadow-lift backdrop-blur-xl sm:h-44 sm:w-44"
        >
          {callState === 'processing' ? (
            <div className="flex flex-col items-center gap-2">
              <CircleNotch size={48} className="animate-spin text-amber-400" />
            </div>
          ) : callState === 'speaking' ? (
            <div className="flex flex-col items-center gap-2">
              <AudioWaveform
                isPlaying={true}
                barCount={7}
                size="md"
                color="bg-emerald-400"
              />
            </div>
          ) : isMuted ? (
            <MicrophoneSlash size={44} weight="fill" className="text-red-400" />
          ) : (
            <div className="flex flex-col items-center gap-2">
              <AudioWaveform
                isPlaying={true}
                barCount={5}
                size="sm"
                color="bg-accent"
              />
              <Waveform size={36} weight="duotone" className="text-accent" />
            </div>
          )}
        </motion.div>
      </div>

      {/* Reactive Status Badge */}
      <motion.div
        layout
        className={`mt-6 inline-flex items-center gap-2 rounded-full border bg-white/5 px-4 py-1.5 text-xs font-semibold backdrop-blur-md ${getStatusColor()}`}
      >
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-75" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-current" />
        </span>
        <span>{getStatusLabel()}</span>
      </motion.div>
    </div>
  )
}
