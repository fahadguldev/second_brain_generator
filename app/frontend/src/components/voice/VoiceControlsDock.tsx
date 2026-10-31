import {
  Keyboard,
  Microphone,
  MicrophoneSlash,
  PhoneDisconnect,
  Waveform,
} from '@phosphor-icons/react'
import { motion } from 'motion/react'
import type { CallState } from '../../hooks/useVoiceCall'

interface VoiceControlsDockProps {
  callState: CallState
  isMuted: boolean
  isChatOpen: boolean
  onToggleMute: () => void
  onToggleChat: () => void
  onInterrupt: () => void
  onEndCall: () => void
}

export function VoiceControlsDock({
  callState,
  isMuted,
  isChatOpen,
  onToggleMute,
  onToggleChat,
  onInterrupt,
  onEndCall,
}: VoiceControlsDockProps) {
  return (
    <div className="flex items-center justify-around rounded-3xl border border-white/20 bg-white/10 p-3 shadow-lift backdrop-blur-2xl">
      {/* Mute Toggle Button */}
      <motion.button
        type="button"
        onClick={onToggleMute}
        whileHover={{ scale: 1.08 }}
        whileTap={{ scale: 0.92 }}
        aria-label={isMuted ? 'Unmute microphone' : 'Mute microphone'}
        title={isMuted ? 'Unmute (Space)' : 'Mute (Space)'}
        className={`grid h-12 w-12 place-items-center rounded-2xl transition-colors ${
          isMuted
            ? 'bg-red-500 text-white shadow-lift'
            : 'bg-white/10 text-white hover:bg-white/20'
        }`}
      >
        {isMuted ? (
          <MicrophoneSlash size={22} weight="fill" />
        ) : (
          <Microphone size={22} weight="bold" />
        )}
      </motion.button>

      {/* In-Call Keyboard / Text Chat Toggle */}
      <motion.button
        type="button"
        onClick={onToggleChat}
        whileHover={{ scale: 1.08 }}
        whileTap={{ scale: 0.92 }}
        aria-label="Toggle text input"
        title={isChatOpen ? 'Close keyboard' : 'Type message instead'}
        className={`grid h-12 w-12 place-items-center rounded-2xl transition-colors ${
          isChatOpen
            ? 'bg-accent text-accent-ink shadow-soft'
            : 'bg-white/10 text-white hover:bg-white/20'
        }`}
      >
        <Keyboard size={22} weight={isChatOpen ? 'fill' : 'bold'} />
      </motion.button>

      {/* Interrupt / Cut-in Button */}
      <motion.button
        type="button"
        onClick={onInterrupt}
        disabled={callState !== 'speaking'}
        whileHover={callState === 'speaking' ? { scale: 1.08 } : undefined}
        whileTap={callState === 'speaking' ? { scale: 0.92 } : undefined}
        aria-label="Interrupt GanjiSwag"
        title="Interrupt and speak"
        className="grid h-12 w-12 place-items-center rounded-2xl bg-white/10 text-white transition-all hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-30"
      >
        <Waveform size={22} weight="bold" />
      </motion.button>

      {/* End Call Button */}
      <motion.button
        type="button"
        onClick={onEndCall}
        whileHover={{ scale: 1.08 }}
        whileTap={{ scale: 0.92 }}
        aria-label="End voice call"
        title="End call (Esc)"
        className="grid h-13 w-13 place-items-center rounded-full bg-red-500 text-white shadow-lift transition-transform hover:bg-red-600"
      >
        <PhoneDisconnect size={22} weight="fill" />
      </motion.button>
    </div>
  )
}
