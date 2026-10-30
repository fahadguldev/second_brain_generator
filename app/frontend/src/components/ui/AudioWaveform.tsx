import { motion, useReducedMotion } from 'motion/react'

interface AudioWaveformProps {
  isPlaying: boolean
  barCount?: number
  className?: string
  color?: string
  size?: 'sm' | 'md' | 'lg'
}

export function AudioWaveform({
  isPlaying,
  barCount = 5,
  className = '',
  color = 'bg-accent',
  size = 'sm',
}: AudioWaveformProps) {
  const reduce = useReducedMotion()

  const heights = {
    sm: { min: 4, max: 16, width: 2, gap: 1.5 },
    md: { min: 6, max: 24, width: 3, gap: 2 },
    lg: { min: 8, max: 36, width: 4, gap: 3 },
  }[size]

  // Predetermined rhythmic animation patterns for natural audio visualizer look
  const patterns = [
    [0.3, 0.9, 0.4, 0.8, 0.3],
    [0.7, 0.4, 1.0, 0.5, 0.6],
    [0.4, 0.8, 0.5, 0.9, 0.4],
    [0.9, 0.5, 0.8, 0.3, 0.7],
    [0.5, 1.0, 0.3, 0.7, 0.5],
  ]

  const bars = Array.from({ length: barCount }, (_, index) => {
    const pattern = patterns[index % patterns.length]
    return { id: index, pattern }
  })

  return (
    <div
      className={`inline-flex items-center justify-center ${className}`}
      style={{ gap: `${heights.gap}px`, height: `${heights.max}px` }}
      aria-hidden="true"
    >
      {bars.map(({ id, pattern }) => (
        <motion.span
          key={id}
          className={`rounded-full ${color}`}
          style={{ width: `${heights.width}px` }}
          initial={{ height: heights.min }}
          animate={
            isPlaying && !reduce
              ? {
                  height: pattern.map(p => Math.max(heights.min, p * heights.max)),
                  opacity: [0.7, 1, 0.8, 1, 0.7],
                }
              : {
                  height: heights.min,
                  opacity: 0.5,
                }
          }
          transition={
            isPlaying && !reduce
              ? {
                  duration: 0.85 + (id % 3) * 0.15,
                  repeat: Infinity,
                  repeatType: 'reverse',
                  ease: 'easeInOut',
                  delay: id * 0.08,
                }
              : {
                  duration: 0.25,
                  ease: 'easeOut',
                }
          }
        />
      ))}
    </div>
  )
}
