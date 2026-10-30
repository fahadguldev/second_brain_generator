import { motion, useReducedMotion } from 'motion/react'

export function ThinkingDots() {
  const reduce = useReducedMotion()
  return (
    <span className="flex items-center gap-1.5 py-1" aria-label="Thinking">
      {[0, 1, 2].map(i => (
        <motion.span
          key={i}
          animate={reduce ? undefined : { opacity: [0.25, 1, 0.25], y: [0, -3, 0] }}
          transition={{ duration: 1.1, repeat: Infinity, delay: i * 0.18, ease: 'easeInOut' }}
          className="h-1.5 w-1.5 rounded-full bg-accent"
        />
      ))}
    </span>
  )
}