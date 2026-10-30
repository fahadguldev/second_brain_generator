import { Brain } from '@phosphor-icons/react'
import { motion, useReducedMotion } from 'motion/react'

export function BrandMark({ size = 36 }: { size?: number }) {
  const reduce = useReducedMotion()
  return (
    <motion.div
      initial={reduce ? false : { scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 260, damping: 20 }}
      className="grid shrink-0 place-items-center rounded-xl border border-white/10 bg-raised text-accent shadow-xs"
      style={{ width: size, height: size }}
    >
      <Brain size={Math.round(size * 0.54)} weight="duotone" />
    </motion.div>
  )
}

export function BrandBlock() {
  return (
    <div className="flex items-center gap-2.5">
      <BrandMark size={32} />
      <div className="min-w-0">
        <h1 className="font-display text-sm font-bold tracking-tight text-ink">
          Ask GanjiSwag
        </h1>
        <p className="text-[10px] font-mono tracking-tight text-muted">
          AI Second Brain
        </p>
      </div>
    </div>
  )
}
