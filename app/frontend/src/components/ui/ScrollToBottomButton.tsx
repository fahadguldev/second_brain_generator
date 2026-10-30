import { ArrowDown } from '@phosphor-icons/react'
import { AnimatePresence, motion } from 'motion/react'

interface ScrollToBottomButtonProps {
  visible: boolean
  onClick: () => void
}

export function ScrollToBottomButton({
  visible,
  onClick,
}: ScrollToBottomButtonProps) {
  return (
    <AnimatePresence>
      {visible && (
        <motion.button
          type="button"
          onClick={onClick}
          initial={{ opacity: 0, y: 12, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 12, scale: 0.9 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          transition={{ type: 'spring', stiffness: 400, damping: 25 }}
          aria-label="Scroll to newest messages"
          title="Scroll to bottom"
          className="fixed bottom-24 right-6 z-30 inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-raised/90 px-3 py-1.5 text-xs font-medium text-ink shadow-lift backdrop-blur-xl transition-all hover:border-white/20 sm:right-10"
        >
          <motion.span
            animate={{ y: [0, 2, 0] }}
            transition={{ repeat: Infinity, duration: 1.5, ease: 'easeInOut' }}
          >
            <ArrowDown size={14} weight="bold" />
          </motion.span>
          <span>Latest</span>
        </motion.button>
      )}
    </AnimatePresence>
  )
}
