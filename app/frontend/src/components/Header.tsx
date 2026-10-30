import {
  List,
  MoonStars,
  PhoneCall,
  Sun,
} from '@phosphor-icons/react'
import { AnimatePresence, motion } from 'motion/react'

export interface HeaderProps {
  isThinking: boolean
  theme: 'light' | 'dark'
  onToggleTheme: () => void
  onMenu?: () => void
  onStartCall?: () => void
  onFocusSearch?: () => void
}

export function Header({
  isThinking,
  theme,
  onToggleTheme,
  onMenu,
  onStartCall,
}: HeaderProps) {
  return (
    <header className="sticky top-0 z-20 flex h-[58px] shrink-0 items-center justify-between border-b border-line/60 bg-bg/80 px-4 backdrop-blur-xl sm:px-6">
      {/* Left: Mobile Menu & Status */}
      <div className="flex items-center gap-3">
        {onMenu && (
          <button
            onClick={onMenu}
            aria-label="Open menu"
            className="grid h-8 w-8 place-items-center rounded-lg border border-line/60 bg-surface text-muted transition-colors hover:text-ink lg:hidden"
          >
            <List size={16} weight="bold" />
          </button>
        )}

        {/* Subtle Activity Beacon */}
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span
              className={`absolute inline-flex h-full w-full rounded-full ${
                isThinking ? 'animate-ping bg-accent opacity-75' : 'bg-transparent'
              }`}
            />
            <span
              className={`relative inline-flex h-2 w-2 rounded-full ${
                isThinking ? 'bg-accent' : 'bg-ok'
              }`}
            />
          </span>
          <span className="font-mono text-[11px] text-muted">
            {isThinking ? 'Searching archive…' : 'Active'}
          </span>
        </div>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-2">
        {/* 1-on-1 Voice Call Pill */}
        {onStartCall && (
          <motion.button
            type="button"
            onClick={onStartCall}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.96 }}
            aria-label="Start voice call with Fahad"
            className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-raised/80 px-3 py-1.5 text-xs font-medium text-ink shadow-xs backdrop-blur-md transition-all hover:border-accent/50 hover:bg-raised"
          >
            <PhoneCall size={13} weight="fill" className="text-accent" />
            <span>Call Fahad</span>
          </motion.button>
        )}

        {/* Theme Toggle */}
        <button
          onClick={onToggleTheme}
          aria-label={
            theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'
          }
          className="grid h-8 w-8 place-items-center rounded-lg border border-line/60 bg-surface text-muted transition-colors hover:border-line-strong hover:text-ink"
        >
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={theme}
              initial={{ rotate: -30, opacity: 0, scale: 0.7 }}
              animate={{ rotate: 0, opacity: 1, scale: 1 }}
              exit={{ rotate: 30, opacity: 0, scale: 0.7 }}
              transition={{ duration: 0.15 }}
              className="grid place-items-center"
            >
              {theme === 'dark' ? (
                <Sun size={15} weight="bold" className="text-amber-300" />
              ) : (
                <MoonStars size={15} weight="bold" className="text-zinc-700" />
              )}
            </motion.span>
          </AnimatePresence>
        </button>
      </div>
    </header>
  )
}
