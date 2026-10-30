import { useState } from 'react'
import { Check, Copy } from '@phosphor-icons/react'
import { AnimatePresence, motion } from 'motion/react'

interface CopyButtonProps {
  text: string
  className?: string
  title?: string
}

export function CopyButton({
  text,
  className = '',
  title = 'Copy message',
}: CopyButtonProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Fallback if clipboard API fails
      const textArea = document.createElement('textarea')
      textArea.value = text
      document.body.appendChild(textArea)
      textArea.select()
      document.execCommand('copy')
      document.body.removeChild(textArea)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <motion.button
      type="button"
      onClick={handleCopy}
      whileHover={{ scale: 1.08 }}
      whileTap={{ scale: 0.92 }}
      aria-label={copied ? 'Copied to clipboard' : title}
      title={copied ? 'Copied!' : title}
      className={`relative grid h-7 w-7 place-items-center rounded-md text-muted transition-colors hover:bg-raised hover:text-ink ${className}`}
    >
      <AnimatePresence mode="wait" initial={false}>
        {copied ? (
          <motion.span
            key="check"
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.5, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 500, damping: 25 }}
            className="text-ok"
          >
            <Check size={13} weight="bold" />
          </motion.span>
        ) : (
          <motion.span
            key="copy"
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.5, opacity: 0 }}
            transition={{ duration: 0.15 }}
          >
            <Copy size={13} weight="bold" />
          </motion.span>
        )}
      </AnimatePresence>
    </motion.button>
  )
}
