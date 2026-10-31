import { useState, useMemo } from 'react'
import {
  ChatCircle,
  MagnifyingGlass,
  Plus,
  X,
} from '@phosphor-icons/react'
import { motion, useReducedMotion } from 'motion/react'
import { BrandBlock } from './BrandMark'
import type { Conversation } from '../types'

export interface SidebarProps {
  turnCount: number
  onSuggest: (text: string) => void
  conversations: Conversation[]
  activeConversationId: string | null
  onSelectConversation: (id: string) => void
  onNewConversation: () => void
}

export function Sidebar({
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewConversation,
}: SidebarProps) {
  const reduce = useReducedMotion()
  const [filterQuery, setFilterQuery] = useState('')

  const filteredConversations = useMemo(() => {
    if (!filterQuery.trim()) return conversations
    const q = filterQuery.toLowerCase()
    return conversations.filter(c => c.title.toLowerCase().includes(q))
  }, [conversations, filterQuery])

  return (
    <motion.aside
      initial={reduce ? false : { opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
      className="flex h-full flex-col border-r border-line/60 bg-bg px-4 py-5"
    >
      {/* Brand Header */}
      <div className="flex items-center justify-between px-1">
        <BrandBlock />
      </div>

      {/* Primary New Chat Action */}
      <motion.button
        type="button"
        onClick={onNewConversation}
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.98 }}
        className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-raised/70 px-3.5 py-2.5 text-xs font-semibold text-ink shadow-xs backdrop-blur-md transition-all hover:border-accent/40 hover:bg-raised"
      >
        <Plus size={14} weight="bold" className="text-accent" />
        <span>New conversation</span>
      </motion.button>

      {/* History Section */}
      <div className="mt-6 flex min-h-0 flex-1 flex-col">
        <div className="flex items-center justify-between px-1">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-muted">
            History
          </span>
          {conversations.length > 0 && (
            <span className="font-mono text-[10px] text-faint">
              {filteredConversations.length}
            </span>
          )}
        </div>

        {/* Filter input (only when there are several conversations) */}
        {conversations.length > 2 && (
          <div className="relative mt-2">
            <MagnifyingGlass
              size={12}
              className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted"
            />
            <input
              type="text"
              value={filterQuery}
              onChange={e => setFilterQuery(e.target.value)}
              placeholder="Search history…"
              className="w-full rounded-lg border border-line/60 bg-surface/50 py-1.5 pl-7 pr-6 text-xs text-ink placeholder:text-muted/60 focus:border-accent/40 focus:outline-none"
            />
            {filterQuery && (
              <button
                type="button"
                onClick={() => setFilterQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted hover:text-ink"
              >
                <X size={11} weight="bold" />
              </button>
            )}
          </div>
        )}

        {/* Conversation List */}
        <div className="mt-2.5 flex-1 space-y-0.5 overflow-y-auto pr-1">
          {conversations.length === 0 ? (
            <div className="py-8 text-center text-xs text-muted">
              No previous conversations
            </div>
          ) : filteredConversations.length === 0 ? (
            <div className="py-6 text-center text-xs text-muted">
              No matching chats
            </div>
          ) : (
            filteredConversations.map(conversation => {
              const isActive = activeConversationId === conversation.id
              return (
                <button
                  key={conversation.id}
                  type="button"
                  onClick={() => onSelectConversation(conversation.id)}
                  className={`relative flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-xs transition-colors ${
                    isActive
                      ? 'font-medium text-ink'
                      : 'text-muted hover:text-ink hover:bg-raised/40'
                  }`}
                >
                  {isActive && (
                    <motion.div
                      layoutId="activeHistoryPill"
                      className="absolute inset-0 rounded-lg border border-white/10 bg-raised"
                      transition={{
                        type: 'spring',
                        stiffness: 450,
                        damping: 35,
                      }}
                    />
                  )}
                  <ChatCircle
                    size={13}
                    weight={isActive ? 'fill' : 'regular'}
                    className={`relative z-10 shrink-0 ${
                      isActive ? 'text-accent' : 'text-muted'
                    }`}
                  />
                  <span className="relative z-10 truncate">
                    {conversation.title}
                  </span>
                </button>
              )
            })
          )}
        </div>
      </div>

      {/* Clean Minimalist Footer */}
      <div className="mt-auto border-t border-line/50 pt-3">
        <div className="flex items-center gap-2 px-1 text-[11px] text-muted">
          <span className="h-1.5 w-1.5 rounded-full bg-ok" />
          <span>GanjiSwag&apos;s Knowledge Base</span>
          <span className="ml-auto font-mono text-[10px] text-faint">v2</span>
        </div>
      </div>
    </motion.aside>
  )
}
