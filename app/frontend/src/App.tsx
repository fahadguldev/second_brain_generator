import { Suspense, useEffect, useState } from 'react'
import { lazy } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ConversationProvider } from '@elevenlabs/react'
import { useChat } from './hooks/useChat'
import { useTheme } from './hooks/useTheme'
import { useVoicePlayer } from './hooks/useVoicePlayer'
import { useVoiceCall } from './hooks/useVoiceCall'
import { Sidebar } from './components/Sidebar'
import { Header } from './components/Header'
import { Composer } from './components/Composer'
import { EmptyState } from './components/EmptyState'
import { MessageBubble } from './components/MessageBubble'
import { VoiceCallModal } from './components/VoiceCallModal'
import { ScrollToBottomButton } from './components/ui/ScrollToBottomButton'

// Split the admin console out of the chat bundle: visitors to / only
// download the chat UI, /admin loads this chunk on demand.
const AdminApp = lazy(() =>
  import('./components/AdminApp').then(m => ({ default: m.AdminApp }))
)

function ChatApp() {
  const {
    messages,
    conversations,
    activeConversationId,
    isThinking,
    isLoading,
    sendMessage,
    selectConversation,
    newConversation,
    scrollRef,
  } = useChat()
  const { theme, toggle } = useTheme()
  const {
    playingId,
    loadingId,
    errorId,
    errorMessage,
    play: playVoice,
    stop: stopVoice,
  } = useVoicePlayer()

  const voiceCall = useVoiceCall({
    onSendMessage: sendMessage,
    isThinking,
  })

  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [showScrollBottom, setShowScrollBottom] = useState(false)

  const turns = messages.filter(m => m.role === 'user').length

  const focusComposer = () => {
    document.querySelector('textarea')?.focus()
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        focusComposer()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 140
    setShowScrollBottom(!isNearBottom && messages.length > 2)
  }

  const scrollToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth',
      })
    }
  }

  const handlePick = (text: string) => {
    stopVoice()
    void sendMessage(text)
  }

  const closeSidebar = () => setSidebarOpen(false)
  const handleSelectConversation = (id: string) => {
    stopVoice()
    selectConversation(id)
    closeSidebar()
  }
  const handleNewConversation = () => {
    stopVoice()
    newConversation()
    closeSidebar()
  }

  const sidebar = (
    <Sidebar
      turnCount={turns}
      onSuggest={handlePick}
      conversations={conversations}
      activeConversationId={activeConversationId}
      onSelectConversation={handleSelectConversation}
      onNewConversation={handleNewConversation}
    />
  )

  return (
    <div className="h-[100dvh] overflow-hidden bg-bg text-ink">
      {/* Mobile Drawer with Animated Backdrop & Spring Slide */}
      <AnimatePresence>
        {sidebarOpen && (
          <div className="fixed inset-0 z-50 lg:hidden">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="absolute inset-0 bg-black/60 backdrop-blur-xs"
              onClick={closeSidebar}
              aria-hidden
            />
            <motion.div
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', stiffness: 350, damping: 30 }}
              className="absolute inset-y-0 left-0 w-[310px] max-w-[85vw] overflow-y-auto bg-bg shadow-lift"
            >
              {sidebar}
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Desktop Layout Grid */}
      <div className="mx-auto grid h-full w-full max-w-[1440px] grid-cols-1 overflow-hidden lg:grid-cols-[250px_minmax(0,1fr)]">
        <div className="hidden lg:flex lg:flex-col lg:overflow-y-auto">
          {sidebar}
        </div>

        <main className="relative flex h-full min-h-0 flex-col overflow-hidden">
          <Header
            isThinking={isThinking}
            theme={theme}
            onToggleTheme={toggle}
            onMenu={() => setSidebarOpen(true)}
            onStartCall={voiceCall.startCall}
            onFocusSearch={focusComposer}
          />

          <div
            ref={scrollRef}
            onScroll={handleScroll}
            className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-6 lg:px-8"
          >
            {isLoading ? (
              <div className="grid h-full place-items-center font-mono text-xs text-muted">
                Loading history…
              </div>
            ) : messages.length === 0 ? (
              <EmptyState onPick={handlePick} />
            ) : (
              <>
                <div className="mx-auto flex w-full max-w-2xl flex-col gap-4">
                  {messages.map(msg => (
                    <MessageBubble
                      key={msg.id}
                      msg={msg}
                      onPlayVoice={playVoice}
                      isPlayingVoice={playingId === msg.id}
                      isVoiceLoading={loadingId === msg.id}
                      voiceError={errorId === msg.id ? errorMessage : null}
                    />
                  ))}
                </div>

                {turns > 0 && (
                  <div className="mx-auto mt-6 flex max-w-2xl justify-center">
                    <motion.button
                      whileHover={{ scale: 1.04, y: -1 }}
                      whileTap={{ scale: 0.96 }}
                      onClick={handleNewConversation}
                      className="rounded-full border border-line bg-surface/90 px-3.5 py-1.5 text-xs font-semibold text-muted shadow-xs transition-colors hover:border-accent/40 hover:text-accent"
                    >
                      New conversation
                    </motion.button>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Floating Scroll to Bottom Button */}
          <ScrollToBottomButton
            visible={showScrollBottom}
            onClick={scrollToBottom}
          />

          <Composer onSend={handlePick} disabled={isThinking} />
        </main>
      </div>

      {/* 1-on-1 Voice Call Modal Overlay */}
      <VoiceCallModal
        isOpen={voiceCall.isCallActive}
        callState={voiceCall.callState}
        duration={voiceCall.duration}
        isMuted={voiceCall.isMuted}
        currentTranscript={voiceCall.currentTranscript}
        lastAIResponse={voiceCall.lastAIResponse}
        error={voiceCall.error}
        provider={voiceCall.provider}
        hasConvAIAgent={voiceCall.hasConvAIAgent}
        callMode={voiceCall.callMode}
        onSwitchMode={voiceCall.setCallMode}
        onEndCall={voiceCall.endCall}
        onToggleMute={voiceCall.toggleMute}
        onInterrupt={voiceCall.interrupt}
        onSendTextMessage={voiceCall.sendTextMessage}
      />
    </div>
  )
}

export default function App() {
  if (!window.location.pathname.startsWith('/admin')) {
    return (
      <ConversationProvider>
        <ChatApp />
      </ConversationProvider>
    )
  }
  return (
    <Suspense
      fallback={
        <div className="grid min-h-[100dvh] place-items-center bg-bg font-mono text-xs text-muted">
          Loading admin portal…
        </div>
      }
    >
      <AdminApp />
    </Suspense>
  )
}
