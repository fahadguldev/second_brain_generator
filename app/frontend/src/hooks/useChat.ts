import { useCallback, useEffect, useRef, useState } from 'react'
import type { AssistantInfo, ChatMessage, Conversation, StreamEvent } from '../types'

import { API_URL } from '../lib/api'
const VISITOR_KEY = 'second_brain_visitor_id'

function getStoredVisitorId(): string | null {
  try {
    return localStorage.getItem(VISITOR_KEY)
  } catch {
    return null
  }
}

function setStoredVisitorId(id: string | null) {
  if (!id) return
  try {
    localStorage.setItem(VISITOR_KEY, id)
  } catch {
    // ignore storage restrictions
  }
}

const request = async (path: string, init?: RequestInit): Promise<Response> => {
  const visitorId = getStoredVisitorId()
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(visitorId ? { 'X-Visitor-Id': visitorId } : {}),
    ...(init?.headers as Record<string, string> || {}),
  }

  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    credentials: 'include',
    headers,
  })

  const newVisitorId = res.headers.get('x-visitor-id')
  if (newVisitorId) {
    setStoredVisitorId(newVisitorId)
  }

  return res
}

function makeId() {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function normalizeMessage(item: Record<string, any>): ChatMessage {
  return {
    id: item.id,
    role: item.role,
    text: item.text,
    createdAt: item.created_at,
    info: item.info ? {
      latency: item.info.latency,
      model: item.info.model,
      embeddingModel: item.info.embedding_model,
      sources: item.info.sources || [],
    } : undefined,
  }
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null)
  const [isThinking, setIsThinking] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const scrollRef = useRef<HTMLDivElement>(null)
  const streamAbortRef = useRef<AbortController | null>(null)

  // Abort any in-flight stream when the chat unmounts.
  useEffect(() => () => streamAbortRef.current?.abort(), [])

  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [])

  const refreshConversations = useCallback(async () => {
    const res = await request('/api/conversations')
    if (!res.ok) throw new Error('Could not load conversations')
    const items: Conversation[] = await res.json()
    setConversations(items)
    return items
  }, [])

  const selectConversation = useCallback(async (id: string) => {
    setIsLoading(true)
    try {
      const res = await request(`/api/conversations/${id}/messages`)
      if (!res.ok) throw new Error('Could not load messages')
      setMessages((await res.json()).map(normalizeMessage))
      setActiveConversationId(id)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    refreshConversations()
      .then(items => items[0] ? selectConversation(items[0].id) : setIsLoading(false))
      .catch(() => setIsLoading(false))
  }, [refreshConversations, selectConversation])

  useEffect(scrollToBottom, [messages, scrollToBottom])

  const newConversation = useCallback(() => {
    setActiveConversationId(null)
    setMessages([])
  }, [])

  const sendMessage = useCallback(async (text: string): Promise<string | null> => {
    const trimmed = text.trim()
    if (!trimmed || isThinking) return null
    let accumulatedAnswer = ''
    const optimisticId = makeId()
    const assistantId = makeId()
    const controller = new AbortController()
    streamAbortRef.current?.abort()
    streamAbortRef.current = controller
    setMessages(prev => [
      ...prev,
      { id: optimisticId, role: 'user', text: trimmed },
      { id: assistantId, role: 'assistant', text: '', pending: true },
    ])
    setIsThinking(true)
    try {
      const res = await request('/api/ask/stream', {
        method: 'POST',
        body: JSON.stringify({ question: trimmed, conversation_id: activeConversationId }),
        signal: controller.signal,
      })
      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}))
        if (res.status === 404 && activeConversationId) {
          setActiveConversationId(null)
        }
        throw new Error(data.detail || 'Request failed')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let info: AssistantInfo = {}

      const handleEvent = (event: StreamEvent) => {
        if (event.type === 'metadata') {
          info = {
            latency: event.latency, model: event.model,
            embeddingModel: event.embedding_model, sources: event.sources || [],
          }
          return
        }
        if (event.type === 'error') throw new Error(event.message)
        if (event.type === 'delta') {
          accumulatedAnswer += event.text
        }
        if (event.type === 'done') {
          setMessages(prev => prev.map(message => {
            if (message.id === optimisticId) return { ...message, id: event.user_message_id }
            if (message.id === assistantId) {
              return { ...message, id: event.assistant_message_id, pending: false, info }
            }
            return message
          }))
          setActiveConversationId(event.conversation_id)
          void refreshConversations()
          return
        }
        setMessages(prev => prev.map(message => message.id === assistantId
          ? { ...message, pending: false, text: message.text + event.text, info }
          : message))
      }

      while (true) {
        const { value, done } = await reader.read()
        buffer += decoder.decode(value, { stream: !done })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.trim()) {
            try {
              handleEvent(JSON.parse(line) as StreamEvent)
            } catch (err) {
              console.warn('Could not parse stream event line:', line, err)
            }
          }
        }
        if (done) break
      }
      if (buffer.trim()) {
        try {
          handleEvent(JSON.parse(buffer) as StreamEvent)
        } catch (err) {
          console.warn('Could not parse trailing stream event:', buffer, err)
        }
      }
      return accumulatedAnswer.trim() || null
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        // Stream was superseded or the chat unmounted: drop the optimistic pair.
        setMessages(prev => prev.filter(m => m.id !== optimisticId && m.id !== assistantId))
      } else {
        console.error('API error:', error)
        setMessages(prev => prev.map(message => message.id === assistantId
          ? { ...message, pending: false, error: true,
              text: message.text || 'Could not generate a response. Please try again.' }
          : message))
      }
      return null
    } finally {
      if (streamAbortRef.current === controller) streamAbortRef.current = null
      setIsThinking(false)
    }
  }, [activeConversationId, isThinking, refreshConversations])

  return {
    messages, conversations, activeConversationId, isThinking, isLoading,
    sendMessage, selectConversation, newConversation, scrollRef,
  }
}
