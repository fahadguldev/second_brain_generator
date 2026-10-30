import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import type { Session } from '@supabase/supabase-js'
import {
  ArrowClockwise,
  ArrowRight,
  ArrowUUpLeft,
  ChatCircle,
  Check,
  CheckCircle,
  CircleNotch,
  Clock,
  CloudArrowUp,
  Cpu,
  Database,
  Eye,
  EyeSlash,
  FileText,
  MagnifyingGlass,
  PaperPlaneTilt,
  PencilSimple,
  Play,
  Plus,
  SignOut,
  Sparkle,
  Stack,
  Tag,
  Terminal,
  Trash,
  WarningCircle,
  X,
} from '@phosphor-icons/react'
import { supabase, supabaseConfigured } from '../lib/supabase'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

export type KnowledgeItem = {
  id: string
  question?: string
  content: string
  source_type: string
  topics?: string[] | null
  domain?: 'personal' | 'professional' | string
  source_message_id?: string
  status: 'draft' | 'approved' | 'indexed' | string
  chunks_count?: number
  created_by: string
  created_at: string
  updated_at: string
  approved_at?: string
}

export type JobLog = {
  timestamp: string
  step: string
  level: 'info' | 'warning' | 'error' | string
  message: string
}

export type Job = {
  id: string
  knowledge_item_id: string
  status: 'queued' | 'indexing' | 'indexed' | 'failed' | string
  current_step?: 'queued' | 'chunking' | 'embedding' | 'pushing_qdrant' | 'indexed' | 'failed' | string
  chunks_total: number
  chunks_indexed: number
  logs?: JobLog[]
  error?: string
  created_at: string
  finished_at?: string
}

export type QdrantStats = {
  status: string
  collection: string
  points_count: number
  indexed_vectors_count?: number
  vector_size: number
  pushed_records_count?: number
  pushed_chunks_count?: number
  error?: string
}

export type SinglePushTracker = {
  jobId: string
  knowledgeItemId?: string
  messageId?: string
  title: string
  step: 'queued' | 'chunking' | 'embedding' | 'pushing_qdrant' | 'indexed' | 'failed' | string
  status: 'queued' | 'indexing' | 'indexed' | 'failed' | string
  chunksIndexed?: number
  chunksTotal?: number
  lastMessage?: string
  error?: string
}

export type CategoryItem = {
  name: string
  keywords: string[]
  domain: 'personal' | 'professional' | string
}

type AdminMessage = {
  id: string
  role: 'user' | 'assistant'
  text: string
  pipeline_status?: string | null
  is_pushed?: boolean
  knowledge_item_id?: string | null
}

type AdminConversation = {
  id: string
  title: string
  user_id: string
  messages: AdminMessage[]
}

// Labels are the action itself. No "Step 1 / Stage 2" numbering.
const PIPELINE_STEPS = [
  { id: 'queued', label: 'Queued', icon: Clock, desc: 'Scheduled in background' },
  { id: 'chunking', label: 'Chunk', icon: FileText, desc: 'Text split into RAG chunks' },
  { id: 'embedding', label: 'Embed', icon: Sparkle, desc: 'Generating vectors via Gemini' },
  { id: 'pushing_qdrant', label: 'Upsert', icon: CloudArrowUp, desc: 'Writing vectors to the collection' },
  { id: 'indexed', label: 'Verified', icon: CheckCircle, desc: 'Confirmed in Qdrant and the database' },
] as const

/**
 * One status system for the whole console.
 * Hue encodes urgency, nothing else:
 *   running  -> accent   (work in flight, the one interactive colour)
 *   indexed  -> emerald  (finished and verified)
 *   failed   -> rose     (broken, needs a human)
 *   waiting  -> neutral  (queued on a human: draft, approved)
 * Anything waiting on a person is deliberately colourless so that colour
 * always means "look here".
 */
type Tone = 'running' | 'indexed' | 'failed' | 'waiting'

const TONE_TEXT: Record<Tone, string> = {
  running: 'text-accent',
  indexed: 'text-ok',
  failed: 'text-danger',
  waiting: 'text-muted',
}

const TONE_SURFACE: Record<Tone, string> = {
  running: 'bg-accent-soft',
  indexed: 'bg-ok-soft',
  failed: 'bg-danger-soft',
  waiting: 'bg-raised',
}

// Solid variant of the same tones, for progress fills.
const TONE_FILL: Record<Tone, string> = {
  running: 'bg-accent',
  indexed: 'bg-ok',
  failed: 'bg-danger',
  waiting: 'bg-muted',
}

const TONE_LABEL: Record<Tone, string> = {
  running: 'Running',
  indexed: 'Indexed',
  failed: 'Failed',
  waiting: 'Waiting',
}

function toneOf(status?: string | null): Tone {
  if (status === 'indexed') return 'indexed'
  if (status === 'failed') return 'failed'
  if (status === 'queued' || status === 'indexing') return 'running'
  return 'waiting'
}

// The single status renderer. Anything that needs to show state uses this so a
// record, a row, a tab and a toast never disagree about what "running" looks like.
function StatusPill({
  status,
  label,
  className = '',
}: {
  status?: string | null
  label?: string
  className?: string
}) {
  const tone = toneOf(status)
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-2 py-[3px] text-[11px] font-semibold leading-none ${TONE_SURFACE[tone]} ${TONE_TEXT[tone]} ${className}`}
    >
      {tone === 'running' && (
        <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-accent" />
      )}
      {label ?? TONE_LABEL[tone]}
    </span>
  )
}

function ClassificationBadges({
  topics,
  domain,
}: {
  topics?: string[] | null
  domain?: string
}) {
  if (!topics?.length) {
    return <span className="text-[11px] text-muted">Not classified yet</span>
  }

  const label = domain === 'personal' ? 'Personal' : 'Professional'

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="rounded-full bg-accent-soft px-2 py-0.5 text-[11px] font-semibold text-accent">
        {label}
      </span>
      {topics.map(topic => (
        <span key={topic} className="rounded-full bg-raised px-2 py-0.5 text-[11px] text-muted">
          {topic}
        </span>
      ))}
    </div>
  )
}

// Metrics sit in plain layout, not in boxes. At this density a card per number
// is noise; the number and its label carry it. 22px is the only genuinely large
// type in the app, which is what makes a stat read as a stat.
function Stat({ label, value, unit, tone }: { label: string; value: string; unit?: string; tone?: Tone }) {
  return (
    <div className="min-w-0 pr-7 first:pl-0">
      <p className="eyebrow">{label}</p>
      <p className="mt-1.5 flex items-baseline gap-1.5">
        <span
          className={`font-mono text-[22px] font-semibold leading-none tracking-tight tabular-nums ${
            tone ? TONE_TEXT[tone] : 'text-ink'
          }`}
        >
          {value}
        </span>
        {unit && <span className="text-[12px] text-muted">{unit}</span>}
      </p>
    </div>
  )
}

// Label/value pairs on a grid instead of a middle-dot run, so metadata scans
// vertically and never wraps into an unreadable ribbon.
function MetaGrid({ entries }: { entries: { label: string; value: string; mono?: boolean }[] }) {
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-3">
      {entries.map(e => (
        <div key={e.label} className="min-w-0">
          <dt className="eyebrow">{e.label}</dt>
          <dd className={`mt-1 truncate text-[13px] text-ink ${e.mono ? 'font-mono' : ''}`} title={e.value}>
            {e.value}
          </dd>
        </div>
      ))}
    </dl>
  )
}

type AdminTab = 'chats' | 'review' | 'ready' | 'pipeline' | 'categories' | 'legacy'

export type LegacyPoint = {
  id: string
  text: string
  source_type: string
  topics: string[]
  domain?: string | null
}

function shortId(id?: string | null) {
  return id ? id.slice(0, 8) : 'none'
}

function Login({ onLogin }: { onLogin: (session: Session) => void }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [signingIn, setSigningIn] = useState(false)

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setError('')
    setSigningIn(true)
    try {
      const { data, error } = await supabase.auth.signInWithPassword({ email, password })
      if (error || !data.session) {
        setError(error?.message || 'Could not sign in')
      } else {
        onLogin(data.session)
      }
    } finally {
      setSigningIn(false)
    }
  }

  return (
    <main className="grid min-h-[100dvh] place-items-center bg-bg p-4 text-ink">
      <form onSubmit={submit} className="w-full max-w-sm rounded-lg border border-line bg-surface p-6">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-accent-ink">
            <Database size={18} weight="bold" />
          </div>
          <p className="text-[13px] font-semibold">Second Brain admin</p>
        </div>
        <h1 className="mt-6 text-[22px] font-semibold leading-tight tracking-tight">Sign in</h1>
        <p className="mt-2 text-[13px] leading-relaxed text-muted">
          Use your Supabase credentials to reach the curation and ingestion tools.
        </p>

        {supabaseConfigured ? (
          <div className="mt-6 space-y-4">
            <div>
              <label htmlFor="admin-email" className="eyebrow mb-1.5 block">
                Email
              </label>
              <input
                id="admin-email"
                className="h-10 w-full rounded-lg border border-line bg-bg px-3 text-[13.5px] text-ink placeholder:text-muted focus:border-accent focus:outline-none"
                type="email"
                autoComplete="username"
                placeholder="you@company.com"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
              />
            </div>
            <div>
              <label htmlFor="admin-password" className="eyebrow mb-1.5 block">
                Password
              </label>
              <input
                id="admin-password"
                className="h-10 w-full rounded-lg border border-line bg-bg px-3 text-[13.5px] text-ink placeholder:text-muted focus:border-accent focus:outline-none"
                type="password"
                autoComplete="current-password"
                placeholder="Your password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
              />
            </div>
            {error && (
              <p className="rounded-lg bg-danger-soft px-3 py-2.5 text-[12.5px] leading-relaxed text-danger">
                {error}
              </p>
            )}
            <button
              type="submit"
              disabled={!supabaseConfigured || signingIn}
              className="ctl ctl-primary mt-1 h-10 w-full text-[13.5px]"
            >
              {signingIn ? 'Signing in...' : 'Sign in'}
            </button>
          </div>
        ) : (
          <p className="mt-6 rounded-lg bg-danger-soft px-3 py-2.5 text-[12.5px] leading-relaxed text-danger">
            Supabase environment variables are missing, so sign in is unavailable.
          </p>
        )}
      </form>
    </main>
  )
}

export function AdminApp() {
  const [session, setSession] = useState<Session | null>(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState<AdminTab>('chats')

  const [knowledge, setKnowledge] = useState<KnowledgeItem[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [conversations, setConversations] = useState<AdminConversation[]>([])
  const [qdrantStats, setQdrantStats] = useState<QdrantStats | null>(null)

  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(null)
  const [recordSearch, setRecordSearch] = useState('')
  const [recordStatus, setRecordStatus] = useState<'all' | 'active' | 'indexed' | 'failed'>('all')
  const [logLevelFilter, setLogLevelFilter] = useState<'all' | 'error' | 'warning'>('all')
  const [selectedItemIds, setSelectedItemIds] = useState<string[]>([])
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null)
  const [searchChat, setSearchChat] = useState('')
  const [hidePushedInChats, setHidePushedInChats] = useState(true)
  const [actionNotice, setActionNotice] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [isPushingBatch, setIsPushingBatch] = useState(false)
  const [isRefreshingStats, setIsRefreshingStats] = useState(false)
  const [isUploading, setIsUploading] = useState(false)

  // Inline editing state for Chats tab
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null)
  const [messageDrafts, setMessageDrafts] = useState<Record<string, { question: string; content: string }>>({})
  const [isSavingInline, setIsSavingInline] = useState(false)
  const [singlePushTracker, setSinglePushTracker] = useState<SinglePushTracker | null>(null)
  const [pushingMessageId, setPushingMessageId] = useState<string | null>(null)
  const [isPushingConversation, setIsPushingConversation] = useState(false)
  const chatScrollRef = useRef<HTMLDivElement | null>(null)
  const logScrollRef = useRef<HTMLDivElement | null>(null)
  // Category taxonomy management state
  const [categories, setCategories] = useState<CategoryItem[]>([])
  const [categoryName, setCategoryName] = useState('')
  const [categoryDomain, setCategoryDomain] = useState<'personal' | 'professional'>('professional')
  const [categoryKeywords, setCategoryKeywords] = useState('')
  const [editingCategory, setEditingCategory] = useState<string | null>(null)
  const [categorySaving, setCategorySaving] = useState(false)
  const [categorySearch, setCategorySearch] = useState('')
  const [isCategoryModalOpen, setIsCategoryModalOpen] = useState(false)

  const [legacyPoints, setLegacyPoints] = useState<LegacyPoint[]>([])
  const [legacyTotal, setLegacyTotal] = useState(0)
  const [legacyNextOffset, setLegacyNextOffset] = useState<string | null>(null)
  const [legacyLoading, setLegacyLoading] = useState(false)
  const [legacyLoaded, setLegacyLoaded] = useState(false)
  const [legacySourceFilter, setLegacySourceFilter] = useState('')
  const [legacyTopicFilter, setLegacyTopicFilter] = useState('')
  const [legacyEditingId, setLegacyEditingId] = useState<string | null>(null)
  const [legacyTopicsDraft, setLegacyTopicsDraft] = useState('')
  const [legacyDomainDraft, setLegacyDomainDraft] = useState('professional')
  const [legacySavingId, setLegacySavingId] = useState<string | null>(null)

  const pollIntervalRef = useRef<number | null>(null)

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session)
      setLoading(false)
    })
    const { data } = supabase.auth.onAuthStateChange((_event, next) => setSession(next))
    return () => data.subscription.unsubscribe()
  }, [])

  const api = useCallback(
    async (path: string, init?: RequestInit) => {
      if (!session) throw new Error('Not signed in')
      const response = await fetch(`${API_URL}/api/admin${path}`, {
        ...init,
        headers: { Authorization: `Bearer ${session.access_token}`, ...init?.headers },
      })
      if (response.status === 401) {
        // Token expired or revoked: drop the session so the login screen
        // comes back instead of failing every admin call behind a banner.
        await supabase.auth.signOut().catch(() => undefined)
        throw new Error('Session expired. Please sign in again.')
      }
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body.detail || 'Admin request failed')
      return body
    },
    [session]
  )

  const fetchQdrantStats = useCallback(async () => {
    if (!session) return
    setIsRefreshingStats(true)
    try {
      const stats = await api('/qdrant/stats')
      setQdrantStats(stats)
    } catch {
      // Non-blocking fallback
    } finally {
      setIsRefreshingStats(false)
    }
  }, [api, session])

  const refresh = useCallback(async () => {
    if (!session) return
    try {
      const [chatRows, knowledgeRows, jobRows, catRows, stats] = await Promise.all([
        api('/conversations'),
        api('/knowledge'),
        api('/jobs'),
        api('/categories').catch(() => []),
        api('/qdrant/stats').catch(() => null),
      ])
      setConversations(chatRows)
      setKnowledge(knowledgeRows)
      setJobs(jobRows)
      if (Array.isArray(catRows)) setCategories(catRows)
      if (stats) setQdrantStats(stats)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load admin data')
    }
  }, [api, session])

  const handleSaveCategory = async (e: FormEvent) => {
    e.preventDefault()
    if (!categoryName.trim()) return
    setCategorySaving(true)
    try {
      const kws = categoryKeywords
        .split(',')
        .map(k => k.trim())
        .filter(Boolean)
      const updated = await api('/categories', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: categoryName.trim().toLowerCase(),
          keywords: kws,
          domain: categoryDomain,
        }),
      })
      setCategories(prev => {
        const filtered = prev.filter(c => c.name !== updated.name)
        return [...filtered, updated].sort((a, b) => a.name.localeCompare(b.name))
      })
      setActionNotice(`Category '${updated.name}' saved successfully`)
      setTimeout(() => setActionNotice(null), 3000)
      setIsCategoryModalOpen(false)
      setCategoryName('')
      setCategoryKeywords('')
      setEditingCategory(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save category')
    } finally {
      setCategorySaving(false)
    }
  }

  const handleDeleteCategory = async (name: string) => {
    if (!window.confirm(`Are you sure you want to delete category '${name}'?`)) return
    try {
      await api(`/categories/${encodeURIComponent(name)}`, { method: 'DELETE' })
      setCategories(prev => prev.filter(c => c.name !== name))
      setActionNotice(`Category '${name}' deleted`)
      setTimeout(() => setActionNotice(null), 3000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete category')
    }
  }

  const handleEditCategory = (cat: CategoryItem) => {
    setCategoryName(cat.name)
    setCategoryDomain(cat.domain === 'personal' ? 'personal' : 'professional')
    setCategoryKeywords(cat.keywords.join(', '))
    setEditingCategory(cat.name)
    setIsCategoryModalOpen(true)
  }

  const fetchLegacyPoints = useCallback(
    async (opts: { reset: boolean; sourceType?: string; topic?: string }) => {
      if (!session) return
      setLegacyLoading(true)
      try {
        const params = new URLSearchParams({ limit: '24' })
        const source = opts.sourceType ?? legacySourceFilter
        const topic = opts.topic ?? legacyTopicFilter
        if (source) params.set('source_type', source)
        if (topic.trim()) params.set('topic', topic.trim().toLowerCase())
        if (!opts.reset && legacyNextOffset) params.set('offset', legacyNextOffset)
        const body = await api(`/legacy-points?${params.toString()}`)
        setLegacyPoints(prev => (opts.reset ? body.points : [...prev, ...body.points]))
        setLegacyTotal(body.total_matched)
        setLegacyNextOffset(body.next_offset ?? null)
        setLegacyLoaded(true)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load legacy points')
      } finally {
        setLegacyLoading(false)
      }
    },
    [api, session, legacyNextOffset, legacySourceFilter, legacyTopicFilter]
  )

  // Load the legacy corpus the first time the tab opens
  useEffect(() => {
    if (tab === 'legacy' && !legacyLoaded && !legacyLoading) {
      fetchLegacyPoints({ reset: true })
    }
  }, [tab, legacyLoaded, legacyLoading, fetchLegacyPoints])

  const handleSaveLegacyTags = async (pointId: string) => {
    setLegacySavingId(pointId)
    try {
      const topics = legacyTopicsDraft
        .split(',')
        .map(t => t.trim().toLowerCase())
        .filter(Boolean)
      const updated = await api(`/legacy-points/${encodeURIComponent(pointId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topics, domain: legacyDomainDraft }),
      })
      setLegacyPoints(prev =>
        prev.map(p => (p.id === pointId ? { ...p, topics: updated.topics, domain: updated.domain } : p))
      )
      setLegacyEditingId(null)
      setActionNotice(`Point ${pointId} re-tagged`)
      setTimeout(() => setActionNotice(null), 3000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save tags')
    } finally {
      setLegacySavingId(null)
    }
  }

  useEffect(() => {
    refresh()
  }, [refresh])

  // Categorized knowledge items
  const draftItems = useMemo(() => knowledge.filter(k => k.status === 'draft'), [knowledge])
  const approvedItems = useMemo(() => knowledge.filter(k => k.status === 'approved'), [knowledge])
  const indexedItems = useMemo(() => knowledge.filter(k => k.status === 'indexed'), [knowledge])

  const filteredCategories = useMemo(() => {
    if (!categorySearch.trim()) return categories
    const q = categorySearch.toLowerCase().trim()
    return categories.filter(
      c => c.name.toLowerCase().includes(q) || c.keywords.some(k => k.toLowerCase().includes(q))
    )
  }, [categories, categorySearch])

  // Set of message IDs that have been pushed to Qdrant
  const pushedMessageIds = useMemo(() => {
    const set = new Set<string>()
    knowledge.forEach(k => {
      if ((k.status === 'indexed' || (k.chunks_count && k.chunks_count > 0)) && k.source_message_id) {
        set.add(k.source_message_id)
      }
    })
    return set
  }, [knowledge])

  // Set of canonical contents for pushed items (content fallback in case source_message_id was lost)
  const pushedContents = useMemo(() => {
    const set = new Set<string>()
    knowledge.forEach(k => {
      if ((k.status === 'indexed' || (k.chunks_count && k.chunks_count > 0)) && k.content) {
        set.add(k.content.trim())
      }
    })
    return set
  }, [knowledge])

  // Helper to check whether an assistant message has been pushed to Qdrant
  const isAssistantPushed = useCallback(
    (m: AdminMessage) => {
      if (m.role !== 'assistant') return false
      return (
        m.is_pushed === true ||
        pushedMessageIds.has(m.id) ||
        pushedContents.has(m.text.trim())
      )
    },
    [pushedMessageIds, pushedContents]
  )

  // Set of message IDs that are currently in draft or approved review
  const inPipelineMessageIds = useMemo(() => {
    const set = new Set<string>()
    knowledge.forEach(k => {
      if ((k.status === 'draft' || k.status === 'approved') && k.source_message_id) {
        set.add(k.source_message_id)
      }
    })
    return set
  }, [knowledge])

  // Active jobs checking (queued or indexing)
  const hasActiveJobs = useMemo(() => {
    const hasJobActive = jobs.some(j => j.status === 'queued' || j.status === 'indexing')
    const hasSingleTrackerActive = !!(
      singlePushTracker &&
      (singlePushTracker.status === 'queued' || singlePushTracker.status === 'indexing')
    )
    return hasJobActive || hasSingleTrackerActive
  }, [jobs, singlePushTracker])

  // Synchronize single push tracker with background job polling updates
  useEffect((): (() => void) | undefined => {
    if (!singlePushTracker) return undefined
    const matchingJob = jobs.find(j => j.id === singlePushTracker.jobId)
    if (matchingJob) {
      const lastLog =
        matchingJob.logs && matchingJob.logs.length > 0
          ? matchingJob.logs[matchingJob.logs.length - 1].message
          : undefined

      setSinglePushTracker(prev => {
        if (!prev) return null
        return {
          ...prev,
          step: matchingJob.current_step || prev.step,
          status: matchingJob.status,
          chunksIndexed: matchingJob.chunks_indexed,
          chunksTotal: matchingJob.chunks_total,
          lastMessage: lastLog || prev.lastMessage,
          error: matchingJob.error || prev.error,
        }
      })

      if (matchingJob.status === 'indexed' || matchingJob.status === 'failed') {
        setPushingMessageId(null)
        // Auto-dismiss the side notification after 6 seconds
        const timer = setTimeout(() => {
          setSinglePushTracker(current => (current?.jobId === matchingJob.id ? null : current))
        }, 6000)
        return () => clearTimeout(timer)
      }
    }
    return undefined
  }, [jobs, singlePushTracker?.jobId])

  // Real-time polling when jobs are executing
  useEffect(() => {
    // Clear any previous interval first: this effect re-runs when `api`
    // changes identity, and overwriting the ref would orphan the old timer.
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = null
    }
    if (!hasActiveJobs) return

    pollIntervalRef.current = window.setInterval(async () => {
      try {
        const [jobRows, knowledgeRows, chatRows, stats] = await Promise.all([
          api('/jobs'),
          api('/knowledge'),
          api('/conversations'),
          api('/qdrant/stats').catch(() => null),
        ])
        setJobs(jobRows)
        setKnowledge(knowledgeRows)
        setConversations(chatRows)
        if (stats) setQdrantStats(stats)
      } catch {
        // Silently continue polling
      }
    }, 1500)

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
      }
    }
  }, [api, hasActiveJobs])

  // Clear notice after 4 seconds
  useEffect(() => {
    if (!actionNotice) return undefined
    const t = setTimeout(() => setActionNotice(null), 4000)
    return () => clearTimeout(t)
  }, [actionNotice])

  // Filter conversations for Chats tab: completely omit pushed conversations and pushed turns
  const { conversationsToDisplay, fullyPushedConvCount } = useMemo(() => {
    let pushedConvCount = 0

    const list = conversations
      .map(conv => {
        // 1. Identify which messages belong to already pushed Q&A dialogue turns
        const pushedTurnMessageIds = new Set<string>()

        for (let i = 0; i < conv.messages.length; i++) {
          const msg = conv.messages[i]
          if (msg.role === 'assistant' && isAssistantPushed(msg)) {
            pushedTurnMessageIds.add(msg.id)
            // Walk backwards to find and mark user message(s) that prompted this pushed assistant answer
            for (let j = i - 1; j >= 0; j--) {
              const prev = conv.messages[j]
              if (prev.role === 'user') {
                pushedTurnMessageIds.add(prev.id)
              } else {
                break
              }
            }
          }
        }

        const totalAssistants = conv.messages.filter(m => m.role === 'assistant').length
        const pushedAssistants = conv.messages.filter(m => isAssistantPushed(m)).length
        const unindexedAssistants = totalAssistants - pushedAssistants
        const isFullyPushed = totalAssistants > 0 && unindexedAssistants === 0

        if (isFullyPushed) {
          pushedConvCount++
        }

        // Visible messages: when hiding pushed, omit any message from already pushed turns
        const visibleMessages = conv.messages.filter(msg => {
          if (!hidePushedInChats) return true
          return !pushedTurnMessageIds.has(msg.id)
        })

        return {
          ...conv,
          visibleMessages,
          unindexedAssistants,
          isFullyPushed,
        }
      })
      .filter(conv => {
        // When hiding pushed chats, omit conversations that are completely pushed or have 0 visible messages
        if (hidePushedInChats && (conv.isFullyPushed || conv.visibleMessages.length === 0)) {
          return false
        }
        if (!searchChat) return true
        const matchTitle = conv.title.toLowerCase().includes(searchChat.toLowerCase())
        const matchText = conv.messages.some(m => m.text.toLowerCase().includes(searchChat.toLowerCase()))
        return matchTitle || matchText
      })

    return { conversationsToDisplay: list, fullyPushedConvCount: pushedConvCount }
  }, [conversations, hidePushedInChats, isAssistantPushed, searchChat])

  // Keep a valid conversation selected as the list changes
  useEffect(() => {
    if (conversationsToDisplay.length === 0) {
      setSelectedConversationId(null)
      return
    }
    if (selectedConversationId && conversationsToDisplay.some(c => c.id === selectedConversationId)) {
      return
    }
    setSelectedConversationId(conversationsToDisplay[0].id)
  }, [conversationsToDisplay, selectedConversationId])

  const selectedConversation = useMemo(
    () => conversationsToDisplay.find(c => c.id === selectedConversationId) || null,
    [conversationsToDisplay, selectedConversationId]
  )

  // Scroll the transcript to the latest message when the selection changes
  useEffect(() => {
    chatScrollRef.current?.scrollTo({ top: chatScrollRef.current.scrollHeight })
  }, [selectedConversationId])


  // Resolve the editable draft for a message: unsaved edits win, then an existing
  // knowledge item, then the raw chat text.
  const draftFor = useCallback(
    (conv: AdminConversation, message: AdminMessage) => {
      const saved = messageDrafts[message.id]
      if (saved) return saved

      const item = knowledge.find(k => k.source_message_id === message.id)
      const idx = conv.messages.findIndex(m => m.id === message.id)
      let question = item?.question ?? ''
      if (!question) {
        for (let j = (idx >= 0 ? idx : conv.messages.length) - 1; j >= 0; j--) {
          if (conv.messages[j].role === 'user') {
            question = conv.messages[j].text
            break
          }
        }
      }
      return { question, content: item?.content || message.text || '' }
    },
    [messageDrafts, knowledge]
  )

  const updateDraft = (messageId: string, patch: Partial<{ question: string; content: string }>) => {
    setMessageDrafts(prev => ({
      ...prev,
      [messageId]: {
        question: patch.question ?? prev[messageId]?.question ?? '',
        content: patch.content ?? prev[messageId]?.content ?? '',
      },
    }))
  }

  const toggleMessageEdit = (conv: AdminConversation, message: AdminMessage) => {
    if (editingMessageId === message.id) {
      setEditingMessageId(null)
      return
    }
    const draft = draftFor(conv, message)
    setMessageDrafts(prev => ({ ...prev, [message.id]: draft }))
    setEditingMessageId(message.id)
  }

  // Save an edited message: 'draft' stages it, 'approve' stages it as ready to push.
  const saveMessageEdit = async (
    conv: AdminConversation,
    message: AdminMessage,
    action: 'draft' | 'approve'
  ) => {
    setError('')
    const draft = draftFor(conv, message)
    const content = draft.content.trim()
    if (!content) {
      setError('Message content cannot be empty')
      return
    }
    setIsSavingInline(true)
    try {
      await api('/knowledge/quick-action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: draft.question.trim() || null,
          content,
          source_type: 'chat',
          source_message_id: message.id,
          action,
        }),
      })
      setEditingMessageId(null)
      setMessageDrafts(prev => {
        const next = { ...prev }
        delete next[message.id]
        return next
      })
      setActionNotice(
        action === 'approve'
          ? 'Approved! Message moved to Ready to Push.'
          : 'Draft saved to In Review.'
      )
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Action failed')
    } finally {
      setIsSavingInline(false)
    }
  }

  // Push a single message straight to Qdrant, using the current edits as written.
  const directPushMessage = async (conv: AdminConversation, message: AdminMessage) => {
    setError('')
    const draft = draftFor(conv, message)
    const content = draft.content.trim()
    if (!content) {
      setError('Message content cannot be empty')
      return
    }
    setPushingMessageId(message.id)
    const titleText = (draft.question.trim() || content).slice(0, 45)
    const title = titleText.length >= 45 ? `${titleText}...` : titleText

    setSinglePushTracker({
      jobId: 'pending',
      messageId: message.id,
      title,
      step: 'queued',
      status: 'queued',
      lastMessage: 'Starting ingestion pipeline...',
    })

    try {
      const res = await api('/knowledge/quick-action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_message_id: message.id,
          question: draft.question.trim() || null,
          content,
          source_type: 'chat',
          action: 'push',
        }),
      })

      if (res.job) {
        if (res.item?.id) setSelectedRecordId(res.item.id)
        setSinglePushTracker({
          jobId: res.job.id,
          knowledgeItemId: res.item?.id,
          messageId: message.id,
          title,
          step: res.job.current_step || 'queued',
          status: res.job.status || 'queued',
          chunksIndexed: res.job.chunks_indexed,
          chunksTotal: res.job.chunks_total,
          lastMessage: res.job.logs?.[res.job.logs.length - 1]?.message || 'Queued for processing',
        })
      }
      setEditingMessageId(null)
      setMessageDrafts(prev => {
        const next = { ...prev }
        delete next[message.id]
        return next
      })
      await refresh()
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Could not push message'
      setError(errMsg)
      setSinglePushTracker(prev =>
        prev
          ? { ...prev, status: 'failed', step: 'failed', error: errMsg, lastMessage: errMsg }
          : null
      )
      setPushingMessageId(null)
    }
  }

  // Push every not-yet-indexed assistant message in the conversation as one batch.
  const pushWholeConversation = async (conv: AdminConversation) => {
    setError('')
    const pending = conv.messages.filter(m => m.role === 'assistant' && !isAssistantPushed(m))
    if (pending.length === 0) {
      setActionNotice('Every answer in this chat is already indexed in Qdrant.')
      return
    }

    const payload = pending
      .map(m => {
        const draft = draftFor(conv, m)
        return {
          source_message_id: m.id,
          question: draft.question.trim() || null,
          content: draft.content.trim(),
        }
      })
      .filter(p => p.content.length > 0)

    if (payload.length === 0) {
      setError('Nothing to push: every pending answer is empty')
      return
    }

    setIsPushingConversation(true)
    try {
      const res = await api('/knowledge/push-messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: payload, source_type: 'chat' }),
      })
      setMessageDrafts(prev => {
        const next = { ...prev }
        payload.forEach(p => delete next[p.source_message_id])
        return next
      })
      setEditingMessageId(null)
      setActionNotice(res.message || `Queued ${payload.length} message(s) for ingestion`)
      setRecordStatus('active')
      setSelectedRecordId(null)
      await refresh()
      setTab('pipeline')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not push chat to Qdrant')
    } finally {
      setIsPushingConversation(false)
    }
  }

  // Which message ids a single delete removes: a user turn takes its answer with it.
  const messageDeleteTargets = (conv: AdminConversation, message: AdminMessage): string[] => {
    if (message.role !== 'user') return [message.id]
    const idx = conv.messages.findIndex(m => m.id === message.id)
    if (idx < 0) return [message.id]
    const next = conv.messages[idx + 1]
    return next && next.role === 'assistant' ? [message.id, next.id] : [message.id]
  }

  // Delete one message (and its answer when a user question is removed).
  const deleteChatMessage = async (conv: AdminConversation, message: AdminMessage) => {
    const targetIds = messageDeleteTargets(conv, message)
    const willDeleteAnswer = targetIds.length > 1
    const ok = confirm(
      willDeleteAnswer
        ? 'Delete this message and its answer? Anything pushed from this turn will also be removed from Qdrant.'
        : 'Delete this answer? If it was pushed to Qdrant it will be removed from the brain memory.'
    )
    if (!ok) return

    setError('')
    try {
      const res = await api(`/messages/${message.id}`, { method: 'DELETE' })
      setEditingMessageId(prev => (prev && targetIds.includes(prev) ? null : prev))
      setMessageDrafts(prev => {
        const next = { ...prev }
        targetIds.forEach(id => delete next[id])
        return next
      })
      setPushingMessageId(prev => (prev && targetIds.includes(prev) ? null : prev))
      setSinglePushTracker(prev =>
        prev && prev.messageId && targetIds.includes(prev.messageId) ? null : prev
      )
      setActionNotice(
        res.deleted_conversation_id
          ? 'Chat deleted'
          : willDeleteAnswer
          ? 'Message and answer deleted'
          : 'Message deleted'
      )
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not delete message')
    }
  }

  // Delete an entire conversation plus everything pushed from it.
  const deleteWholeChat = async (conv: AdminConversation) => {
    const ok = confirm(
      `Delete the whole chat "${conv.title || 'Untitled conversation'}"? All of its messages and everything pushed from it will be removed from Qdrant.`
    )
    if (!ok) return

    setError('')
    try {
      await api(`/conversations/${conv.id}`, { method: 'DELETE' })
      setEditingMessageId(null)
      setMessageDrafts(prev => {
        const next = { ...prev }
        conv.messages.forEach(m => delete next[m.id])
        return next
      })
      setSelectedConversationId(prev => (prev === conv.id ? null : prev))
      setActionNotice('Chat deleted from history and brain memory')
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not delete chat')
    }
  }

  // Other Action Handlers
  const updateDraftItem = async (item: KnowledgeItem) => {
    setError('')
    try {
      await api(`/knowledge/${item.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: item.question || null, content: item.content }),
      })
      setActionNotice('Draft saved successfully')
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save draft')
    }
  }

  const approveItem = async (item: KnowledgeItem) => {
    setError('')
    try {
      await api(`/knowledge/${item.id}/approve`, { method: 'POST' })
      setActionNotice('Approved! Moved to Ready to Push tab.')
      await refresh()
      setTab('ready')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not approve item')
    }
  }

  const revertItem = async (item: KnowledgeItem, stayOnTab = false) => {
    setError('')
    try {
      await api(`/knowledge/${item.id}/revert`, { method: 'POST' })
      setActionNotice('Reverted to Review draft')
      await refresh()
      if (!stayOnTab) setTab('review')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not revert item')
    }
  }

  const deleteItem = async (item: KnowledgeItem) => {
    if (!confirm('Are you sure you want to discard this item?')) return
    setError('')
    try {
      await api(`/knowledge/${item.id}`, { method: 'DELETE' })
      setActionNotice('Item removed')
      setSelectedItemIds(ids => ids.filter(id => id !== item.id))
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not delete item')
    }
  }

  const pushSingleItem = async (item: KnowledgeItem) => {
    setError('')
    try {
      const titleText = (item.question || item.content).slice(0, 45)
      const title = titleText.length >= 45 ? `${titleText}...` : titleText
      setSinglePushTracker({
        jobId: 'pending',
        knowledgeItemId: item.id,
        messageId: item.source_message_id || undefined,
        title,
        step: 'queued',
        status: 'queued',
        lastMessage: 'Starting ingestion...',
      })
      const job = await api(`/knowledge/${item.id}/ingest`, { method: 'POST' })
      setSinglePushTracker({
        jobId: job.id,
        knowledgeItemId: item.id,
        messageId: item.source_message_id || undefined,
        title,
        step: job.current_step || 'queued',
        status: job.status || 'queued',
        chunksIndexed: job.chunks_indexed,
        chunksTotal: job.chunks_total,
        lastMessage: job.logs?.[job.logs.length - 1]?.message || 'Queued for processing',
      })
      setActionNotice('Ingestion started! Tracking progress in side notification.')
      await refresh()
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Could not start ingestion'
      setError(errMsg)
      setSinglePushTracker(prev =>
        prev
          ? { ...prev, status: 'failed', step: 'failed', error: errMsg, lastMessage: errMsg }
          : null
      )
    }
  }

  const pushBatch = async (itemIds?: string[]) => {
    setError('')
    setIsPushingBatch(true)
    try {
      const payload = itemIds && itemIds.length > 0 ? { item_ids: itemIds } : {}
      const res = await api('/knowledge/batch-ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      setActionNotice(res.message || 'Batch ingestion started!')
      setRecordStatus('active')
      setSelectedRecordId(null)
      setSelectedItemIds([])
      await refresh()
      setTab('pipeline')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not start batch ingestion')
    } finally {
      setIsPushingBatch(false)
    }
  }

  const uploadFile = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError('')
    const form = event.currentTarget
    const input = form.elements.namedItem('file') as HTMLInputElement
    if (!input.files?.[0]) return
    const body = new FormData()
    body.append('file', input.files[0])
    setIsUploading(true)
    try {
      await api('/uploads', { method: 'POST', body })
      form.reset()
      setActionNotice('Document uploaded and added to Review drafts!')
      await refresh()
      setTab('review')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setIsUploading(false)
    }
  }

  // Every knowledge item that has ever entered the ingestion pipeline, paired
  // with its most recent job so records and their logs live in one list.
  const pipelineRecords = useMemo(() => {
    return knowledge
      .map(item => {
        const itemJobs = jobs
          .filter(j => j.knowledge_item_id === item.id)
          .sort((a, b) => +new Date(b.created_at) - +new Date(a.created_at))
        return { item, job: itemJobs[0] || null, jobCount: itemJobs.length }
      })
      .filter(r => r.job !== null || r.item.status === 'indexed' || (r.item.chunks_count ?? 0) > 0)
      .sort((a, b) => +new Date(b.item.updated_at) - +new Date(a.item.updated_at))
  }, [knowledge, jobs])

  const recordStatusOf = useCallback((record: (typeof pipelineRecords)[number]) => {
    if (record.job) return record.job.status
    return record.item.status
  }, [])

  const filteredRecords = useMemo(() => {
    const q = recordSearch.trim().toLowerCase()
    return pipelineRecords.filter(record => {
      const status = recordStatusOf(record)
      if (recordStatus === 'active' && status !== 'queued' && status !== 'indexing') return false
      if (recordStatus === 'indexed' && status !== 'indexed') return false
      if (recordStatus === 'failed' && status !== 'failed') return false
      if (!q) return true
      return (
        (record.item.question || '').toLowerCase().includes(q) ||
        record.item.content.toLowerCase().includes(q) ||
        record.item.id.toLowerCase().includes(q) ||
        (record.job?.id || '').toLowerCase().includes(q) ||
        record.item.source_type.toLowerCase().includes(q)
      )
    })
  }, [pipelineRecords, recordSearch, recordStatus, recordStatusOf])

  const recordCounts = useMemo(() => {
    const counts = { all: pipelineRecords.length, active: 0, indexed: 0, failed: 0 }
    pipelineRecords.forEach(r => {
      const status = recordStatusOf(r)
      if (status === 'queued' || status === 'indexing') counts.active++
      else if (status === 'indexed') counts.indexed++
      else if (status === 'failed') counts.failed++
    })
    return counts
  }, [pipelineRecords, recordStatusOf])

  // Keep a valid record selected as the list changes
  useEffect(() => {
    if (filteredRecords.length === 0) {
      setSelectedRecordId(null)
      return
    }
    if (selectedRecordId && filteredRecords.some(r => r.item.id === selectedRecordId)) return
    setSelectedRecordId(filteredRecords[0].item.id)
  }, [filteredRecords, selectedRecordId])

  const activeRecord = useMemo(
    () => filteredRecords.find(r => r.item.id === selectedRecordId) || null,
    [filteredRecords, selectedRecordId]
  )

  const activeJob = activeRecord?.job || null

  const visibleLogs = useMemo(() => {
    if (!activeJob?.logs) return []
    if (logLevelFilter === 'all') return activeJob.logs
    return activeJob.logs.filter(l => l.level === logLevelFilter)
  }, [activeJob, logLevelFilter])

  // Keep the log console pinned to the newest line while a job streams
  useEffect(() => {
    const el = logScrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [visibleLogs.length, selectedRecordId])

  // Helper for pipeline step visual state
  const stepOrder = ['queued', 'chunking', 'embedding', 'pushing_qdrant', 'indexed']
  const getStepState = (stepIndex: number, currentStep?: string, jobStatus?: string) => {
    if (jobStatus === 'failed') {
      const failIndex = stepOrder.indexOf(currentStep || 'queued')
      if (stepIndex < failIndex) return 'done'
      if (stepIndex === failIndex) return 'failed'
      return 'pending'
    }
    if (jobStatus === 'indexed') return 'done'

    const activeIndex = stepOrder.indexOf(currentStep || 'queued')
    if (stepIndex < activeIndex) return 'done'
    if (stepIndex === activeIndex) return 'active'
    return 'pending'
  }

  const navItems: { id: AdminTab; label: string; icon: any; count: number }[] = [
    { id: 'chats', label: 'Chats', icon: ChatCircle, count: conversationsToDisplay.length },
    { id: 'review', label: 'In Review', icon: FileText, count: draftItems.length },
    { id: 'ready', label: 'Ready to Push', icon: CloudArrowUp, count: approvedItems.length },
    { id: 'pipeline', label: 'Pipeline', icon: Cpu, count: indexedItems.length },
    { id: 'categories', label: 'Categories', icon: Tag, count: categories.length },
    { id: 'legacy', label: 'Legacy Corpus', icon: Stack, count: legacyTotal },
  ]

  if (loading) return <div className="grid min-h-[100dvh] place-items-center bg-bg text-muted">Loading admin portal...</div>
  if (!session) return <Login onLogin={setSession} />

  return (
    // h-[100dvh] + a flex column + flex-1 min-h-0 down the tree: every tab
    // fills the viewport instead of floating in the top half of it.
    <div className="flex h-[100dvh] flex-col overflow-hidden bg-bg text-ink">
      {/* Top Header */}
      <header className="z-20 shrink-0 border-b border-line bg-surface">
        <div className="mx-auto flex h-16 max-w-[1600px] items-center justify-between gap-4 px-5">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent text-accent-ink">
              <Database size={18} weight="bold" />
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-[19px] font-semibold leading-tight tracking-tight">
                Second Brain
              </h1>
              <p className="mt-0.5 hidden truncate text-[12.5px] leading-tight text-muted sm:block">
                Curate chat knowledge, push embeddings to Qdrant
              </p>
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-1.5">
            {/* Only shown when there is genuine live work, otherwise it is decoration */}
            {hasActiveJobs && (
              <span className="mr-1 hidden items-center gap-1.5 rounded-full bg-accent-soft px-2.5 py-1 text-[12px] font-semibold text-accent sm:flex">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
                Ingesting
              </span>
            )}
            <button
              onClick={refresh}
              title="Refresh all data"
              aria-label="Refresh all data"
              className="ctl ctl-icon ctl-ghost"
            >
              <ArrowClockwise size={17} />
            </button>
            <button
              onClick={() => supabase.auth.signOut()}
              title="Sign out"
              aria-label="Sign out"
              className="ctl ctl-icon ctl-ghost ctl-danger"
            >
              <SignOut size={17} />
            </button>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="z-10 shrink-0 border-b border-line bg-surface">
        <div className="mx-auto flex max-w-[1600px] gap-1 overflow-x-auto px-4">
          {navItems.map(item => {
            const isOn = tab === item.id
            return (
              <button
                key={item.id}
                onClick={() => setTab(item.id)}
                aria-current={isOn ? 'page' : undefined}
                className={`relative flex shrink-0 items-center gap-2 whitespace-nowrap px-3 py-3 text-[13px] font-medium transition-colors ${
                  isOn ? 'text-ink' : 'text-muted hover:text-ink'
                }`}
              >
                <item.icon size={16} weight={isOn ? 'bold' : 'regular'} />
                <span>{item.label}</span>
                {item.count > 0 && (
                  <span
                    className={`rounded-full px-1.5 py-px font-mono text-[11px] font-semibold tabular-nums ${
                      isOn ? 'bg-accent-soft text-accent' : 'bg-raised text-muted'
                    }`}
                  >
                    {item.count}
                  </span>
                )}
                {isOn && <span className="absolute inset-x-2 bottom-0 h-0.5 rounded-full bg-accent" />}
              </button>
            )
          })}
        </div>
      </nav>

      {/* Main Content Area. flex-1 + min-h-0 is what makes panes fill height. */}
      <main className="mx-auto flex w-full max-w-[1600px] min-h-0 flex-1 flex-col p-4 sm:p-5">
        {/* Flash Notifications & Errors */}
        {actionNotice && (
          <div className="mb-3 flex shrink-0 items-center justify-between rounded-lg border border-line bg-raised px-3.5 py-2.5 text-[13px] font-medium text-ink">
            <div className="flex min-w-0 items-center gap-2">
              <CheckCircle size={16} weight="fill" className="shrink-0 text-ok" />
              <span className="truncate">{actionNotice}</span>
            </div>
            <button
              onClick={() => setActionNotice(null)}
              aria-label="Dismiss"
              className="ctl ctl-icon ctl-ghost ml-3 shrink-0"
            >
              <X size={14} />
            </button>
          </div>
        )}

        {error && (
          <div className="mb-3 flex shrink-0 items-center justify-between rounded-lg border border-danger/30 bg-danger-soft px-3.5 py-2.5 text-[13px] font-medium text-ink">
            <div className="flex min-w-0 items-center gap-2">
              <WarningCircle size={16} weight="fill" className="shrink-0 text-danger" />
              <span className="truncate">{error}</span>
            </div>
            <button
              onClick={() => setError('')}
              aria-label="Dismiss"
              className="ctl ctl-icon ctl-ghost ml-3 shrink-0"
            >
              <X size={14} />
            </button>
          </div>
        )}

        {/* TAB 1: CHATS */}
        {tab === 'chats' && (
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-lg border border-line bg-surface">
            <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[21rem_minmax(0,1fr)]">
              {/* LEFT: conversation list */}
              <aside className="flex min-h-0 flex-col border-b border-line bg-bg lg:border-b-0 lg:border-r">
                <div className="space-y-2.5 border-b border-line p-3">
                  <div className="relative">
                    <label htmlFor="chat-search" className="sr-only">
                      Search chats
                    </label>
                    <MagnifyingGlass size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
                    <input
                      id="chat-search"
                      type="search"
                      placeholder="Search chats"
                      value={searchChat}
                      onChange={e => setSearchChat(e.target.value)}
                      className="h-9 w-full rounded-lg border border-line bg-surface pl-9 pr-3 text-[13px] text-ink placeholder:text-muted focus:border-accent focus:outline-none"
                    />
                  </div>
                  <button
                    onClick={() => setHidePushedInChats(h => !h)}
                    aria-pressed={hidePushedInChats}
                    className={`inline-flex h-8 w-full items-center gap-1.5 rounded-lg px-2.5 text-[12px] font-medium transition ${
                      hidePushedInChats ? 'bg-raised text-ink' : 'text-muted hover:bg-raised hover:text-ink'
                    }`}
                    title="Toggle visibility of conversations already pushed to Qdrant"
                  >
                    {hidePushedInChats ? <EyeSlash size={14} /> : <Eye size={14} />}
                    <span>
                      {hidePushedInChats
                        ? `Hiding Pushed (${fullyPushedConvCount})`
                        : 'Showing All'}
                    </span>
                  </button>
                </div>

                <div className="min-h-0 flex-1 divide-y divide-line overflow-y-auto">
                  {conversationsToDisplay.length === 0 ? (
                    <div className="px-5 py-14 text-center">
                      <CheckCircle size={26} className="mx-auto mb-2.5 text-ok/70" />
                      <p className="text-[13px] font-medium text-ink">Queue is clear</p>
                      <p className="mx-auto mt-1 max-w-[26ch] text-[12px] leading-relaxed text-muted">
                        {hidePushedInChats
                          ? 'Every chat has been pushed. Toggle above to see them.'
                          : 'No conversations match your search.'}
                      </p>
                    </div>
                  ) : (
                    conversationsToDisplay.map(conv => {
                      const isActive = conv.id === selectedConversationId
                      const firstUnpushed = conv.visibleMessages.find(
                        m => m.role === 'assistant' && !isAssistantPushed(m)
                      )
                      const preview =
                        firstUnpushed?.text ||
                        conv.visibleMessages[conv.visibleMessages.length - 1]?.text ||
                        'No messages'

                      return (
                        <div
                          key={conv.id}
                          role="button"
                          tabIndex={0}
                          onClick={() => setSelectedConversationId(conv.id)}
                          onKeyDown={e => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault()
                              setSelectedConversationId(conv.id)
                            }
                          }}
                          className={`relative block w-full cursor-pointer px-4 py-3 text-left transition-colors ${
                            isActive ? 'bg-accent/[0.07]' : 'hover:bg-raised/60'
                          }`}
                        >
                          {isActive && <span className="absolute inset-y-0 left-0 w-0.5 bg-accent" />}
                          <div className="flex items-start justify-between gap-2">
                            <p
                              className={`line-clamp-1 min-w-0 flex-1 text-[13.5px] font-semibold leading-snug ${
                                isActive ? 'text-accent' : 'text-ink'
                              }`}
                            >
                              {conv.title || 'Untitled conversation'}
                            </p>
                            {conv.isFullyPushed && (
                              <CheckCircle
                                size={14}
                                weight="fill"
                                className="mt-px shrink-0 text-ok"
                              />
                            )}
                          </div>
                          <p className="mt-1 line-clamp-2 text-[12px] leading-relaxed text-muted">
                            {preview}
                          </p>
                          <div className="mt-2 flex items-center gap-2 text-[11px] text-muted">
                            <span className="font-mono">{conv.id.slice(0, 8)}</span>
                            <span aria-hidden className="h-2.5 w-px shrink-0 bg-line" />
                            <span className="font-mono">{conv.messages.length} msgs</span>
                            <span className="ml-auto flex items-center gap-1.5">
                              {conv.unindexedAssistants > 0 && (
                                <span className="rounded-full bg-accent-soft px-1.5 py-px font-semibold text-accent">
                                  {conv.unindexedAssistants} to push
                                </span>
                              )}
                              <button
                                onClick={e => {
                                  e.stopPropagation()
                                  deleteWholeChat(conv)
                                }}
                                onKeyDown={e => e.stopPropagation()}
                                className="flex h-6 w-6 items-center justify-center rounded-md text-muted transition hover:bg-danger-soft hover:text-danger focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-danger/40"
                                title="Delete this chat and everything pushed from it"
                                aria-label="Delete chat"
                              >
                                <Trash size={13} />
                              </button>
                            </span>
                          </div>
                        </div>
                      )
                    })
                  )}
                </div>
              </aside>

              {/* RIGHT: full chat transcript */}
              <section className="flex min-h-0 flex-col bg-bg">
                {!selectedConversation ? (
                  <div className="grid flex-1 place-items-center p-10 text-center text-muted">
                    <div>
                      <ChatCircle size={40} className="mx-auto mb-3 opacity-50" />
                      <p className="text-[15px] font-semibold text-ink">Select a conversation</p>
                      <p className="mt-1 text-[13px]">
                        Pick a chat on the left to read its full history and curate answers.
                      </p>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Transcript header */}
                    <header className="shrink-0 border-b border-line bg-surface px-5 py-3.5">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="min-w-0">
                          <h2 className="truncate text-[16px] font-semibold leading-tight tracking-tight">
                            {selectedConversation.title || 'Untitled conversation'}
                          </h2>
                          <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[12px] text-muted">
                            <span className="font-mono">{selectedConversation.id.slice(0, 8)}</span>
                            <span aria-hidden className="h-3 w-px bg-line" />
                            <span className="font-mono">
                              user {selectedConversation.user_id.slice(0, 8)}
                            </span>
                            <span aria-hidden className="h-3 w-px bg-line" />
                            <span>{selectedConversation.messages.length} messages</span>
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          {selectedConversation.isFullyPushed && (
                            <span className="inline-flex items-center gap-1.5 rounded-full bg-ok-soft px-2.5 py-1 text-[11px] font-semibold text-ok">
                              <CheckCircle size={12} weight="fill" />
                              Pushed to Qdrant
                            </span>
                          )}
                          <button
                            onClick={() => deleteWholeChat(selectedConversation)}
                            title="Delete this chat permanently, including anything pushed from it"
                            className="ctl ctl-md ctl-ghost ctl-danger"
                          >
                            <Trash size={14} />
                            Delete chat
                          </button>
                          <button
                            onClick={() => pushWholeConversation(selectedConversation)}
                            disabled={
                              isPushingConversation ||
                              selectedConversation.unindexedAssistants === 0
                            }
                            title="Push every un-indexed answer in this chat to Qdrant in one batch"
                            className="ctl ctl-md ctl-primary"
                          >
                            {isPushingConversation ? (
                              <CircleNotch size={14} className="animate-spin" />
                            ) : (
                              <PaperPlaneTilt size={14} weight="fill" />
                            )}
                            {isPushingConversation
                              ? 'Pushing...'
                              : `Push whole chat (${selectedConversation.unindexedAssistants})`}
                          </button>
                        </div>
                      </div>
                    </header>

                    {/* Messages */}
                    <div ref={chatScrollRef} className="min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-6">
                      {selectedConversation.visibleMessages.length === 0 ? (
                        <p className="py-10 text-center text-[13px] text-muted">
                          Every message in this chat has been pushed to Qdrant.
                        </p>
                      ) : (
                        <div className="mx-auto flex w-full max-w-3xl flex-col gap-4">
                          {selectedConversation.visibleMessages.map(message => {
                            const isUser = message.role === 'user'
                            const isAlreadyPushed = isAssistantPushed(message)
                            const isInPipeline = inPipelineMessageIds.has(message.id)
                            const isEditingThis = editingMessageId === message.id
                            const draft = draftFor(selectedConversation, message)

                            const isThisMsgPushing =
                              pushingMessageId === message.id ||
                              ((singlePushTracker?.status === 'indexing' ||
                                singlePushTracker?.status === 'queued') &&
                                singlePushTracker?.messageId === message.id)

                            // Reuse the shared stage labels so the bubble, the toast and the
                            // pipeline rail never describe the same step differently.
                            const stepLabel =
                              PIPELINE_STEPS.find(s => s.id === singlePushTracker?.step)?.label ?? 'Queued'

                            return (
                              <div
                                key={message.id}
                                className={`group flex items-end gap-2 ${
                                  isUser ? 'justify-end' : ''
                                }`}
                              >
                                {!isUser && (
                                  <div className="pointer-events-none grid h-7 w-7 shrink-0 place-items-center rounded-full bg-accent/10 text-accent">
                                    <Sparkle size={14} weight="duotone" />
                                  </div>
                                )}

                                <div
                                  className={`relative max-w-[min(42rem,90%)] rounded-lg px-4 py-3 text-[13.5px] leading-relaxed ${
                                    isUser
                                      ? 'rounded-br-[3px] bg-accent text-accent-ink'
                                      : isEditingThis
                                      ? 'rounded-bl-[3px] border border-accent bg-surface ring-2 ring-accent/15'
                                      : 'rounded-bl-[3px] border border-line bg-surface'
                                  }`}
                                >
                                  {/* Role + status row */}
                                  <div className="mb-2 flex flex-wrap items-center gap-2">
                                    <span
                                      className={`text-[11px] font-semibold uppercase tracking-[0.09em] ${
                                        isUser ? 'text-accent-ink/70' : 'text-accent'
                                      }`}
                                    >
                                      {isUser ? 'User' : 'Assistant'}
                                    </span>

                                    {!isUser && isAlreadyPushed && <StatusPill status="indexed" label="Pushed" />}

                                    {!isUser && !isAlreadyPushed && isInPipeline && (
                                      <StatusPill status="indexing" label="In pipeline" />
                                    )}

                                    {/* Pencil + Paper plane actions, and delete for every message */}
                                    {!isEditingThis && (
                                      <div className="ml-auto flex items-center gap-1 transition lg:opacity-0 lg:group-hover:opacity-100 lg:group-focus-within:opacity-100">
                                        {!isUser && !isAlreadyPushed && (
                                          <>
                                            <button
                                              onClick={() =>
                                                toggleMessageEdit(selectedConversation, message)
                                              }
                                              className="ctl ctl-icon"
                                              title="Edit this message"
                                              aria-label="Edit message"
                                            >
                                              <PencilSimple size={15} />
                                            </button>
                                            <button
                                              onClick={() =>
                                                directPushMessage(selectedConversation, message)
                                              }
                                              disabled={
                                                isThisMsgPushing ||
                                                (!!pushingMessageId && !isThisMsgPushing)
                                              }
                                              className="ctl ctl-icon ctl-primary"
                                              title="Push this message to Qdrant"
                                              aria-label="Push message to Qdrant"
                                            >
                                              {isThisMsgPushing ? (
                                                <CircleNotch size={15} className="animate-spin" />
                                              ) : (
                                                <PaperPlaneTilt size={15} weight="fill" />
                                              )}
                                            </button>
                                          </>
                                        )}
                                        <button
                                          onClick={() =>
                                            deleteChatMessage(selectedConversation, message)
                                          }
                                          disabled={isThisMsgPushing}
                                          className="ctl ctl-icon ctl-ghost ctl-danger"
                                          title={
                                            isUser
                                              ? 'Delete this message and its answer'
                                              : 'Delete this message'
                                          }
                                          aria-label="Delete message"
                                        >
                                          <Trash size={15} />
                                        </button>
                                      </div>
                                    )}

                                    {!isUser && isThisMsgPushing && (
                                      <StatusPill status="indexing" label={stepLabel} />
                                    )}
                                  </div>

                                  {/* Body or inline editor */}
                                  {!isEditingThis ? (
                                    <p className="whitespace-pre-wrap">{message.text}</p>
                                  ) : (
                                    <div className="mt-1 space-y-3">
                                      <div>
                                        <label className="eyebrow mb-1.5 block">Question or title</label>
                                        <input
                                          value={draft.question}
                                          onChange={e =>
                                            updateDraft(message.id, { question: e.target.value })
                                          }
                                          className="w-full rounded-lg border border-line bg-bg px-3 py-2 text-[13px] font-medium text-ink placeholder:text-muted focus:border-accent focus:outline-none"
                                          placeholder="What question does this answer?"
                                        />
                                      </div>

                                      <div>
                                        <label className="eyebrow mb-1.5 block">Canonical answer</label>
                                        <textarea
                                          value={draft.content}
                                          onChange={e =>
                                            updateDraft(message.id, { content: e.target.value })
                                          }
                                          rows={5}
                                          className="w-full rounded-lg border border-line bg-bg px-3 py-2 text-[13px] leading-relaxed text-ink placeholder:text-muted focus:border-accent focus:outline-none"
                                          placeholder="Polish the text that will be embedded."
                                        />
                                      </div>

                                      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line pt-3">
                                        <button
                                          onClick={() => toggleMessageEdit(selectedConversation, message)}
                                          className="ctl ctl-sm ctl-ghost"
                                        >
                                          <X size={14} /> Close
                                        </button>

                                        <div className="flex flex-wrap items-center gap-2">
                                          <button
                                            disabled={isSavingInline}
                                            onClick={() =>
                                              saveMessageEdit(selectedConversation, message, 'draft')
                                            }
                                            className="ctl ctl-sm"
                                          >
                                            Save draft
                                          </button>
                                          <button
                                            disabled={isSavingInline}
                                            onClick={() =>
                                              saveMessageEdit(selectedConversation, message, 'approve')
                                            }
                                            className="ctl ctl-sm"
                                          >
                                            <Check size={14} /> Approve
                                          </button>
                                          <button
                                            disabled={isSavingInline}
                                            onClick={() =>
                                              directPushMessage(selectedConversation, message)
                                            }
                                            className="ctl ctl-sm ctl-primary"
                                          >
                                            <PaperPlaneTilt size={14} weight="fill" />
                                            Push to Qdrant
                                          </button>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </section>
            </div>
          </div>
        )}

        {/* TAB 2: IN REVIEW */}
        {tab === 'review' && (
          <div className="min-h-0 flex-1 space-y-5 overflow-y-auto pb-2">
            {/* Upload Document Draft box */}
            <form onSubmit={uploadFile} className="rounded-lg border border-dashed border-line bg-surface p-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-3">
                  <div className="rounded-lg bg-accent-soft p-2.5 text-accent">
                    <CloudArrowUp size={22} />
                  </div>
                  <div>
                    <h3 className="text-[14px] font-semibold leading-tight">Upload a document for review</h3>
                    <p className="mt-0.5 text-[12px] text-muted">Add a .txt or .md file to the review queue.</p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <input
                    name="file"
                    type="file"
                    accept=".txt,.md,text/plain,text/markdown"
                    required
                    aria-label="Document file"
                    className="text-[12px] file:mr-2 file:rounded-lg file:border file:border-line file:bg-surface file:px-3 file:py-1.5 file:text-[12px] file:font-medium file:text-ink hover:file:bg-raised"
                  />
                  <button type="submit" disabled={isUploading} className="ctl ctl-md ctl-primary disabled:opacity-50">
                    {isUploading && <CircleNotch size={14} className="animate-spin" />}
                    {isUploading ? 'Uploading...' : 'Upload draft'}
                  </button>
                </div>
              </div>
            </form>

            <div>
              <h2 className="text-[16px] font-semibold leading-tight tracking-tight text-ink">
                Drafts awaiting review
                <span className="ml-2 font-mono text-[13px] font-medium tabular-nums text-muted">
                  {draftItems.length}
                </span>
              </h2>
              <p className="mt-1.5 max-w-[68ch] text-[13px] leading-relaxed text-muted">
                Edit the question and the canonical answer. Approving moves the item to Ready to Push.
              </p>
            </div>

            {draftItems.length === 0 ? (
              <div className="rounded-lg border border-line bg-surface px-5 py-14 text-center">
                <FileText size={28} className="mx-auto mb-2.5 text-muted/60" />
                <p className="text-[14px] font-semibold text-ink">Nothing in review</p>
                <p className="mx-auto mt-1.5 max-w-[38ch] text-[13px] leading-relaxed text-muted">
                  Edit any assistant message in the Chats tab, or upload a document above.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {draftItems.map((item) => (
                  <article key={item.id} className="rounded-lg border border-line bg-surface p-4">
                    <div className="mb-3.5 flex flex-wrap items-center gap-2">
                      <span className="rounded-full bg-raised px-2 py-0.5 font-mono text-[11px] text-muted">
                        {item.source_type}
                      </span>
                      <span className="font-mono text-[11px] text-muted">{item.id.slice(0, 8)}</span>
                      <StatusPill status={item.status} label="In review" className="ml-auto" />
                    </div>
                    <ClassificationBadges topics={item.topics} domain={item.domain} />

                    <div className="space-y-3">
                      <div>
                        <label className="eyebrow mb-1.5 block">Question or title</label>
                        <input
                          value={item.question || ''}
                          onChange={e => {
                            const val = e.target.value
                            setKnowledge(prev => prev.map(k => (k.id === item.id ? { ...k, question: val } : k)))
                          }}
                          className="w-full rounded-lg border border-line bg-bg px-3.5 py-2 text-[13.5px] font-medium text-ink placeholder:text-muted focus:border-accent focus:outline-none"
                          placeholder="What question does this content answer?"
                        />
                      </div>

                      <div>
                        <label className="eyebrow mb-1.5 block">Canonical answer</label>
                        <textarea
                          value={item.content}
                          rows={5}
                          onChange={e => {
                            const val = e.target.value
                            setKnowledge(prev => prev.map(k => (k.id === item.id ? { ...k, content: val } : k)))
                          }}
                          className="w-full rounded-lg border border-line bg-bg px-3.5 py-2 font-sans text-[13.5px] leading-relaxed text-ink placeholder:text-muted focus:border-accent focus:outline-none"
                          placeholder="Polish the text that will be embedded."
                        />
                      </div>
                    </div>

                    <div className="mt-3.5 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-3">
                      <button onClick={() => deleteItem(item)} className="ctl ctl-sm ctl-ghost ctl-danger">
                        <Trash size={14} /> Discard
                      </button>

                      <div className="flex items-center gap-2">
                        <button onClick={() => updateDraftItem(item)} className="ctl ctl-sm">
                          Save draft
                        </button>
                        <button onClick={() => approveItem(item)} className="ctl ctl-sm ctl-primary">
                          <Check size={14} weight="bold" /> Approve
                        </button>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </div>
        )}

        {/* TAB 3: READY TO PUSH */}
        {tab === 'ready' && (
          <div className="flex min-h-0 flex-1 flex-col">
            <div className="mb-4 shrink-0 rounded-lg border border-line bg-surface p-4">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h2 className="text-[16px] font-semibold leading-tight tracking-tight text-ink">
                    Ready to push
                    <span className="ml-2 font-mono text-[13px] font-medium tabular-nums text-muted">
                      {approvedItems.length}
                    </span>
                  </h2>
                  <p className="mt-1.5 max-w-[58ch] text-[13px] leading-relaxed text-muted">
                    Approved answers waiting to be embedded. Push one at a time, or push everything in
                    safe batches.
                  </p>
                </div>

                <div className="flex shrink-0 flex-wrap items-center gap-2">
                  {selectedItemIds.length > 0 && (
                    <button
                      onClick={() => pushBatch(selectedItemIds)}
                      disabled={isPushingBatch}
                      className="ctl ctl-md"
                    >
                      <Play size={14} weight="fill" />
                      Push selected ({selectedItemIds.length})
                    </button>
                  )}

                  <button
                    onClick={() => pushBatch()}
                    disabled={approvedItems.length === 0 || isPushingBatch}
                    className="ctl ctl-md ctl-primary"
                  >
                    <Stack size={15} weight="bold" />
                    Push all ({approvedItems.length})
                  </button>
                </div>
              </div>

              {approvedItems.length > 5 && (
                <div className="mt-3.5 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-3 text-[12px] text-muted">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="select-all"
                      checked={selectedItemIds.length === approvedItems.length && approvedItems.length > 0}
                      onChange={e => {
                        if (e.target.checked) setSelectedItemIds(approvedItems.map(i => i.id))
                        else setSelectedItemIds([])
                      }}
                      className="h-4 w-4 rounded border-line accent-[rgb(var(--c-accent))]"
                    />
                    <label htmlFor="select-all" className="cursor-pointer font-medium text-ink">
                      Select all {approvedItems.length}
                    </label>
                  </div>
                  <span className="text-[12px]">Rate limits are handled automatically</span>
                </div>
              )}
            </div>

            {approvedItems.length === 0 ? (
              <div className="flex-1 rounded-lg border border-line bg-surface px-5 py-14 text-center">
                <CheckCircle size={28} className="mx-auto mb-2.5 text-muted/60" />
                <p className="text-[14px] font-semibold text-ink">Nothing queued for push</p>
                <p className="mx-auto mt-1.5 max-w-[42ch] text-[13px] leading-relaxed text-muted">
                  Approve a draft from the In Review tab, or edit and approve a message in Chats.
                </p>
              </div>
            ) : (
              <div className="min-h-0 flex-1 overflow-hidden rounded-lg border border-line bg-surface">
                <div className="h-full divide-y divide-line overflow-y-auto">
                  {approvedItems.map(item => {
                    const isChecked = selectedItemIds.includes(item.id)
                    return (
                      <article
                        key={item.id}
                        className={`flex flex-wrap items-start gap-3 px-4 py-3.5 transition-colors ${
                          isChecked ? 'bg-accent/[0.07]' : 'hover:bg-raised/50'
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={isChecked}
                          aria-label={`Select ${item.question || 'untitled record'}`}
                          onChange={e => {
                            if (e.target.checked) setSelectedItemIds(ids => [...ids, item.id])
                            else setSelectedItemIds(ids => ids.filter(id => id !== item.id))
                          }}
                          className="mt-1 h-4 w-4 shrink-0 rounded border-line accent-[rgb(var(--c-accent))]"
                        />

                        <div className="min-w-0 flex-1 basis-64">
                          <h4 className="truncate text-[13.5px] font-semibold leading-snug text-ink">
                            {item.question || 'Untitled record'}
                          </h4>
                          <p className="mt-1 line-clamp-2 text-[12.5px] leading-relaxed text-muted">
                            {item.content}
                          </p>
                          <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-muted">
                            <span className="font-mono">{item.source_type}</span>
                            <span aria-hidden className="h-2.5 w-px shrink-0 bg-line" />
                            <span className="font-mono">{item.id.slice(0, 8)}</span>
                            <span aria-hidden className="h-2.5 w-px shrink-0 bg-line" />
                            <span className="font-mono">
                              approved{' '}
                              {item.approved_at ? new Date(item.approved_at).toLocaleTimeString() : 'recently'}
                            </span>
                          </div>
                          <div className="mt-2.5">
                            <ClassificationBadges topics={item.topics} domain={item.domain} />
                          </div>
                        </div>

                        <div className="ml-auto flex shrink-0 items-center gap-1.5">
                          <button
                            onClick={() => revertItem(item)}
                            title="Send this item back to In Review"
                            className="ctl ctl-sm"
                          >
                            <ArrowUUpLeft size={14} /> Revert
                          </button>
                          <button
                            onClick={() => deleteItem(item)}
                            title="Delete this item"
                            aria-label="Delete this item"
                            className="ctl ctl-icon ctl-ghost ctl-danger"
                          >
                            <Trash size={14} />
                          </button>
                          <button onClick={() => pushSingleItem(item)} className="ctl ctl-sm ctl-primary">
                            <Play size={14} weight="fill" /> Push
                          </button>
                        </div>
                      </article>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 4: PIPELINE FLOW, LOGS & PUSHED RECORDS */}
        {tab === 'pipeline' && (
          <div className="flex min-h-0 flex-1 flex-col gap-4">
            {/* Metrics sit in plain layout. A box per number is noise at this density. */}
            <div className="flex shrink-0 flex-wrap items-center justify-between gap-y-4 rounded-lg border border-line bg-surface px-5 py-4">
              <div className="flex flex-wrap items-center gap-y-4">
                <Stat
                  label="Qdrant vectors"
                  value={qdrantStats?.points_count !== undefined ? qdrantStats.points_count.toLocaleString() : 'unavailable'}
                  unit={qdrantStats?.vector_size ? `${qdrantStats.vector_size}d` : undefined}
                />
                <Stat
                  label="Records pushed"
                  value={String(qdrantStats?.pushed_records_count ?? indexedItems.length)}
                  unit="verified"
                  tone="indexed"
                />
                <Stat label="Chunks stored" value={String(qdrantStats?.pushed_chunks_count ?? 0)} unit="chunks" />
                <Stat
                  label="Active jobs"
                  value={String(recordCounts.active)}
                  unit={hasActiveJobs ? 'processing' : 'idle'}
                  tone={hasActiveJobs ? 'running' : undefined}
                />
                <Stat
                  label="Failed"
                  value={String(recordCounts.failed)}
                  unit="all time"
                  tone={recordCounts.failed > 0 ? 'failed' : undefined}
                />
              </div>
              <button
                onClick={() => {
                  fetchQdrantStats()
                  refresh()
                }}
                className="ctl ctl-sm ml-auto shrink-0"
              >
                <ArrowClockwise size={14} className={isRefreshingStats ? 'animate-spin' : ''} />
                Refresh
              </button>
            </div>

            {/* RECORDS + LOGS WORKSPACE */}
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-lg border border-line bg-surface">
              <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[minmax(0,25rem)_minmax(0,1fr)]">
                {/* LEFT: searchable record list */}
                <div className="flex min-h-0 flex-col border-b border-line lg:border-b-0 lg:border-r">
                  <div className="shrink-0 space-y-3 border-b border-line p-4">
                    <div className="flex items-baseline justify-between gap-2">
                      <h3 className="text-[14px] font-semibold leading-tight tracking-tight text-ink">
                        Records
                      </h3>
                      <span className="font-mono text-[11px] tabular-nums text-muted">
                        {filteredRecords.length} of {pipelineRecords.length}
                      </span>
                    </div>

                    <div className="relative">
                      <label htmlFor="record-search" className="sr-only">
                        Search records
                      </label>
                      <MagnifyingGlass size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
                      <input
                        id="record-search"
                        type="search"
                        value={recordSearch}
                        onChange={e => setRecordSearch(e.target.value)}
                        placeholder="Search records"
                        className="h-9 w-full rounded-lg border border-line bg-bg pl-9 pr-3 text-[13px] text-ink placeholder:text-muted focus:border-accent focus:outline-none"
                      />
                    </div>

                    <div className="flex items-center gap-0.5 rounded-lg bg-raised p-0.5">
                      {(
                        [
                          { key: 'all', label: 'All', count: recordCounts.all },
                          { key: 'active', label: 'Active', count: recordCounts.active },
                          { key: 'indexed', label: 'Indexed', count: recordCounts.indexed },
                          { key: 'failed', label: 'Failed', count: recordCounts.failed },
                        ] as const
                      ).map(f => {
                        const isOn = recordStatus === f.key
                        return (
                          <button
                            key={f.key}
                            onClick={() => setRecordStatus(f.key)}
                            aria-pressed={isOn}
                            className={`flex h-7 flex-1 items-center justify-center gap-1.5 rounded-lg px-2 text-[12px] font-medium transition ${
                              isOn ? 'bg-surface text-ink' : 'text-muted hover:text-ink'
                            }`}
                          >
                            {f.label}
                            <span className="font-mono text-[11px] tabular-nums opacity-70">{f.count}</span>
                          </button>
                        )
                      })}
                    </div>
                  </div>

                  <div className="min-h-0 flex-1 divide-y divide-line overflow-y-auto max-lg:max-h-[22rem]">
                    {filteredRecords.length === 0 ? (
                      <div className="px-5 py-14 text-center">
                        <Stack size={26} className="mx-auto mb-2.5 text-muted/60" />
                        <p className="text-[14px] font-semibold text-ink">No matching records</p>
                        <p className="mx-auto mt-1.5 max-w-[30ch] text-[13px] leading-relaxed text-muted">
                          {pipelineRecords.length === 0
                            ? 'Push a chat or an approved item and it will appear here.'
                            : 'Try a different search term or status filter.'}
                        </p>
                      </div>
                    ) : (
                      filteredRecords.map(record => {
                        const status = recordStatusOf(record)
                        const isActive = record.item.id === selectedRecordId
                        const lastLog =
                          record.job?.logs && record.job.logs.length > 0
                            ? record.job.logs[record.job.logs.length - 1].message
                            : null

                        return (
                          <button
                            key={record.item.id}
                            onClick={() => setSelectedRecordId(record.item.id)}
                            className={`relative block w-full px-4 py-3 text-left transition-colors ${
                              isActive ? 'bg-accent/[0.07]' : 'hover:bg-raised/60'
                            }`}
                          >
                            {isActive && <span className="absolute inset-y-0 left-0 w-0.5 bg-accent" />}
                            <div className="flex items-start justify-between gap-3">
                              {/* line-clamp-2, not truncate: a truncated record title
                                  hid which record you were actually looking at. */}
                              <p
                                className={`line-clamp-2 min-w-0 flex-1 text-[13.5px] font-semibold leading-snug ${
                                  isActive ? 'text-accent' : 'text-ink'
                                }`}
                              >
                                {record.item.question || 'Untitled record'}
                              </p>
                              <StatusPill status={status} />
                            </div>
                            <p className="mt-1 line-clamp-1 text-[12px] leading-relaxed text-muted">
                              {lastLog || record.item.content}
                            </p>
                            <div className="mt-2 flex items-center gap-2 text-[11px] text-muted">
                              <span className="font-mono">{record.item.source_type}</span>
                              <span aria-hidden className="h-2.5 w-px shrink-0 bg-line" />
                              <span className="font-mono">{record.item.chunks_count || 0} chunks</span>
                              <span aria-hidden className="h-2.5 w-px shrink-0 bg-line" />
                              <span className="ml-auto font-mono">
                                {new Date(record.item.updated_at).toLocaleDateString()}
                              </span>
                            </div>
                            <div className="mt-2.5">
                              <ClassificationBadges
                                topics={record.item.topics}
                                domain={record.item.domain}
                              />
                            </div>
                          </button>
                        )
                      })
                    )}
                  </div>
                </div>

                {/* RIGHT: one record, one scroll column */}
                <div className="min-h-0 overflow-y-auto bg-bg">
                  {!activeRecord ? (
                    <div className="grid h-full place-items-center px-6 py-16 text-center">
                      <div>
                        <Terminal size={28} className="mx-auto mb-3 text-muted/60" />
                        <p className="text-[14px] font-semibold text-ink">Select a record</p>
                        <p className="mx-auto mt-1.5 max-w-[34ch] text-[13px] leading-relaxed text-muted">
                          Choose a record to see its ingestion stages, vector metadata and full execution log.
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-6 p-5">
                      {/* Title + actions */}
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div className="min-w-0 flex-1">
                          <h4 className="text-[18px] font-semibold leading-snug tracking-tight text-ink">
                            {activeRecord.item.question || 'Untitled record'}
                          </h4>
                          <div className="mt-2.5 flex items-center gap-2.5">
                            <StatusPill status={recordStatusOf(activeRecord)} />
                            {activeRecord.item.source_message_id && (
                              <button
                                onClick={() => {
                                  const owner = conversations.find(c =>
                                    c.messages.some(m => m.id === activeRecord.item.source_message_id)
                                  )
                                  setSearchChat('')
                                  setHidePushedInChats(false)
                                  if (owner) setSelectedConversationId(owner.id)
                                  setTab('chats')
                                }}
                                className="ctl ctl-sm ctl-ghost"
                              >
                                <ChatCircle size={14} /> View in chat
                              </button>
                            )}
                          </div>
                        </div>

                        <div className="flex shrink-0 items-center gap-2">
                          <button
                            onClick={() => revertItem(activeRecord.item, true)}
                            title="Send this record back to In Review"
                            className="ctl ctl-sm"
                          >
                            <ArrowUUpLeft size={14} /> Revert
                          </button>
                          <button
                            onClick={() => deleteItem(activeRecord.item)}
                            title="Delete this record and purge its Qdrant points"
                            className="ctl ctl-sm ctl-danger"
                          >
                            <Trash size={14} /> Delete
                          </button>
                        </div>
                      </div>

                      {/* Stage rail. Icons carry the step, order carries the sequence. */}
                      <div>
                        <p className="eyebrow">Ingestion stages</p>
                        <ol className="mt-3.5 flex items-start">
                          {PIPELINE_STEPS.map((step, idx) => {
                            const Icon = step.icon
                            const state = activeJob
                              ? getStepState(idx, activeJob.current_step, activeJob.status)
                              : 'pending'
                            const isLast = idx === PIPELINE_STEPS.length - 1
                            return (
                              <li key={step.id} className={`flex min-w-0 flex-col ${isLast ? 'shrink-0' : 'flex-1'}`}>
                                <div className="flex items-center">
                                  <span
                                    title={step.desc}
                                    className={`grid h-6 w-6 shrink-0 place-items-center rounded-full ${
                                      state === 'active'
                                        ? 'bg-accent text-accent-ink'
                                        : state === 'done'
                                        ? 'bg-ok-soft text-ok'
                                        : state === 'failed'
                                        ? 'bg-danger text-white'
                                        : 'bg-raised text-muted'
                                    }`}
                                  >
                                    <Icon size={12} weight={state === 'done' || state === 'active' ? 'bold' : 'regular'} />
                                  </span>
                                  {!isLast && (
                                    <span
                                      aria-hidden
                                      className={`mx-1.5 h-px min-w-3 flex-1 ${
                                        state === 'done' ? 'bg-ok/40' : 'bg-line'
                                      }`}
                                    />
                                  )}
                                </div>
                                <span
                                  className={`mt-2 truncate pr-3 text-[12px] font-medium ${
                                    state === 'active'
                                      ? 'text-accent'
                                      : state === 'done'
                                      ? 'text-ink'
                                      : state === 'failed'
                                      ? 'text-danger'
                                      : 'text-muted'
                                  }`}
                                >
                                  {step.label}
                                </span>
                              </li>
                            )
                          })}
                        </ol>

                        {activeJob && activeJob.chunks_total > 0 && (
                          <div className="mt-4">
                            <div className="flex items-baseline justify-between text-[12px] text-muted">
                              <span>Chunks embedded</span>
                              <span className="font-mono tabular-nums">
                                {activeJob.chunks_indexed} / {activeJob.chunks_total}
                              </span>
                            </div>
                            <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-raised">
                              <div
                                className={`h-full transition-all duration-500 ${
                                  TONE_FILL[toneOf(activeJob.status)]
                                }`}
                                style={{
                                  width: `${Math.min(
                                    100,
                                    (activeJob.chunks_indexed / activeJob.chunks_total) * 100
                                  )}%`,
                                }}
                              />
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Vector metadata as a scannable grid, not a separator run */}
                      <div>
                        <p className="eyebrow">Metadata</p>
                        <div className="mt-3.5">
                          <MetaGrid
                            entries={[
                              { label: 'Item', value: shortId(activeRecord.item.id), mono: true },
                              { label: 'Job', value: shortId(activeJob?.id), mono: true },
                              { label: 'Chunks', value: String(activeRecord.item.chunks_count || 0) },
                              { label: 'Source', value: activeRecord.item.source_type },
                              { label: 'Attempts', value: String(activeRecord.jobCount) },
                              { label: 'Updated', value: new Date(activeRecord.item.updated_at).toLocaleString() },
                            ]}
                          />
                        </div>
                      </div>

                      <div>
                        <p className="eyebrow">Classification</p>
                        <div className="mt-3.5">
                          <ClassificationBadges
                            topics={activeRecord.item.topics}
                            domain={activeRecord.item.domain}
                          />
                        </div>
                      </div>

                      {/* What actually got embedded */}
                      <div>
                        <p className="eyebrow">Embedded content</p>
                        <p className="mt-3.5 whitespace-pre-wrap rounded-lg border border-line bg-surface px-4 py-3.5 text-[13.5px] leading-relaxed text-ink">
                          {activeRecord.item.content}
                        </p>
                      </div>

                      {/* Log console: aligned columns so the stream reads as a log.
                          Colours come from the console tokens, so it themes with the app. */}
                      <div>
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <p className="eyebrow">Execution log</p>
                          <div className="flex items-center gap-1.5">
                            {activeJob?.logs && activeJob.logs.length > 1 && (
                              <div className="flex items-center gap-0.5 rounded-lg bg-raised p-0.5">
                                {(
                                  [
                                    { key: 'all', label: 'All' },
                                    { key: 'warning', label: 'Warnings' },
                                    { key: 'error', label: 'Errors' },
                                  ] as const
                                ).map(f => {
                                  const isOn = logLevelFilter === f.key
                                  return (
                                    <button
                                      key={f.key}
                                      onClick={() => setLogLevelFilter(f.key)}
                                      aria-pressed={isOn}
                                      className={`h-7 rounded-lg px-2.5 text-[12px] font-medium transition ${
                                        isOn ? 'bg-surface text-ink' : 'text-muted hover:text-ink'
                                      }`}
                                    >
                                      {f.label}
                                    </button>
                                  )
                                })}
                              </div>
                            )}
                            <button
                              onClick={refresh}
                              title="Refresh logs"
                              aria-label="Refresh logs"
                              className="ctl ctl-icon ctl-ghost"
                            >
                              <ArrowClockwise size={14} />
                            </button>
                          </div>
                        </div>

                        <div className="mt-3.5 overflow-hidden rounded-lg border border-console-line bg-console">
                          <div
                            ref={logScrollRef}
                            className="max-h-80 overflow-y-auto p-3.5 font-mono text-[12px] leading-relaxed"
                          >
                            {!activeJob ? (
                              <p className="text-console-muted">
                                This record has no ingestion job. Push it from the Ready to Push tab to
                                generate logs.
                              </p>
                            ) : visibleLogs.length === 0 ? (
                              <p className="text-console-muted">
                                [{new Date(activeJob.created_at).toLocaleTimeString()}] [QUEUED] Job
                                initialized. Processing chunks: {activeJob.chunks_indexed}/
                                {activeJob.chunks_total}...
                              </p>
                            ) : (
                              <div className="space-y-1.5">
                                {visibleLogs.map((log, lIdx) => (
                                  <div
                                    key={lIdx}
                                    className="grid grid-cols-[auto_4.5rem_minmax(0,1fr)] items-baseline gap-x-3"
                                  >
                                    <span className="whitespace-nowrap text-console-muted">
                                      {log.timestamp}
                                    </span>
                                    <span
                                      className={`text-[11px] font-semibold uppercase tracking-wider ${
                                        log.level === 'error'
                                          ? 'text-danger'
                                          : log.level === 'warning'
                                          ? 'text-warn'
                                          : 'text-console-muted'
                                      }`}
                                    >
                                      {log.level}
                                    </span>
                                    <span className="break-words text-console-ink">{log.message}</span>
                                  </div>
                                ))}
                                {logLevelFilter === 'all' && activeJob.status === 'indexed' && (
                                  <p className="flex items-center gap-1.5 pt-1.5 font-semibold text-ok">
                                    <CheckCircle size={13} weight="bold" />
                                    Ingestion completed and verified in the Qdrant collection.
                                  </p>
                                )}
                                {logLevelFilter === 'all' && activeJob.status === 'failed' && (
                                  <p className="flex items-center gap-1.5 pt-1.5 font-semibold text-danger">
                                    <WarningCircle size={13} weight="bold" />
                                    Job failed: {activeJob.error || 'Unknown error'}
                                  </p>
                                )}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: CATEGORIES & TAXONOMY MANAGER */}
        {tab === 'categories' && (
          <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pb-8">
            {/* Top Toolbar / Overview */}
            <div className="flex shrink-0 flex-col gap-4 rounded-lg border border-line bg-surface p-5 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-[16px] font-semibold tracking-tight text-ink">Categories & Keyword Taxonomy</h2>
                <p className="mt-1 text-[13px] text-muted">
                  Topic keywords configured in <code className="rounded bg-raised px-1.5 py-0.5 font-mono text-[12px] text-accent">brain_config.yaml</code>. Uploads and chat items are automatically tagged with these categories.
                </p>
              </div>
              <div className="flex items-center gap-2.5">
                <div className="relative min-w-[220px]">
                  <MagnifyingGlass size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
                  <input
                    type="text"
                    value={categorySearch}
                    onChange={e => setCategorySearch(e.target.value)}
                    placeholder="Search categories or keywords..."
                    className="h-9 w-full rounded-lg border border-line bg-bg pl-8.5 pr-3 text-[13px] text-ink placeholder:text-muted focus:border-accent focus:outline-none"
                  />
                </div>
                <button
                  onClick={() => {
                    setCategoryName('')
                    setCategoryDomain('professional')
                    setCategoryKeywords('')
                    setEditingCategory(null)
                    setIsCategoryModalOpen(true)
                  }}
                  className="ctl ctl-primary flex shrink-0 items-center gap-1.5"
                >
                  <Plus size={15} weight="bold" />
                  <span>Add Category</span>
                </button>
              </div>
            </div>

            {/* Metric counters */}
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div className="rounded-lg border border-line bg-surface p-4">
                <p className="eyebrow">Total Categories</p>
                <p className="mt-1 font-mono text-[22px] font-semibold text-ink">{categories.length}</p>
              </div>
              <div className="rounded-lg border border-line bg-surface p-4">
                <p className="eyebrow">Personal Categories</p>
                <p className="mt-1 font-mono text-[22px] font-semibold text-accent">
                  {categories.filter(c => c.domain === 'personal').length}
                </p>
              </div>
              <div className="rounded-lg border border-line bg-surface p-4">
                <p className="eyebrow">Professional Categories</p>
                <p className="mt-1 font-mono text-[22px] font-semibold text-ok">
                  {categories.filter(c => c.domain === 'professional').length}
                </p>
              </div>
            </div>

            {/* Categories Grid */}
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              {filteredCategories.map(cat => {
                const isPersonal = cat.domain === 'personal'
                return (
                  <div
                    key={cat.name}
                    className="flex flex-col justify-between rounded-lg border border-line bg-surface p-4.5 transition-all hover:border-line-active"
                  >
                    <div>
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className="flex h-7 w-7 items-center justify-center rounded-md bg-raised text-accent">
                            <Tag size={15} weight="bold" />
                          </span>
                          <h3 className="font-mono text-[14.5px] font-semibold capitalize text-ink">
                            {cat.name}
                          </h3>
                        </div>
                        <span
                          className={`rounded-full px-2 py-0.5 text-[11px] font-medium uppercase tracking-wider ${
                            isPersonal
                              ? 'bg-accent-soft text-accent'
                              : 'bg-ok-soft text-ok'
                          }`}
                        >
                          {cat.domain}
                        </span>
                      </div>

                      {/* Keywords list */}
                      <div className="mt-3.5">
                        <p className="text-[11px] font-medium uppercase tracking-wider text-muted">
                          Trigger Keywords ({cat.keywords.length})
                        </p>
                        <div className="mt-2 flex max-h-36 flex-wrap gap-1.5 overflow-y-auto">
                          {cat.keywords.map((kw, i) => (
                            <span
                              key={i}
                              className="rounded bg-raised px-2 py-0.5 font-mono text-[11.5px] text-ink"
                            >
                              {kw}
                            </span>
                          ))}
                          {cat.keywords.length === 0 && (
                            <span className="text-[12px] italic text-muted">No keywords assigned</span>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="mt-4 flex items-center justify-end gap-2 border-t border-line/60 pt-3">
                      <button
                        onClick={() => handleEditCategory(cat)}
                        className="ctl ctl-ghost ctl-sm flex items-center gap-1 text-[12px]"
                      >
                        <PencilSimple size={14} />
                        Edit
                      </button>
                      <button
                        onClick={() => handleDeleteCategory(cat.name)}
                        className="ctl ctl-ghost ctl-sm ctl-danger flex items-center gap-1 text-[12px]"
                      >
                        <Trash size={14} />
                        Delete
                      </button>
                    </div>
                  </div>
                )
              })}
              {filteredCategories.length === 0 && (
                <div className="col-span-full rounded-lg border border-dashed border-line bg-surface/50 p-8 text-center text-muted">
                  No categories found matching &quot;{categorySearch}&quot;.
                </div>
              )}
            </div>
          </div>
        )}

        {tab === 'legacy' && (
          <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pb-8">
            {/* Toolbar */}
            <div className="flex shrink-0 flex-col gap-3 rounded-lg border border-line bg-surface p-5 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <h2 className="text-[16px] font-semibold tracking-tight text-ink">Legacy Corpus</h2>
                <p className="mt-1 text-[13px] text-muted">
                  TikTok DMs, comment replies, LinkedIn posts and video transcripts imported from the old knowledge base.
                  These chunks are searchable via topic tags — fix a tag and retrieval routes to them.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2.5">
                <div>
                  <label className="eyebrow mb-1 block">Source</label>
                <select
                  value={legacySourceFilter}
                  onChange={e => {
                    setLegacySourceFilter(e.target.value)
                    setLegacyPoints([])
                    setLegacyNextOffset(null)
                    setLegacyLoaded(false)
                  }}
                  className="h-9 rounded-lg border border-line bg-bg px-2.5 text-[13px] text-ink focus:border-accent focus:outline-none"
                >
                    <option value="">All sources</option>
                    <option value="tiktok_dm">TikTok DMs</option>
                    <option value="tiktok_comment_reply">TikTok Comments</option>
                    <option value="linkedin_post">LinkedIn Posts</option>
                    <option value="video_transcript">Video Transcripts</option>
                  </select>
                </div>
                <div>
                  <label className="eyebrow mb-1 block">Topic tag</label>
                  <input
                    type="text"
                    value={legacyTopicFilter}
                    onChange={e => setLegacyTopicFilter(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') {
                        setLegacyPoints([])
                        setLegacyNextOffset(null)
                        setLegacyLoaded(false)
                      }
                    }}
                    placeholder="e.g. cybersecurity"
                    className="h-9 w-44 rounded-lg border border-line bg-bg px-2.5 text-[13px] text-ink placeholder:text-muted focus:border-accent focus:outline-none"
                  />
                </div>
                <button
                  onClick={() => fetchLegacyPoints({ reset: true })}
                  disabled={legacyLoading}
                  className="ctl ctl-ghost flex h-9 items-center gap-1.5 self-end"
                >
                  {legacyLoading ? <CircleNotch size={15} className="animate-spin" /> : <ArrowClockwise size={15} />}
                  <span>Reload</span>
                </button>
              </div>
            </div>

            {/* Count strip */}
            <div className="shrink-0 rounded-lg border border-line bg-surface px-4 py-3 text-[13px] text-muted">
              Showing <span className="font-mono font-semibold text-ink">{legacyPoints.length}</span> of{' '}
              <span className="font-mono font-semibold text-ink">{legacyTotal}</span> legacy chunks
            </div>

            {/* Point list */}
            <div className="flex flex-col gap-3">
              {legacyPoints.map(point => {
                const isEditing = legacyEditingId === point.id
                return (
                  <div key={point.id} className="rounded-lg border border-line bg-surface p-4">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="rounded bg-raised px-2 py-0.5 font-mono text-[11.5px] text-accent">
                        {point.source_type}
                      </span>
                      {point.domain && (
                        <span className="rounded-full bg-raised px-2 py-0.5 text-[11px] uppercase tracking-wider text-muted">
                          {point.domain}
                        </span>
                      )}
                      <span className="ml-auto font-mono text-[11.5px] text-muted">#{point.id}</span>
                      <button
                        onClick={() => {
                          if (isEditing) {
                            setLegacyEditingId(null)
                          } else {
                            setLegacyEditingId(point.id)
                            setLegacyTopicsDraft(point.topics.join(', '))
                            setLegacyDomainDraft(point.domain === 'personal' ? 'personal' : 'professional')
                          }
                        }}
                        className="ctl ctl-ghost ctl-sm flex items-center gap-1 text-[12px]"
                      >
                        <PencilSimple size={14} />
                        {isEditing ? 'Cancel' : 'Re-tag'}
                      </button>
                    </div>

                    <p className="mt-2.5 whitespace-pre-wrap text-[13px] leading-relaxed text-ink/90">
                      {point.text.length > 600 ? `${point.text.slice(0, 600)}…` : point.text}
                    </p>

                    {isEditing ? (
                      <div className="mt-3 flex flex-wrap items-end gap-3 rounded-md border border-line bg-bg p-3">
                        <div className="min-w-[240px] flex-1">
                          <label className="eyebrow mb-1 block">Topics (comma-separated)</label>
                          <input
                            type="text"
                            value={legacyTopicsDraft}
                            onChange={e => setLegacyTopicsDraft(e.target.value)}
                            placeholder="e.g. cybersecurity, studies"
                            className="h-9 w-full rounded-lg border border-line bg-surface px-2.5 font-mono text-[13px] text-ink placeholder:text-muted focus:border-accent focus:outline-none"
                          />
                        </div>
                        <div>
                          <label className="eyebrow mb-1 block">Domain</label>
                          <select
                            value={legacyDomainDraft}
                            onChange={e => setLegacyDomainDraft(e.target.value)}
                            className="h-9 rounded-lg border border-line bg-surface px-2.5 text-[13px] text-ink focus:border-accent focus:outline-none"
                          >
                            <option value="professional">professional</option>
                            <option value="personal">personal</option>
                          </select>
                        </div>
                        <button
                          onClick={() => handleSaveLegacyTags(point.id)}
                          disabled={legacySavingId === point.id}
                          className="ctl ctl-primary flex h-9 items-center gap-1.5"
                        >
                          {legacySavingId === point.id ? <CircleNotch size={15} className="animate-spin" /> : <Check size={15} weight="bold" />}
                          <span>Save tags</span>
                        </button>
                      </div>
                    ) : (
                      <div className="mt-2.5 flex flex-wrap gap-1.5">
                        {point.topics.length > 0 ? (
                          point.topics.map(t => (
                            <span key={t} className="rounded bg-accent-soft px-2 py-0.5 font-mono text-[11.5px] text-accent">
                              {t}
                            </span>
                          ))
                        ) : (
                          <span className="text-[12px] italic text-muted">untagged — only reachable via fallback search</span>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
              {legacyPoints.length === 0 && !legacyLoading && (
                <div className="rounded-lg border border-dashed border-line bg-surface/50 p-8 text-center text-muted">
                  No legacy chunks match the current filters.
                </div>
              )}
              {legacyLoading && legacyPoints.length > 0 && (
                <div className="flex items-center justify-center gap-2 py-3 text-[13px] text-muted">
                  <CircleNotch size={16} className="animate-spin" /> Loading…
                </div>
              )}
            </div>

            {legacyNextOffset && (
              <button
                onClick={() => fetchLegacyPoints({ reset: false })}
                disabled={legacyLoading}
                className="ctl ctl-ghost mx-auto flex items-center gap-1.5"
              >
                {legacyLoading ? <CircleNotch size={15} className="animate-spin" /> : <ArrowRight size={15} />}
                <span>Load more</span>
              </button>
            )}
          </div>
        )}
      </main>

      {/* Category Add/Edit Modal */}
      {isCategoryModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-xs">
          <div className="w-full max-w-md rounded-xl border border-line bg-surface p-6 shadow-xl">
            <div className="flex items-center justify-between border-b border-line pb-3">
              <h3 className="text-[15px] font-semibold text-ink">
                {editingCategory ? `Edit Category: ${editingCategory}` : 'New Category'}
              </h3>
              <button
                onClick={() => setIsCategoryModalOpen(false)}
                className="ctl ctl-icon ctl-ghost"
              >
                <X size={16} />
              </button>
            </div>

            <form onSubmit={handleSaveCategory} className="mt-4 space-y-4">
              <div>
                <label className="eyebrow mb-1 block">Category Name</label>
                <input
                  type="text"
                  value={categoryName}
                  onChange={e => setCategoryName(e.target.value)}
                  placeholder="e.g. fitness, investing, family"
                  className="h-9 w-full rounded-lg border border-line bg-bg px-3 font-mono text-[13px] text-ink placeholder:text-muted focus:border-accent focus:outline-none"
                  required
                  disabled={!!editingCategory}
                />
              </div>

              <div>
                <label className="eyebrow mb-1 block">Domain Classification</label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setCategoryDomain('professional')}
                    className={`flex items-center justify-center rounded-lg border py-2 text-[12.5px] font-medium transition-colors ${
                      categoryDomain === 'professional'
                        ? 'border-accent bg-accent-soft text-accent'
                        : 'border-line bg-bg text-muted hover:text-ink'
                    }`}
                  >
                    Professional
                  </button>
                  <button
                    type="button"
                    onClick={() => setCategoryDomain('personal')}
                    className={`flex items-center justify-center rounded-lg border py-2 text-[12.5px] font-medium transition-colors ${
                      categoryDomain === 'personal'
                        ? 'border-accent bg-accent-soft text-accent'
                        : 'border-line bg-bg text-muted hover:text-ink'
                    }`}
                  >
                    Personal
                  </button>
                </div>
              </div>

              <div>
                <label className="eyebrow mb-1 block">Trigger Keywords (comma-separated)</label>
                <textarea
                  value={categoryKeywords}
                  onChange={e => setCategoryKeywords(e.target.value)}
                  placeholder="e.g. workout, gym, health, fitness, running, cardio"
                  rows={3}
                  className="w-full rounded-lg border border-line bg-bg p-3 font-mono text-[13px] text-ink placeholder:text-muted focus:border-accent focus:outline-none"
                />
                <p className="mt-1 text-[11.5px] text-muted">
                  Content matching any of these keywords will be tagged with this category.
                </p>
              </div>

              <div className="flex items-center justify-end gap-2 border-t border-line pt-3">
                <button
                  type="button"
                  onClick={() => setIsCategoryModalOpen(false)}
                  className="ctl ctl-ghost"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={categorySaving || !categoryName.trim()}
                  className="ctl ctl-primary flex items-center gap-1.5"
                >
                  {categorySaving ? <CircleNotch size={14} className="animate-spin" /> : <Check size={14} />}
                  <span>{editingCategory ? 'Update Category' : 'Create Category'}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* FLOATING SIDE NOTIFICATION FOR SINGLE PUSH PROGRESS */}
      {singlePushTracker && (() => {
        const trackerDone = singlePushTracker.status === 'indexed'
        const trackerFailed = singlePushTracker.status === 'failed'
        const currentStepId = trackerDone ? 'indexed' : singlePushTracker.step
        const currentIdx = Math.max(
          0,
          PIPELINE_STEPS.findIndex(s => s.id === currentStepId)
        )
        const trackerTone: Tone = trackerDone ? 'indexed' : trackerFailed ? 'failed' : 'running'
        const chunkPct =
          singlePushTracker.chunksTotal && singlePushTracker.chunksTotal > 0
            ? Math.min(100, ((singlePushTracker.chunksIndexed || 0) / singlePushTracker.chunksTotal) * 100)
            : null

        return (
        <aside
          role="status"
          aria-live="polite"
          className="fixed bottom-5 right-5 z-50 w-[23rem] max-w-[calc(100vw-2rem)] rounded-lg border border-line bg-surface p-4 shadow-lift"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex min-w-0 items-center gap-2.5">
              {trackerDone ? (
                <CheckCircle size={18} weight="fill" className="shrink-0 text-ok" />
              ) : trackerFailed ? (
                <WarningCircle size={18} weight="fill" className="shrink-0 text-danger" />
              ) : (
                <CircleNotch size={18} className="shrink-0 animate-spin text-accent" weight="bold" />
              )}
              <div className="min-w-0">
                <h4 className="truncate text-[14px] font-semibold leading-tight tracking-tight text-ink">
                  {trackerDone
                    ? 'Indexed in Qdrant'
                    : trackerFailed
                    ? 'Ingestion failed'
                    : 'Pushing to Qdrant'}
                </h4>
                <p className="mt-0.5 text-[12px] text-muted">
                  {trackerDone
                    ? 'Vectors stored and verified'
                    : trackerFailed
                    ? 'Open the pipeline to inspect the error'
                    : PIPELINE_STEPS[currentIdx]?.label || 'Queued'}
                </p>
              </div>
            </div>
            <button
              onClick={() => setSinglePushTracker(null)}
              className="ctl ctl-icon ctl-ghost -mr-1 -mt-1 shrink-0"
              title="Dismiss"
              aria-label="Dismiss progress"
            >
              <X size={14} />
            </button>
          </div>

          <p className="mt-3.5 line-clamp-2 text-[12.5px] leading-relaxed text-ink" title={singlePushTracker.title}>
            {singlePushTracker.title}
          </p>

          {/* Same stage rail as the pipeline tab, so the two never disagree */}
          <ol className="mt-4 flex items-start">
            {PIPELINE_STEPS.map((step, idx) => {
              const Icon = step.icon
              const state = getStepState(idx, currentStepId, singlePushTracker.status)
              const isLast = idx === PIPELINE_STEPS.length - 1
              return (
                <li key={step.id} className={`flex min-w-0 flex-col ${isLast ? 'shrink-0' : 'flex-1'}`}>
                  <div className="flex items-center">
                    <span
                      className={`grid h-5 w-5 shrink-0 place-items-center rounded-full ${
                        state === 'active'
                          ? 'bg-accent text-accent-ink'
                          : state === 'done'
                          ? 'bg-ok-soft text-ok'
                          : state === 'failed'
                          ? 'bg-danger text-white'
                          : 'bg-raised text-muted'
                      }`}
                    >
                      <Icon size={10} weight={state === 'done' || state === 'active' ? 'bold' : 'regular'} />
                    </span>
                    {!isLast && (
                      <span
                        aria-hidden
                        className={`mx-1 h-px min-w-2 flex-1 ${state === 'done' ? 'bg-ok/40' : 'bg-line'}`}
                      />
                    )}
                  </div>
                  <span
                    className={`mt-1.5 truncate pr-2 text-[12px] font-medium ${
                      state === 'active' ? 'text-accent' : state === 'done' ? 'text-ink' : 'text-muted'
                    }`}
                  >
                    {step.label}
                  </span>
                </li>
              )
            })}
          </ol>

          {/* Real chunk progress only. No invented percentages. */}
          {chunkPct !== null && !trackerDone && !trackerFailed && (
            <div className="mt-4">
              <div className="flex items-baseline justify-between text-[12px] text-muted">
                <span>Chunks embedded</span>
                <span className="font-mono tabular-nums">
                  {singlePushTracker.chunksIndexed || 0} / {singlePushTracker.chunksTotal}
                </span>
              </div>
              <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-raised">
                <div
                  className={`h-full transition-all duration-500 ${TONE_FILL[trackerTone]}`}
                  style={{ width: `${chunkPct}%` }}
                />
              </div>
            </div>
          )}

          {singlePushTracker.error && (
            <p className="mt-3.5 break-words rounded-lg bg-danger-soft px-3 py-2.5 text-[12px] leading-relaxed text-danger">
              {singlePushTracker.error}
            </p>
          )}

          <p className="mt-3.5 line-clamp-2 text-[12px] leading-relaxed text-muted">
            {singlePushTracker.lastMessage || 'Waiting for the background worker'}
          </p>

          <div className="mt-4 flex items-center justify-between border-t border-line pt-3">
            <span className="font-mono text-[12px] text-muted">
              {singlePushTracker.chunksTotal
                ? `${singlePushTracker.chunksIndexed || 0}/${singlePushTracker.chunksTotal} chunks`
                : 'single message'}
            </span>
            <button
              onClick={() => {
                if (singlePushTracker.knowledgeItemId) {
                  setSelectedRecordId(singlePushTracker.knowledgeItemId)
                  setRecordStatus('all')
                  setRecordSearch('')
                }
                setTab('pipeline')
              }}
              className="ctl ctl-sm ctl-ghost text-accent hover:text-accent"
            >
              View logs
              <ArrowRight size={13} />
            </button>
          </div>
        </aside>
        )
      })()}
    </div>
  )
}
