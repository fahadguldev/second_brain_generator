import { useEffect, useMemo, useState } from 'react'
import { ArrowSquareOut, Play } from '@phosphor-icons/react'
import type { SourceItem } from '../../types'

function sourceMetadata(item: SourceItem) {
  return item.payload?.metadata ?? item.metadata
}

interface VideoPreview {
  url: string
  title: string
}

export function VideoRecommendations({ sources }: { sources: SourceItem[] }) {
  const videoUrls = useMemo(
    () =>
      Array.from(
        new Set(
          sources
            .map(source => sourceMetadata(source)?.source?.url)
            .filter((url): url is string => Boolean(url)),
        ),
      ),
    [sources],
  )
  const [videos, setVideos] = useState<VideoPreview[]>([])

  useEffect(() => {
    let cancelled = false

    const initialVideos: VideoPreview[] = videoUrls.slice(0, 2).map(url => ({
      url,
      title: 'Watch related discussion video',
    }))
    setVideos(initialVideos)

    Promise.all(
      videoUrls.slice(0, 2).map(async url => {
        if (!url.includes('tiktok.com/')) {
          return { url, title: 'Watch related video' } satisfies VideoPreview
        }
        try {
          const response = await fetch(
            `https://www.tiktok.com/oembed?url=${encodeURIComponent(url)}`,
          )
          if (!response.ok) return { url, title: 'Watch on TikTok' }
          const data = await response.json()
          return {
            url,
            title: data.title?.trim() || 'Watch on TikTok',
          } satisfies VideoPreview
        } catch {
          return { url, title: 'Watch on TikTok' } satisfies VideoPreview
        }
      }),
    ).then(results => {
      if (!cancelled) setVideos(results)
    })

    return () => {
      cancelled = true
    }
  }, [videoUrls])

  if (videos.length === 0) return null

  return (
    <div className="mt-2.5 flex flex-wrap gap-2">
      {videos.map(video => (
        <a
          key={video.url}
          href={video.url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 rounded-md border border-white/8 bg-raised/50 px-2.5 py-1 text-xs text-muted transition-colors hover:border-accent/40 hover:text-accent"
        >
          <Play size={10} weight="fill" className="text-accent" />
          <span className="line-clamp-1 max-w-[260px] text-[11px]">
            {video.title}
          </span>
          <ArrowSquareOut size={11} className="opacity-60" />
        </a>
      ))}
    </div>
  )
}
