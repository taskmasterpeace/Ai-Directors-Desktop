import { Download, Heart, Search, Trash2, X, FolderOpen, Package } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'

// ── Types ──────────────────────────────────────────────────────────

interface CivitaiLoraResult {
  civitaiModelId: number
  civitaiVersionId: number
  name: string
  description: string
  thumbnailUrl: string
  triggerPhrase: string
  baseModel: string
  downloadUrl: string
  fileSizeBytes: number
  fileName: string
  stats: {
    downloadCount: number
    favoriteCount: number
    thumbsUpCount: number
    rating: number
  }
  isDownloaded: boolean
}

interface LoraLibraryEntry {
  id: string
  name: string
  file_path: string
  file_size_bytes: number
  thumbnail_url: string
  trigger_phrase: string
  base_model: string
  civitai_model_id: number | null
  civitai_version_id: number | null
  description: string
}

interface LoraBrowserProps {
  isOpen: boolean
  onClose: () => void
  onSelectLora: (filePath: string, triggerPhrase: string, weight: number) => void
}

// ── Helpers ────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`
}

function formatNumber(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
  return String(n)
}

// ── Component ──────────────────────────────────────────────────────

export function LoraBrowser({ isOpen, onClose, onSelectLora }: LoraBrowserProps) {
  const [activeTab, setActiveTab] = useState<'browse' | 'library'>('library')
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<CivitaiLoraResult[]>([])
  const [library, setLibrary] = useState<LoraLibraryEntry[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [downloading, setDownloading] = useState<Set<number>>(new Set())
  const [error, setError] = useState<string | null>(null)
  const [backendUrl, setBackendUrl] = useState('')
  const [baseModelFilter, setBaseModelFilter] = useState('Flux.1 D')
  const [sortBy, setSortBy] = useState('Most Downloaded')

  useEffect(() => {
    window.electronAPI.getBackendUrl().then(setBackendUrl)
  }, [])

  // Load library on open
  useEffect(() => {
    if (isOpen && backendUrl) {
      loadLibrary()
    }
  }, [isOpen, backendUrl])

  const loadLibrary = useCallback(async () => {
    if (!backendUrl) return
    try {
      const resp = await fetch(`${backendUrl}/api/lora/library`)
      if (resp.ok) {
        const data = await resp.json()
        setLibrary(data.entries || [])
      }
    } catch { /* ignore */ }
  }, [backendUrl])

  const searchCivitai = useCallback(async () => {
    if (!backendUrl) return
    setIsSearching(true)
    setError(null)
    try {
      const resp = await fetch(`${backendUrl}/api/lora/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          baseModel: baseModelFilter,
          sort: sortBy,
          limit: 20,
          nsfw: false,
        }),
      })
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}))
        throw new Error(errData.error || `Search failed (${resp.status})`)
      }
      const data = await resp.json()
      setResults(data.items || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed')
    } finally {
      setIsSearching(false)
    }
  }, [backendUrl, query, baseModelFilter, sortBy])

  const downloadLora = useCallback(async (item: CivitaiLoraResult) => {
    if (!backendUrl) return
    setDownloading(prev => new Set(prev).add(item.civitaiModelId))
    try {
      const resp = await fetch(`${backendUrl}/api/lora/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          downloadUrl: item.downloadUrl,
          fileName: item.fileName,
          name: item.name,
          thumbnailUrl: item.thumbnailUrl,
          triggerPhrase: item.triggerPhrase,
          baseModel: item.baseModel,
          civitaiModelId: item.civitaiModelId,
          civitaiVersionId: item.civitaiVersionId,
          description: item.description,
        }),
      })
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}))
        throw new Error(errData.error || 'Download failed')
      }
      // Mark as downloaded in results
      setResults(prev => prev.map(r =>
        r.civitaiModelId === item.civitaiModelId ? { ...r, isDownloaded: true } : r
      ))
      await loadLibrary()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed')
    } finally {
      setDownloading(prev => {
        const next = new Set(prev)
        next.delete(item.civitaiModelId)
        return next
      })
    }
  }, [backendUrl, loadLibrary])

  const deleteLora = useCallback(async (id: string) => {
    if (!backendUrl) return
    try {
      await fetch(`${backendUrl}/api/lora/library/${id}`, { method: 'DELETE' })
      await loadLibrary()
    } catch { /* ignore */ }
  }, [backendUrl, loadLibrary])

  const selectLibraryLora = useCallback((entry: LoraLibraryEntry) => {
    onSelectLora(entry.file_path, entry.trigger_phrase, 1.0)
    onClose()
  }, [onSelectLora, onClose])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-[900px] max-h-[80vh] bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-800">
          <div className="flex items-center gap-3">
            <Package className="w-5 h-5 text-purple-400" />
            <h2 className="text-sm font-semibold text-zinc-200">LoRA Library</h2>
          </div>
          <button onClick={onClose} className="p-1 text-zinc-500 hover:text-zinc-300 rounded-lg hover:bg-zinc-800">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 px-5 pt-3">
          {(['library', 'browse'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                activeTab === tab
                  ? 'bg-purple-600/20 text-purple-400 border border-purple-500/30'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800'
              }`}
            >
              {tab === 'library' ? 'My LoRAs' : 'Browse CivitAI'}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {activeTab === 'browse' ? (
            <BrowseTab
              query={query}
              setQuery={setQuery}
              results={results}
              isSearching={isSearching}
              error={error}
              baseModelFilter={baseModelFilter}
              setBaseModelFilter={setBaseModelFilter}
              sortBy={sortBy}
              setSortBy={setSortBy}
              downloading={downloading}
              onSearch={searchCivitai}
              onDownload={downloadLora}
            />
          ) : (
            <LibraryTab
              library={library}
              backendUrl={backendUrl}
              onSelect={selectLibraryLora}
              onDelete={deleteLora}
            />
          )}
        </div>
      </div>
    </div>
  )
}

// ── Browse Tab ─────────────────────────────────────────────────────

function BrowseTab({
  query, setQuery, results, isSearching, error,
  baseModelFilter, setBaseModelFilter, sortBy, setSortBy,
  downloading, onSearch, onDownload,
}: {
  query: string
  setQuery: (q: string) => void
  results: CivitaiLoraResult[]
  isSearching: boolean
  error: string | null
  baseModelFilter: string
  setBaseModelFilter: (f: string) => void
  sortBy: string
  setSortBy: (s: string) => void
  downloading: Set<number>
  onSearch: () => void
  onDownload: (item: CivitaiLoraResult) => void
}) {
  return (
    <div className="space-y-4">
      {/* Search Bar */}
      <div className="flex gap-2">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-500" />
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && onSearch()}
            placeholder="Search LoRAs on CivitAI..."
            className="w-full pl-9 pr-3 py-2 text-xs bg-zinc-800 border border-zinc-700 rounded-lg text-zinc-300 placeholder-zinc-600 focus:outline-none focus:border-purple-500/50"
          />
        </div>
        <select
          value={baseModelFilter}
          onChange={e => setBaseModelFilter(e.target.value)}
          className="px-2 py-2 text-xs bg-zinc-800 border border-zinc-700 rounded-lg text-zinc-300"
        >
          <option value="">All Models</option>
          <option value="Flux.1 D">FLUX.1 Dev</option>
          <option value="Flux.1 S">FLUX.1 Schnell</option>
          <option value="SD 1.5">SD 1.5</option>
          <option value="SDXL 1.0">SDXL</option>
        </select>
        <select
          value={sortBy}
          onChange={e => setSortBy(e.target.value)}
          className="px-2 py-2 text-xs bg-zinc-800 border border-zinc-700 rounded-lg text-zinc-300"
        >
          <option value="Most Downloaded">Most Downloaded</option>
          <option value="Highest Rated">Highest Rated</option>
          <option value="Newest">Newest</option>
        </select>
        <button
          onClick={onSearch}
          disabled={isSearching}
          className="px-4 py-2 text-xs font-medium bg-purple-600 hover:bg-purple-500 text-white rounded-lg disabled:opacity-50 transition-colors"
        >
          {isSearching ? 'Searching...' : 'Search'}
        </button>
      </div>

      {error && (
        <div className="px-3 py-2 text-xs text-red-400 bg-red-900/20 border border-red-800/30 rounded-lg">
          {error}
        </div>
      )}

      {/* Results Grid */}
      <div className="grid grid-cols-2 gap-3">
        {results.map(item => (
          <CivitaiCard
            key={item.civitaiModelId}
            item={item}
            isDownloading={downloading.has(item.civitaiModelId)}
            onDownload={() => onDownload(item)}
          />
        ))}
      </div>

      {results.length === 0 && !isSearching && !error && (
        <div className="flex flex-col items-center justify-center py-12 text-zinc-600">
          <Search className="w-8 h-8 mb-3 opacity-40" />
          <p className="text-sm">Search CivitAI for LoRAs</p>
          <p className="text-xs mt-1">Try "anime style", "realistic", "DC animation"...</p>
        </div>
      )}
    </div>
  )
}

function CivitaiCard({ item, isDownloading, onDownload }: {
  item: CivitaiLoraResult
  isDownloading: boolean
  onDownload: () => void
}) {
  return (
    <div className="group bg-zinc-800/50 border border-zinc-700/50 rounded-xl overflow-hidden hover:border-zinc-600 transition-colors">
      {/* Thumbnail */}
      <div className="relative aspect-square bg-zinc-900 overflow-hidden">
        {item.thumbnailUrl ? (
          <img
            src={item.thumbnailUrl}
            alt={item.name}
            className="w-full h-full object-cover"
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-zinc-700">
            <Package className="w-12 h-12" />
          </div>
        )}
        {/* Base model badge */}
        {item.baseModel && (
          <span className="absolute top-2 left-2 px-2 py-0.5 text-[10px] font-medium bg-black/60 backdrop-blur-sm text-zinc-300 rounded-md">
            {item.baseModel}
          </span>
        )}
        {/* Downloaded badge */}
        {item.isDownloaded && (
          <span className="absolute top-2 right-2 px-2 py-0.5 text-[10px] font-medium bg-green-600/80 text-white rounded-md">
            Downloaded
          </span>
        )}
      </div>

      {/* Info */}
      <div className="p-3 space-y-2">
        <h3 className="text-xs font-medium text-zinc-200 truncate" title={item.name}>
          {item.name}
        </h3>

        {item.triggerPhrase && (
          <p className="text-[10px] text-zinc-500 truncate" title={item.triggerPhrase}>
            Trigger: {item.triggerPhrase}
          </p>
        )}

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 text-[10px] text-zinc-600">
            <span className="flex items-center gap-1">
              <Download className="w-3 h-3" />
              {formatNumber(item.stats.downloadCount)}
            </span>
            <span className="flex items-center gap-1">
              <Heart className="w-3 h-3" />
              {formatNumber(item.stats.favoriteCount)}
            </span>
            <span>{formatBytes(item.fileSizeBytes)}</span>
          </div>

          {item.isDownloaded ? (
            <span className="text-[10px] text-green-500">Installed</span>
          ) : (
            <button
              onClick={onDownload}
              disabled={isDownloading}
              className="flex items-center gap-1 px-2.5 py-1 text-[10px] font-medium bg-purple-600 hover:bg-purple-500 text-white rounded-md disabled:opacity-50 transition-colors"
            >
              <Download className="w-3 h-3" />
              {isDownloading ? 'Downloading...' : 'Download'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Library Tab ────────────────────────────────────────────────────

function LibraryTab({ library, backendUrl, onSelect, onDelete }: {
  library: LoraLibraryEntry[]
  backendUrl: string
  onSelect: (entry: LoraLibraryEntry) => void
  onDelete: (id: string) => void
}) {
  if (library.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-zinc-600">
        <FolderOpen className="w-10 h-10 mb-3 opacity-30" />
        <p className="text-sm font-medium text-zinc-500">No LoRAs installed</p>
        <p className="text-xs mt-1">Browse CivitAI to download LoRAs, or import local files</p>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-3 gap-3">
      {library.map(entry => (
        <div
          key={entry.id}
          className="group relative bg-zinc-800/50 border border-zinc-700/50 rounded-xl overflow-hidden hover:border-purple-500/40 cursor-pointer transition-colors"
          onClick={() => onSelect(entry)}
        >
          {/* Thumbnail */}
          <div className="relative aspect-square bg-zinc-900 overflow-hidden">
            {entry.thumbnail_url ? (
              entry.thumbnail_url.startsWith('http') ? (
                <img src={entry.thumbnail_url} alt={entry.name} className="w-full h-full object-cover" loading="lazy" />
              ) : (
                <img src={`${backendUrl}/api/lora/thumbnail/${entry.id}`} alt={entry.name} className="w-full h-full object-cover" loading="lazy" />
              )
            ) : (
              <div className="w-full h-full flex items-center justify-center text-zinc-700">
                <Package className="w-10 h-10" />
              </div>
            )}
            {entry.base_model && (
              <span className="absolute top-2 left-2 px-1.5 py-0.5 text-[9px] font-medium bg-black/60 backdrop-blur-sm text-zinc-400 rounded">
                {entry.base_model}
              </span>
            )}
          </div>

          {/* Info */}
          <div className="p-2.5 space-y-1">
            <h3 className="text-[11px] font-medium text-zinc-200 truncate">{entry.name}</h3>
            {entry.trigger_phrase && (
              <p className="text-[9px] text-zinc-500 truncate">{entry.trigger_phrase}</p>
            )}
            <p className="text-[9px] text-zinc-600">{formatBytes(entry.file_size_bytes)}</p>
          </div>

          {/* Delete button (appears on hover) */}
          <button
            onClick={(e) => { e.stopPropagation(); onDelete(entry.id) }}
            className="absolute top-2 right-2 p-1 bg-black/60 backdrop-blur-sm rounded-md text-zinc-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
            title="Delete LoRA"
          >
            <Trash2 className="w-3 h-3" />
          </button>
        </div>
      ))}
    </div>
  )
}
