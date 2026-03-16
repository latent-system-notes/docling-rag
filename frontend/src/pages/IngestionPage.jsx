import { useState, useEffect, useRef, useCallback } from 'react'
import { api } from '../api/client'
import { Play, Square, FolderInput, RefreshCw } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Switch } from '@/components/ui/switch'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from '@/lib/utils'

const STATUS_COLORS = {
  running: 'bg-blue-600',
  completed: 'bg-emerald-600',
  failed: 'bg-red-600',
  cancelled: 'bg-amber-600',
}

const LOG_COLORS = {
  ERROR: 'text-red-600',
  WARNING: 'text-amber-600',
  SKIP: 'text-muted-foreground',
  INFO: '',
}

export default function IngestionPage() {
  const [status, setStatus] = useState(null)
  const [logs, setLogs] = useState([])
  const [logOffset, setLogOffset] = useState(0)
  const [starting, setStarting] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [folders, setFolders] = useState('')
  const [force, setForce] = useState(false)
  const [workers, setWorkers] = useState(3)
  const [ocrMode, setOcrMode] = useState('smart')
  const [showOptions, setShowOptions] = useState(false)

  const logsEndRef = useRef(null)
  const pollRef = useRef(null)

  const fetchStatus = useCallback(async () => {
    try {
      const data = await api.getIngestionStatus()
      setStatus(data)
      return data
    } catch {
      return null
    }
  }, [])

  const fetchLogs = useCallback(async (offset) => {
    try {
      const data = await api.getIngestionLogs(offset)
      if (data.logs.length > 0) {
        setLogs(prev => [...prev, ...data.logs])
        setLogOffset(data.total)
      }
    } catch {
      // ignore
    }
  }, [])

  useEffect(() => {
    fetchStatus().then(s => {
      if (s && s.id) fetchLogs(0)
    })
  }, [fetchStatus, fetchLogs])

  useEffect(() => {
    if (status?.active) {
      pollRef.current = setInterval(async () => {
        const s = await fetchStatus()
        await fetchLogs(logOffset)
        if (s && !s.active) clearInterval(pollRef.current)
      }, 2000)
      return () => clearInterval(pollRef.current)
    }
  }, [status?.active, logOffset, fetchStatus, fetchLogs])

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const handleStart = async () => {
    setStarting(true)
    setLogs([])
    setLogOffset(0)
    try {
      const opts = { workers, ocr_mode: ocrMode }
      if (folders.trim()) opts.folders = folders.trim()
      if (force) opts.force = true
      await api.startIngestion(opts)
      toast.success('Ingestion started.')
      await fetchStatus()
    } catch (e) {
      toast.error(e.message)
    } finally {
      setStarting(false)
    }
  }

  const handleCancel = async () => {
    setCancelling(true)
    try {
      await api.cancelIngestion()
      toast.warning('Cancel signal sent. Waiting for current files to finish...')
      await fetchStatus()
    } catch (e) {
      toast.error(e.message)
    } finally {
      setCancelling(false)
    }
  }

  const isRunning = status?.active === true
  const handled = (status?.processed ?? 0) + (status?.skipped ?? 0) + (status?.failed ?? 0)
  const progress = status?.scan_complete && status?.total_files > 0
    ? Math.round((handled / status.total_files) * 100)
    : (status?.total_files > 0 ? Math.min(Math.round((handled / status.total_files) * 100), 99) : 0)

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight">Ingestion</h1>

      {/* Controls */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center gap-3 flex-wrap">
            {!isRunning ? (
              <Button onClick={handleStart} disabled={starting}>
                <Play className="h-4 w-4" /> {starting ? 'Starting...' : 'Start Ingestion'}
              </Button>
            ) : (
              <Button variant="destructive" onClick={handleCancel} disabled={cancelling}>
                <Square className="h-4 w-4" /> {cancelling ? 'Cancelling...' : 'Cancel'}
              </Button>
            )}
            <Button variant="outline" size="sm" onClick={() => setShowOptions(o => !o)}>
              <FolderInput className="h-4 w-4" /> Options
            </Button>
            {!isRunning && status?.id && (
              <Button variant="outline" size="sm" onClick={() => { fetchStatus(); setLogs([]); setLogOffset(0); fetchLogs(0) }}>
                <RefreshCw className="h-4 w-4" /> Refresh
              </Button>
            )}
          </div>

          {showOptions && (
            <div className="flex items-end gap-4 flex-wrap mt-4 pt-4 border-t">
              <div className="space-y-2">
                <Label className="text-xs font-semibold">Folders (pipe-separated)</Label>
                <Input
                  value={folders}
                  onChange={e => setFolders(e.target.value)}
                  placeholder="e.g. regulations|policies"
                  className="w-[250px] font-mono text-sm"
                  disabled={isRunning}
                />
              </div>
              <div className="space-y-2">
                <Label className="text-xs font-semibold">Workers</Label>
                <Input
                  type="number"
                  value={workers}
                  onChange={e => setWorkers(Math.max(1, parseInt(e.target.value) || 1))}
                  min={1} max={10}
                  className="w-[70px] text-sm"
                  disabled={isRunning}
                />
              </div>
              <div className="flex items-center gap-2">
                <Switch checked={force} onCheckedChange={setForce} disabled={isRunning} />
                <Label className="text-sm cursor-pointer">Force re-ingest</Label>
              </div>
              <div className="space-y-2">
                <Label className="text-xs font-semibold">OCR Mode</Label>
                <div className="flex rounded-md border overflow-hidden">
                  {[['smart', 'Smart OCR'], ['simple', 'Simple OCR']].map(([value, label]) => (
                    <button
                      key={value}
                      type="button"
                      disabled={isRunning}
                      onClick={() => setOcrMode(value)}
                      className={cn(
                        'px-3 py-1.5 text-xs font-medium transition-colors',
                        ocrMode === value
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-background text-muted-foreground hover:bg-accent',
                        isRunning && 'opacity-50 cursor-not-allowed'
                      )}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Status card */}
      {status?.id && (
        <Card>
          <CardHeader className="flex-row items-center gap-3 space-y-0 pb-4">
            <CardTitle className="text-base">Job {status.id}</CardTitle>
            <Badge className={cn('text-white text-xs', STATUS_COLORS[status.status] || 'bg-gray-500')}>
              {status.status}
            </Badge>
          </CardHeader>
          <CardContent className="space-y-4">
            {status.folders_resolved && (
              <div className="rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-sm">
                <span className="font-medium text-amber-600">Folder filter active:</span>{' '}
                {status.folders_resolved.map((f, i) => (
                  <Badge key={i} variant="outline" className="ml-1 font-mono text-xs">{f}</Badge>
                ))}
                <p className="text-xs text-muted-foreground mt-1">
                  Only files inside these subfolders are ingested. Files in the root or other folders are not included.
                </p>
              </div>
            )}
            {status.force && (
              <div className="rounded-md border border-blue-500/30 bg-blue-500/5 px-3 py-2 text-sm">
                <span className="font-medium text-blue-600">Force re-ingest:</span>{' '}
                All files will be re-processed even if already ingested.
              </div>
            )}
            <div className={cn(
              "rounded-md border px-3 py-2 text-sm",
              status.ocr_mode === 'simple'
                ? "border-gray-500/30 bg-gray-500/5"
                : "border-violet-500/30 bg-violet-500/5"
            )}>
              <span className={cn("font-medium", status.ocr_mode === 'simple' ? "text-gray-600" : "text-violet-600")}>
                {status.ocr_mode === 'simple' ? 'Simple OCR:' : 'Smart OCR:'}
              </span>{' '}
              {status.ocr_mode === 'simple'
                ? 'Fast mode — no OCR probe, image description, or image classification.'
                : 'Full mode — OCR probe per PDF, image description & classification enabled.'}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Stat label={status.scan_complete ? "Total files" : "Discovered"} value={status.scan_complete ? status.total_files : `${status.total_files}...`} />
              <Stat label="Processed" value={status.processed} className="text-emerald-600" />
              <Stat label="Skipped" value={status.skipped} className="text-muted-foreground" />
              <Stat label="Failed" value={status.failed} className="text-red-600" />
            </div>

            {status.total_files > 0 && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{isRunning && status.current_file ? `Processing: ${status.current_file}` : ''}</span>
                  <span>{progress}%</span>
                </div>
                <Progress value={progress} />
              </div>
            )}

            <p className="text-xs text-muted-foreground">
              Started by {status.started_by} at {new Date(status.started_at).toLocaleString()}
              {status.finished_at && <> &mdash; finished at {new Date(status.finished_at).toLocaleString()}</>}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Logs */}
      {logs.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Logs</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="max-h-[400px] rounded-md">
              <div className="font-mono text-xs leading-6 bg-muted rounded-md p-3">
                {logs.map((log, i) => (
                  <div key={i} className={LOG_COLORS[log.level] || ''}>
                    <span className="text-muted-foreground">{log.ts.split('T')[1]}</span>{' '}
                    {!['INFO'].includes(log.level) && <span>[{log.level}] </span>}
                    {log.msg}
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}

      {!status?.id && (
        <p className="text-muted-foreground">No ingestion jobs yet. Click "Start Ingestion" to begin.</p>
      )}
    </div>
  )
}

function Stat({ label, value, className }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground mb-0.5">{label}</p>
      <p className={cn("text-xl font-bold", className)}>{value}</p>
    </div>
  )
}
