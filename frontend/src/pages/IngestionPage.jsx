import { useState, useEffect, useRef, useCallback } from 'react'
import { api } from '../api/client'
import { Play, Square, FolderInput, RefreshCw } from 'lucide-react'

const STATUS_COLORS = {
  running: 'var(--primary, #2563eb)',
  completed: 'var(--success, #155724)',
  failed: 'var(--danger, #c00)',
  cancelled: 'var(--warning, #856404)',
}

const LOG_COLORS = {
  ERROR: 'var(--danger, #c00)',
  WARNING: 'var(--warning, #856404)',
  INFO: 'inherit',
}

export default function IngestionPage() {
  const [status, setStatus] = useState(null)
  const [logs, setLogs] = useState([])
  const [logOffset, setLogOffset] = useState(0)
  const [starting, setStarting] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [message, setMessage] = useState(null)
  const [folders, setFolders] = useState('')
  const [force, setForce] = useState(false)
  const [workers, setWorkers] = useState(3)
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

  // Initial load
  useEffect(() => {
    fetchStatus().then(s => {
      if (s && s.id) fetchLogs(0)
    })
  }, [fetchStatus, fetchLogs])

  // Polling while active
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

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const handleStart = async () => {
    setStarting(true)
    setMessage(null)
    setLogs([])
    setLogOffset(0)
    try {
      const opts = { workers }
      if (folders.trim()) opts.folders = folders.trim()
      if (force) opts.force = true
      await api.startIngestion(opts)
      setMessage({ type: 'success', text: 'Ingestion started.' })
      await fetchStatus()
    } catch (e) {
      setMessage({ type: 'error', text: e.message })
    } finally {
      setStarting(false)
    }
  }

  const handleCancel = async () => {
    setCancelling(true)
    try {
      await api.cancelIngestion()
      setMessage({ type: 'warning', text: 'Cancel signal sent. Waiting for current files to finish...' })
      await fetchStatus()
    } catch (e) {
      setMessage({ type: 'error', text: e.message })
    } finally {
      setCancelling(false)
    }
  }

  const isRunning = status?.active === true
  const progress = status?.total_files > 0
    ? Math.round(((status.processed + status.skipped + status.failed) / status.total_files) * 100)
    : 0

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
        <h1 style={{ margin: 0 }}>Ingestion</h1>
      </div>

      {message && (
        <div style={{ marginBottom: '1rem', padding: '0.75rem 1rem', borderRadius: '6px', background: message.type === 'error' ? 'var(--danger-bg, #fee)' : message.type === 'warning' ? 'var(--warning-bg, #fff3cd)' : 'var(--success-bg, #d4edda)', color: message.type === 'error' ? 'var(--danger, #c00)' : message.type === 'warning' ? 'var(--warning, #856404)' : 'var(--success, #155724)', border: '1px solid currentColor', fontSize: '0.875rem' }}>
          {message.text}
          <button onClick={() => setMessage(null)} style={{ float: 'right', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 'bold', color: 'inherit' }}>&times;</button>
        </div>
      )}

      {/* Controls */}
      <div className="card" style={{ padding: '1.25rem', marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          {!isRunning ? (
            <button className="btn btn-primary" onClick={handleStart} disabled={starting} style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
              <Play size={16} /> {starting ? 'Starting...' : 'Start Ingestion'}
            </button>
          ) : (
            <button className="btn btn-danger" onClick={handleCancel} disabled={cancelling} style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
              <Square size={16} /> {cancelling ? 'Cancelling...' : 'Cancel'}
            </button>
          )}
          <button className="btn btn-sm" onClick={() => setShowOptions(o => !o)} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <FolderInput size={14} /> Options
          </button>
          {!isRunning && status?.id && (
            <button className="btn btn-sm" onClick={() => { fetchStatus(); fetchLogs(0).then(() => setLogs([])); setLogOffset(0); fetchLogs(0) }} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <RefreshCw size={14} /> Refresh
            </button>
          )}
        </div>

        {showOptions && (
          <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, marginBottom: '0.25rem' }}>Folders (pipe-separated)</label>
              <input type="text" value={folders} onChange={e => setFolders(e.target.value)} placeholder="e.g. regulations|policies" style={{ width: '250px', fontFamily: 'monospace', fontSize: '0.8rem' }} disabled={isRunning} />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, marginBottom: '0.25rem' }}>Workers</label>
              <input type="number" value={workers} onChange={e => setWorkers(Math.max(1, parseInt(e.target.value) || 1))} min={1} max={10} style={{ width: '70px', fontSize: '0.8rem' }} disabled={isRunning} />
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', fontSize: '0.8rem', cursor: 'pointer' }}>
              <input type="checkbox" checked={force} onChange={e => setForce(e.target.checked)} disabled={isRunning} />
              Force re-ingest
            </label>
          </div>
        )}
      </div>

      {/* Status card */}
      {status?.id && (
        <div className="card" style={{ padding: '1.25rem', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
            <h3 style={{ margin: 0, fontSize: '1rem' }}>Job {status.id}</h3>
            <span className="badge" style={{ background: STATUS_COLORS[status.status] || '#888', color: '#fff', fontSize: '0.7rem', padding: '0.125rem 0.5rem' }}>
              {status.status}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '0.75rem', marginBottom: '0.75rem' }}>
            <Stat label="Total files" value={status.total_files} />
            <Stat label="Processed" value={status.processed} color="var(--success, #155724)" />
            <Stat label="Skipped" value={status.skipped} color="var(--text-dim, #888)" />
            <Stat label="Failed" value={status.failed} color="var(--danger, #c00)" />
          </div>

          {/* Progress bar */}
          {status.total_files > 0 && (
            <div style={{ marginBottom: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '0.25rem' }}>
                <span>{isRunning && status.current_file ? `Processing: ${status.current_file}` : ''}</span>
                <span>{progress}%</span>
              </div>
              <div style={{ height: '8px', background: 'var(--border, #e0e0e0)', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${progress}%`, background: STATUS_COLORS[status.status] || 'var(--primary)', borderRadius: '4px', transition: 'width 0.3s ease' }} />
              </div>
            </div>
          )}

          <div style={{ fontSize: '0.7rem', color: 'var(--text-dim, #888)' }}>
            Started by {status.started_by} at {new Date(status.started_at).toLocaleString()}
            {status.finished_at && <> &mdash; finished at {new Date(status.finished_at).toLocaleString()}</>}
          </div>
        </div>
      )}

      {/* Logs */}
      {logs.length > 0 && (
        <div className="card" style={{ padding: '1.25rem' }}>
          <h3 style={{ margin: '0 0 0.75rem 0', fontSize: '1rem', borderBottom: '1px solid var(--border, #e0e0e0)', paddingBottom: '0.5rem' }}>Logs</h3>
          <div style={{ maxHeight: '400px', overflow: 'auto', fontFamily: 'monospace', fontSize: '0.75rem', lineHeight: '1.6', background: 'var(--bg-secondary, #f8f9fa)', padding: '0.75rem', borderRadius: '4px' }}>
            {logs.map((log, i) => (
              <div key={i} style={{ color: LOG_COLORS[log.level] || 'inherit' }}>
                <span style={{ color: 'var(--text-dim, #888)' }}>{log.ts.split('T')[1]}</span>{' '}
                {log.level !== 'INFO' && <span>[{log.level}] </span>}
                {log.msg}
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}

      {!status?.id && (
        <p className="text-muted">No ingestion jobs yet. Click "Start Ingestion" to begin.</p>
      )}
    </div>
  )
}

function Stat({ label, value, color }) {
  return (
    <div>
      <div style={{ fontSize: '0.7rem', color: 'var(--text-dim, #888)', marginBottom: '0.125rem' }}>{label}</div>
      <div style={{ fontSize: '1.25rem', fontWeight: 700, color: color || 'inherit' }}>{value}</div>
    </div>
  )
}
