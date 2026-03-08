import { useEffect, useRef } from 'react'
import { X } from 'lucide-react'

export function Dialog({ open, onClose, title, children, width = 420 }) {
  const overlayRef = useRef(null)

  useEffect(() => {
    if (!open) return
    const handleKey = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div ref={overlayRef} className="dialog-overlay" onClick={(e) => { if (e.target === overlayRef.current) onClose() }}>
      <div className="card dialog-card" style={{ width, maxWidth: '95vw' }}>
        <div className="dialog-header">
          <h3 style={{ margin: 0 }}>{title}</h3>
          <button className="btn-icon" onClick={onClose}><X size={18} /></button>
        </div>
        {children}
      </div>
    </div>
  )
}

export function ConfirmDialog({ open, onClose, onConfirm, title = 'Confirm', message, confirmLabel = 'Confirm', danger = false }) {
  return (
    <Dialog open={open} onClose={onClose} title={title} width={380}>
      <p style={{ margin: '1rem 0', fontSize: '0.9rem', color: 'var(--text-muted)' }}>{message}</p>
      <div className="flex gap-2" style={{ justifyContent: 'flex-end' }}>
        <button style={{ padding: '0.5rem 1rem', cursor: 'pointer', background: 'var(--bg-input)', color: 'var(--text)' }} onClick={onClose}>Cancel</button>
        <button className={danger ? 'btn-danger' : 'btn-primary'} onClick={() => { onConfirm(); onClose() }}>{confirmLabel}</button>
      </div>
    </Dialog>
  )
}

export function AlertDialog({ open, onClose, title = 'Info', message }) {
  return (
    <Dialog open={open} onClose={onClose} title={title} width={380}>
      <p style={{ margin: '1rem 0', fontSize: '0.9rem', color: 'var(--text-muted)' }}>{message}</p>
      <div className="flex gap-2" style={{ justifyContent: 'flex-end' }}>
        <button className="btn-primary" onClick={onClose}>OK</button>
      </div>
    </Dialog>
  )
}
