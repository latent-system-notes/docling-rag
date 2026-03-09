import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import { Save, RotateCcw, RefreshCw } from 'lucide-react'

export default function SettingsPage() {
  const [settings, setSettings] = useState([])
  const [edits, setEdits] = useState({})
  const [saving, setSaving] = useState({})
  const [message, setMessage] = useState(null)
  const [mcpDirty, setMcpDirty] = useState(false)
  const [reloading, setReloading] = useState(false)

  const load = useCallback(async () => {
    try {
      const data = await api.getSettings()
      setSettings(data)
      setEdits({})
    } catch (e) {
      setMessage({ type: 'error', text: e.message })
    }
  }, [])

  useEffect(() => { load() }, [load])

  const handleChange = (key, value) => {
    setEdits(prev => ({ ...prev, [key]: value }))
  }

  const handleSave = async (key) => {
    const value = edits[key]
    if (value === undefined) return
    setSaving(prev => ({ ...prev, [key]: true }))
    try {
      const res = await api.updateSetting(key, value)
      if (res.reload_mcp) {
        setMcpDirty(true)
        setMessage({ type: 'warning', text: `"${key}" saved. Click "Reload MCP" to apply changes.` })
      } else if (res.restart_required) {
        setMessage({ type: 'warning', text: `"${key}" saved. Server restart required for changes to take effect.` })
      } else {
        setMessage({ type: 'success', text: `"${key}" updated successfully.` })
      }
      await load()
    } catch (e) {
      setMessage({ type: 'error', text: e.message })
    } finally {
      setSaving(prev => ({ ...prev, [key]: false }))
    }
  }

  const handleReset = async (key) => {
    setSaving(prev => ({ ...prev, [key]: true }))
    try {
      const res = await api.deleteSetting(key)
      if (res.reload_mcp) {
        setMcpDirty(true)
      }
      setMessage({ type: 'success', text: `"${key}" reverted to default.` })
      await load()
    } catch (e) {
      setMessage({ type: 'error', text: e.message })
    } finally {
      setSaving(prev => ({ ...prev, [key]: false }))
    }
  }

  const handleReloadMcp = async () => {
    setReloading(true)
    try {
      await api.reloadMcp()
      setMcpDirty(false)
      setMessage({ type: 'success', text: 'MCP reloaded successfully. New settings are now active.' })
    } catch (e) {
      setMessage({ type: 'error', text: `Failed to reload MCP: ${e.message}` })
    } finally {
      setReloading(false)
    }
  }

  // Group settings by category
  const groups = {}
  for (const s of settings) {
    if (!groups[s.group]) groups[s.group] = []
    groups[s.group].push(s)
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
        <h1 style={{ margin: 0 }}>Settings</h1>
      </div>

      {message && (
        <div className={`alert alert-${message.type}`} style={{ marginBottom: '1rem', padding: '0.75rem 1rem', borderRadius: '6px', background: message.type === 'error' ? 'var(--danger-bg, #fee)' : message.type === 'warning' ? 'var(--warning-bg, #fff3cd)' : 'var(--success-bg, #d4edda)', color: message.type === 'error' ? 'var(--danger, #c00)' : message.type === 'warning' ? 'var(--warning, #856404)' : 'var(--success, #155724)', border: '1px solid currentColor', fontSize: '0.875rem' }}>
          {message.text}
          <button onClick={() => setMessage(null)} style={{ float: 'right', background: 'none', border: 'none', cursor: 'pointer', fontWeight: 'bold', color: 'inherit' }}>&times;</button>
        </div>
      )}

      {Object.entries(groups).map(([groupName, items]) => {
        const groupHasReloadMcp = items.some(s => s.reload_mcp)
        return (
          <div key={groupName} className="card" style={{ marginBottom: '1.5rem', padding: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', margin: '0 0 1rem 0', borderBottom: '1px solid var(--border, #e0e0e0)', paddingBottom: '0.5rem' }}>
              <h3 style={{ margin: 0, fontSize: '1rem' }}>{groupName}</h3>
              {groupHasReloadMcp && mcpDirty && (
                <button
                  className="btn btn-primary btn-sm"
                  onClick={handleReloadMcp}
                  disabled={reloading}
                  style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}
                >
                  <RefreshCw size={14} className={reloading ? 'spinning' : ''} />
                  {reloading ? 'Reloading...' : 'Reload MCP'}
                </button>
              )}
            </div>
            {items.map(s => {
              const editValue = edits[s.key] !== undefined ? edits[s.key] : s.value
              const isDirty = edits[s.key] !== undefined && edits[s.key] !== s.value
              return (
                <div key={s.key} style={{ marginBottom: '1rem', paddingBottom: '1rem', borderBottom: '1px solid var(--border-light, #f0f0f0)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.375rem' }}>
                    <label style={{ fontWeight: 600, fontSize: '0.875rem' }}>{s.label}</label>
                    {s.reload_mcp && mcpDirty && (
                      <span className="badge badge-yellow" style={{ fontSize: '0.65rem', padding: '0.125rem 0.375rem' }}>MCP reload needed</span>
                    )}
                    {s.restart_required && (
                      <span className="badge badge-yellow" style={{ fontSize: '0.65rem', padding: '0.125rem 0.375rem' }}>restart required</span>
                    )}
                    {s.has_override && (
                      <span className="badge badge-blue" style={{ fontSize: '0.65rem', padding: '0.125rem 0.375rem' }}>overridden</span>
                    )}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-dim, #888)', marginBottom: '0.375rem' }}>{s.key}</div>
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
                    {s.multiline ? (
                      <textarea
                        value={editValue}
                        onChange={e => handleChange(s.key, e.target.value)}
                        rows={4}
                        style={{ flex: 1, fontFamily: 'monospace', fontSize: '0.8rem' }}
                      />
                    ) : (
                      <input
                        type="text"
                        value={editValue}
                        onChange={e => handleChange(s.key, e.target.value)}
                        style={{ flex: 1, fontFamily: 'monospace', fontSize: '0.8rem' }}
                      />
                    )}
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => handleSave(s.key)}
                      disabled={!isDirty || saving[s.key]}
                      title="Save"
                      style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}
                    >
                      <Save size={14} /> Save
                    </button>
                    <button
                      className="btn btn-sm"
                      onClick={() => handleReset(s.key)}
                      disabled={!s.has_override || saving[s.key]}
                      title="Reset to default"
                      style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}
                    >
                      <RotateCcw size={14} /> Reset
                    </button>
                  </div>
                  {s.has_override && s.updated_by && (
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-dim, #888)', marginTop: '0.25rem' }}>
                      Last updated by {s.updated_by} {s.updated_at ? `at ${new Date(s.updated_at).toLocaleString()}` : ''}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )
      })}

      {settings.length === 0 && (
        <p className="text-muted">Loading settings...</p>
      )}
    </div>
  )
}
