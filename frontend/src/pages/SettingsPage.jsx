import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import { Save, RotateCcw, RefreshCw } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Separator } from '@/components/ui/separator'
import DirectoryPicker from '@/components/DirectoryPicker'
import FoldersPicker from '@/components/FoldersPicker'

export default function SettingsPage() {
  const [settings, setSettings] = useState([])
  const [edits, setEdits] = useState({})
  const [saving, setSaving] = useState({})
  const [mcpDirty, setMcpDirty] = useState(false)
  const [reloading, setReloading] = useState(false)

  const load = useCallback(async () => {
    try {
      const data = await api.getSettings()
      setSettings(data)
      setEdits({})
    } catch (e) {
      toast.error(e.message)
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
        toast.warning(`"${key}" saved. Click "Reload MCP" to apply changes.`)
      } else if (res.restart_required) {
        toast.warning(`"${key}" saved. Server restart required for changes to take effect.`)
      } else {
        toast.success(`"${key}" updated successfully.`)
      }
      await load()
    } catch (e) {
      toast.error(e.message)
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
      toast.success(`"${key}" reverted to default.`)
      await load()
    } catch (e) {
      toast.error(e.message)
    } finally {
      setSaving(prev => ({ ...prev, [key]: false }))
    }
  }

  const handleReloadMcp = async () => {
    setReloading(true)
    try {
      await api.reloadMcp()
      setMcpDirty(false)
      toast.success('MCP reloaded successfully. New settings are now active.')
    } catch (e) {
      toast.error(`Failed to reload MCP: ${e.message}`)
    } finally {
      setReloading(false)
    }
  }

  const groups = {}
  for (const s of settings) {
    if (!groups[s.group]) groups[s.group] = []
    groups[s.group].push(s)
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>

      {Object.entries(groups).map(([groupName, items]) => {
        const groupHasReloadMcp = items.some(s => s.reload_mcp)
        return (
          <Card key={groupName}>
            <CardHeader className="flex-row items-center justify-between space-y-0 pb-4">
              <CardTitle className="text-base">{groupName}</CardTitle>
              {groupHasReloadMcp && mcpDirty && (
                <Button size="sm" onClick={handleReloadMcp} disabled={reloading}>
                  <RefreshCw className={`h-3.5 w-3.5 ${reloading ? 'animate-spin-slow' : ''}`} />
                  {reloading ? 'Reloading...' : 'Reload MCP'}
                </Button>
              )}
            </CardHeader>
            <CardContent className="space-y-4">
              {items.map((s, i) => {
                const editValue = edits[s.key] !== undefined ? edits[s.key] : s.value
                const isDirty = edits[s.key] !== undefined && edits[s.key] !== s.value
                return (
                  <div key={s.key}>
                    {i > 0 && <Separator className="mb-4" />}
                    <div className="flex items-center gap-2 mb-1">
                      <Label className="font-semibold">{s.label}</Label>
                      {s.reload_mcp && mcpDirty && <Badge variant="warning" className="text-[10px] px-1.5 py-0">MCP reload needed</Badge>}
                      {s.restart_required && <Badge variant="warning" className="text-[10px] px-1.5 py-0">restart required</Badge>}
                      {s.has_override && <Badge variant="info" className="text-[10px] px-1.5 py-0">overridden</Badge>}
                    </div>
                    <p className="text-xs text-muted-foreground mb-2">{s.key}</p>
                    <div className="space-y-2">
                      {s.multiline ? (
                        <Textarea
                          value={editValue}
                          onChange={e => handleChange(s.key, e.target.value)}
                          rows={4}
                          className="w-full font-mono text-sm"
                        />
                      ) : s.browse ? (
                        <div className="flex gap-2">
                          <Input
                            value={editValue}
                            onChange={e => handleChange(s.key, e.target.value)}
                            className="flex-1 font-mono text-sm"
                          />
                          <DirectoryPicker
                            value={editValue}
                            onSelect={(path) => handleChange(s.key, path)}
                          />
                        </div>
                      ) : s.browse_folders ? (
                        <div className="flex gap-2">
                          <Input
                            value={editValue}
                            onChange={e => handleChange(s.key, e.target.value)}
                            className="flex-1 font-mono text-sm"
                          />
                          <FoldersPicker
                            value={editValue}
                            onSelect={(val) => handleChange(s.key, val)}
                          />
                        </div>
                      ) : (
                        <Input
                          value={editValue}
                          onChange={e => handleChange(s.key, e.target.value)}
                          className="w-full font-mono text-sm"
                        />
                      )}
                      <div className="flex gap-2">
                        <Button size="sm" onClick={() => handleSave(s.key)} disabled={!isDirty || saving[s.key]}>
                          <Save className="h-3.5 w-3.5" /> Save
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => handleReset(s.key)} disabled={!s.has_override || saving[s.key]}>
                          <RotateCcw className="h-3.5 w-3.5" /> Reset
                        </Button>
                      </div>
                    </div>
                    {s.has_override && s.updated_by && (
                      <p className="text-xs text-muted-foreground mt-1">
                        Last updated by {s.updated_by} {s.updated_at ? `at ${new Date(s.updated_at).toLocaleString()}` : ''}
                      </p>
                    )}
                  </div>
                )
              })}
            </CardContent>
          </Card>
        )
      })}

      {settings.length === 0 && (
        <p className="text-muted-foreground">Loading settings...</p>
      )}
    </div>
  )
}
