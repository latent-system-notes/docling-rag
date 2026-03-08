import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import { Folder, File, ChevronRight, ChevronDown, RefreshCw, Loader, Plus } from 'lucide-react'
import { ConfirmDialog, AlertDialog } from '../components/Dialog'

function TreeNode({ node, groups, onAssign, onRemove, onLoadChildren, level = 0 }) {
  const [expanded, setExpanded] = useState(level === 0)
  const [loading, setLoading] = useState(false)
  const isDir = node.type === 'directory'
  const hasChildren = isDir && (node.children?.length > 0 || node.has_children)

  const handleToggle = async () => {
    if (!isDir) return
    if (!expanded && !node.children && node.has_children) {
      setLoading(true)
      await onLoadChildren(node.path)
      setLoading(false)
    }
    setExpanded(!expanded)
  }

  return (
    <div style={{ marginLeft: level > 0 ? '1.25rem' : 0 }}>
      <div className="tree-toggle" onClick={handleToggle}>
        {isDir ? (
          loading ? <Loader size={16} className="spin" /> :
          hasChildren ? (expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />) :
          <span style={{ width: 16, display: 'inline-block' }} />
        ) : <span style={{ width: 16, display: 'inline-block' }} />}
        {isDir ? <Folder size={16} style={{ color: 'var(--warning)' }} /> : <File size={16} style={{ color: 'var(--text-muted)' }} />}
        <span>{node.name}</span>
        <span className="tree-badges">
          {node.groups?.map(g => (
            <span key={g.group_id} className="tree-badge"
              onClick={(e) => { e.stopPropagation(); onRemove(node.path, g.group_id, g.group_name) }}
              title={`Click to remove ${g.group_name}`}>
              {g.group_name} ×
            </span>
          ))}
        </span>
        <GroupAssigner path={node.path} groups={groups} assigned={node.groups || []} onAssign={onAssign} />
      </div>
      {isDir && expanded && node.children?.map((child) => (
        <TreeNode
          key={child.path}
          node={child}
          groups={groups}
          onAssign={onAssign}
          onRemove={onRemove}
          onLoadChildren={onLoadChildren}
          level={level + 1}
        />
      ))}
    </div>
  )
}

function GroupAssigner({ path, groups, assigned, onAssign }) {
  const [open, setOpen] = useState(false)
  const available = groups.filter(g => !assigned.some(a => a.group_id === g.id))

  if (available.length === 0) return null

  return (
    <span style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }} onClick={e => e.stopPropagation()}>
      <button className="tree-add-btn" onClick={() => setOpen(!open)} title="Assign group">
        <Plus size={14} />
      </button>
      {open && (
        <div className="tree-dropdown">
          {available.map(g => (
            <div key={g.id} className="tree-dropdown-item"
              onClick={() => { onAssign(path, g.id); setOpen(false) }}>
              {g.name}
            </div>
          ))}
        </div>
      )}
    </span>
  )
}

export default function PermissionsPage() {
  const [root, setRoot] = useState(null)
  const [groups, setGroups] = useState([])
  const [refreshing, setRefreshing] = useState(false)
  const [confirmRemove, setConfirmRemove] = useState(null)
  const [alert, setAlert] = useState({ open: false, title: '', message: '' })

  const loadRoot = async () => {
    const [tree, g] = await Promise.all([api.getTreeChildren(''), api.listAllGroups()])
    setRoot(tree)
    setGroups(g.items)
  }

  useEffect(() => { loadRoot() }, [])

  const loadChildren = useCallback(async (parentPath) => {
    const children = await api.getTreeChildren(parentPath)
    setRoot(prev => {
      if (!prev) return prev
      return _injectChildren(prev, parentPath, children)
    })
  }, [])

  const handleAssign = async (path, groupId) => {
    await api.addPathPermission(path, groupId)
    await _reloadNode(path)
  }

  const handleRemoveConfirmed = async () => {
    if (!confirmRemove) return
    await api.removePathPermission(confirmRemove.path, confirmRemove.groupId)
    await _reloadNode(confirmRemove.path)
  }

  const _reloadNode = async (nodePath) => {
    const parts = nodePath.split('/')
    const parentPath = parts.length > 1 ? parts.slice(0, -1).join('/') : ''
    const children = await api.getTreeChildren(parentPath || root?.path || '')
    setRoot(prev => {
      if (!prev) return prev
      if (!parentPath || parentPath === prev.path) {
        return { ...prev, children }
      }
      return _injectChildren(prev, parentPath, children)
    })
  }

  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      const result = await api.refreshPermissions()
      setAlert({ open: true, title: 'Refresh Complete', message: `Refreshed permissions for ${result.refreshed} documents.` })
    } finally {
      setRefreshing(false)
    }
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h2>Path Permissions</h2>
        <button className="btn-primary" onClick={handleRefresh} disabled={refreshing}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <RefreshCw size={14} className={refreshing ? 'spin' : ''} />
          {refreshing ? 'Refreshing...' : 'Refresh Document Permissions'}
        </button>
      </div>

      <div className="card">
        <div className="text-sm text-muted mb-4">
          Click a folder to expand. Click [+] to assign a group. Click a badge to remove it. Permissions inherit downward.
        </div>
        {root ? (
          <TreeNode
            node={root}
            groups={groups}
            onAssign={handleAssign}
            onRemove={(path, groupId, groupName) => setConfirmRemove({ path, groupId, groupName })}
            onLoadChildren={loadChildren}
          />
        ) : (
          <div className="text-muted">Loading...</div>
        )}
      </div>

      <ConfirmDialog
        open={!!confirmRemove}
        onClose={() => setConfirmRemove(null)}
        onConfirm={handleRemoveConfirmed}
        title="Remove Permission"
        message={`Remove "${confirmRemove?.groupName}" from this path?`}
        confirmLabel="Remove"
        danger
      />

      <AlertDialog open={alert.open} onClose={() => setAlert({ ...alert, open: false })} title={alert.title} message={alert.message} />
    </div>
  )
}

function _injectChildren(node, parentPath, children) {
  if (node.path === parentPath) {
    return { ...node, children }
  }
  if (!node.children) return node
  return {
    ...node,
    children: node.children.map(c => _injectChildren(c, parentPath, children)),
  }
}
