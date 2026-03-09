import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import { Folder, File, ChevronRight, ChevronDown, RefreshCw, Loader, Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog'
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip'
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem } from '@/components/ui/dropdown-menu'
import { cn } from '@/lib/utils'

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
      <div
        className="flex items-center gap-2 py-1 px-1 rounded-md cursor-pointer hover:bg-accent transition-colors"
        onClick={handleToggle}
      >
        {isDir ? (
          loading ? <Loader className="h-4 w-4 animate-spin-slow" /> :
          hasChildren ? (expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />) :
          <span className="w-4 inline-block" />
        ) : <span className="w-4 inline-block" />}

        {isDir
          ? <Folder className="h-4 w-4 text-amber-500" />
          : <File className="h-4 w-4 text-muted-foreground" />
        }
        <span className="text-sm">{node.name}</span>

        <div className="flex items-center gap-1 ml-1">
          {node.groups?.map(g => (
            <Tooltip key={g.group_id}>
              <TooltipTrigger asChild>
                <Badge
                  variant="info"
                  className="cursor-pointer hover:opacity-80 text-xs"
                  onClick={(e) => { e.stopPropagation(); onRemove(node.path, g.group_id, g.group_name) }}
                >
                  {g.group_name} &times;
                </Badge>
              </TooltipTrigger>
              <TooltipContent>Click to remove {g.group_name}</TooltipContent>
            </Tooltip>
          ))}
        </div>

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
  const available = groups.filter(g => !assigned.some(a => a.group_id === g.id))

  if (available.length === 0) return null

  return (
    <span className="inline-flex items-center" onClick={e => e.stopPropagation()}>
      <DropdownMenu>
        <Tooltip>
          <TooltipTrigger asChild>
            <DropdownMenuTrigger asChild>
              <button className="inline-flex items-center justify-center h-5 w-5 rounded-full bg-primary text-primary-foreground hover:bg-primary/90 transition-colors">
                <Plus className="h-3 w-3" />
              </button>
            </DropdownMenuTrigger>
          </TooltipTrigger>
          <TooltipContent>Assign group</TooltipContent>
        </Tooltip>
        <DropdownMenuContent align="start">
          {available.map(g => (
            <DropdownMenuItem key={g.id} onClick={() => onAssign(path, g.id)}>
              {g.name}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
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
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">Path Permissions</h1>
        <Button onClick={handleRefresh} disabled={refreshing}>
          <RefreshCw className={cn("h-4 w-4", refreshing && "animate-spin-slow")} />
          {refreshing ? 'Refreshing...' : 'Refresh Document Permissions'}
        </Button>
      </div>

      <Card>
        <CardContent className="pt-6">
          <p className="text-sm text-muted-foreground mb-4">
            Click a folder to expand. Click [+] to assign a group. Click a badge to remove it. Permissions inherit downward.
          </p>
          {root ? (
            <TreeNode
              node={root}
              groups={groups}
              onAssign={handleAssign}
              onRemove={(path, groupId, groupName) => setConfirmRemove({ path, groupId, groupName })}
              onLoadChildren={loadChildren}
            />
          ) : (
            <p className="text-muted-foreground">Loading...</p>
          )}
        </CardContent>
      </Card>

      {/* Confirm Remove Dialog */}
      <Dialog open={!!confirmRemove} onOpenChange={() => setConfirmRemove(null)}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Remove Permission</DialogTitle>
            <DialogDescription>Remove "{confirmRemove?.groupName}" from this path?</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmRemove(null)}>Cancel</Button>
            <Button variant="destructive" onClick={() => { handleRemoveConfirmed(); setConfirmRemove(null) }}>Remove</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Alert Dialog */}
      <Dialog open={alert.open} onOpenChange={() => setAlert({ ...alert, open: false })}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{alert.title}</DialogTitle>
            <DialogDescription>{alert.message}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button onClick={() => setAlert({ ...alert, open: false })}>OK</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
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
