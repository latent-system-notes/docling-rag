import { useState, useEffect, useCallback, useMemo } from 'react'
import { api } from '../api/client'
import { Pencil, Trash2, Plus, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip'
import { DataTable, DataTablePagination, DataTableCard, SortableHeader } from '@/components/ui/data-table'

function toKebabInput(str) {
  return str.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '').replace(/-{2,}/g, '-').replace(/^-/, '')
}
function toKebabFinal(str) {
  return toKebabInput(str).replace(/-$/, '')
}

const PAGE_SIZES = [10, 15, 20, 50]

export default function GroupsPage() {
  const [groups, setGroups] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(15)
  const [search, setSearch] = useState('')
  const [showCreate, setShowCreate] = useState(false)
  const [form, setForm] = useState({ name: '', description: '' })
  const [editGroup, setEditGroup] = useState(null)
  const [editForm, setEditForm] = useState({ name: '', description: '' })
  const [confirmDelete, setConfirmDelete] = useState(null)

  const loadGroups = useCallback(async () => {
    const res = await api.listGroups(search, page, pageSize)
    setGroups(res.items)
    setTotal(res.total)
  }, [search, page, pageSize])

  useEffect(() => { loadGroups() }, [loadGroups])

  const handleSearch = (v) => { setSearch(v); setPage(1) }

  const handleCreate = async (e) => {
    e.preventDefault()
    const name = toKebabFinal(form.name)
    if (!name) return
    await api.createGroup({ name, description: form.description })
    setForm({ name: '', description: '' })
    setShowCreate(false)
    loadGroups()
  }

  const handleEdit = (g) => {
    setEditGroup(g)
    setEditForm({ name: g.name, description: g.description })
  }

  const handleUpdate = async (e) => {
    e.preventDefault()
    const name = toKebabFinal(editForm.name)
    if (!name) return
    await api.updateGroup(editGroup.id, { name, description: editForm.description })
    setEditGroup(null)
    loadGroups()
  }

  const handleDelete = async (id) => {
    await api.deleteGroup(id)
    if (editGroup?.id === id) setEditGroup(null)
    loadGroups()
  }

  const columns = useMemo(() => [
    {
      accessorKey: 'id',
      header: ({ column }) => <SortableHeader column={column} title="ID" />,
      cell: ({ getValue }) => <span className="text-muted-foreground">{getValue()}</span>,
      size: 70,
    },
    {
      accessorKey: 'name',
      header: ({ column }) => <SortableHeader column={column} title="Name" />,
      cell: ({ getValue }) => <Badge variant="info">{getValue()}</Badge>,
      size: 180,
    },
    {
      accessorKey: 'description',
      header: ({ column }) => <SortableHeader column={column} title="Description" />,
      size: 300,
    },
    {
      accessorKey: 'created_at',
      header: ({ column }) => <SortableHeader column={column} title="Created" />,
      cell: ({ getValue }) => <span className="text-muted-foreground text-sm">{new Date(getValue()).toLocaleDateString()}</span>,
      size: 120,
    },
    {
      id: 'actions',
      header: () => <div className="text-right">Actions</div>,
      enableSorting: false,
      enableResizing: false,
      size: 100,
      cell: ({ row }) => {
        const g = row.original
        return (
          <div className="flex gap-1 justify-end">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={(e) => { e.stopPropagation(); handleEdit(g); setShowCreate(false) }}>
                  <Pencil className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Edit</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive" onClick={(e) => { e.stopPropagation(); setConfirmDelete(g) }}>
                  <Trash2 className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Delete</TooltipContent>
            </Tooltip>
          </div>
        )
      },
    },
  ], [])

  return (
    <div className="flex flex-col h-full gap-4 overflow-hidden">
      <div className="flex items-center justify-between shrink-0">
        <h1 className="text-2xl font-semibold tracking-tight">Groups</h1>
        <Button onClick={() => { setShowCreate(!showCreate); setEditGroup(null) }}>
          {showCreate ? <><X className="h-4 w-4" /> Cancel</> : <><Plus className="h-4 w-4" /> New Group</>}
        </Button>
      </div>

      {showCreate && (
        <Card className="shrink-0">
          <CardHeader><CardTitle>Create Group</CardTitle></CardHeader>
          <CardContent>
            <form onSubmit={handleCreate} className="flex flex-col sm:flex-row sm:items-end gap-4">
              <div className="space-y-2 flex-1">
                <Label>Name <span className="text-muted-foreground text-xs">(kebab-case)</span></Label>
                <Input value={form.name} onChange={e => setForm({ ...form, name: toKebabInput(e.target.value) })} placeholder="e.g. finance-team" required />
              </div>
              <div className="space-y-2 flex-1">
                <Label>Description</Label>
                <Input value={form.description} onChange={e => setForm({ ...form, description: e.target.value })} placeholder="Optional description" />
              </div>
              <Button type="submit"><Plus className="h-4 w-4" /> Create</Button>
            </form>
          </CardContent>
        </Card>
      )}

      <Card className="shrink-0">
        <CardContent className="py-3">
          <Input
            value={search}
            onChange={e => handleSearch(e.target.value)}
            placeholder="Search by name or description..."
            className="border-0 shadow-none focus-visible:ring-0"
          />
        </CardContent>
      </Card>

      <DataTableCard>
        <DataTable
          columns={columns}
          data={groups}
          noResultsMessage="No groups found"
        />
        <DataTablePagination
          total={total}
          page={page}
          pageSize={pageSize}
          onPageChange={setPage}
          onPageSizeChange={setPageSize}
          pageSizes={PAGE_SIZES}
          noun="group"
        />
      </DataTableCard>

      {/* Edit Group Dialog */}
      <Dialog open={!!editGroup} onOpenChange={() => setEditGroup(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Group: {editGroup?.name}</DialogTitle>
          </DialogHeader>
          <form onSubmit={handleUpdate} className="space-y-4">
            <div className="space-y-2">
              <Label>Name <span className="text-muted-foreground text-xs">(kebab-case)</span></Label>
              <Input value={editForm.name} onChange={e => setEditForm({ ...editForm, name: toKebabInput(e.target.value) })} required autoFocus />
            </div>
            <div className="space-y-2">
              <Label>Description</Label>
              <Input value={editForm.description} onChange={e => setEditForm({ ...editForm, description: e.target.value })} />
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setEditGroup(null)}>Cancel</Button>
              <Button type="submit">Save</Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Confirm Delete Dialog */}
      <Dialog open={!!confirmDelete} onOpenChange={() => setConfirmDelete(null)}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Delete Group</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground py-2">
            Delete group "{confirmDelete?.name}"? All associated permissions will be removed.
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmDelete(null)}>Cancel</Button>
            <Button variant="destructive" onClick={() => { handleDelete(confirmDelete?.id); setConfirmDelete(null) }}>Delete</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
