import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import { Pencil, Trash2, Plus, X, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip'
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select'

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
  const totalPages = Math.max(1, Math.ceil(total / pageSize))

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

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">Groups</h1>
        <Button onClick={() => { setShowCreate(!showCreate); setEditGroup(null) }}>
          {showCreate ? <><X className="h-4 w-4" /> Cancel</> : <><Plus className="h-4 w-4" /> New Group</>}
        </Button>
      </div>

      {showCreate && (
        <Card>
          <CardHeader>
            <CardTitle>Create Group</CardTitle>
          </CardHeader>
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

      {/* Search */}
      <Card>
        <CardContent className="py-3">
          <Input
            value={search}
            onChange={e => handleSearch(e.target.value)}
            placeholder="Search by name or description..."
            className="border-0 shadow-none focus-visible:ring-0"
          />
        </CardContent>
      </Card>

      {/* Groups table */}
      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="hidden sm:table-cell">ID</TableHead>
                <TableHead>Name</TableHead>
                <TableHead className="hidden md:table-cell">Description</TableHead>
                <TableHead className="hidden sm:table-cell">Created</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {groups.map(g => (
                <TableRow key={g.id}>
                  <TableCell className="hidden sm:table-cell text-muted-foreground">{g.id}</TableCell>
                  <TableCell><Badge variant="info">{g.name}</Badge></TableCell>
                  <TableCell className="hidden md:table-cell">{g.description}</TableCell>
                  <TableCell className="hidden sm:table-cell text-muted-foreground text-sm">{new Date(g.created_at).toLocaleDateString()}</TableCell>
                  <TableCell className="text-right">
                    <div className="flex gap-1 justify-end">
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => { handleEdit(g); setShowCreate(false) }}>
                            <Pencil className="h-4 w-4" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>Edit</TooltipContent>
                      </Tooltip>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive" onClick={() => setConfirmDelete(g)}>
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>Delete</TooltipContent>
                      </Tooltip>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
              {groups.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="text-center text-muted-foreground py-8">No groups found</TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>

          {/* Pagination footer */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 px-4 py-3 border-t">
            <div className="text-sm text-muted-foreground">
              {total} group{total !== 1 ? 's' : ''}
            </div>
            <div className="flex items-center gap-3 sm:gap-6 flex-wrap justify-center">
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground whitespace-nowrap">Rows per page</span>
                <Select value={String(pageSize)} onValueChange={(v) => { setPageSize(Number(v)); setPage(1) }}>
                  <SelectTrigger className="h-8 w-[70px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {PAGE_SIZES.map(s => (
                      <SelectItem key={s} value={String(s)}>{s}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <span className="text-sm text-muted-foreground whitespace-nowrap">
                Page {page} of {totalPages}
              </span>
              <div className="flex items-center gap-1">
                <Button variant="outline" size="icon" className="h-8 w-8" disabled={page <= 1} onClick={() => setPage(1)}>
                  <ChevronsLeft className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="icon" className="h-8 w-8" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="icon" className="h-8 w-8" disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>
                  <ChevronRight className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="icon" className="h-8 w-8" disabled={page >= totalPages} onClick={() => setPage(totalPages)}>
                  <ChevronsRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

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
