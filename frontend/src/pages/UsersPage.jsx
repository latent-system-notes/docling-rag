import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import { Pencil, Trash2, KeyRound, Plus, UserPlus, X, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog'
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select'
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip'

function sanitizeUsername(str) {
  return str.toLowerCase().replace(/[^a-z0-9_-]/g, '')
}

const PAGE_SIZES = [10, 15, 20, 50]

export default function UsersPage() {
  const [users, setUsers] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(15)
  const [search, setSearch] = useState('')
  const [allGroups, setAllGroups] = useState([])
  const [showCreate, setShowCreate] = useState(false)
  const [form, setForm] = useState({ username: '', password: '', display_name: '', email: '', is_admin: false, auth_type: 'local' })
  const [selectedUser, setSelectedUser] = useState(null)
  const [userGroups, setUserGroups] = useState([])
  const [resetPw, setResetPw] = useState({ show: false, userId: null, username: '', password: '' })
  const [editUser, setEditUser] = useState(null)
  const [editForm, setEditForm] = useState({ display_name: '', email: '', is_admin: false, is_active: true, auth_type: 'local' })
  const [confirmDelete, setConfirmDelete] = useState(null)
  const [alert, setAlert] = useState({ open: false, title: '', message: '' })

  const loadUsers = useCallback(async () => {
    const res = await api.listUsers(search, page, pageSize)
    setUsers(res.items)
    setTotal(res.total)
  }, [search, page, pageSize])

  const loadGroups = async () => {
    const res = await api.listAllGroups()
    setAllGroups(res.items)
  }

  useEffect(() => { loadUsers() }, [loadUsers])
  useEffect(() => { loadGroups() }, [])

  const handleSearch = (v) => { setSearch(v); setPage(1) }
  const totalPages = Math.max(1, Math.ceil(total / pageSize))

  const handleCreate = async (e) => {
    e.preventDefault()
    const username = sanitizeUsername(form.username)
    if (!username) return
    await api.createUser({ ...form, username })
    setForm({ username: '', password: '', display_name: '', email: '', is_admin: false, auth_type: 'local' })
    setShowCreate(false)
    loadUsers()
  }

  const handleDelete = async (id) => {
    await api.deleteUser(id)
    if (selectedUser?.id === id) setSelectedUser(null)
    if (editUser?.id === id) setEditUser(null)
    loadUsers()
  }

  const handleEdit = (u) => {
    setEditUser(u)
    setEditForm({
      display_name: u.display_name || '',
      email: u.email || '',
      is_admin: u.is_admin,
      is_active: u.is_active,
      auth_type: u.auth_type,
    })
  }

  const handleUpdate = async (e) => {
    e.preventDefault()
    await api.updateUser(editUser.id, editForm)
    setEditUser(null)
    loadUsers()
  }

  const selectUser = async (user) => {
    setSelectedUser(user)
    const ug = await api.getUserGroups(user.id)
    setUserGroups(ug)
  }

  const assignGroup = async (groupId) => {
    await api.assignUserGroup(selectedUser.id, groupId)
    const ug = await api.getUserGroups(selectedUser.id)
    setUserGroups(ug)
  }

  const removeGroup = async (groupId) => {
    await api.removeUserGroup(selectedUser.id, groupId)
    const ug = await api.getUserGroups(selectedUser.id)
    setUserGroups(ug)
  }

  const handleResetPassword = async (e) => {
    e.preventDefault()
    if (resetPw.password.length < 6) {
      setAlert({ open: true, title: 'Validation Error', message: 'Password must be at least 6 characters' })
      return
    }
    try {
      await api.resetUserPassword(resetPw.userId, resetPw.password)
      setResetPw({ show: false, userId: null, username: '', password: '' })
      setAlert({ open: true, title: 'Success', message: 'Password reset successfully. User will be prompted to change on next login.' })
    } catch (err) {
      setAlert({ open: true, title: 'Error', message: err.message || 'Failed to reset password' })
    }
  }

  const availableGroups = allGroups.filter(g => !userGroups.some(ug => ug.id === g.id))

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">Users</h1>
        <Button onClick={() => setShowCreate(!showCreate)}>
          {showCreate ? <><X className="h-4 w-4" /> Cancel</> : <><UserPlus className="h-4 w-4" /> New User</>}
        </Button>
      </div>

      {/* Create User Form */}
      {showCreate && (
        <Card>
          <CardHeader>
            <CardTitle>Create User</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleCreate} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Username <span className="text-muted-foreground text-xs">(a-z, 0-9, dash, underscore)</span></Label>
                  <Input value={form.username} onChange={e => setForm({ ...form, username: sanitizeUsername(e.target.value) })} placeholder="e.g. john-doe" required />
                </div>
                <div className="space-y-2">
                  <Label>Password</Label>
                  <Input type="password" value={form.password} onChange={e => setForm({ ...form, password: e.target.value })} />
                </div>
                <div className="space-y-2">
                  <Label>Auth Type</Label>
                  <Select value={form.auth_type} onValueChange={v => setForm({ ...form, auth_type: v })}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="local">Local</SelectItem>
                      <SelectItem value="ldap">LDAP</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Display Name</Label>
                  <Input value={form.display_name} onChange={e => setForm({ ...form, display_name: e.target.value })} />
                </div>
                <div className="space-y-2">
                  <Label>Email</Label>
                  <Input value={form.email} onChange={e => setForm({ ...form, email: e.target.value })} />
                </div>
                <div className="space-y-2">
                  <Label>Admin</Label>
                  <Select value={String(form.is_admin)} onValueChange={v => setForm({ ...form, is_admin: v === 'true' })}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="false">No</SelectItem>
                      <SelectItem value="true">Yes</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
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
            placeholder="Search by username, name, or email..."
            className="border-0 shadow-none focus-visible:ring-0"
          />
        </CardContent>
      </Card>

      {/* Users table + Groups panel */}
      <div className="flex flex-col lg:flex-row gap-6">
        <Card className="flex-[2] min-w-0">
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Username</TableHead>
                  <TableHead className="hidden sm:table-cell">Name</TableHead>
                  <TableHead className="hidden lg:table-cell">Email</TableHead>
                  <TableHead className="hidden md:table-cell">Auth</TableHead>
                  <TableHead className="hidden md:table-cell">Admin</TableHead>
                  <TableHead className="hidden md:table-cell">Active</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {users.map(u => (
                  <TableRow
                    key={u.id}
                    className="cursor-pointer"
                    onClick={() => selectUser(u)}
                    data-state={selectedUser?.id === u.id ? 'selected' : undefined}
                  >
                    <TableCell className="font-medium">{u.username}</TableCell>
                    <TableCell className="hidden sm:table-cell">{u.display_name}</TableCell>
                    <TableCell className="hidden lg:table-cell text-muted-foreground text-sm">{u.email}</TableCell>
                    <TableCell className="hidden md:table-cell"><Badge variant="info">{u.auth_type}</Badge></TableCell>
                    <TableCell className="hidden md:table-cell">{u.is_admin ? <Badge variant="success">Yes</Badge> : <span className="text-muted-foreground">No</span>}</TableCell>
                    <TableCell className="hidden md:table-cell">{u.is_active ? <Badge variant="success">Yes</Badge> : <Badge variant="destructive">No</Badge>}</TableCell>
                    <TableCell className="text-right">
                      <div className="flex gap-1 justify-end">
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={(e) => { e.stopPropagation(); handleEdit(u) }}>
                              <Pencil className="h-4 w-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Edit</TooltipContent>
                        </Tooltip>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8 text-amber-600" onClick={(e) => { e.stopPropagation(); setResetPw({ show: true, userId: u.id, username: u.username, password: '' }) }}>
                              <KeyRound className="h-4 w-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Reset Password</TooltipContent>
                        </Tooltip>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive" onClick={(e) => { e.stopPropagation(); setConfirmDelete(u) }}>
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Delete</TooltipContent>
                        </Tooltip>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
                {users.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center text-muted-foreground py-8">No users found</TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>

            {/* Pagination footer */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-3 px-4 py-3 border-t">
              <div className="text-sm text-muted-foreground">
                {total} user{total !== 1 ? 's' : ''}
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

        {/* Groups side panel */}
        {selectedUser && (
          <Card className="flex-1">
            <CardHeader className="flex-row items-center justify-between space-y-0">
              <CardTitle className="text-base">{selectedUser.username} &mdash; Groups</CardTitle>
              <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setSelectedUser(null)}>
                <X className="h-4 w-4" />
              </Button>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 mb-4">
                {userGroups.length === 0 && <p className="text-sm text-muted-foreground">No groups assigned</p>}
                {userGroups.map(g => (
                  <div key={g.id} className="flex items-center justify-between">
                    <Badge variant="info">{g.name}</Badge>
                    <Button variant="ghost" size="icon" className="h-7 w-7 text-destructive" onClick={() => removeGroup(g.id)}>
                      <X className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                ))}
              </div>
              {availableGroups.length > 0 && (
                <div>
                  <p className="text-sm text-muted-foreground mb-2">Add group:</p>
                  <div className="flex flex-wrap gap-2">
                    {availableGroups.map(g => (
                      <Button key={g.id} variant="outline" size="sm" onClick={() => assignGroup(g.id)}>
                        <Plus className="h-3 w-3" /> {g.name}
                      </Button>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>

      {/* Edit User Dialog */}
      <Dialog open={!!editUser} onOpenChange={() => setEditUser(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit User: {editUser?.username}</DialogTitle>
          </DialogHeader>
          <form onSubmit={handleUpdate} className="space-y-4">
            <div className="space-y-2">
              <Label>Display Name</Label>
              <Input value={editForm.display_name} onChange={e => setEditForm({ ...editForm, display_name: e.target.value })} autoFocus />
            </div>
            <div className="space-y-2">
              <Label>Email</Label>
              <Input value={editForm.email} onChange={e => setEditForm({ ...editForm, email: e.target.value })} />
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label>Auth Type</Label>
                <Select value={editForm.auth_type} onValueChange={v => setEditForm({ ...editForm, auth_type: v })}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="local">Local</SelectItem>
                    <SelectItem value="ldap">LDAP</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Admin</Label>
                <Select value={String(editForm.is_admin)} onValueChange={v => setEditForm({ ...editForm, is_admin: v === 'true' })}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="false">No</SelectItem>
                    <SelectItem value="true">Yes</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Active</Label>
                <Select value={String(editForm.is_active)} onValueChange={v => setEditForm({ ...editForm, is_active: v === 'true' })}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="true">Yes</SelectItem>
                    <SelectItem value="false">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setEditUser(null)}>Cancel</Button>
              <Button type="submit">Save</Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Reset Password Dialog */}
      <Dialog open={resetPw.show} onOpenChange={() => setResetPw({ show: false, userId: null, username: '', password: '' })}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Reset Password: {resetPw.username}</DialogTitle>
            <DialogDescription>User will be forced to change password on next login.</DialogDescription>
          </DialogHeader>
          <form onSubmit={handleResetPassword} className="space-y-4">
            <div className="space-y-2">
              <Label>New Password</Label>
              <Input type="password" value={resetPw.password} onChange={e => setResetPw({ ...resetPw, password: e.target.value })} autoFocus required />
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setResetPw({ show: false, userId: null, username: '', password: '' })}>Cancel</Button>
              <Button type="submit">Reset Password</Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Confirm Delete Dialog */}
      <Dialog open={!!confirmDelete} onOpenChange={() => setConfirmDelete(null)}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Delete User</DialogTitle>
            <DialogDescription>Are you sure you want to delete user "{confirmDelete?.username}"?</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmDelete(null)}>Cancel</Button>
            <Button variant="destructive" onClick={() => { handleDelete(confirmDelete?.id); setConfirmDelete(null) }}>Delete</Button>
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
