import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import { Pencil, Trash2, KeyRound, Plus, UserPlus, X } from 'lucide-react'
import { Dialog, ConfirmDialog, AlertDialog } from '../components/Dialog'

function sanitizeUsername(str) {
  return str.toLowerCase().replace(/[^a-z0-9_-]/g, '')
}

const PAGE_SIZE = 15

export default function UsersPage() {
  const [users, setUsers] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
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
    const res = await api.listUsers(search, page, PAGE_SIZE)
    setUsers(res.items)
    setTotal(res.total)
  }, [search, page])

  const loadGroups = async () => {
    const res = await api.listAllGroups()
    setAllGroups(res.items)
  }

  useEffect(() => { loadUsers() }, [loadUsers])
  useEffect(() => { loadGroups() }, [])

  const handleSearch = (v) => { setSearch(v); setPage(1) }
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

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
    <div>
      <div className="flex justify-between items-center mb-4">
        <h2>Users</h2>
        <button className="btn-primary" onClick={() => setShowCreate(!showCreate)}
          style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          {showCreate ? <><X size={14} /> Cancel</> : <><UserPlus size={14} /> New User</>}
        </button>
      </div>

      {showCreate && (
        <div className="card">
          <h3>Create User</h3>
          <form onSubmit={handleCreate}>
            <div className="form-row">
              <div className="form-group">
                <label>Username <span className="text-muted text-sm">(a-z, 0-9, dash, underscore)</span></label>
                <input
                  value={form.username}
                  onChange={e => setForm({ ...form, username: sanitizeUsername(e.target.value) })}
                  placeholder="e.g. john-doe"
                  required
                />
              </div>
              <div className="form-group">
                <label>Password</label>
                <input type="password" value={form.password} onChange={e => setForm({ ...form, password: e.target.value })} />
              </div>
              <div className="form-group">
                <label>Auth Type</label>
                <select value={form.auth_type} onChange={e => setForm({ ...form, auth_type: e.target.value })}>
                  <option value="local">Local</option>
                  <option value="ldap">LDAP</option>
                </select>
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>Display Name</label>
                <input value={form.display_name} onChange={e => setForm({ ...form, display_name: e.target.value })} />
              </div>
              <div className="form-group">
                <label>Email</label>
                <input value={form.email} onChange={e => setForm({ ...form, email: e.target.value })} />
              </div>
              <div className="form-group">
                <label>Admin</label>
                <select value={form.is_admin} onChange={e => setForm({ ...form, is_admin: e.target.value === 'true' })}>
                  <option value="false">No</option>
                  <option value="true">Yes</option>
                </select>
              </div>
            </div>
            <button className="btn-primary" type="submit" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Plus size={14} /> Create
            </button>
          </form>
        </div>
      )}

      <div className="card" style={{ padding: '0.75rem 1rem', marginBottom: '1rem' }}>
        <input
          value={search}
          onChange={e => handleSearch(e.target.value)}
          placeholder="Search by username, name, or email..."
          style={{ width: '100%', border: 'none', outline: 'none', background: 'transparent', color: 'inherit', fontSize: '0.9rem' }}
        />
      </div>

      <div className="flex gap-4">
        <div className="card" style={{ flex: 2, minWidth: 0 }}>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Username</th><th>Name</th><th>Email</th><th>Auth</th><th>Admin</th><th>Active</th><th>Actions</th></tr>
              </thead>
              <tbody>
                {users.map(u => (
                  <tr key={u.id} onClick={() => selectUser(u)} style={{ cursor: 'pointer', background: selectedUser?.id === u.id ? 'var(--bg-input)' : '' }}>
                    <td>{u.username}</td>
                    <td>{u.display_name}</td>
                    <td className="text-muted text-sm">{u.email}</td>
                    <td><span className="badge badge-blue">{u.auth_type}</span></td>
                    <td>{u.is_admin ? <span className="badge badge-green">Yes</span> : 'No'}</td>
                    <td>{u.is_active ? <span className="badge badge-green">Yes</span> : <span className="badge badge-red">No</span>}</td>
                    <td>
                      <div className="flex gap-2">
                        <button className="btn-icon-primary" title="Edit" onClick={(e) => { e.stopPropagation(); handleEdit(u) }}><Pencil size={15} /></button>
                        <button className="btn-icon-warning" title="Reset Password" onClick={(e) => { e.stopPropagation(); setResetPw({ show: true, userId: u.id, username: u.username, password: '' }) }}><KeyRound size={15} /></button>
                        <button className="btn-icon-danger" title="Delete" onClick={(e) => { e.stopPropagation(); setConfirmDelete(u) }}><Trash2 size={15} /></button>
                      </div>
                    </td>
                  </tr>
                ))}
                {users.length === 0 && <tr><td colSpan={7} className="text-muted" style={{ textAlign: 'center' }}>No users found</td></tr>}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <div className="flex justify-between items-center" style={{ marginTop: '0.75rem', fontSize: '0.85rem' }}>
              <span className="text-muted">{total} user{total !== 1 ? 's' : ''}</span>
              <div className="flex gap-2 items-center">
                <button className="btn-sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>Prev</button>
                <span>{page} / {totalPages}</span>
                <button className="btn-sm" disabled={page >= totalPages} onClick={() => setPage(page + 1)}>Next</button>
              </div>
            </div>
          )}
        </div>

        {selectedUser && (
          <div className="card" style={{ flex: 1 }}>
            <div className="flex justify-between items-center mb-4">
              <h3 style={{ margin: 0 }}>{selectedUser.username} — Groups</h3>
              <button className="btn-icon" onClick={() => setSelectedUser(null)}><X size={16} /></button>
            </div>
            <div className="mb-4">
              {userGroups.length === 0 && <div className="text-muted text-sm">No groups assigned</div>}
              {userGroups.map(g => (
                <div key={g.id} className="flex justify-between items-center" style={{ padding: '0.3rem 0' }}>
                  <span className="badge badge-blue">{g.name}</span>
                  <button className="btn-icon-danger" title="Remove group" onClick={() => removeGroup(g.id)}><X size={14} /></button>
                </div>
              ))}
            </div>
            {availableGroups.length > 0 && (
              <div>
                <div className="text-sm text-muted mb-2">Add group:</div>
                <div className="flex gap-2" style={{ flexWrap: 'wrap' }}>
                  {availableGroups.map(g => (
                    <button key={g.id} className="btn-primary btn-sm" onClick={() => assignGroup(g.id)}
                      style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <Plus size={12} /> {g.name}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Edit User Dialog */}
      <Dialog open={!!editUser} onClose={() => setEditUser(null)} title={`Edit User: ${editUser?.username || ''}`} width={480}>
        <form onSubmit={handleUpdate}>
          <div className="form-group mb-4">
            <label>Display Name</label>
            <input value={editForm.display_name} onChange={e => setEditForm({ ...editForm, display_name: e.target.value })} autoFocus />
          </div>
          <div className="form-group mb-4">
            <label>Email</label>
            <input value={editForm.email} onChange={e => setEditForm({ ...editForm, email: e.target.value })} />
          </div>
          <div className="form-row">
            <div className="form-group">
              <label>Auth Type</label>
              <select value={editForm.auth_type} onChange={e => setEditForm({ ...editForm, auth_type: e.target.value })}>
                <option value="local">Local</option>
                <option value="ldap">LDAP</option>
              </select>
            </div>
            <div className="form-group">
              <label>Admin</label>
              <select value={editForm.is_admin} onChange={e => setEditForm({ ...editForm, is_admin: e.target.value === 'true' })}>
                <option value="false">No</option>
                <option value="true">Yes</option>
              </select>
            </div>
            <div className="form-group">
              <label>Active</label>
              <select value={editForm.is_active} onChange={e => setEditForm({ ...editForm, is_active: e.target.value === 'true' })}>
                <option value="true">Yes</option>
                <option value="false">No</option>
              </select>
            </div>
          </div>
          <div className="flex gap-2" style={{ marginTop: '1rem', justifyContent: 'flex-end' }}>
            <button type="button" style={{ padding: '0.5rem 1rem', cursor: 'pointer', background: 'var(--bg-input)', color: 'var(--text)' }} onClick={() => setEditUser(null)}>Cancel</button>
            <button className="btn-primary" type="submit">Save</button>
          </div>
        </form>
      </Dialog>

      {/* Reset Password Dialog */}
      <Dialog open={resetPw.show} onClose={() => setResetPw({ show: false, userId: null, username: '', password: '' })} title={`Reset Password: ${resetPw.username}`} width={400}>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
          User will be forced to change password on next login.
        </p>
        <form onSubmit={handleResetPassword}>
          <div className="form-group mb-4">
            <label>New Password</label>
            <input type="password" value={resetPw.password} onChange={e => setResetPw({ ...resetPw, password: e.target.value })} autoFocus required />
          </div>
          <div className="flex gap-2" style={{ justifyContent: 'flex-end' }}>
            <button type="button" style={{ padding: '0.5rem 1rem', cursor: 'pointer', background: 'var(--bg-input)', color: 'var(--text)' }} onClick={() => setResetPw({ show: false, userId: null, username: '', password: '' })}>Cancel</button>
            <button className="btn-primary" type="submit">Reset Password</button>
          </div>
        </form>
      </Dialog>

      {/* Confirm Delete Dialog */}
      <ConfirmDialog
        open={!!confirmDelete}
        onClose={() => setConfirmDelete(null)}
        onConfirm={() => handleDelete(confirmDelete?.id)}
        title="Delete User"
        message={`Are you sure you want to delete user "${confirmDelete?.username}"?`}
        confirmLabel="Delete"
        danger
      />

      {/* Alert Dialog */}
      <AlertDialog open={alert.open} onClose={() => setAlert({ ...alert, open: false })} title={alert.title} message={alert.message} />
    </div>
  )
}
