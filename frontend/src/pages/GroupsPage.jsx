import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import { Pencil, Trash2, Plus, X } from 'lucide-react'
import { Dialog, ConfirmDialog } from '../components/Dialog'

// During typing: allow trailing dash so user can type "finance-team"
function toKebabInput(str) {
  return str.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '').replace(/-{2,}/g, '-').replace(/^-/, '')
}
// On submit: strip trailing dash
function toKebabFinal(str) {
  return toKebabInput(str).replace(/-$/, '')
}

const PAGE_SIZE = 15

export default function GroupsPage() {
  const [groups, setGroups] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [showCreate, setShowCreate] = useState(false)
  const [form, setForm] = useState({ name: '', description: '' })
  const [editGroup, setEditGroup] = useState(null)
  const [editForm, setEditForm] = useState({ name: '', description: '' })
  const [confirmDelete, setConfirmDelete] = useState(null)

  const loadGroups = useCallback(async () => {
    const res = await api.listGroups(search, page, PAGE_SIZE)
    setGroups(res.items)
    setTotal(res.total)
  }, [search, page])

  useEffect(() => { loadGroups() }, [loadGroups])

  const handleSearch = (v) => { setSearch(v); setPage(1) }
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

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
    <div>
      <div className="flex justify-between items-center mb-4">
        <h2>Groups</h2>
        <button className="btn-primary" onClick={() => { setShowCreate(!showCreate); setEditGroup(null) }}
          style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          {showCreate ? <><X size={14} /> Cancel</> : <><Plus size={14} /> New Group</>}
        </button>
      </div>

      {showCreate && (
        <div className="card">
          <h3>Create Group</h3>
          <form onSubmit={handleCreate}>
            <div className="form-row">
              <div className="form-group">
                <label>Name <span className="text-muted text-sm">(kebab-case)</span></label>
                <input
                  value={form.name}
                  onChange={e => setForm({ ...form, name: toKebabInput(e.target.value) })}
                  placeholder="e.g. finance-team"
                  required
                />
              </div>
              <div className="form-group">
                <label>Description</label>
                <input value={form.description} onChange={e => setForm({ ...form, description: e.target.value })} placeholder="Optional description" />
              </div>
              <button className="btn-primary" type="submit" style={{ alignSelf: 'end', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Plus size={14} /> Create
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="card" style={{ padding: '0.75rem 1rem', marginBottom: '1rem' }}>
        <input
          value={search}
          onChange={e => handleSearch(e.target.value)}
          placeholder="Search by name or description..."
          style={{ width: '100%', border: 'none', outline: 'none', background: 'transparent', color: 'inherit', fontSize: '0.9rem' }}
        />
      </div>

      <div className="card">
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>ID</th><th>Name</th><th>Description</th><th>Created</th><th>Actions</th></tr>
            </thead>
            <tbody>
              {groups.map(g => (
                <tr key={g.id}>
                  <td className="text-muted">{g.id}</td>
                  <td><span className="badge badge-blue">{g.name}</span></td>
                  <td>{g.description}</td>
                  <td className="text-muted text-sm">{new Date(g.created_at).toLocaleDateString()}</td>
                  <td>
                    <div className="flex gap-2">
                      <button className="btn-icon-primary" title="Edit" onClick={() => { handleEdit(g); setShowCreate(false) }}><Pencil size={15} /></button>
                      <button className="btn-icon-danger" title="Delete" onClick={() => setConfirmDelete(g)}><Trash2 size={15} /></button>
                    </div>
                  </td>
                </tr>
              ))}
              {groups.length === 0 && <tr><td colSpan={5} className="text-muted" style={{ textAlign: 'center' }}>No groups found</td></tr>}
            </tbody>
          </table>
        </div>

        {totalPages > 1 && (
          <div className="flex justify-between items-center" style={{ marginTop: '0.75rem', fontSize: '0.85rem' }}>
            <span className="text-muted">{total} group{total !== 1 ? 's' : ''}</span>
            <div className="flex gap-2 items-center">
              <button className="btn-sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>Prev</button>
              <span>{page} / {totalPages}</span>
              <button className="btn-sm" disabled={page >= totalPages} onClick={() => setPage(page + 1)}>Next</button>
            </div>
          </div>
        )}
      </div>

      {/* Edit Group Dialog */}
      <Dialog open={!!editGroup} onClose={() => setEditGroup(null)} title={`Edit Group: ${editGroup?.name || ''}`} width={460}>
        <form onSubmit={handleUpdate}>
          <div className="form-group mb-4">
            <label>Name <span className="text-muted text-sm">(kebab-case)</span></label>
            <input
              value={editForm.name}
              onChange={e => setEditForm({ ...editForm, name: toKebabInput(e.target.value) })}
              required
              autoFocus
            />
          </div>
          <div className="form-group mb-4">
            <label>Description</label>
            <input value={editForm.description} onChange={e => setEditForm({ ...editForm, description: e.target.value })} />
          </div>
          <div className="flex gap-2" style={{ justifyContent: 'flex-end' }}>
            <button type="button" style={{ padding: '0.5rem 1rem', cursor: 'pointer', background: 'var(--bg-input)', color: 'var(--text)' }} onClick={() => setEditGroup(null)}>Cancel</button>
            <button className="btn-primary" type="submit">Save</button>
          </div>
        </form>
      </Dialog>

      {/* Confirm Delete Dialog */}
      <ConfirmDialog
        open={!!confirmDelete}
        onClose={() => setConfirmDelete(null)}
        onConfirm={() => handleDelete(confirmDelete?.id)}
        title="Delete Group"
        message={`Delete group "${confirmDelete?.name}"? All associated permissions will be removed.`}
        confirmLabel="Delete"
        danger
      />
    </div>
  )
}
