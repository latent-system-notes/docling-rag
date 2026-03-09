const BASE = '/api'

function getToken() {
  return localStorage.getItem('token')
}

async function request(path, options = {}) {
  const token = getToken()
  const headers = { 'Content-Type': 'application/json', ...options.headers }
  if (token) headers['Authorization'] = `Bearer ${token}`

  const res = await fetch(`${BASE}${path}`, { ...options, headers })

  if (res.status === 401) {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    window.location.href = '/login'
    return
  }

  if (res.status === 204) return null

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }

  return res.json()
}

export const api = {
  // Auth
  login: (username, password) =>
    request('/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) }),
  me: () => request('/auth/me'),
  changePassword: (currentPassword, newPassword) =>
    request('/auth/change-password', { method: 'POST', body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }) }),

  // Users
  listUsers: (q = '', page = 1, pageSize = 20) =>
    request(`/users?q=${encodeURIComponent(q)}&page=${page}&page_size=${pageSize}`),
  createUser: (data) => request('/users', { method: 'POST', body: JSON.stringify(data) }),
  updateUser: (id, data) => request(`/users/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
  deleteUser: (id) => request(`/users/${id}`, { method: 'DELETE' }),
  resetUserPassword: (id, newPassword) =>
    request(`/users/${id}/reset-password`, { method: 'POST', body: JSON.stringify({ new_password: newPassword }) }),
  getUserGroups: (id) => request(`/users/${id}/groups`),
  assignUserGroup: (userId, groupId) =>
    request(`/users/${userId}/groups`, { method: 'POST', body: JSON.stringify({ group_id: groupId }) }),
  removeUserGroup: (userId, groupId) =>
    request(`/users/${userId}/groups/${groupId}`, { method: 'DELETE' }),

  // Groups
  listGroups: (q = '', page = 1, pageSize = 20) =>
    request(`/groups?q=${encodeURIComponent(q)}&page=${page}&page_size=${pageSize}`),
  listAllGroups: () => request('/groups?page_size=100'),
  createGroup: (data) => request('/groups', { method: 'POST', body: JSON.stringify(data) }),
  updateGroup: (id, data) => request(`/groups/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
  deleteGroup: (id) => request(`/groups/${id}`, { method: 'DELETE' }),

  // Permissions
  listPathPermissions: () => request('/permissions/paths'),
  addPathPermission: (path, groupId) =>
    request('/permissions/paths', { method: 'POST', body: JSON.stringify({ path, group_id: groupId }) }),
  removePathPermission: (path, groupId) =>
    request('/permissions/paths', { method: 'DELETE', body: JSON.stringify({ path, group_id: groupId }) }),
  getTreeChildren: (path = '') => request(`/permissions/tree/children?path=${encodeURIComponent(path)}`),
  refreshPermissions: () => request('/permissions/refresh', { method: 'POST' }),

  // Search (user-facing, auto-filtered by groups)
  search: (q, topK = 10) => request(`/search?q=${encodeURIComponent(q)}&top_k=${topK}`),
  myDocuments: (limit = 50, offset = 0) => request(`/search/documents?limit=${limit}&offset=${offset}`),

  // Settings
  getSettings: () => request('/settings'),
  updateSetting: (key, value) =>
    request(`/settings/${encodeURIComponent(key)}`, { method: 'PUT', body: JSON.stringify({ value }) }),
  deleteSetting: (key) =>
    request(`/settings/${encodeURIComponent(key)}`, { method: 'DELETE' }),
  reloadMcp: () =>
    request('/settings/reload-mcp', { method: 'POST' }),

  // Browse
  browseDirectories: (path = '') =>
    request(`/browse/directories?path=${encodeURIComponent(path)}`),
  browseDocumentFolders: (path = '') =>
    request(`/browse/document-folders?path=${encodeURIComponent(path)}`),

  // Files
  listFiles: (path = '') => request(`/files?path=${encodeURIComponent(path)}`),
  uploadFiles: (path, fileList) => {
    const token = getToken()
    const form = new FormData()
    for (const f of fileList) form.append('files', f)
    return fetch(`${BASE}/files/upload?path=${encodeURIComponent(path)}`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
      body: form,
    }).then(async res => {
      if (res.status === 401) { localStorage.removeItem('token'); localStorage.removeItem('user'); window.location.href = '/login'; return }
      if (!res.ok) { const err = await res.json().catch(() => ({ detail: res.statusText })); throw new Error(err.detail || 'Upload failed') }
      return res.json()
    })
  },
  createDirectory: (path, name) =>
    request('/files/mkdir', { method: 'POST', body: JSON.stringify({ path, name }) }),
  deleteFile: (path) =>
    request('/files', { method: 'DELETE', body: JSON.stringify({ path }) }),
  renameFile: (path, newName) =>
    request('/files/rename', { method: 'PUT', body: JSON.stringify({ path, new_name: newName }) }),

  // Ingestion
  startIngestion: (opts = {}) =>
    request('/ingestion/start', { method: 'POST', body: JSON.stringify(opts) }),
  getIngestionStatus: () => request('/ingestion/status'),
  getIngestionLogs: (offset = 0) => request(`/ingestion/logs?offset=${offset}`),
  cancelIngestion: () => request('/ingestion/cancel', { method: 'POST' }),
}
