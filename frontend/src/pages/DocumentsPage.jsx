import { useState, useEffect } from 'react'
import { api } from '../api/client'
import { FileText, ChevronLeft, ChevronRight } from 'lucide-react'

export default function DocumentsPage() {
  const [docs, setDocs] = useState([])
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(true)
  const limit = 25

  const load = async (newOffset = 0) => {
    setLoading(true)
    try {
      const data = await api.myDocuments(limit, newOffset)
      setDocs(data.documents)
      setOffset(newOffset)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const fileName = (path) => {
    const parts = path.replace(/\\/g, '/').split('/')
    return parts[parts.length - 1] || path
  }

  const folderPath = (path) => {
    const normalized = path.replace(/\\/g, '/')
    const idx = normalized.lastIndexOf('/')
    return idx > 0 ? normalized.substring(0, idx) : ''
  }

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem' }}>My Documents</h2>

      <div className="card">
        <div className="text-sm text-muted mb-4">
          Documents you have access to based on your group permissions.
        </div>

        {loading && <div className="text-muted">Loading...</div>}

        {!loading && docs.length === 0 && (
          <div className="text-muted" style={{ textAlign: 'center', padding: '2rem' }}>
            No documents available. You may not have any group permissions assigned yet.
          </div>
        )}

        {!loading && docs.length > 0 && (
          <>
            <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>File</th>
                  <th>Folder</th>
                  <th>Type</th>
                  <th>Language</th>
                  <th style={{ textAlign: 'right' }}>Chunks</th>
                  <th>Ingested</th>
                </tr>
              </thead>
              <tbody>
                {docs.map((d) => (
                  <tr key={d.doc_id}>
                    <td>
                      <div className="flex items-center gap-2">
                        <FileText size={14} style={{ color: 'var(--warning)', flexShrink: 0 }} />
                        <span style={{ fontWeight: 500 }}>{fileName(d.file_path)}</span>
                      </div>
                    </td>
                    <td className="text-sm text-muted" style={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {folderPath(d.file_path)}
                    </td>
                    <td><span className="badge badge-green">{d.doc_type}</span></td>
                    <td><span className="badge badge-blue">{d.language}</span></td>
                    <td style={{ textAlign: 'right' }}>{d.num_chunks}</td>
                    <td className="text-sm text-muted">
                      {d.ingested_at ? new Date(d.ingested_at).toLocaleString() : ''}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>

            <div className="flex justify-between items-center mt-2" style={{ padding: '0.5rem 0' }}>
              <button
                className="btn-primary btn-sm"
                disabled={offset === 0}
                onClick={() => load(Math.max(0, offset - limit))}
                style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}
              >
                <ChevronLeft size={14} /> Previous
              </button>
              <span className="text-sm text-muted">
                Showing {offset + 1}–{offset + docs.length}
              </span>
              <button
                className="btn-primary btn-sm"
                disabled={docs.length < limit}
                onClick={() => load(offset + limit)}
                style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}
              >
                Next <ChevronRight size={14} />
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
