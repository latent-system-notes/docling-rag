import { useState } from 'react'
import { api } from '../api/client'
import { Search, FileText } from 'lucide-react'

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(10)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError('')
    try {
      const data = await api.search(query.trim(), topK)
      setResults(data)
    } catch (err) {
      setError(err.message || 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem' }}>Search Documents</h2>

      <div className="card">
        <form onSubmit={handleSearch}>
          <div className="form-group mb-4">
            <label>Query</label>
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Type your question..."
              autoFocus
            />
          </div>
          <div className="flex gap-2 items-center" style={{ flexWrap: 'wrap' }}>
            <div className="form-group" style={{ flex: 'none', width: 80 }}>
              <label>Results</label>
              <input type="number" value={topK} onChange={e => setTopK(Number(e.target.value))} min={1} max={100} />
            </div>
            <button className="btn-primary" type="submit" disabled={loading}
              style={{ alignSelf: 'end', display: 'flex', alignItems: 'center', gap: '0.5rem', whiteSpace: 'nowrap' }}>
              <Search size={14} />
              {loading ? 'Searching...' : 'Search'}
            </button>
          </div>
        </form>
      </div>

      {error && <div className="card" style={{ borderColor: 'var(--danger)', color: 'var(--danger)' }}>{error}</div>}

      {results && (
        <div>
          <div className="text-sm text-muted mb-4">
            Found {results.total_results} result{results.total_results !== 1 ? 's' : ''} for "{results.query}"
          </div>

          {results.results.length === 0 && (
            <div className="card text-muted" style={{ textAlign: 'center' }}>
              No results found. You may not have access to relevant documents, or try different keywords.
            </div>
          )}

          {results.results.map((r) => (
            <div key={r.rank} className="card">
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ color: 'var(--primary)', fontWeight: 700, fontSize: '1.1rem' }}>#{r.rank}</span>
                <FileText size={16} style={{ color: 'var(--warning)', flexShrink: 0 }} />
                <span style={{ fontWeight: 500, wordBreak: 'break-all' }}>{r.file}</span>
                {r.page && <span className="badge badge-blue">Page {r.page}</span>}
                {r.doc_type && <span className="badge badge-green">{r.doc_type}</span>}
                <span className="text-sm text-muted" style={{ marginLeft: 'auto' }}>Score: {r.score}</span>
              </div>
              <div style={{ fontSize: '0.9rem', lineHeight: 1.7, color: 'var(--text)', whiteSpace: 'pre-wrap', background: 'var(--bg)', padding: '0.75rem', borderRadius: 'var(--radius)', overflowX: 'auto' }}>
                {r.text}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
