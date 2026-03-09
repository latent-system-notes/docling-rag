import { useState } from 'react'
import { api } from '../api/client'
import { Search, FileText } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'

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
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight">Search Documents</h1>

      <Card>
        <CardContent className="pt-6">
          <form onSubmit={handleSearch} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="query">Query</Label>
              <Input
                id="query"
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="Type your question..."
                autoFocus
              />
            </div>
            <div className="flex items-end gap-3">
              <div className="space-y-2 w-20">
                <Label htmlFor="topk">Results</Label>
                <Input id="topk" type="number" value={topK} onChange={e => setTopK(Number(e.target.value))} min={1} max={100} />
              </div>
              <Button type="submit" disabled={loading}>
                <Search className="h-4 w-4" />
                {loading ? 'Searching...' : 'Search'}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {results && (
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Found {results.total_results} result{results.total_results !== 1 ? 's' : ''} for "{results.query}"
          </p>

          {results.results.length === 0 && (
            <Card>
              <CardContent className="pt-6 text-center text-muted-foreground">
                No results found. You may not have access to relevant documents, or try different keywords.
              </CardContent>
            </Card>
          )}

          {results.results.map((r) => (
            <Card key={r.rank}>
              <CardContent className="pt-6">
                <div className="flex flex-wrap items-center gap-2 mb-3">
                  <span className="text-lg font-bold text-primary">#{r.rank}</span>
                  <FileText className="h-4 w-4 text-amber-500 shrink-0" />
                  <span className="font-medium break-all">{r.file}</span>
                  {r.page && <Badge variant="info">Page {r.page}</Badge>}
                  {r.doc_type && <Badge variant="success">{r.doc_type}</Badge>}
                  <span className="text-sm text-muted-foreground ml-auto">Score: {r.score}</span>
                </div>
                <div className="text-sm leading-relaxed whitespace-pre-wrap bg-muted rounded-md p-3 overflow-x-auto">
                  {r.text}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
