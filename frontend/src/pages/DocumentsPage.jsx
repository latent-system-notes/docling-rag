import { useState, useEffect } from 'react'
import { api } from '../api/client'
import { FileText, ChevronLeft, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

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
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight">My Documents</h1>

      <Card>
        <CardContent className="pt-6">
          <p className="text-sm text-muted-foreground mb-4">
            Documents you have access to based on your group permissions.
          </p>

          {loading && <p className="text-muted-foreground">Loading...</p>}

          {!loading && docs.length === 0 && (
            <p className="text-center text-muted-foreground py-8">
              No documents available. You may not have any group permissions assigned yet.
            </p>
          )}

          {!loading && docs.length > 0 && (
            <>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>File</TableHead>
                    <TableHead>Folder</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Language</TableHead>
                    <TableHead className="text-right">Chunks</TableHead>
                    <TableHead>Ingested</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {docs.map((d) => (
                    <TableRow key={d.doc_id}>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <FileText className="h-4 w-4 text-amber-500 shrink-0" />
                          <span className="font-medium">{fileName(d.file_path)}</span>
                        </div>
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground max-w-[300px] truncate">
                        {folderPath(d.file_path)}
                      </TableCell>
                      <TableCell><Badge variant="success">{d.doc_type}</Badge></TableCell>
                      <TableCell><Badge variant="info">{d.language}</Badge></TableCell>
                      <TableCell className="text-right">{d.num_chunks}</TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {d.ingested_at ? new Date(d.ingested_at).toLocaleString() : ''}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              <div className="flex items-center justify-between pt-4">
                <Button variant="outline" size="sm" disabled={offset === 0} onClick={() => load(Math.max(0, offset - limit))}>
                  <ChevronLeft className="h-4 w-4" /> Previous
                </Button>
                <span className="text-sm text-muted-foreground">
                  Showing {offset + 1}&ndash;{offset + docs.length}
                </span>
                <Button variant="outline" size="sm" disabled={docs.length < limit} onClick={() => load(offset + limit)}>
                  Next <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
