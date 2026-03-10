import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { DataTable, DataTablePagination, DataTableCard, SortableHeader } from '@/components/ui/data-table'

const PAGE_SIZES = [10, 20, 50, 100]

const truncText = (text, len = 100) => text && text.length > len ? text.slice(0, len) + '...' : text

const columns = [
  {
    accessorKey: 'doc_id',
    header: ({ column }) => <SortableHeader column={column} title="Doc ID" />,
    cell: ({ getValue }) => <span className="font-mono text-xs" title={getValue()}>{getValue()}</span>,
    size: 280,
  },
  {
    accessorKey: 'file_path',
    header: ({ column }) => <SortableHeader column={column} title="File" />,
    cell: ({ getValue }) => {
      const v = getValue()
      return <span className="text-sm text-muted-foreground truncate block max-w-[200px]">{v ? v.split('/').pop() : ''}</span>
    },
    size: 160,
  },
  {
    accessorKey: 'page_num',
    header: ({ column }) => <SortableHeader column={column} title="Pg" />,
    cell: ({ getValue }) => getValue() ?? '-',
    size: 50,
  },
  {
    accessorKey: 'chunk_index',
    header: ({ column }) => <SortableHeader column={column} title="#" />,
    size: 50,
  },
  {
    accessorKey: 'language',
    header: ({ column }) => <SortableHeader column={column} title="Lang" />,
    cell: ({ getValue }) => {
      const v = getValue()
      return v && v !== 'unknown' ? <Badge variant="info">{v}</Badge> : <span className="text-muted-foreground">-</span>
    },
    size: 60,
  },
  {
    accessorKey: 'text',
    header: ({ column }) => <SortableHeader column={column} title="Text" />,
    cell: ({ getValue }) => <span className="line-clamp-2 text-sm">{truncText(getValue(), 200)}</span>,
    enableSorting: false,
  },
  {
    accessorKey: 'ingested_at',
    header: ({ column }) => <SortableHeader column={column} title="Ingested" />,
    cell: ({ getValue }) => {
      const v = getValue()
      return <span className="text-sm text-muted-foreground whitespace-nowrap">{v ? new Date(v).toLocaleDateString() : ''}</span>
    },
    size: 100,
  },
]

export default function ChunksPage() {
  const [chunks, setChunks] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)
  const [search, setSearch] = useState('')
  const [fileFilter, setFileFilter] = useState('')
  const [selectedChunk, setSelectedChunk] = useState(null)

  const loadChunks = useCallback(async () => {
    const res = await api.listChunks(search, '', fileFilter, page, pageSize)
    setChunks(res.items)
    setTotal(res.total)
  }, [search, fileFilter, page, pageSize])

  useEffect(() => { loadChunks() }, [loadChunks])

  const handleSearch = (v) => { setSearch(v); setPage(1) }
  const handleFileFilter = (v) => { setFileFilter(v); setPage(1) }

  return (
    <div className="flex flex-col h-full gap-4 overflow-hidden">
      <h1 className="text-2xl font-semibold tracking-tight shrink-0">Chunks</h1>

      <Card className="shrink-0">
        <CardContent className="py-3">
          <div className="flex flex-col sm:flex-row gap-3">
            <Input
              value={search}
              onChange={e => handleSearch(e.target.value)}
              placeholder="Search chunk text..."
              className="border-0 shadow-none focus-visible:ring-0 flex-1"
            />
            <Input
              value={fileFilter}
              onChange={e => handleFileFilter(e.target.value)}
              placeholder="Search by file name..."
              className="border-0 shadow-none focus-visible:ring-0 sm:w-48"
            />
          </div>
        </CardContent>
      </Card>

      <DataTableCard>
        <DataTable
          columns={columns}
          data={chunks}
          onRowClick={(chunk) => setSelectedChunk(chunk)}
          noResultsMessage="No chunks found"
        />
        <DataTablePagination
          total={total}
          page={page}
          pageSize={pageSize}
          onPageChange={setPage}
          onPageSizeChange={setPageSize}
          pageSizes={PAGE_SIZES}
          noun="chunk"
        />
      </DataTableCard>

      <Dialog open={!!selectedChunk} onOpenChange={() => setSelectedChunk(null)}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Chunk Detail</DialogTitle>
            <DialogDescription>
              {selectedChunk && (
                <span className="font-mono text-xs">
                  Doc: {selectedChunk.doc_id} | Page: {selectedChunk.page_num ?? '-'} | Chunk#: {selectedChunk.chunk_index}
                </span>
              )}
            </DialogDescription>
          </DialogHeader>
          {selectedChunk && (
            <div className="space-y-3">
              <div className="text-sm text-muted-foreground">
                <strong>File:</strong> {selectedChunk.file_path}
              </div>
              <div className="rounded-md bg-muted p-4 text-sm whitespace-pre-wrap break-words">
                {selectedChunk.text}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
