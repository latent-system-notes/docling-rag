import { useState, useEffect, useCallback, useMemo } from 'react'
import { api } from '../api/client'
import { FileText } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { DataTable, DataTablePagination, DataTableCard, SortableHeader } from '@/components/ui/data-table'

const PAGE_SIZES = [10, 25, 50, 100]

export default function DocumentsPage() {
  const [docs, setDocs] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(25)
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.myDocuments(page, pageSize)
      setDocs(data.items)
      setTotal(data.total)
    } finally {
      setLoading(false)
    }
  }, [page, pageSize])

  useEffect(() => { load() }, [load])

  const fileName = (path) => {
    const parts = path.replace(/\\/g, '/').split('/')
    return parts[parts.length - 1] || path
  }

  const folderPath = (path) => {
    const normalized = path.replace(/\\/g, '/')
    const idx = normalized.lastIndexOf('/')
    return idx > 0 ? normalized.substring(0, idx) : ''
  }

  const columns = useMemo(() => [
    {
      accessorKey: 'file_path',
      header: ({ column }) => <SortableHeader column={column} title="File" />,
      cell: ({ getValue }) => (
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-amber-500 shrink-0" />
          <span className="font-medium">{fileName(getValue())}</span>
        </div>
      ),
      size: 220,
      sortingFn: (a, b) => fileName(a.original.file_path).localeCompare(fileName(b.original.file_path)),
    },
    {
      id: 'folder',
      header: ({ column }) => <SortableHeader column={column} title="Folder" />,
      accessorFn: (row) => folderPath(row.file_path),
      cell: ({ getValue }) => <span className="text-sm text-muted-foreground max-w-[300px] truncate block">{getValue()}</span>,
      size: 250,
    },
    {
      accessorKey: 'doc_type',
      header: ({ column }) => <SortableHeader column={column} title="Type" />,
      cell: ({ getValue }) => <Badge variant="success">{getValue()}</Badge>,
      size: 100,
    },
    {
      accessorKey: 'language',
      header: ({ column }) => <SortableHeader column={column} title="Language" />,
      cell: ({ getValue }) => <Badge variant="info">{getValue()}</Badge>,
      size: 100,
    },
    {
      accessorKey: 'num_chunks',
      header: ({ column }) => <SortableHeader column={column} title="Chunks" className="justify-end" />,
      cell: ({ getValue }) => <div className="text-right">{getValue()}</div>,
      size: 90,
    },
    {
      accessorKey: 'ingested_at',
      header: ({ column }) => <SortableHeader column={column} title="Ingested" />,
      cell: ({ getValue }) => {
        const v = getValue()
        return <span className="text-sm text-muted-foreground">{v ? new Date(v).toLocaleString() : ''}</span>
      },
      size: 170,
    },
  ], [])

  return (
    <div className="flex flex-col h-full gap-4 overflow-hidden">
      <h1 className="text-2xl font-semibold tracking-tight shrink-0">My Documents</h1>

      <p className="text-sm text-muted-foreground shrink-0">
        Documents you have access to based on your group permissions.
      </p>

      {loading && <p className="text-muted-foreground">Loading...</p>}

      {!loading && docs.length === 0 && (
        <p className="text-center text-muted-foreground py-8">
          No documents available. You may not have any group permissions assigned yet.
        </p>
      )}

      {!loading && docs.length > 0 && (
        <DataTableCard>
          <DataTable
            columns={columns}
            data={docs}
            noResultsMessage="No documents found"
          />
          <DataTablePagination
            total={total}
            page={page}
            pageSize={pageSize}
            onPageChange={setPage}
            onPageSizeChange={setPageSize}
            pageSizes={PAGE_SIZES}
            noun="document"
          />
        </DataTableCard>
      )}
    </div>
  )
}
