import * as React from 'react'
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table'
import { ArrowDown, ArrowUp, ChevronsUpDown, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// DataTableCard — flex-1 card wrapper for viewport-constrained table layouts
// ---------------------------------------------------------------------------
export function DataTableCard({ children, className }) {
  return (
    <Card className={cn('flex-1 min-h-0 flex flex-col overflow-hidden', className)}>
      <CardContent className="flex-1 min-h-0 flex flex-col overflow-hidden p-0">
        {children}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// SortableHeader — click to toggle sort, shows arrow indicator
// ---------------------------------------------------------------------------
export function SortableHeader({ column, title, className }) {
  if (!column.getCanSort()) {
    return <div className={className}>{title}</div>
  }

  return (
    <Button
      variant="ghost"
      size="sm"
      className={cn('-ml-3 h-8 font-medium', className)}
      onClick={() => column.toggleSorting(column.getIsSorted() === 'asc')}
    >
      {title}
      {column.getIsSorted() === 'desc' ? (
        <ArrowDown className="ml-1 h-3.5 w-3.5" />
      ) : column.getIsSorted() === 'asc' ? (
        <ArrowUp className="ml-1 h-3.5 w-3.5" />
      ) : (
        <ChevronsUpDown className="ml-1 h-3.5 w-3.5 text-muted-foreground/50" />
      )}
    </Button>
  )
}

// ---------------------------------------------------------------------------
// DataTablePagination — reusable pagination footer
// ---------------------------------------------------------------------------
const DEFAULT_PAGE_SIZES = [10, 20, 50, 100]

export function DataTablePagination({
  total,
  page,
  pageSize,
  onPageChange,
  onPageSizeChange,
  pageSizes = DEFAULT_PAGE_SIZES,
  noun = 'row',
}) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize))

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-3 px-4 py-3 border-t">
      <div className="text-sm text-muted-foreground">
        {total} {noun}{total !== 1 ? 's' : ''}
      </div>
      <div className="flex items-center gap-3 sm:gap-6 flex-wrap justify-center">
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground whitespace-nowrap">Rows per page</span>
          <Select value={String(pageSize)} onValueChange={(v) => { onPageSizeChange(Number(v)); onPageChange(1) }}>
            <SelectTrigger className="h-8 w-[70px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {pageSizes.map(s => (
                <SelectItem key={s} value={String(s)}>{s}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <span className="text-sm text-muted-foreground whitespace-nowrap">
          Page {page} of {totalPages}
        </span>
        <div className="flex items-center gap-1">
          <Button variant="outline" size="icon" className="h-8 w-8" disabled={page <= 1} onClick={() => onPageChange(1)}>
            <ChevronsLeft className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon" className="h-8 w-8" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon" className="h-8 w-8" disabled={page >= totalPages} onClick={() => onPageChange(page + 1)}>
            <ChevronRight className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon" className="h-8 w-8" disabled={page >= totalPages} onClick={() => onPageChange(totalPages)}>
            <ChevronsRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// DataTable — generic table with TanStack sorting + column resizing
// ---------------------------------------------------------------------------
export function DataTable({
  columns,
  data,
  onRowClick,
  selectedRowId,
  noResultsMessage = 'No results.',
  state: externalState = {},
  ...tableOptions
}) {
  const [sorting, setSorting] = React.useState([])
  const [columnSizing, setColumnSizing] = React.useState({})

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting,
      columnSizing,
      ...externalState,
    },
    onSortingChange: setSorting,
    onColumnSizingChange: setColumnSizing,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    columnResizeMode: 'onChange',
    ...tableOptions,
  })

  // Check if any column has been manually resized
  const isResized = Object.keys(columnSizing).length > 0

  return (
    <div className="flex-1 min-h-0 overflow-auto">
      <Table style={{ width: isResized ? Math.max(table.getCenterTotalSize(), '100%') : '100%', tableLayout: isResized ? 'fixed' : 'auto' }}>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <TableHead
                  key={header.id}
                  className="relative group"
                  style={isResized ? { width: header.getSize() } : { minWidth: header.column.columnDef.minSize }}
                >
                  {header.isPlaceholder
                    ? null
                    : flexRender(header.column.columnDef.header, header.getContext())}
                  {header.column.getCanResize() && (
                    <div
                      onMouseDown={header.getResizeHandler()}
                      onTouchStart={header.getResizeHandler()}
                      className={cn(
                        'absolute right-0 top-0 h-full w-1 cursor-col-resize select-none touch-none',
                        'opacity-0 group-hover:opacity-100 bg-border hover:bg-primary',
                        header.column.getIsResizing() && 'opacity-100 bg-primary'
                      )}
                    />
                  )}
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows?.length ? (
            table.getRowModel().rows.map((row) => (
              <TableRow
                key={row.id}
                className={onRowClick ? 'cursor-pointer' : ''}
                onClick={() => onRowClick?.(row.original)}
                data-state={selectedRowId && row.original.id === selectedRowId ? 'selected' : undefined}
              >
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id} style={isResized ? { width: cell.column.getSize() } : { minWidth: cell.column.columnDef.minSize }}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))
          ) : (
            <TableRow>
              <TableCell colSpan={columns.length} className="h-24 text-center text-muted-foreground">
                {noResultsMessage}
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  )
}
