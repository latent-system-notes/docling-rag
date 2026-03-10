import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { api } from '../api/client'
import { Folder, FileText, Upload, FolderPlus, Trash2, Download, Pencil, Home, AlertTriangle, Eye } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog'
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip'
import { Breadcrumb, BreadcrumbList, BreadcrumbItem, BreadcrumbLink, BreadcrumbPage, BreadcrumbSeparator } from '@/components/ui/breadcrumb'
import { DataTable, DataTablePagination, DataTableCard, SortableHeader } from '@/components/ui/data-table'
import { cn } from '@/lib/utils'
import FilePreview from '../components/FilePreview'

const EXT_COLORS = {
  '.pdf': 'text-red-500',
  '.docx': 'text-blue-500',
  '.xlsx': 'text-emerald-500',
  '.pptx': 'text-orange-500',
  '.md': 'text-purple-500',
  '.html': 'text-amber-500',
  '.htm': 'text-amber-500',
}

const PAGE_SIZES = [10, 20, 50, 100]

function formatSize(bytes) {
  if (bytes === 0) return '-'
  const units = ['B', 'KB', 'MB', 'GB']
  let i = 0
  while (bytes >= 1024 && i < units.length - 1) { bytes /= 1024; i++ }
  return `${bytes.toFixed(i > 0 ? 1 : 0)} ${units[i]}`
}

function formatDate(iso) {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export default function FilesPage() {
  const [currentPath, setCurrentPath] = useState('')
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState('')
  const fileInputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)
  const [showMkdir, setShowMkdir] = useState(false)
  const [newFolderName, setNewFolderName] = useState('')
  const [renameItem, setRenameItem] = useState(null)
  const [renameName, setRenameName] = useState('')
  const [deleteItem, setDeleteItem] = useState(null)
  const [previewItem, setPreviewItem] = useState(null)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)

  const fetchItems = useCallback(async (path) => {
    setLoading(true)
    try {
      const data = await api.listFiles(path)
      setItems(data.items)
      setPage(1)
    } catch (e) {
      toast.error(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchItems(currentPath) }, [currentPath, fetchItems])

  const navigateTo = (path) => setCurrentPath(path)

  const breadcrumbs = () => {
    if (!currentPath) return []
    return currentPath.split('/').filter(Boolean)
  }

  const breadcrumbPath = (index) => breadcrumbs().slice(0, index + 1).join('/')

  const handleUpload = async (fileList) => {
    if (!fileList || fileList.length === 0) return
    setUploading(true)
    setUploadProgress(`Uploading ${fileList.length} file(s)...`)
    try {
      const result = await api.uploadFiles(currentPath, fileList)
      const parts = []
      if (result.uploaded.length > 0) parts.push(`${result.uploaded.length} file(s) uploaded`)
      if (result.errors.length > 0) parts.push(`${result.errors.length} error(s)`)
      const msg = parts.join(', ') + (result.errors.length > 0 ? ': ' + result.errors.join('; ') : '')
      if (result.errors.length > 0) toast.warning(msg)
      else toast.success(msg)
      await fetchItems(currentPath)
    } catch (e) {
      toast.error(e.message)
    } finally {
      setUploading(false)
      setUploadProgress('')
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const onFileInputChange = (e) => handleUpload(e.target.files)

  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    handleUpload(e.dataTransfer.files)
  }

  const handleMkdir = async () => {
    if (!newFolderName.trim()) return
    try {
      await api.createDirectory(currentPath, newFolderName.trim())
      setShowMkdir(false)
      setNewFolderName('')
      toast.success(`Folder "${newFolderName.trim()}" created`)
      await fetchItems(currentPath)
    } catch (e) {
      toast.error(e.message)
    }
  }

  const handleDelete = async () => {
    if (!deleteItem) return
    try {
      await api.deleteFile(deleteItem.path)
      setDeleteItem(null)
      toast.success(`"${deleteItem.name}" deleted`)
      await fetchItems(currentPath)
    } catch (e) {
      toast.error(e.message)
    }
  }

  const handleRename = async () => {
    if (!renameItem || !renameName.trim()) return
    try {
      await api.renameFile(renameItem.path, renameName.trim())
      setRenameItem(null)
      setRenameName('')
      toast.success('Renamed successfully')
      await fetchItems(currentPath)
    } catch (e) {
      toast.error(e.message)
    }
  }

  const handleDownload = (item) => {
    const token = localStorage.getItem('token')
    const url = `/api/files/download?path=${encodeURIComponent(item.path)}`
    fetch(url, { headers: { 'Authorization': `Bearer ${token}` } })
      .then(res => {
        if (!res.ok) throw new Error('Download failed')
        return res.blob()
      })
      .then(blob => {
        const a = document.createElement('a')
        a.href = URL.createObjectURL(blob)
        a.download = item.name
        a.click()
        URL.revokeObjectURL(a.href)
      })
      .catch(e => toast.error(e.message))
  }

  const crumbs = breadcrumbs()

  const totalItems = items.length
  const pagedItems = items.slice((page - 1) * pageSize, page * pageSize)

  const columns = useMemo(() => [
    {
      accessorKey: 'name',
      header: ({ column }) => <SortableHeader column={column} title="Name" />,
      cell: ({ row }) => {
        const item = row.original
        return (
          <div
            className={cn("flex items-center gap-2", "cursor-pointer")}
            onClick={(e) => { e.stopPropagation(); if (item.type === 'directory') navigateTo(item.path); else setPreviewItem(item) }}
          >
            {item.type === 'directory'
              ? <Folder className="h-4 w-4 text-amber-500 shrink-0" />
              : <FileText className={cn("h-4 w-4 shrink-0", EXT_COLORS[item.extension] || "text-muted-foreground")} />
            }
            <span className={cn(item.type === 'directory' ? 'font-medium' : '', 'hover:underline')}>{item.name}</span>
            {item.extension && (
              <Badge variant="secondary" className="text-[10px] px-1.5 py-0">{item.extension}</Badge>
            )}
          </div>
        )
      },
      size: 350,
    },
    {
      accessorKey: 'size',
      header: ({ column }) => <SortableHeader column={column} title="Size" className="justify-end" />,
      cell: ({ row }) => (
        <div className="text-right text-sm text-muted-foreground">
          {row.original.type === 'file' ? formatSize(row.original.size) : '-'}
        </div>
      ),
      size: 100,
    },
    {
      accessorKey: 'modified',
      header: ({ column }) => <SortableHeader column={column} title="Modified" className="justify-end" />,
      cell: ({ getValue }) => <div className="text-right text-sm text-muted-foreground">{formatDate(getValue())}</div>,
      size: 180,
    },
    {
      id: 'actions',
      header: () => <div className="text-right">Actions</div>,
      enableSorting: false,
      enableResizing: false,
      size: 165,
      cell: ({ row }) => {
        const item = row.original
        return (
          <div className="flex gap-1 justify-end">
            {item.type === 'file' && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon" className="h-8 w-8" onClick={(e) => { e.stopPropagation(); setPreviewItem(item) }}>
                    <Eye className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Preview</TooltipContent>
              </Tooltip>
            )}
            {item.type === 'file' && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon" className="h-8 w-8" onClick={(e) => { e.stopPropagation(); handleDownload(item) }}>
                    <Download className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Download</TooltipContent>
              </Tooltip>
            )}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8" onClick={(e) => { e.stopPropagation(); setRenameItem(item); setRenameName(item.name) }}>
                  <Pencil className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Rename</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive" onClick={(e) => { e.stopPropagation(); setDeleteItem(item) }}>
                  <Trash2 className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Delete</TooltipContent>
            </Tooltip>
          </div>
        )
      },
    },
  ], [currentPath])

  return (
    <div className="flex flex-col h-full gap-4 overflow-hidden">
      <div className="flex items-center justify-between flex-wrap gap-3 shrink-0">
        <h1 className="text-2xl font-semibold tracking-tight">Files</h1>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
            <Upload className="h-4 w-4" /> Upload
          </Button>
          <Button size="sm" variant="outline" onClick={() => { setShowMkdir(true); setNewFolderName('') }}>
            <FolderPlus className="h-4 w-4" /> New Folder
          </Button>
        </div>
        <input ref={fileInputRef} type="file" multiple className="hidden" onChange={onFileInputChange} />
      </div>

      <Breadcrumb className="shrink-0">
        <BreadcrumbList>
          <BreadcrumbItem>
            <Tooltip>
              <TooltipTrigger asChild>
                <BreadcrumbLink onClick={() => navigateTo('')}>
                  <Home className="h-4 w-4" />
                </BreadcrumbLink>
              </TooltipTrigger>
              <TooltipContent>Root</TooltipContent>
            </Tooltip>
          </BreadcrumbItem>
          {crumbs.map((name, i) => (
            <span key={i} className="contents">
              <BreadcrumbSeparator />
              <BreadcrumbItem>
                {i === crumbs.length - 1 ? (
                  <BreadcrumbPage>{name}</BreadcrumbPage>
                ) : (
                  <BreadcrumbLink onClick={() => navigateTo(breadcrumbPath(i))}>
                    {name}
                  </BreadcrumbLink>
                )}
              </BreadcrumbItem>
            </span>
          ))}
        </BreadcrumbList>
      </Breadcrumb>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={cn(
          "rounded-lg transition-all text-center flex items-center justify-center shrink-0",
          dragOver ? "border-2 border-dashed border-primary bg-primary/5 min-h-[80px] p-4" : "",
          uploading ? "p-4" : ""
        )}
      >
        {dragOver && <span className="text-primary font-medium">Drop files here to upload</span>}
        {uploading && <span className="text-sm text-muted-foreground">{uploadProgress}</span>}
      </div>

      {loading ? (
        <div className="p-8 text-center text-muted-foreground">Loading...</div>
      ) : items.length === 0 ? (
        <div className="p-8 text-center text-muted-foreground">
          This folder is empty. Upload files or create a subfolder.
        </div>
      ) : (
        <DataTableCard>
          <DataTable
            columns={columns}
            data={pagedItems}
            noResultsMessage="No files found"
          />
          <DataTablePagination
            total={totalItems}
            page={page}
            pageSize={pageSize}
            onPageChange={setPage}
            onPageSizeChange={setPageSize}
            pageSizes={PAGE_SIZES}
            noun="item"
          />
        </DataTableCard>
      )}

      {/* Mkdir Dialog */}
      <Dialog open={showMkdir} onOpenChange={setShowMkdir}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>New Folder</DialogTitle>
          </DialogHeader>
          <div className="space-y-2">
            <Label>Folder name</Label>
            <Input
              autoFocus
              value={newFolderName}
              onChange={e => setNewFolderName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleMkdir()}
              placeholder="Folder name"
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowMkdir(false)}>Cancel</Button>
            <Button onClick={handleMkdir} disabled={!newFolderName.trim()}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Rename Dialog */}
      <Dialog open={!!renameItem} onOpenChange={() => setRenameItem(null)}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Rename</DialogTitle>
            <DialogDescription>Renaming: {renameItem?.name}</DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <Label>New name</Label>
            <Input
              autoFocus
              value={renameName}
              onChange={e => setRenameName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleRename()}
              placeholder="New name"
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenameItem(null)}>Cancel</Button>
            <Button onClick={handleRename} disabled={!renameName.trim() || renameName.trim() === renameItem?.name}>Rename</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Preview Dialog */}
      <FilePreview
        open={!!previewItem}
        onClose={() => setPreviewItem(null)}
        filePath={previewItem?.path}
        fileName={previewItem?.name}
      />

      {/* Delete Dialog */}
      <Dialog open={!!deleteItem} onOpenChange={() => setDeleteItem(null)}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-destructive" />
              Delete {deleteItem?.type === 'directory' ? 'Folder' : 'File'}
            </DialogTitle>
          </DialogHeader>
          <p className="text-sm">
            Are you sure you want to delete <strong>{deleteItem?.name}</strong>?
          </p>
          {deleteItem?.type === 'directory' && (
            <p className="text-sm text-destructive">
              This will delete the folder and all its contents.
            </p>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteItem(null)}>Cancel</Button>
            <Button variant="destructive" onClick={handleDelete}>Delete</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
