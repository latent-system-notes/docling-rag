import { useState, useCallback } from 'react'
import { api } from '../api/client'
import { FolderSearch, Folder, File, ArrowUp, X, Loader2 } from 'lucide-react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'

export default function FoldersPicker({ value, onSelect }) {
  const [open, setOpen] = useState(false)
  const [currentPath, setCurrentPath] = useState('')
  const [parentPath, setParentPath] = useState(null)
  const [basePath, setBasePath] = useState('')
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState(new Set())

  const parseValue = (v) => {
    if (!v) return new Set()
    return new Set(v.split('|').map(s => s.trim()).filter(Boolean))
  }

  const navigate = useCallback(async (path) => {
    setLoading(true)
    setError('')
    try {
      const data = await api.browseDocumentFolders(path)
      setCurrentPath(data.relative_path)
      setParentPath(data.parent)
      setBasePath(data.base)
      setItems(data.items)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleOpen = () => {
    setSelected(parseValue(value))
    setOpen(true)
    navigate('')
  }

  const toggleFolder = (relativePath) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(relativePath)) {
        next.delete(relativePath)
      } else {
        next.add(relativePath)
      }
      return next
    })
  }

  const removeFolder = (relativePath) => {
    setSelected(prev => {
      const next = new Set(prev)
      next.delete(relativePath)
      return next
    })
  }

  const handleApply = () => {
    onSelect([...selected].sort().join('|'))
    setOpen(false)
  }

  return (
    <>
      <Button variant="outline" size="sm" onClick={handleOpen} type="button">
        <FolderSearch className="h-3.5 w-3.5" /> Select Folders
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Select Folders to Include</DialogTitle>
          </DialogHeader>

          <div className="rounded-md border overflow-hidden">
            {/* Path bar */}
            <div className="flex items-center gap-2 p-2 border-b bg-muted/30">
              <Button
                variant="ghost"
                size="icon"
                className="shrink-0 h-8 w-8"
                disabled={parentPath === null}
                onClick={() => navigate(parentPath ?? '')}
              >
                <ArrowUp className="h-4 w-4" />
              </Button>
              <div className="flex-1 truncate font-mono text-sm px-2 py-1 rounded bg-background border">
                {basePath}{currentPath ? `/${currentPath}` : ''}
              </div>
            </div>

            {/* Selected tags */}
            {selected.size > 0 && (
              <div className="flex flex-wrap gap-1 p-2 border-b bg-muted/10">
                {[...selected].sort().map(f => (
                  <Badge key={f} variant="secondary" className="text-xs gap-1 pr-1">
                    {f}
                    <button
                      onClick={() => removeFolder(f)}
                      className="ml-0.5 rounded-full hover:bg-muted p-0.5"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}

            {error && <p className="text-sm text-destructive px-3 pt-2">{error}</p>}

            <ScrollArea className="h-64">
              {loading ? (
                <div className="flex items-center justify-center h-full">
                  <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                </div>
              ) : items.length === 0 ? (
                <div className="flex items-center justify-center h-full">
                  <p className="text-sm text-muted-foreground">Empty folder</p>
                </div>
              ) : (
                <div className="p-1.5 space-y-0.5">
                  {items.map((item) => (
                    item.type === 'directory' ? (
                      <div
                        key={item.relative_path}
                        className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-sm hover:bg-accent hover:text-accent-foreground transition-colors"
                      >
                        <Checkbox
                          checked={selected.has(item.relative_path)}
                          onCheckedChange={() => toggleFolder(item.relative_path)}
                        />
                        <button
                          onClick={() => navigate(item.relative_path)}
                          className="flex items-center gap-2 flex-1 text-left min-w-0"
                        >
                          <Folder className="h-4 w-4 shrink-0 text-muted-foreground" />
                          <span className="truncate">{item.name}</span>
                        </button>
                      </div>
                    ) : (
                      <div
                        key={item.name}
                        className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-sm text-muted-foreground/60"
                      >
                        <div className="h-4 w-4 shrink-0" />
                        <File className="h-4 w-4 shrink-0" />
                        <span className="truncate">{item.name}</span>
                      </div>
                    )
                  ))}
                </div>
              )}
            </ScrollArea>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
            <Button onClick={handleApply}>
              Apply Selection{selected.size > 0 && ` (${selected.size})`}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
