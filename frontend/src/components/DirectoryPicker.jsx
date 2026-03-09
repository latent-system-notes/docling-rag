import { useState, useCallback } from 'react'
import { api } from '../api/client'
import { FolderSearch, Folder, ArrowUp, Loader2 } from 'lucide-react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

export default function DirectoryPicker({ value, onSelect }) {
  const [open, setOpen] = useState(false)
  const [currentPath, setCurrentPath] = useState('')
  const [parent, setParent] = useState(null)
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const navigate = useCallback(async (path) => {
    setLoading(true)
    setError('')
    try {
      const data = await api.browseDirectories(path)
      setCurrentPath(data.path)
      setParent(data.parent)
      setItems(data.items)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleOpen = () => {
    setOpen(true)
    navigate(value || '')
  }

  const handleSelect = () => {
    onSelect(currentPath)
    setOpen(false)
  }

  return (
    <>
      <Button variant="outline" size="sm" onClick={handleOpen} type="button">
        <FolderSearch className="h-3.5 w-3.5" /> Browse
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Browse Server Directories</DialogTitle>
          </DialogHeader>

          <div className="rounded-md border overflow-hidden">
            <div className="flex items-center gap-2 p-2 border-b bg-muted/30">
              <Button
                variant="ghost"
                size="icon"
                className="shrink-0 h-8 w-8"
                disabled={parent === null && !currentPath}
                onClick={() => navigate(parent ?? '')}
              >
                <ArrowUp className="h-4 w-4" />
              </Button>
              <div className="flex-1 truncate font-mono text-sm px-2 py-1 rounded bg-background border">
                {currentPath || '(drives)'}
              </div>
            </div>

            {error && <p className="text-sm text-destructive px-3 pt-2">{error}</p>}

            <ScrollArea className="h-64">
              {loading ? (
                <div className="flex items-center justify-center h-full">
                  <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                </div>
              ) : items.length === 0 ? (
                <div className="flex items-center justify-center h-full">
                  <p className="text-sm text-muted-foreground">No subdirectories</p>
                </div>
              ) : (
                <div className="p-1.5 space-y-0.5">
                  {items.map((item) => (
                    <button
                      key={item.path}
                      onClick={() => navigate(item.path)}
                      className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-sm hover:bg-accent hover:text-accent-foreground transition-colors text-left"
                    >
                      <Folder className="h-4 w-4 shrink-0 text-muted-foreground" />
                      {item.name}
                    </button>
                  ))}
                </div>
              )}
            </ScrollArea>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
            <Button onClick={handleSelect} disabled={!currentPath}>
              Select This Directory
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
