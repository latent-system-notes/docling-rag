import { useState, useEffect, useRef, useCallback } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'
import { renderAsync } from 'docx-preview'
import * as XLSX from 'xlsx'
import { api } from '../api/client'
import { Download, X, ChevronLeft, ChevronRight, FileWarning, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent } from '@/components/ui/dialog'
import { cn } from '@/lib/utils'

pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs'

const PREVIEW_EXTENSIONS = {
  pdf: ['.pdf'],
  docx: ['.docx'],
  xlsx: ['.xlsx'],
  image: ['.png', '.jpg', '.jpeg'],
  tiff: ['.tiff', '.tif'],
  audio: ['.wav', '.mp3'],
  html: ['.html', '.htm'],
  markdown: ['.md'],
}

function getPreviewType(fileName) {
  const ext = (fileName || '').toLowerCase().match(/\.[^.]+$/)?.[0] || ''
  for (const [type, exts] of Object.entries(PREVIEW_EXTENSIONS)) {
    if (exts.includes(ext)) return type
  }
  return 'unsupported'
}

// --- PDF Renderer ---
function PdfRenderer({ url }) {
  const [numPages, setNumPages] = useState(null)
  const [pageNumber, setPageNumber] = useState(1)

  return (
    <div className="flex flex-col items-center gap-3 h-full overflow-auto">
      <Document file={url} onLoadSuccess={({ numPages: n }) => setNumPages(n)} loading={<Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />}>
        <Page pageNumber={pageNumber} width={750} />
      </Document>
      {numPages && (
        <div className="flex items-center gap-3 pb-3 shrink-0">
          <Button variant="outline" size="icon" className="h-8 w-8" disabled={pageNumber <= 1} onClick={() => setPageNumber(p => p - 1)}>
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-sm text-muted-foreground">Page {pageNumber} of {numPages}</span>
          <Button variant="outline" size="icon" className="h-8 w-8" disabled={pageNumber >= numPages} onClick={() => setPageNumber(p => p + 1)}>
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  )
}

// --- DOCX Renderer ---
function DocxRenderer({ blob }) {
  const containerRef = useRef(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    if (!blob || !containerRef.current) return
    containerRef.current.innerHTML = ''
    renderAsync(blob, containerRef.current).catch(() => setError(true))
  }, [blob])

  if (error) return <UnsupportedMessage message="Failed to render DOCX" />
  return <div ref={containerRef} className="bg-white text-black p-4 rounded overflow-auto h-full" />
}

// --- XLSX Renderer ---
function XlsxRenderer({ blob }) {
  const [sheets, setSheets] = useState([])
  const [activeSheet, setActiveSheet] = useState(0)
  const [error, setError] = useState(false)

  useEffect(() => {
    if (!blob) return
    blob.arrayBuffer().then(buf => {
      try {
        const wb = XLSX.read(buf, { type: 'array' })
        const parsed = wb.SheetNames.map(name => {
          const sheet = wb.Sheets[name]
          const rows = XLSX.utils.sheet_to_json(sheet, { header: 1 })
          return { name, rows: rows.slice(0, 1000) }
        })
        setSheets(parsed)
      } catch {
        setError(true)
      }
    })
  }, [blob])

  if (error) return <UnsupportedMessage message="Failed to parse spreadsheet" />
  if (sheets.length === 0) return <Loader2 className="h-8 w-8 animate-spin text-muted-foreground m-auto" />

  const current = sheets[activeSheet]

  return (
    <div className="flex flex-col h-full gap-2">
      {sheets.length > 1 && (
        <div className="flex gap-1 shrink-0 overflow-x-auto pb-1">
          {sheets.map((s, i) => (
            <Button key={i} variant={i === activeSheet ? 'default' : 'outline'} size="sm" className="text-xs h-7" onClick={() => setActiveSheet(i)}>
              {s.name}
            </Button>
          ))}
        </div>
      )}
      <div className="overflow-auto flex-1 rounded border">
        <table className="w-full text-sm border-collapse">
          <tbody>
            {current.rows.map((row, ri) => (
              <tr key={ri} className={ri === 0 ? 'bg-muted font-medium' : 'border-t border-border'}>
                {(Array.isArray(row) ? row : []).map((cell, ci) => (
                  <td key={ci} className="px-3 py-1.5 border-r border-border whitespace-nowrap">{cell != null ? String(cell) : ''}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {current.rows.length >= 1000 && <p className="text-xs text-muted-foreground text-center">Showing first 1,000 rows</p>}
    </div>
  )
}

// --- Markdown Renderer ---
function MarkdownRenderer({ blob }) {
  const [text, setText] = useState('')

  useEffect(() => {
    if (!blob) return
    blob.text().then(setText)
  }, [blob])

  return <pre className="whitespace-pre-wrap text-sm p-4 overflow-auto h-full font-mono">{text}</pre>
}

// --- Unsupported ---
function UnsupportedMessage({ message, onDownload }) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 h-full text-muted-foreground">
      <FileWarning className="h-16 w-16" />
      <p className="text-lg">{message || 'Preview not available for this file type'}</p>
      {onDownload && (
        <Button onClick={onDownload}>
          <Download className="h-4 w-4 mr-2" /> Download File
        </Button>
      )}
    </div>
  )
}

export default function FilePreview({ open, onClose, filePath, fileName }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [blobData, setBlobData] = useState(null) // { blob, url }

  const previewType = getPreviewType(fileName)
  const ext = (fileName || '').toLowerCase().match(/\.[^.]+$/)?.[0] || ''

  // Fetch blob when dialog opens
  useEffect(() => {
    if (!open || !filePath) return
    let revoked = false
    setLoading(true)
    setError(null)
    setBlobData(null)

    api.getPreviewBlob(filePath)
      .then(data => {
        if (!revoked) setBlobData(data)
      })
      .catch(e => {
        if (!revoked) setError(e.message)
      })
      .finally(() => {
        if (!revoked) setLoading(false)
      })

    return () => {
      revoked = true
    }
  }, [open, filePath])

  // Cleanup blob URL on close
  useEffect(() => {
    if (!open && blobData?.url) {
      URL.revokeObjectURL(blobData.url)
      setBlobData(null)
    }
  }, [open])

  const handleDownload = useCallback(() => {
    if (!blobData?.url) return
    const a = document.createElement('a')
    a.href = blobData.url
    a.download = fileName || 'file'
    a.click()
  }, [blobData, fileName])

  const renderContent = () => {
    if (loading) {
      return <div className="flex items-center justify-center h-full"><Loader2 className="h-8 w-8 animate-spin text-muted-foreground" /></div>
    }
    if (error) {
      return <UnsupportedMessage message={`Error: ${error}`} onDownload={blobData ? handleDownload : undefined} />
    }
    if (!blobData) return null

    switch (previewType) {
      case 'pdf':
        return <PdfRenderer url={blobData.url} />
      case 'docx':
        return <DocxRenderer blob={blobData.blob} />
      case 'xlsx':
        return <XlsxRenderer blob={blobData.blob} />
      case 'image':
        return (
          <div className="flex items-center justify-center h-full overflow-auto p-4">
            <img src={blobData.url} alt={fileName} className="max-w-full max-h-full object-contain" />
          </div>
        )
      case 'tiff':
        return (
          <div className="flex items-center justify-center h-full overflow-auto p-4">
            <img
              src={blobData.url}
              alt={fileName}
              className="max-w-full max-h-full object-contain"
              onError={(e) => { e.target.style.display = 'none'; e.target.parentElement.innerHTML = '<div class="flex flex-col items-center gap-4 text-muted-foreground"><p class="text-lg">TIFF preview not supported by this browser</p></div>' }}
            />
          </div>
        )
      case 'audio':
        return (
          <div className="flex items-center justify-center h-full">
            <audio controls src={blobData.url} className="w-full max-w-lg" />
          </div>
        )
      case 'html':
        return <iframe src={blobData.url} sandbox="" className="w-full h-full rounded bg-white" title="HTML Preview" />
      case 'markdown':
        return <MarkdownRenderer blob={blobData.blob} />
      default:
        return <UnsupportedMessage message="Preview not available for this file type" onDownload={handleDownload} />
    }
  }

  return (
    <Dialog open={open} onOpenChange={(v) => { if (!v) onClose() }}>
      <DialogContent className="max-w-5xl w-[90vw] h-[85vh] flex flex-col p-0 gap-0 [&>button]:hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <span className="font-medium truncate">{fileName}</span>
            {ext && <Badge variant="secondary" className="text-[10px] px-1.5 py-0 shrink-0">{ext}</Badge>}
          </div>
          <div className="flex items-center gap-1 shrink-0">
            <Button variant="ghost" size="sm" onClick={handleDownload} disabled={!blobData}>
              <Download className="h-4 w-4 mr-1" /> Download
            </Button>
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
        {/* Body */}
        <div className="flex-1 overflow-hidden p-4">
          {renderContent()}
        </div>
      </DialogContent>
    </Dialog>
  )
}
