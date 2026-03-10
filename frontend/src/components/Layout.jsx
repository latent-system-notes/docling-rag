import { useState, useEffect } from 'react'
import { Outlet, NavLink, useNavigate, useLocation, Link } from 'react-router-dom'
import { Users, Shield, FolderTree, LogOut, Search, FileText, Menu, PanelLeftClose, PanelLeftOpen, Settings, Upload, FolderOpen, Database } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { Sheet, SheetTrigger, SheetContent } from '@/components/ui/sheet'
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

const SIDEBAR_EXPANDED = 240  // w-60 = 15rem = 240px
const SIDEBAR_COLLAPSED = 48  // w-12 = 3rem = 48px

export default function Layout() {
  const navigate = useNavigate()
  const location = useLocation()
  const user = JSON.parse(localStorage.getItem('user') || '{}')
  const isAdmin = user.is_admin === true
  const [expanded, setExpanded] = useState(true)
  const [isMobile, setIsMobile] = useState(false)
  const [sheetOpen, setSheetOpen] = useState(false)

  useEffect(() => {
    const check = () => {
      const mobile = window.innerWidth <= 768
      setIsMobile(mobile)
    }
    check()
    window.addEventListener('resize', check)
    return () => window.removeEventListener('resize', check)
  }, [])

  const logout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    navigate('/login')
  }

  const handleNavClick = () => {
    if (isMobile) setSheetOpen(false)
  }

  // --- Nav items config ---
  const mainNav = [
    { to: '/search', icon: Search, label: 'Search' },
    { to: '/documents', icon: FileText, label: 'Documents' },
    { to: '/chunks', icon: Database, label: 'Chunks' },
  ]
  const adminNav = [
    { to: '/users', icon: Users, label: 'Users' },
    { to: '/groups', icon: Shield, label: 'Groups' },
    { to: '/permissions', icon: FolderTree, label: 'Permissions' },
    { to: '/files', icon: FolderOpen, label: 'Files' },
    { to: '/ingestion', icon: Upload, label: 'Ingestion' },
    { to: '/settings', icon: Settings, label: 'Settings' },
  ]

  // --- Expanded link class ---
  const navLinkClass = ({ isActive }) =>
    cn(
      'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
      isActive
        ? 'bg-accent text-accent-foreground'
        : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
    )

  // --- Collapsed icon link class (static, computed from location) ---
  const iconLinkClass = (to) =>
    cn(
      'flex items-center justify-center rounded-md h-9 w-9 transition-colors',
      location.pathname.startsWith(to)
        ? 'bg-accent text-accent-foreground'
        : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
    )

  // --- Full sidebar content (expanded, used in desktop expanded + mobile sheet) ---
  const expandedContent = (
    <>
      <div className="flex-1 overflow-y-auto px-3 py-4">
        <div className="space-y-1">
          {mainNav.map(({ to, icon: Icon, label }) => (
            <NavLink key={to} to={to} className={navLinkClass} onClick={handleNavClick}>
              <Icon className="h-4 w-4 shrink-0" /> {label}
            </NavLink>
          ))}
        </div>
        {isAdmin && (
          <>
            <div className="px-3 py-2 mt-6 mb-1">
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Administration
              </span>
            </div>
            <div className="space-y-1">
              {adminNav.map(({ to, icon: Icon, label }) => (
                <NavLink key={to} to={to} className={navLinkClass} onClick={handleNavClick}>
                  <Icon className="h-4 w-4 shrink-0" /> {label}
                </NavLink>
              ))}
            </div>
          </>
        )}
      </div>
      <div className="border-t p-4">
        <div className="flex items-center gap-2 mb-3">
          <Avatar>
            <AvatarFallback>{(user.username || 'U')[0].toUpperCase()}</AvatarFallback>
          </Avatar>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium truncate">{user.username || 'user'}</div>
          </div>
          {isAdmin && <Badge variant="success" className="text-[10px] px-1.5 py-0">admin</Badge>}
        </div>
        <Button variant="outline" size="sm" className="w-full gap-2" onClick={logout}>
          <LogOut className="h-3.5 w-3.5" /> Logout
        </Button>
      </div>
    </>
  )

  // --- Icon-only sidebar content (collapsed desktop) ---
  const collapsedContent = (
    <>
      <div className="flex-1 py-4 flex flex-col gap-1">
        {mainNav.map(({ to, icon: Icon, label }) => (
          <div key={to} className="flex items-center justify-center">
            <Tooltip>
              <TooltipTrigger asChild>
                <Link to={to} className={iconLinkClass(to)}>
                  <Icon className="h-4 w-4" />
                </Link>
              </TooltipTrigger>
              <TooltipContent side="right">{label}</TooltipContent>
            </Tooltip>
          </div>
        ))}
        {isAdmin && (
          <>
            <div className="my-2 border-t mx-2" />
            {adminNav.map(({ to, icon: Icon, label }) => (
              <div key={to} className="flex items-center justify-center">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Link to={to} className={iconLinkClass(to)}>
                      <Icon className="h-4 w-4" />
                    </Link>
                  </TooltipTrigger>
                  <TooltipContent side="right">{label}</TooltipContent>
                </Tooltip>
              </div>
            ))}
          </>
        )}
      </div>
      <div className="border-t py-3 flex flex-col items-center gap-2">
        <Tooltip>
          <TooltipTrigger asChild>
            <Avatar className="cursor-default">
              <AvatarFallback>{(user.username || 'U')[0].toUpperCase()}</AvatarFallback>
            </Avatar>
          </TooltipTrigger>
          <TooltipContent side="right">{user.username || 'user'}</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground" onClick={logout}>
              <LogOut className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="right">Logout</TooltipContent>
        </Tooltip>
      </div>
    </>
  )

  return (
    <div className="flex min-h-screen">
      {/* Mobile: Sheet sidebar */}
      {isMobile && (
        <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
          {!sheetOpen && (
            <SheetTrigger asChild>
              <Button variant="outline" size="icon" className="fixed top-3 left-3 z-[60]">
                <Menu className="h-5 w-5" />
              </Button>
            </SheetTrigger>
          )}
          <SheetContent side="left" className="w-60 p-0 flex flex-col">
            <div className="flex h-14 items-center px-4 border-b">
              <span className="text-lg font-semibold tracking-tight">Docling RAG</span>
            </div>
            {expandedContent}
          </SheetContent>
        </Sheet>
      )}

      {/* Desktop: sidebar (expanded or collapsed) */}
      {!isMobile && (
        <nav
          className={cn(
            'fixed inset-y-0 left-0 z-50 flex flex-col border-r bg-sidebar sidebar-transition',
            expanded ? 'w-60' : 'w-12'
          )}
        >
          {expanded ? (
            <>
              {/* Expanded header */}
              <div className="flex h-14 items-center justify-between px-4 border-b">
                <span className="text-lg font-semibold tracking-tight">Docling RAG</span>
                <Button variant="ghost" size="icon" onClick={() => setExpanded(false)} className="h-8 w-8">
                  <PanelLeftClose className="h-4 w-4" />
                </Button>
              </div>
              {expandedContent}
            </>
          ) : (
            <>
              {/* Collapsed header */}
              <div className="flex h-14 items-center justify-center border-b">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon" onClick={() => setExpanded(true)} className="h-8 w-8">
                      <PanelLeftOpen className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="right">Expand sidebar</TooltipContent>
                </Tooltip>
              </div>
              {collapsedContent}
            </>
          )}
        </nav>
      )}

      {/* Main content */}
      <main
        className={cn(
          'flex-1 bg-muted/40 sidebar-transition h-screen overflow-hidden',
          !isMobile ? (expanded ? 'ml-60' : 'ml-12') : 'ml-0',
          isMobile ? 'pt-14' : ''
        )}
      >
        <div className="flex flex-col h-full p-3 md:p-6 overflow-y-auto">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
